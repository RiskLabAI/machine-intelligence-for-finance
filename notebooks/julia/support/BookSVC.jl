module BookSVC

using LinearAlgebra, Statistics
using ..BookEstimators: Standardization, SVCModel, fit_standardization, transform

function pairwise_total(values,first_index=1,count=length(values))
    if count<8
        total = -0.0
        for index in first_index:first_index+count-1
            total += values[index]
        end
        return total
    elseif count<=128
        partial = [values[first_index+offset] for offset in 0:7]
        boundary = count-count%8
        for offset in 8:8:boundary-1, lane in 1:8
            partial[lane] += values[first_index+offset+lane-1]
        end
        total = ((partial[1]+partial[2])+(partial[3]+partial[4]))+((partial[5]+partial[6])+(partial[7]+partial[8]))
        for offset in boundary:count-1
            total += values[first_index+offset]
        end
        return total
    end
    half = count÷2
    half -= half%8
    return pairwise_total(values,first_index,half)+pairwise_total(values,first_index+half,count-half)
end

function fit_scaler(values)
    count = size(values,1)
    center = [pairwise_total(view(values,:,column))/count for column in axes(values,2)]
    variance = map(axes(values,2)) do column
        shifted = values[:,column].-center[column]
        correction = pairwise_total(shifted)
        return (pairwise_total(shifted.^2)-correction^2/count)/count
    end
    bound = count*eps(Float64).*variance .+ (count*eps(Float64).*center).^2
    return Standardization(center,[value<=limit ? 1. : sqrt(value) for (value,limit) in zip(variance,bound)])
end

function kernel_cache(values,kernel,gamma,labels)
    packed = permutedims(values)
    diagonal = [BLAS.dot(view(packed,:,row),view(packed,:,row)) for row in axes(values,1)]
    count = length(labels)
    cache = Matrix{Float32}(undef,count,count)
    for column in 1:count, row in 1:count
        product = BLAS.dot(view(packed,:,row),view(packed,:,column))
        value = kernel==:linear ? product : exp(-gamma*(diagonal[row]+diagonal[column]-2.0*product))
        cache[row,column] = labels[row]*labels[column]*value
    end
    return cache,kernel==:linear ? diagonal : ones(count)
end

function working_pair(order,active,labels,alpha,bounds,gradient,cache,diagonal)
    highest,lowest = -Inf,Inf
    first_index,second_index = 0,0
    best_change = Inf
    @inbounds for position in 1:active
        index = order[position]
        eligible = labels[index]>0 ? alpha[index]<bounds[index] : alpha[index]>0
        value = -labels[index]*gradient[index]
        if eligible && value>=highest
            highest,first_index = value,index
        end
    end
    @inbounds for position in 1:active
        index = order[position]
        eligible = labels[index]>0 ? alpha[index]>0 : alpha[index]<bounds[index]
        if eligible
            value = -labels[index]*gradient[index]
            lowest = min(lowest,value)
            difference = highest-value
            if difference>0 && first_index!=0
                curvature = diagonal[first_index]+diagonal[index]-2.0*labels[first_index]*labels[index]*cache[index,first_index]
                change = -difference*difference/(curvature>0 ? curvature : 1e-12)
                if change<=best_change
                    best_change,second_index = change,index
                end
            end
        end
    end
    return first_index,second_index,highest-lowest
end

function reconstruct!(gradient,bar,order,active,alpha,bounds,cache)
    active==length(order) && return
    @inbounds for position in active+1:length(order)
        index = order[position]
        gradient[index] = bar[index]-1.
    end
    @inbounds for position in 1:active
        index = order[position]
        if 0<alpha[index]<bounds[index]
            for other_position in active+1:length(order)
                other = order[other_position]
                gradient[other] += alpha[index]*cache[other,index]
            end
        end
    end
end

function shrinkable(index,upper,lower,labels,alpha,bounds,gradient)
    if alpha[index]>=bounds[index]
        return -gradient[index]>(labels[index]>0 ? upper : lower)
    elseif alpha[index]<=0
        return gradient[index]>(labels[index]>0 ? lower : upper)
    end
    return false
end

function shrink!(order,active,restored,labels,alpha,bounds,gradient,bar,cache,tolerance)
    upper,lower = -Inf,-Inf
    @inbounds for position in 1:active
        index = order[position]
        score = -labels[index]*gradient[index]
        (labels[index]>0 ? alpha[index]<bounds[index] : alpha[index]>0) && (upper=max(upper,score))
        (labels[index]>0 ? alpha[index]>0 : alpha[index]<bounds[index]) && (lower=max(lower,-score))
    end
    if !restored && upper+lower<=10tolerance
        reconstruct!(gradient,bar,order,active,alpha,bounds,cache)
        active,restored = length(order),true
    end
    position = 1
    @inbounds while position<=active
        if shrinkable(order[position],upper,lower,labels,alpha,bounds,gradient)
            active -= 1
            while active>=position
                if !shrinkable(order[active+1],upper,lower,labels,alpha,bounds,gradient)
                    order[position],order[active+1] = order[active+1],order[position]
                    break
                end
                active -= 1
            end
        end
        position += 1
    end
    return active,restored
end

function update_pair!(alpha,first_index,second_index,labels,bounds,gradient,cache,diagonal)
    i,j = first_index,second_index
    ai,aj,ci,cj = alpha[i],alpha[j],bounds[i],bounds[j]
    if labels[i]!=labels[j]
        curvature = diagonal[i]+diagonal[j]+2.0*cache[j,i]
        delta = (-gradient[i]-gradient[j])/(curvature>0 ? curvature : 1e-12)
        difference = ai-aj
        ai,aj = ai+delta,aj+delta
        if difference>0
            aj<0 && ((ai,aj)=(difference,0.))
        else
            ai<0 && ((ai,aj)=(0.,-difference))
        end
        if difference>ci-cj
            ai>ci && ((ai,aj)=(ci,ci-difference))
        else
            aj>cj && ((ai,aj)=(cj+difference,cj))
        end
    else
        curvature = diagonal[i]+diagonal[j]-2.0*cache[j,i]
        delta = (gradient[i]-gradient[j])/(curvature>0 ? curvature : 1e-12)
        total = ai+aj
        ai,aj = ai-delta,aj+delta
        if total>ci
            ai>ci && ((ai,aj)=(ci,total-ci))
        else
            aj<0 && ((ai,aj)=(total,0.))
        end
        if total>cj
            aj>cj && ((ai,aj)=(total-cj,cj))
        else
            ai<0 && ((ai,aj)=(0.,total))
        end
    end
    alpha[i],alpha[j] = ai,aj
end

"""
Fit the public binary weighted C-SVC with class-grouped working sets, a Float32
kernel cache, Float64 gradients, shrinking and the public 1e-3 stopping rule.
Standardization is unweighted over all training rows. Zero-weight rows enter
standardization and gamma estimation but are excluded from dual optimization.
This adapter supports the public linear/RBF experiments without calibration.
"""
function fit_svc(x::AbstractMatrix,y::AbstractVector;sample_weight=ones(length(y)),C::Real=1.,kernel::Symbol=:rbf,gamma=:scale,standardize::Bool=true,tol::Real=1e-3,max_iter::Integer=150000000,shrinking::Bool=true)
    size(x,1)==length(y)==length(sample_weight) || throw(DimensionMismatch("SVC observations must align"))
    classes = sort(unique(y))
    length(classes)==2 || throw(ArgumentError("the book adapter requires two classes"))
    weights = Float64.(sample_weight)
    all(isfinite,x) && all(value->isfinite(value)&&value>=0,weights) && sum(weights)>0 || throw(ArgumentError("features and weights must be finite with positive weight mass"))
    C>0 && isfinite(C) && tol>0 && max_iter>0 || throw(ArgumentError("C, tolerance and iteration limit must be positive"))
    kernel in (:linear,:rbf) || throw(ArgumentError("supported kernels are linear and RBF"))
    scaler = standardize ? fit_scaler(x) : Standardization(zeros(size(x,2)),ones(size(x,2)))
    transformed = transform(scaler,x)
    flattened = vec(permutedims(transformed))
    overall_mean = pairwise_total(flattened)/length(flattened)
    variance = pairwise_total((flattened.-overall_mean).^2)/length(flattened)
    gamma_value = gamma==:scale ? (variance>0 ? 1/(size(x,2)*variance) : 1.) : gamma==:auto ? 1/size(x,2) : Float64(gamma)
    gamma_value>0 && isfinite(gamma_value) || throw(ArgumentError("gamma must be positive and finite"))
    groups = [findall((y.==class).&(weights.>0)) for class in classes]
    all(!isempty,groups) || throw(ArgumentError("both classes require positive weights"))
    positions = vcat(groups...)
    values = transformed[positions,:]
    labels = vcat(ones(Int,length(groups[1])),-ones(Int,length(groups[2])))
    bounds = C.*weights[positions]
    cache,diagonal = kernel_cache(values,kernel,gamma_value,labels)
    count = length(labels)
    alpha,gradient,bar = zeros(count),-ones(count),zeros(count)
    order = collect(1:count)
    active,restored,counter = count,false,min(count,1000)+1
    iterations,converged,violation = 0,false,Inf
    while iterations<max_iter
        counter -= 1
        if counter==0
            counter = min(count,1000)
            shrinking && ((active,restored)=shrink!(order,active,restored,labels,alpha,bounds,gradient,bar,cache,tol))
        end
        first_index,second_index,violation = working_pair(order,active,labels,alpha,bounds,gradient,cache,diagonal)
        if violation<tol || second_index==0
            reconstruct!(gradient,bar,order,active,alpha,bounds,cache)
            active = count
            first_index,second_index,violation = working_pair(order,active,labels,alpha,bounds,gradient,cache,diagonal)
            if violation<tol || second_index==0
                converged = true
                break
            end
            counter = 1
        end
        iterations += 1
        first_before,second_before = alpha[first_index],alpha[second_index]
        first_upper,second_upper = first_before>=bounds[first_index],second_before>=bounds[second_index]
        update_pair!(alpha,first_index,second_index,labels,bounds,gradient,cache,diagonal)
        first_change,second_change = alpha[first_index]-first_before,alpha[second_index]-second_before
        @inbounds for position in 1:active
            index = order[position]
            gradient[index] += cache[index,first_index]*first_change+cache[index,second_index]*second_change
        end
        for (index,was_upper) in ((first_index,first_upper),(second_index,second_upper))
            is_upper = alpha[index]>=bounds[index]
            if is_upper!=was_upper
                change = (was_upper ? -1. : 1.)*bounds[index]
                @inbounds for other in 1:count
                    bar[other] += change*cache[other,index]
                end
            end
        end
    end
    reconstruct!(gradient,bar,order,active,alpha,bounds,cache)
    _,_,violation = working_pair(order,count,labels,alpha,bounds,gradient,cache,diagonal)
    lower,upper,free_sum,free_count = -Inf,Inf,0.,0
    for index in order
        score = labels[index]*gradient[index]
        if alpha[index]>=bounds[index]
            labels[index]<0 ? (upper=min(upper,score)) : (lower=max(lower,score))
        elseif alpha[index]<=0
            labels[index]>0 ? (upper=min(upper,score)) : (lower=max(lower,score))
        else
            free_sum += score
            free_count += 1
        end
    end
    intercept = free_count>0 ? free_sum/free_count : (lower+upper)/2
    signed = -alpha.*labels
    gram = values*values'
    if kernel==:rbf
        norms = sum(abs2,values;dims=2)
        gram = exp.(-gamma_value.*max.(norms.+norms'.-2 .* gram,0.))
    end
    raw = gram*signed
    quadratic = dot(signed,raw)/2
    dual = sum(alpha)-quadratic
    primal = quadratic+sum(bounds.*max.(0.,1 .+ labels.*(raw.+intercept)))
    return SVCModel(values,alpha,signed,intercept,classes,scaler,kernel,gamma_value,dual,primal,violation,converged,iterations)
end

end
