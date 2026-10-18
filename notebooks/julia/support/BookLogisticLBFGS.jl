module BookLogisticLBFGS

using LinearAlgebra, Statistics, RiskLabAI
using ..BookEstimators

function line_search(evaluate,initial_value,initial_gradient,initial_step;maximum_step=1e10,max_evaluations=50)
    best = (0.,initial_value,initial_gradient)
    opposite = best
    step,lower,upper = initial_step,0.,5initial_step
    width,previous_width = maximum_step,2maximum_step
    bracketed,modified = false,true
    decrease_gradient = 1e-3*initial_gradient
    for evaluation in 1:max_evaluations
        value,gradient = evaluate(step)
        isfinite(value) && isfinite(gradient) || return (step=step,converged=false)
        sufficient = initial_value+step*decrease_gradient
        modified && value<=sufficient && gradient>=0 && (modified=false)
        warning = bracketed && (step<=lower || step>=upper || upper-lower<=.1upper)
        warning |= step==maximum_step && value<=sufficient && gradient<=decrease_gradient
        warning |= step==0 && (value>sufficient || gradient>=decrease_gradient)
        wolfe = value<=sufficient && abs(gradient)<=-.9initial_gradient
        (warning || wolfe) && return (step=step,converged=true)
        if modified && value<=best[2] && value>sufficient
            transformed(point) = (point[1],point[2]-point[1]*decrease_gradient,point[3]-decrease_gradient)
            recover(point) = (point[1],point[2]+point[1]*decrease_gradient,point[3]+decrease_gradient)
            best,opposite,step,bracketed = RiskLabAI.Data._mp_interpolate(transformed(best),transformed(opposite),transformed((step,value,gradient)),bracketed,(lower,upper))
            best,opposite = recover(best),recover(opposite)
        else
            best,opposite,step,bracketed = RiskLabAI.Data._mp_interpolate(best,opposite,(step,value,gradient),bracketed,(lower,upper))
        end
        if bracketed
            abs(opposite[1]-best[1])>=.66previous_width && (step=best[1]+.5*(opposite[1]-best[1]))
            previous_width,width = width,abs(opposite[1]-best[1])
            lower,upper = minmax(best[1],opposite[1])
        else
            lower,upper = step+1.1*(step-best[1]),step+4*(step-best[1])
        end
        step = clamp(step,0.,maximum_step)
        bracketed && (step<=lower || step>=upper || upper-lower<=.1upper) && (step=best[1])
    end
    return (step=step,converged=false)
end

function limited_memory_minimize(evaluate,dimension;tol=1e-4,max_iter=1000,memory=10)
    parameters = zeros(dimension)
    value,gradient = evaluate(parameters)
    displacements,changes = Vector{Float64}[],Vector{Float64}[]
    scale = 1.
    for iteration in 1:max_iter
        maximum(abs,gradient)<=tol && return (;parameters,value,gradient,converged=true,iterations=iteration-1)
        direction = copy(gradient)
        coefficients = zeros(length(displacements))
        for index in length(displacements):-1:1
            coefficients[index] = dot(displacements[index],direction)/dot(changes[index],displacements[index])
            direction .-= coefficients[index].*changes[index]
        end
        direction .*= scale
        for index in eachindex(displacements)
            beta = dot(changes[index],direction)/dot(changes[index],displacements[index])
            direction .+= (coefficients[index]-beta).*displacements[index]
        end
        direction .*= -1
        initial_step = iteration==1 ? min(inv(norm(direction)),1e10) : 1.
        evaluate_line = step -> begin
            trial_value,trial_gradient = evaluate(parameters.+step.*direction)
            return trial_value,dot(trial_gradient,direction)
        end
        search = line_search(evaluate_line,value,dot(gradient,direction),initial_step)
        search.converged || return (;parameters,value,gradient,converged=false,iterations=iteration-1)
        next_parameters = parameters.+search.step.*direction
        next_value,next_gradient = evaluate(next_parameters)
        if maximum(abs,next_gradient)<=tol || value-next_value<=64eps(Float64)*max(abs(value),abs(next_value),1.)
            return (parameters=next_parameters,value=next_value,gradient=next_gradient,converged=true,iterations=iteration)
        end
        displacement,change = next_parameters.-parameters,next_gradient.-gradient
        curvature = dot(displacement,change)
        if curvature>eps(Float64)*(-dot(gradient,displacement))
            push!(displacements,displacement)
            push!(changes,change)
            if length(displacements)>memory
                popfirst!(displacements)
                popfirst!(changes)
            end
            scale = curvature/dot(change,change)
        end
        parameters,value,gradient = next_parameters,next_value,next_gradient
    end
    return (;parameters,value,gradient,converged=false,iterations=max_iter)
end

"""
Fit the fixed binary book objective using unbounded limited-memory BFGS.
Preserve the public initial normalized step, ten correction pairs, safeguarded
line search, maximum fifty line evaluations and gradient/function stopping rules.
Observation standardization is unweighted; balanced class weights and the
unpenalized intercept retain the existing native estimator contract.
"""
function fit_logistic(x,y;sample_weight=ones(length(y)),C=1.,class_weight=nothing,standardize=false,tol=1e-4,max_iter=1000)
    isfinite(C) && C>0 || throw(ArgumentError("C must be positive and finite"))
    isfinite(tol) && tol>0 && max_iter>0 || throw(ArgumentError("positive tolerance and iteration count are required"))
    weights = Float64.(sample_weight)
    values,labels,classes,scaler = BookEstimators._prepare(x,y,weights,standardize)
    if class_weight==:balanced
        total = sum(weights)
        masses = [sum(weights[labels.==label]) for label in 0:1]
        weights .*= [total/(2masses[label+1]) for label in labels]
    elseif class_weight!==nothing
        throw(ArgumentError("supported class weights are nothing and :balanced"))
    end
    mass,p = sum(weights),size(values,2)
    evaluate = parameters -> begin
        scores = values*view(parameters,1:p).+parameters[end]
        value = (sum(weights.*(BookEstimators._softplus.(scores).-labels.*scores))+.5sum(abs2,view(parameters,1:p))/C)/mass
        residual = weights.*(BookEstimators._sigmoid.(scores).-labels)
        gradient = vcat((values'*residual.+view(parameters,1:p)./C)./mass,sum(residual)/mass)
        return value,gradient
    end
    result = limited_memory_minimize(evaluate,p+1;tol,max_iter)
    return BookEstimators.LogisticModel(result.parameters[1:p],result.parameters[end],classes,scaler,result.value*mass,maximum(abs,result.gradient),result.converged,result.iterations)
end

export fit_logistic

end
