module BookHierarchy

using LinearAlgebra, Statistics
using RiskLabAI.Optimization: distance_corr, quasi_diagonal, recursive_bisection, hrp, inverse_variance_weights

export linkage_tree, hierarchy, cophenetic_correlation, max_clusters, adjusted_agreement, clustered_covariance, generator_samples, portfolio_set, ordered_weights

function single_tree(distance)
    n = size(distance,1)
    visited = falses(n)
    nearest = fill(Inf,n)
    starts,ends,heights = Int[],Int[],Float64[]
    current = 1
    for step in 1:n-1
        visited[current] = true
        candidate,height = 0,Inf
        for item in 1:n
            visited[item] && continue
            nearest[item] = min(nearest[item],distance[current,item])
            if nearest[item]<height
                candidate,height = item,nearest[item]
            end
        end
        push!(starts,current)
        push!(ends,candidate)
        push!(heights,height)
        current = candidate
    end
    parents = collect(1:2n-1)
    counts = ones(Int,2n-1)
    tree = zeros(n-1,4)
    for (step,edge) in enumerate(sortperm(heights;alg=Base.Sort.MergeSort))
        left,right = starts[edge],ends[edge]
        while parents[left]!=left
            left = parents[left]
        end
        while parents[right]!=right
            right = parents[right]
        end
        merged = n+step
        counts[merged] = counts[left]+counts[right]
        tree[step,:] = [min(left,right)-1,max(left,right)-1,heights[edge],counts[merged]]
        parents[left] = parents[right] = merged
    end
    return tree
end

function linkage_tree(distance,method="single")
    n = size(distance,1)
    size(distance,2)==n && n>1 && all(isfinite,distance) || throw(ArgumentError("Invalid distance matrix"))
    method=="single" && return single_tree(distance)
    method in ("complete","average") || throw(ArgumentError("Unsupported linkage"))
    values = fill(Inf,2n-1,2n-1)
    values[1:n,1:n] = distance
    counts = ones(Int,2n-1)
    active = collect(1:n)
    tree = zeros(n-1,4)
    for step in 1:n-1
        best,left,right = Inf,0,0
        for a in 1:length(active)-1, b in a+1:length(active)
            i,j = active[a],active[b]
            if values[i,j]<best
                best,left,right = values[i,j],i,j
            end
        end
        merged = n+step
        counts[merged] = counts[left]+counts[right]
        tree[step,:] = [left-1,right-1,best,counts[merged]]
        filter!(i->i!=left && i!=right,active)
        for i in active
            updated = method=="complete" ? max(values[left,i],values[right,i]) : (counts[left]*values[left,i]+counts[right]*values[right,i])/counts[merged]
            values[i,merged] = values[merged,i] = updated
        end
        push!(active,merged)
    end
    return tree
end

function hierarchy(correlation,method="single")
    distance = distance_corr(correlation)
    distance = (distance+distance')/2
    distance[diagind(distance)] .= 0.0
    tree = linkage_tree(distance,method)
    return distance,tree,quasi_diagonal(tree)
end

function tree_members(tree)
    n = size(tree,1)+1
    groups = [i<=n ? [i] : Int[] for i in 1:2n-1]
    for step in 1:n-1
        left,right = Int.(tree[step,1:2]).+1
        groups[n+step] = vcat(groups[left],groups[right])
    end
    return groups
end

function cophenetic_correlation(tree,distance)
    n = size(distance,1)
    values = zeros(n,n)
    groups = tree_members(tree)
    for step in 1:n-1
        left,right = Int.(tree[step,1:2]).+1
        for i in groups[left],j in groups[right]
            values[i,j] = values[j,i] = tree[step,3]
        end
    end
    return cor([values[i,j] for i in 1:n-1 for j in i+1:n],[distance[i,j] for i in 1:n-1 for j in i+1:n])
end

function max_clusters(tree,k)
    n = size(tree,1)+1
    1<=k<=n || throw(ArgumentError("Invalid cluster count"))
    labels = collect(1:n)
    k==n && return labels
    threshold = tree[n-k,3]
    groups = tree_members(tree)
    for step in 1:n-1
        tree[step,3]<=threshold || continue
        labels[groups[n+step]] .= minimum(groups[n+step])
    end
    return labels
end

function adjusted_agreement(a,b)
    pairs(n) = n*(n-1)/2
    numerator = sum(pairs(count(i->a[i]==x && b[i]==y,eachindex(a))) for x in unique(a),y in unique(b))
    first_sum = sum(pairs(count(==(x),a)) for x in unique(a))
    second_sum = sum(pairs(count(==(y),b)) for y in unique(b))
    expected = first_sum*second_sum/pairs(length(a))
    denominator = (first_sum+second_sum)/2-expected
    return denominator==0 ? 1.0 : (numerator-expected)/denominator
end

function clustered_covariance(n,clusters,within,between)
    n%clusters==0 || throw(ArgumentError("Unequal clusters"))
    correlation = fill(Float64(between),n,n)
    for block in 0:clusters-1
        indices = block*(n÷clusters).+(1:n÷clusters)
        correlation[indices,indices] .= within
    end
    correlation[diagind(correlation)] .= 1.0
    volatility = 0.12 .+ collect(0:n-1).*((0.28-0.12)/(n-1))
    volatility[end] = 0.28
    return correlation.*(volatility*volatility'),correlation
end

function generator_samples(covariance,innovations,orientation)
    decomposition = svd(covariance)
    vectors = copy(decomposition.U)
    for column in axes(vectors,2)
        sign_here = sign(vectors[argmax(abs.(vectors[:,column])),column])
        vectors[:,column] .*= orientation[column]*sign_here
    end
    return innovations*(vectors.*sqrt.(decomposition.S)')'
end

function ordered_weights(covariance,order)
    weights = zeros(length(order))
    weights[order] = recursive_bisection(covariance,order)
    return weights
end

function portfolio_set(covariance,correlation)
    n = size(covariance,1)
    direction = covariance\ones(n)
    return ["Equal weight"=>fill(1/n,n),"Inverse variance"=>inverse_variance_weights(covariance),"HRP"=>hrp(covariance,correlation),"Minimum variance"=>direction/sum(direction)]
end

end
