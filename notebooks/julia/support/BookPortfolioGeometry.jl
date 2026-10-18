module BookPortfolioGeometry

using LinearAlgebra, Statistics
using RiskLabAI.Optimization: constrained_minimum_variance

export global_minimum_variance, mean_return_weighted, inverse_variance, long_only_minimum_variance, norm_limited_minimum_variance, target_return_portfolio, public_normal_samples

function global_minimum_variance(covariance)
    direction = covariance \ ones(size(covariance,1))
    return direction ./ sum(direction)
end

function mean_return_weighted(covariance, expected_return)
    direction = covariance \ vec(expected_return)
    return direction ./ sum(direction)
end

function inverse_variance(covariance)
    direction = 1 ./ diag(covariance)
    return direction ./ sum(direction)
end

"""Refine a feasible portfolio by solving its active-face optimality equations."""
function _refine_long_only_variance(covariance, weights)
    weights = max.(Float64.(weights), 0.0)
    weights ./= sum(weights)
    free = trues(length(weights))
    for iteration in 1:2000
        indices = findall(free)
        direction = covariance[indices, indices] \ ones(length(indices))
        candidate = zeros(length(weights))
        candidate[indices] = direction ./ sum(direction)
        negative = indices[candidate[indices] .< 0.0]
        if !isempty(negative)
            ratios = weights[negative] ./ (weights[negative] .- candidate[negative])
            position = argmin(ratios)
            weights .+= ratios[position] .* (candidate .- weights)
            weights = max.(weights, 0.0)
            pivot = negative[position]
            weights[pivot] = 0.0
            free[pivot] = false
            continue
        end
        weights = candidate
        marginal = covariance*weights
        multiplier = dot(weights, marginal)
        inactive = findall(.!free)
        if isempty(inactive) || minimum(marginal[inactive]) >= multiplier
            return weights
        end
        free[inactive[argmin(marginal[inactive])]] = true
    end
    error("Long-only optimality refinement did not converge")
end


function long_only_minimum_variance(covariance)
    return _refine_long_only_variance(covariance, constrained_minimum_variance(covariance).weights)
end

"""Solve the stated squared-norm-constrained variance problem through its ridge multiplier."""
function norm_limited_minimum_variance(covariance, squared_norm_limit)
    count = size(covariance,1)
    squared_norm_limit >= 1/count || throw(ArgumentError("norm limit is infeasible"))
    squared_norm_limit == 1/count && return fill(1/count,count)
    unrestricted = global_minimum_variance(covariance)
    dot(unrestricted,unrestricted) <= squared_norm_limit && return unrestricted
    scaled = covariance ./ maximum(abs,covariance)
    lower,upper = 0.0,1.0
    while sum(abs2,global_minimum_variance(scaled + upper*I)) > squared_norm_limit
        upper *= 2
    end
    weights = global_minimum_variance(scaled + upper*I)
    for _ in 1:100
        middle = (lower+upper)/2
        candidate = global_minimum_variance(scaled + middle*I)
        if sum(abs2,candidate) > squared_norm_limit
            lower = middle
        else
            upper,weights = middle,candidate
        end
    end
    return weights
end

function target_return_portfolio(covariance, expected_return, target)
    constraints = hcat(ones(length(expected_return)),expected_return)
    inverse_constraints = covariance \ constraints
    return inverse_constraints * ((constraints' * inverse_constraints) \ [1.0,target])
end

"""Apply the public RandomState multivariate-normal SVD construction to supplied normal innovations."""
function public_normal_samples(expected_return,covariance,innovations;orientation=nothing)
    factorization = svd(covariance)
    directions = copy(factorization.Vt)
    if orientation !== nothing
        length(orientation)==length(factorization.S) && all(value->value in (-1,1),orientation) || throw(ArgumentError("one sign is required per SVD direction"))
        for index in axes(directions,1)
            current = sign(directions[index,argmax(abs.(directions[index,:]))])
            directions[index,:] .*= orientation[index]*current
        end
    end
    factor = sqrt.(factorization.S) .* directions
    return innovations * factor .+ reshape(expected_return,1,:)
end

end
