module BookMarkovRegression

using LinearAlgebra, Statistics, Optim

function _validated(values, parameters)
    length(values) > 0 && all(isfinite, values) || throw(ArgumentError("returns must be nonempty and finite"))
    length(parameters) == 6 && all(isfinite, parameters) || throw(ArgumentError("six finite parameters are required"))
    all(value -> 0 <= value <= 1, parameters[1:2]) && all(>(0), parameters[5:6]) || throw(ArgumentError("transition probabilities and variances are invalid"))
    return Float64.(values), Float64.(parameters)
end

transition_matrix(parameters) = [parameters[1] parameters[2]; 1-parameters[1] 1-parameters[2]]

function stationary_probabilities(parameters)
    a, b = parameters[1:2]
    denominator = 1-a+b
    probabilities = denominator == 0 ? [.5, .5] : [b/denominator, (1-a)/denominator]
    return max.(probabilities, 1e-20)
end

"""Filter two Gaussian states with constant column-stochastic transitions and stationary initialization."""
function filter_markov(values, parameters)
    returns, params = _validated(values, parameters)
    count = length(returns)
    transition = transition_matrix(params)
    initial = stationary_probabilities(params)
    previous = initial
    predicted, filtered = zeros(count, 2), zeros(count, 2)
    contributions = zeros(count)
    for index in eachindex(returns)
        prior = transition*previous
        logs = [-.5*(log(2pi*params[state+4])+(returns[index]-params[state+2])^2/params[state+4])+log(prior[state]) for state in 1:2]
        peak = maximum(logs)
        masses = exp.(logs.-peak)
        contribution = peak+log(sum(masses))
        current = masses./sum(masses)
        predicted[index, :], filtered[index, :] = prior, current
        contributions[index] = contribution
        previous = current
    end
    return (; parameters=params, transition, initial, predicted, filtered, contributions, loglikelihood=sum(contributions))
end

"""Smooth using all supplied observations; smoothed values are retrospective."""
function smooth_markov(values, parameters)
    result = filter_markov(values, parameters)
    count = length(values)
    smoothed = copy(result.filtered)
    joint = zeros(2, 2, count)
    for index in count-1:-1:1
        ratio = [result.predicted[index+1, state] == 0 ? 0. : smoothed[index+1, state]/result.predicted[index+1, state] for state in 1:2]
        smoothed[index, :] = result.filtered[index, :].*(result.transition'*ratio)
        smoothed[index, :] ./= sum(smoothed[index, :])
    end
    for index in 1:count
        previous = index == 1 ? result.initial : result.filtered[index-1, :]
        ratio = [result.predicted[index, state] == 0 ? 0. : smoothed[index, state]/result.predicted[index, state] for state in 1:2]
        joint[:, :, index] = result.transition.*(ratio*previous')
        joint[:, :, index] ./= sum(joint[:, :, index])
    end
    return (; result..., smoothed, joint)
end

function markov_start_parameters(values)
    length(values) >= 2 && all(isfinite, values) || throw(ArgumentError("at least two finite returns are required"))
    variance = var(values; corrected=false)
    variance > 0 || throw(ArgumentError("returns must have positive variance"))
    return [.5, .5, 0., mean(values)/2, variance/10, variance]
end

function markov_em_start(values; iterations=5)
    iterations >= 0 || throw(ArgumentError("EM iterations must be nonnegative"))
    parameters = markov_start_parameters(values)
    likelihoods = Float64[]
    history = [copy(parameters)]
    for iteration in 1:iterations
        result = smooth_markov(values, parameters)
        push!(likelihoods, result.loglikelihood)
        masses = vec(sum(result.smoothed; dims=1))
        probabilities = [sum(result.joint[1, state, :])/masses[state] for state in 1:2]
        probabilities = [value > 1 ? value/(value+1e-6) : value for value in probabilities]
        means = [dot(result.smoothed[:, state], values)/masses[state] for state in 1:2]
        variances = [dot(result.smoothed[:, state], (values.-means[state]).^2)/masses[state] for state in 1:2]
        parameters = vcat(probabilities, means, variances)
        push!(history, copy(parameters))
        if iteration >= 2
            delta = 2*(likelihoods[end]-likelihoods[end-1])/abs(likelihoods[end]+likelihoods[end-1])
            delta > 0 || break
        end
    end
    return (; parameters, history, likelihoods)
end

_sigmoid(value) = value >= 0 ? inv(1+exp(-value)) : exp(value)/(1+exp(value))
_transform(unconstrained) = vcat(_sigmoid.(unconstrained[1:2]), unconstrained[3:4], unconstrained[5:6].^2)
_untransform(parameters) = vcat(log.(parameters[1:2]./(1 .- parameters[1:2])), parameters[3:4], sqrt.(parameters[5:6]))

function markov_objective_gradient(values, unconstrained)
    parameters = _transform(unconstrained)
    a, b = parameters[1:2]
    da, db = a*(1-a), b*(1-b)
    denominator = 1-a+b
    denominator > 0 && all(>(0), parameters[5:6]) || return Inf, fill(NaN, 6)
    previous = [b/denominator, (1-a)/denominator]
    derivative = zeros(6)
    derivative[1] = b/denominator^2*da
    derivative[2] = (1-a)/denominator^2*db
    total, score = 0., zeros(6)
    for observation in values
        prior = [a*previous[1]+b*previous[2], (1-a)*previous[1]+(1-b)*previous[2]]
        prior_derivative = (a-b).*derivative
        prior_derivative[1] += da*previous[1]
        prior_derivative[2] += db*previous[2]
        logs = zeros(2)
        log_derivatives = zeros(2, 6)
        for state in 1:2
            residual, variance = observation-parameters[state+2], parameters[state+4]
            logs[state] = -.5*(log(2pi*variance)+residual^2/variance)+log(prior[state])
            log_derivatives[state, :] = (state == 1 ? 1 : -1).*prior_derivative./prior[state]
            log_derivatives[state, state+2] += residual/variance
            log_derivatives[state, state+4] += (residual^2/variance-1)*unconstrained[state+4]/variance
        end
        peak = maximum(logs)
        masses = exp.(logs.-peak)
        current = masses./sum(masses)
        contribution = peak+log(sum(masses))
        contribution_derivative = vec(current'*log_derivatives)
        derivative = current[1].*(log_derivatives[1, :].-contribution_derivative)
        previous = current
        total += contribution
        score .+= contribution_derivative
    end
    return -total/length(values), -score./length(values)
end

"""
Fit the book's two-state intercept-only Gaussian Markov regression.
Use its deterministic starting values, five EM initialization steps, stationary
initial probabilities, squared-scale variance transformation and full-likelihood
BFGS objective. No state ordering is imposed; labels remain exchangeable.
"""
function fit_markov_regression(values; max_iter=1000, tol=1e-5, em_iter=5)
    max_iter > 0 && isfinite(tol) && tol > 0 || throw(ArgumentError("positive iteration and tolerance settings are required"))
    start = markov_em_start(values; iterations=em_iter)
    initial = _untransform(start.parameters)
    objective = parameters -> first(markov_objective_gradient(values, parameters))
    gradient! = (output, parameters) -> copyto!(output, last(markov_objective_gradient(values, parameters)))
    method = Optim.BFGS(linesearch=Optim.LineSearches.MoreThuente(f_tol=1e-4, gtol=.9, maxfev=50))
    result = Optim.optimize(objective, gradient!, initial, method, Optim.Options(g_abstol=tol, iterations=max_iter))
    unconstrained = Optim.minimizer(result)
    parameters = _transform(unconstrained)
    value, gradient = markov_objective_gradient(values, unconstrained)
    return (; parameters, loglikelihood=-value*length(values), gradient_norm=maximum(abs, gradient),
        converged=Optim.converged(result), iterations=Optim.iterations(result), em_start=start.parameters,
        filtered=filter_markov(values, parameters).filtered)
end

export filter_markov, smooth_markov, fit_markov_regression, transition_matrix,
    stationary_probabilities, markov_start_parameters, markov_em_start,
    markov_objective_gradient

end
