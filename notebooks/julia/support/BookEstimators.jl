module BookEstimators

using LinearAlgebra
using Statistics
using Optim

struct Standardization
    center::Vector{Float64}
    scale::Vector{Float64}
end

function fit_standardization(x::AbstractMatrix)
    count = size(x, 1)
    count > 0 || throw(ArgumentError("standardization requires observations"))
    size(x, 2) > 0 && all(isfinite, x) || throw(ArgumentError("standardization requires finite features"))
    center = vec(mean(x; dims = 1))
    variance = vec(sum(abs2, x .- center'; dims = 1))/count
    bound = count*eps(Float64).*variance .+ (count*eps(Float64).*center).^2
    scale = [value <= tolerance ? 1. : sqrt(value) for (value, tolerance) in zip(variance, bound)]
    return Standardization(center, scale)
end

transform(scaler::Standardization, x::AbstractMatrix) = (x .- scaler.center') ./ scaler.scale'

function _prepare(x, y, weights, scaled)
    size(x, 1) == length(y) == length(weights) || throw(DimensionMismatch("features, labels and observation weights must align"))
    all(isfinite, x) && all(weight -> isfinite(weight) && weight >= 0, weights) || throw(ArgumentError("features must be finite and observation weights nonnegative and finite"))
    classes = sort!(unique(y))
    length(classes) == 2 || throw(ArgumentError("this book adapter requires exactly two classes"))
    all(label -> sum(weights[y .== label]) > 0, classes) || throw(ArgumentError("each class must have positive total weight"))
    scaler = scaled ? fit_standardization(x) : Standardization(zeros(size(x, 2)), ones(size(x, 2)))
    return Matrix{Float64}(transform(scaler, x)), Int.(y .== classes[2]), classes, scaler
end

_sigmoid(value) = value >= 0 ? inv(1+exp(-value)) : exp(value)/(1+exp(value))
_softplus(value) = max(value, 0.)+log1p(exp(-abs(value)))

struct LogisticModel{C}
    coefficients::Vector{Float64}
    intercept::Float64
    classes::Vector{C}
    scaler::Standardization
    objective::Float64
    gradient_norm::Float64
    converged::Bool
    iterations::Int
end

"""
Fit the book's binary L2 logistic objective with an unpenalized intercept.
Standardization, when requested, is fitted without observation weights.
`class_weight=:balanced` uses weighted class frequencies, matching the preserved
sklearn supplier. `C` multiplies the observation-weighted likelihood relative
to the coefficient penalty; probabilities retain the sorted class order.
"""
function fit_logistic(x::AbstractMatrix, y::AbstractVector;
    sample_weight = ones(length(y)), C::Real = 1., class_weight = nothing,
    standardize::Bool = false, tol::Real = 1e-4, max_iter::Integer = 1000)
    isfinite(C) && C > 0 || throw(ArgumentError("C must be positive and finite"))
    isfinite(tol) && tol > 0 && max_iter > 0 || throw(ArgumentError("tol and max_iter must be positive"))
    weights = Float64.(sample_weight)
    values, labels, classes, scaler = _prepare(x, y, weights, standardize)
    if class_weight == :balanced
        total = sum(weights)
        masses = [sum(weights[labels .== label]) for label in 0:1]
        weights .*= [total/(2*masses[label+1]) for label in labels]
    elseif class_weight !== nothing
        throw(ArgumentError("supported class weights are nothing and :balanced"))
    end
    mass = sum(weights)
    p = size(values, 2)
    objective = parameters -> begin
        scores = values*view(parameters, 1:p) .+ parameters[end]
        (sum(weights .* (_softplus.(scores) .- labels.*scores))+.5*sum(abs2, view(parameters, 1:p))/C)/mass
    end
    gradient! = (output, parameters) -> begin
        scores = values*view(parameters, 1:p) .+ parameters[end]
        residual = weights.*(_sigmoid.(scores) .- labels)
        output[1:p] = (values'*residual .+ view(parameters, 1:p)./C)./mass
        output[end] = sum(residual)/mass
        output
    end
    method = Optim.LBFGS(m = 10, linesearch = Optim.LineSearches.MoreThuente(f_tol = 1e-3, gtol = .9, x_tol = .1, maxfev = 50))
    result = Optim.optimize(objective, gradient!, zeros(p+1), method,
        Optim.Options(g_abstol = tol, f_reltol = 64*eps(Float64), iterations = max_iter))
    parameters = Optim.minimizer(result)
    gradient = gradient!(zeros(p+1), parameters)
    return LogisticModel(parameters[1:p], parameters[end], classes, scaler,
        objective(parameters)*mass, maximum(abs, gradient), Optim.converged(result), Optim.iterations(result))
end

decision_function(model::LogisticModel, x::AbstractMatrix) = transform(model.scaler, x)*model.coefficients .+ model.intercept

function predict_proba(model::LogisticModel, x::AbstractMatrix)
    probability = _sigmoid.(decision_function(model, x))
    return hcat(1 .- probability, probability)
end

predict(model::LogisticModel, x::AbstractMatrix) = [model.classes[1+(score > 0)] for score in decision_function(model, x)]

function _kernel_matrix(x, y, kernel, gamma)
    gram = x*y'
    kernel == :linear && return gram
    distance = sum(abs2, x; dims = 2) .+ sum(abs2, y; dims = 2)' .- 2 .* gram
    return exp.(-gamma .* max.(distance, 0.))
end

struct SVCModel{C}
    training::Matrix{Float64}
    alpha::Vector{Float64}
    signed_alpha::Vector{Float64}
    intercept::Float64
    classes::Vector{C}
    scaler::Standardization
    kernel::Symbol
    gamma::Float64
    dual_objective::Float64
    primal_objective::Float64
    kkt_violation::Float64
    converged::Bool
    iterations::Int
end

function _smo_pair(labels, alpha, bounds, gradient, kernel)
    upper_score = -Inf
    first_index = 0
    for index in eachindex(labels)
        active = labels[index] > 0 ? alpha[index] < bounds[index] : alpha[index] > 0
        score = -labels[index]*gradient[index]
        if active && score >= upper_score
            upper_score, first_index = score, index
        end
    end
    first_index == 0 && return 0, 0, 0.
    lower_score = Inf
    best_change = Inf
    second_index = 0
    for index in eachindex(labels)
        active = labels[index] > 0 ? alpha[index] > 0 : alpha[index] < bounds[index]
        if active
            score = -labels[index]*gradient[index]
            lower_score = min(lower_score, score)
            difference = upper_score-score
            if difference > 0
                curvature = max(kernel[first_index, first_index]+kernel[index, index]-2*kernel[first_index, index], 1e-12)
                change = -difference^2/curvature
                if change <= best_change
                    best_change, second_index = change, index
                end
            end
        end
    end
    return first_index, second_index, upper_score-lower_score
end

function _svc_intercept(labels, alpha, bounds, gradient)
    lower, upper = -Inf, Inf
    free = Float64[]
    for index in eachindex(labels)
        bounds[index] == 0 && continue
        score = labels[index]*gradient[index]
        if alpha[index] <= 0
            labels[index] > 0 ? (upper = min(upper, score)) : (lower = max(lower, score))
        elseif alpha[index] >= bounds[index]
            labels[index] > 0 ? (lower = max(lower, score)) : (upper = min(upper, score))
        else
            push!(free, score)
        end
    end
    return isempty(free) ? -(lower+upper)/2 : -mean(free)
end

"""
Fit binary linear/RBF C-SVC with one bound `C*sample_weight[i]` per observation.
Native SMO solves the public dual objective without changing the kernel or
observation weights. The optional scaler is unweighted and fitted on training
rows. `gamma=:scale` uses the variance of the entire transformed training matrix.
The public book uses `tol=1e-3`; kernel-cache roundoff and solver stopping can
produce small differences near a decision boundary. No probability calibration
or multiclass construction is implied by this binary adapter.
"""
function fit_svc(x::AbstractMatrix, y::AbstractVector;
    sample_weight = ones(length(y)), C::Real = 1., kernel::Symbol = :rbf,
    gamma = :scale, standardize::Bool = false, tol::Real = 1e-3,
    max_iter::Integer = 500000)
    isfinite(C) && C > 0 || throw(ArgumentError("C must be positive and finite"))
    kernel in (:linear, :rbf) || throw(ArgumentError("supported kernels are :linear and :rbf"))
    isfinite(tol) && tol > 0 && max_iter > 0 || throw(ArgumentError("tol and max_iter must be positive"))
    weights = Float64.(sample_weight)
    values, encoded, classes, scaler = _prepare(x, y, weights, standardize)
    labels = 2 .* encoded .- 1
    variance = var(vec(values); corrected = false)
    gamma_value = gamma == :scale ? (variance > 0 ? 1/(size(values, 2)*variance) : 1.) : gamma == :auto ? 1/size(values, 2) : Float64(gamma)
    isfinite(gamma_value) && gamma_value > 0 || throw(ArgumentError("gamma must be positive and finite"))
    kernel_values = _kernel_matrix(values, values, kernel, gamma_value)
    bounds = C.*weights
    alpha = zeros(length(y))
    gradient = -ones(length(y))
    converged = false
    violation = Inf
    iteration = 0
    for counter in 1:max_iter
        first_index, second_index, violation = _smo_pair(labels, alpha, bounds, gradient, kernel_values)
        if violation <= tol || first_index == 0 || second_index == 0
            converged = violation <= tol
            break
        end
        difference = -labels[first_index]*gradient[first_index]+labels[second_index]*gradient[second_index]
        curvature = max(kernel_values[first_index, first_index]+kernel_values[second_index, second_index]-2*kernel_values[first_index, second_index], 1e-12)
        first_bound = labels[first_index] > 0 ? bounds[first_index]-alpha[first_index] : alpha[first_index]
        second_bound = labels[second_index] > 0 ? alpha[second_index] : bounds[second_index]-alpha[second_index]
        step = min(difference/curvature, first_bound, second_bound)
        first_before, second_before = alpha[first_index], alpha[second_index]
        alpha[first_index] = clamp(first_before+labels[first_index]*step, 0., bounds[first_index])
        alpha[second_index] = clamp(second_before-labels[second_index]*step, 0., bounds[second_index])
        first_change = labels[first_index]*(alpha[first_index]-first_before)
        second_change = labels[second_index]*(alpha[second_index]-second_before)
        gradient .+= labels.*(first_change.*view(kernel_values, :, first_index) .+ second_change.*view(kernel_values, :, second_index))
        iteration = counter
        if counter % 1000 == 0
            gradient = labels.*(kernel_values*(alpha.*labels)) .- 1
        end
    end
    signed = alpha.*labels
    raw = kernel_values*signed
    gradient = labels.*raw .- 1
    _, _, violation = _smo_pair(labels, alpha, bounds, gradient, kernel_values)
    intercept = _svc_intercept(labels, alpha, bounds, gradient)
    quadratic = .5*dot(signed, raw)
    dual = sum(alpha)-quadratic
    primal = quadratic+sum(bounds.*max.(0., 1 .- labels.*(raw .+ intercept)))
    return SVCModel(values, alpha, signed, intercept, classes, scaler, kernel,
        gamma_value, dual, primal, violation, converged && violation <= tol*(1+1e-6), iteration)
end

decision_function(model::SVCModel, x::AbstractMatrix) = _kernel_matrix(transform(model.scaler, x), model.training, model.kernel, model.gamma)*model.signed_alpha .+ model.intercept
predict(model::SVCModel, x::AbstractMatrix) = [model.classes[1+(score > 0)] for score in decision_function(model, x)]

export Standardization, fit_standardization, transform, fit_logistic,
    LogisticModel, SVCModel, fit_svc, predict, predict_proba, decision_function

end
