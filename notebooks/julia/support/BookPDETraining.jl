using Optimisers

function _training_inner_sum(values)
    length(values) < 8 && return _neural_four_sum(values)
    limit = 8div(length(values), 8)
    partial = Float32[_neural_four_sum(view(values, lane:8:limit)) for lane in 1:8]
    total = 0f0
    for index in limit+1:length(values)
        total += values[index]
    end
    for value in partial
        total += value
    end
    return total
end

_training_row_sum(values) = reshape(Float32[_training_inner_sum(view(values, row, :)) for row in axes(values, 1)], :, 1)
_training_mean(values) = _training_inner_sum(vec(values)) / Float32(length(values))
_training_expand(values, count) = repeat(values, count, 1)

function NeuralChainRules.rrule(::typeof(_training_expand), values, count)
    return _training_expand(values, count), tangent -> (NeuralChainRules.NoTangent(), reshape(_neural_batch_sum(NeuralChainRules.unthunk(tangent)), size(values)), NeuralChainRules.NoTangent())
end

function NeuralChainRules.rrule(::typeof(_training_row_sum), values)
    return _training_row_sum(values), tangent -> (NeuralChainRules.NoTangent(), repeat(NeuralChainRules.unthunk(tangent), 1, size(values, 2)))
end

function NeuralChainRules.rrule(::typeof(_training_mean), values)
    return _training_mean(values), tangent -> (NeuralChainRules.NoTangent(), fill(NeuralChainRules.unthunk(tangent) / Float32(length(values)), size(values)))
end

function _training_norm_row(values, row)
    total = 0f0
    for column in axes(values, 2)
        total += values[row,column] * values[row,column]
    end
    return sqrt(total)
end

_training_norm(values) = reshape(Float32[_training_norm_row(values, row) for row in axes(values, 1)], :, 1)

function _training_log(values)
    return BookPDEMathRuntime.unary(:log, values)
end

function _training_terminal(equation, values)
    equation isa RiskLabAI.Pde.HJBLQ && return _training_log(0.5f0 .* (1f0 .+ _training_norm(values).^2))
    equation isa RiskLabAI.Pde.BlackScholesBarenblatt && return _training_row_sum(values.^2)
    equation isa RiskLabAI.Pde.PricingDefaultRisk && return max.(minimum(values; dims=2), 0f0)
    largest = maximum(values; dims=2)
    return max.(largest .- 120f0, 0f0) .- 2f0 .* max.(largest .- 150f0, 0f0)
end

function _training_driver(equation, states, value, control)
    equation isa RiskLabAI.Pde.HJBLQ && return zeros(Float32, size(states, 1), 1)
    equation isa RiskLabAI.Pde.BlackScholesBarenblatt && return fill(Float32(equation.rate), size(states, 1), 1)
    if equation isa RiskLabAI.Pde.PricingDefaultRisk
        linear = max.(value .- Float32(equation.vh), 0f0) .* Float32(equation.slope)
        piecewise = max.(linear .+ Float32(equation.gammah) .- Float32(equation.gammal), 0f0) .+ Float32(equation.gammal)
        return Float32(1 - equation.delta) .* piecewise .+ Float32(equation.rate)
    end
    temporary = _training_row_sum(control) ./ Float32(equation.sigma) .- value
    return ifelse.(temporary .> 0f0, Float32(equation.rb), Float32(equation.rl))
end

function _training_hamiltonian(equation, states, value, control)
    equation isa RiskLabAI.Pde.HJBLQ && return _training_row_sum(control.^2) ./ Float32(equation.sigma^2)
    equation isa RiskLabAI.Pde.PricingDefaultRisk && return zeros(Float32, size(states, 1), 1)
    sum_control = _training_row_sum(control)
    equation isa RiskLabAI.Pde.BlackScholesBarenblatt && return ((-1f0 .* sum_control) .* Float32(equation.rate)) ./ Float32(equation.sigma)
    temporary = sum_control ./ Float32(equation.sigma) .- value
    upper = (Float32(equation.mu_bar - equation.rb) .* sum_control) ./ Float32(equation.sigma)
    lower = (Float32(equation.mu_bar - equation.rl) .* sum_control) ./ Float32(equation.sigma)
    return ifelse.(temporary .> 0f0, upper, lower)
end

function _training_bsde_loss(equation, method, model, parameters, paths, increments, time, initial_value)
    value = method == "DeepBSDE" ? _training_expand(reshape(parameters.value, 1, 1), size(paths, 1)) : fill(Float32(initial_value), size(paths, 1), 1)
    coefficient, stochastic = ones(Float32, size(value)), zeros(Float32, size(value))
    dt = Float32(equation.delta_t)
    for index in 1:equation.num_time_interval
        states = paths[:, :, index]
        current_time = (time .* dt) .* Float32(index - 1)
        control = if method == "DeepBSDE"
            index == 1 ? _training_expand(parameters.gradient, size(states, 1)) : _pde_hidden_forward(parameters.networks[index-1], states, _pde_relu)
        else
            _pde_dtnn_forward(model, parameters.network, current_time, states; strided_input=true)
        end
        rate = _training_driver(equation, states, value, control)
        hamiltonian = _training_hamiltonian(equation, states, value, control)
        factor = 1f0 .+ rate .* dt
        increment = _training_row_sum(control .* increments[:, :, index])
        value = (value .* factor .+ hamiltonian .* dt) .+ increment
        stochastic = index == 1 ? increment .+ hamiltonian .* dt : (stochastic .* factor .+ increment) .+ hamiltonian .* dt
        coefficient = coefficient .* factor
    end
    payoff = Zygote.ignore() do
        _training_terminal(equation, paths[:, :, end])
    end
    return _training_mean((payoff .- value).^2), coefficient, stochastic, payoff
end

function _training_fbsnn_loss(equation, parameters, paths, increments, time)
    loss, dt = 0f0, Float32(equation.delta_t)
    for index in 1:equation.num_time_interval
        states, next_states = paths[:, :, index], paths[:, :, index+1]
        current_time = (time .* dt) .* Float32(index - 1)
        next_time = (time .* dt) .* Float32(index)
        value, derivative = _pde_sine_value_gradient(parameters.network.layers, hcat(current_time, states))
        next_value = _pde_hidden_forward(parameters.network, hcat(next_time, next_states), sin)
        sigma = equation isa RiskLabAI.Pde.HJBLQ ? Float32(equation.sigma) : Float32(equation.sigma) .* states
        control = derivative[:, 2:end] .* sigma
        rate = _training_driver(equation, states, value, control)
        hamiltonian = _training_hamiltonian(equation, states, value, control)
        predicted = (value .* (1f0 .+ rate .* dt) .+ hamiltonian .* dt) .+ _training_row_sum(control .* increments[:, :, index])
        loss += _training_mean((predicted .- next_value).^2)
    end
    payoff = Zygote.ignore() do
        _training_terminal(equation, paths[:, :, end])
    end
    terminal_value = _pde_hidden_forward(parameters.network, hcat(time .* Float32(equation.total_time), paths[:, :, end]), sin)
    return loss + _training_mean((payoff .- terminal_value).^2)
end

function _training_adam(parameter, gradient, state, learning_rate, betas)
    previous_moment = isnothing(state) ? zero(parameter) : state.moment
    previous_variance = isnothing(state) ? zero(parameter) : state.variance
    step = isnothing(state) ? 1 : state.step + 1
    beta1, beta2 = betas
    moment = _training_lerp_moment(previous_moment, gradient, Float32(1 - beta1))
    scaled_variance = Float32(beta2) .* previous_variance
    variance = scaled_variance .+ (Float32(1 - beta2) .* gradient) .* gradient
    correction = Float32(sqrt(1 - beta2^step))
    denominator = (sqrt.(variance) ./ correction) .+ 1f-8
    scale = Float32(-learning_rate / (1 - beta1^step))
    updated = parameter .+ (scale .* moment) ./ denominator
    return updated, (moment=moment, variance=variance, step=step), moment, variance
end

struct _TrainingPayoffEquation <: RiskLabAI.Pde.Equation
    base::RiskLabAI.Pde.BlackScholesBarenblatt
    payoff::String
end

function Base.getproperty(equation::_TrainingPayoffEquation, name::Symbol)
    return name in (:base, :payoff) ? getfield(equation, name) : getproperty(getfield(equation, :base), name)
end

_training_driver(equation::_TrainingPayoffEquation, states, value, control) = _training_driver(equation.base, states, value, control)
_training_hamiltonian(equation::_TrainingPayoffEquation, states, value, control) = _training_hamiltonian(equation.base, states, value, control)

function _training_terminal(equation::_TrainingPayoffEquation, values)
    equation.payoff == "best" && return maximum(values; dims=2)
    equation.payoff == "maxmin" && return maximum(values; dims=2) .- minimum(values; dims=2)
    return reshape(Float32[_neural_four_sum(view(values, row, :)) / Float32(size(values,2)) for row in axes(values,1)], :, 1)
end
