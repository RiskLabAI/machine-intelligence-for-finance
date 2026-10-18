function _neural_sequential_sum(values)
    result = 0f0
    for value in values
        result += value
    end
    return result
end

function _neural_normalization_statistics(running, inputs, training; backward=false)
    if training
        count, width = size(inputs)
        @assert count > 1
        center = Float32[_neural_sequential_sum(view(inputs, :, column)) / Float32(count) for column in 1:width]
        variance_sum = Float32[_neural_sequential_sum((view(inputs, :, column) .- center[column]).^2) for column in 1:width]
        inverse_std = Float32.(1 ./ sqrt.(Float64.(variance_sum ./ Float32(count)) .+ 1e-6))
        return center, inverse_std, variance_sum
    end
    inverse_std = backward ? Float32.(1 ./ sqrt.(Float64.(running.variance) .+ 1e-6)) : 1f0 ./ sqrt.(running.variance .+ 1f-6)
    return running.mean, inverse_std, nothing
end

function _neural_batch_norm(parameters, running, inputs, training)
    center, inverse_std, variance_sum = _neural_normalization_statistics(running, inputs, training)
    alpha = inverse_std .* parameters.scale
    beta = parameters.offset .- center .* alpha
    values = inputs .* alpha' .+ beta'
    next_state = if training
        count = Float32(size(inputs, 1))
        (mean = 0.01f0 .* center .+ 0.99f0 .* running.mean,
            variance = Float32.(Float64(0.01f0) .* Float64.(variance_sum ./ (count - 1f0)) .+ Float64.(0.99f0 .* running.variance)),
            batches = running.batches + 1)
    else
        running
    end
    return values, next_state
end

function NeuralChainRules.rrule(::typeof(_neural_batch_norm), parameters, running, inputs, training)
    result = _neural_batch_norm(parameters, running, inputs, training)
    center, inverse_std, _ = _neural_normalization_statistics(running, inputs, training; backward=true)
    function pullback(tangent)
        gradient = NeuralChainRules.unthunk(NeuralChainRules.unthunk(tangent)[1])
        sums = Float32[_neural_sequential_sum(view(gradient, :, column)) for column in axes(inputs, 2)]
        products = Float32[_neural_sequential_sum((view(inputs, :, column) .- center[column]) .* view(gradient, :, column)) for column in axes(inputs, 2)]
        input_gradient = if training
            count = Float32(size(inputs, 1))
            coefficient = ((products .* inverse_std) .* inverse_std) ./ count
            means = sums ./ count
            ((gradient .- means' .- (inputs .- center') .* coefficient') .* inverse_std') .* parameters.scale'
        else
            (gradient .* inverse_std') .* parameters.scale'
        end
        return NeuralChainRules.NoTangent(), (scale=products .* inverse_std, offset=sums), NeuralChainRules.NoTangent(), input_gradient, NeuralChainRules.NoTangent()
    end
    return result, pullback
end
