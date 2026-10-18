function _training_strided_statistics(running, inputs)
    count, width = size(inputs)
    center = Float32[column <= 4div(width, 4) ? _neural_cascade(view(inputs, :, column), 1, 1, count) : _neural_four_sum(view(inputs, :, column)) for column in 1:width] ./ Float32(count)
    variance_sum = Float64[sum((Float64(value) - Float64(center[column]))^2 for value in view(inputs, :, column)) for column in 1:width]
    inverse_std = Float32.(1 ./ sqrt.(variance_sum ./ count .+ 1e-6))
    return center, inverse_std, variance_sum
end

function _training_strided_batchnorm(parameters, running, inputs)
    center, inverse_std, variance_sum = _training_strided_statistics(running, inputs)
    normalized = ((inputs .- center') .* inverse_std') .* parameters.scale' .+ parameters.offset'
    state = (mean=Float32.(0.01 .* Float64.(center) .+ 0.99 .* Float64.(running.mean)),
        variance=Float32.(0.01 .* (variance_sum ./ (size(inputs, 1) - 1)) .+ 0.99 .* Float64.(running.variance)), batches=running.batches+1)
    return normalized, state
end

function NeuralChainRules.rrule(::typeof(_training_strided_batchnorm), parameters, running, inputs)
    result = _training_strided_batchnorm(parameters, running, inputs)
    center, inverse_std, _ = _training_strided_statistics(running, inputs)
    function pullback(tangent)
        gradient = NeuralChainRules.unthunk(NeuralChainRules.unthunk(tangent)[1])
        sums = _neural_batch_sum(gradient)
        products = Float64[sum(Float64((inputs[row,column] - center[column]) * gradient[row,column]) for row in axes(inputs,1)) for column in axes(inputs,2)]
        coefficient = ((Float32.(products) .* inverse_std) .* inverse_std) ./ Float32(size(inputs,1))
        means = sums ./ Float32(size(inputs,1))
        input_gradient = ((gradient .- means' .- (inputs .- center') .* coefficient') .* inverse_std') .* parameters.scale'
        scale_gradient = Float32.(products .* Float64.(inverse_std))
        return NeuralChainRules.NoTangent(), (scale=scale_gradient, offset=sums), NeuralChainRules.NoTangent(), input_gradient
    end
    return result, pullback
end

function _training_lerp_moment(previous, gradient, weight)
    count = length(previous)
    limit = 16div(count, 16)
    width = ndims(previous) == 1 ? count : size(previous, 2)
    return map(CartesianIndices(previous)) do index
        ordinal = ndims(previous) == 1 ? index[1] : (index[1]-1)*width + index[2]
        change = gradient[index] - previous[index]
        ordinal <= limit ? fma(weight, change, previous[index]) : previous[index] + weight * change
    end
end
