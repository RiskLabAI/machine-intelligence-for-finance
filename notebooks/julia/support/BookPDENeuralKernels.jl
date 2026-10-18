using Libdl, LinearAlgebra

const NeuralChainRules = Zygote.ChainRulesCore
using ..BookPDEMathRuntime

function _mkl_operand(values)
    transposed = values isa Union{Adjoint, Transpose}
    storage = transposed ? parent(values) : values
    return copy(transpose(storage)), transposed ? 112 : 111, size(storage, 2)
end

function _mkl_product(left::AbstractMatrix{Float32}, right::AbstractMatrix{Float32}, bias)
    rows, shared = size(left)
    @assert shared == size(right, 1)
    columns = size(right, 2)
    a, transa, lda = _mkl_operand(left)
    b, transb, ldb = _mkl_operand(right)
    result = isnothing(bias) ? zeros(Float32, columns, rows) : repeat(reshape(bias, :, 1), 1, rows)
    beta = isnothing(bias) ? 0f0 : 1f0
    @assert max(rows, columns, shared, lda, ldb) <= typemax(Int32)
    function_pointer = BookPDEMathRuntime.runtime().gemm
    ccall(function_pointer, Cvoid, (Cint, Cint, Cint, Cint, Cint, Cint, Float32, Ptr{Float32}, Cint, Ptr{Float32}, Cint, Float32, Ptr{Float32}, Cint), 101, transa, transb, rows, columns, shared, 1f0, a, lda, b, ldb, beta, result, columns)
    return copy(transpose(result))
end

_mkl_multiply(left, right) = _mkl_product(left, right, nothing)
_mkl_affine(inputs, weight, bias) = _mkl_product(inputs, weight', bias)

function _neural_cascade(values, first_index, stride, count)
    accumulator = zeros(Float32, 4)
    step = 1 << max(4, div(ndigits(max(count, 1) - 1; base=2), 4))
    index = 0
    while index + step <= count
        for offset in 0:step-1
            accumulator[1] += values[first_index + (index + offset) * stride]
        end
        index += step
        for level in 2:4
            accumulator[level] += accumulator[level-1]
            accumulator[level-1] = 0f0
            iszero(index & ((step - 1) << ((level - 1) * trailing_zeros(step)))) || break
        end
    end
    for offset in index:count-1
        accumulator[1] += values[first_index + offset * stride]
    end
    for level in 2:4
        accumulator[1] += accumulator[level]
    end
    return accumulator[1]
end

function _neural_four_sum(values)
    count = length(values)
    partial = Float32[_neural_cascade(values, lane, 4, div(count, 4)) for lane in 1:4]
    for index in 4div(count, 4)+1:count
        partial[1] += values[index]
    end
    return ((partial[1] + partial[2]) + partial[3]) + partial[4]
end

function _neural_batch_sum(values::AbstractMatrix{Float32})
    width = size(values, 2)
    if width == 1 && size(values, 1) >= 8
        data = vec(values)
        partial = Float32[_neural_four_sum(view(data, lane:8:8div(length(data), 8))) for lane in 1:8]
        total = 0f0
        for index in 8div(length(data), 8)+1:length(data)
            total += data[index]
        end
        for value in partial
            total += value
        end
        return Float32[total]
    end
    sequential_width = width >= 8 ? 32div(width, 32) : 4div(width, 4)
    return Float32[column <= sequential_width ? _neural_cascade(view(values, :, column), 1, 1, size(values, 1)) : _neural_four_sum(view(values, :, column)) for column in 1:width]
end

function NeuralChainRules.rrule(::typeof(_neural_batch_sum), values)
    result = _neural_batch_sum(values)
    function pullback(tangent)
        gradient = NeuralChainRules.unthunk(tangent)
        return NeuralChainRules.NoTangent(), repeat(reshape(gradient, 1, :), size(values, 1), 1)
    end
    return result, pullback
end

function NeuralChainRules.rrule(::typeof(_mkl_multiply), left, right)
    result = _mkl_multiply(left, right)
    function pullback(tangent)
        gradient = NeuralChainRules.unthunk(tangent)
        left_gradient = left isa Union{Adjoint, Transpose} ? _mkl_multiply(right, gradient')' : _mkl_multiply(gradient, right')
        right_gradient = right isa Union{Adjoint, Transpose} ? _mkl_multiply(gradient', left)' : _mkl_multiply(left', gradient)
        return NeuralChainRules.NoTangent(), NeuralChainRules.ProjectTo(left)(left_gradient), NeuralChainRules.ProjectTo(right)(right_gradient)
    end
    return result, pullback
end

function NeuralChainRules.rrule(::typeof(_mkl_affine), inputs, weight, bias)
    result = _mkl_affine(inputs, weight, bias)
    function pullback(tangent)
        gradient = NeuralChainRules.unthunk(tangent)
        return NeuralChainRules.NoTangent(), _mkl_multiply(gradient, weight), _mkl_multiply(gradient', inputs), _neural_batch_sum(gradient)
    end
    return result, pullback
end

_mkl_unary(name, values::AbstractArray{Float32}) = BookPDEMathRuntime.unary(name, values)

_mkl_sin(values) = _mkl_unary(:sin, values)
_mkl_cos(values) = _mkl_unary(:cos, values)
_mkl_tanh(values) = _mkl_unary(:tanh, values)
_mkl_exp(values) = _mkl_unary(:exp, values)

function NeuralChainRules.rrule(::typeof(_mkl_sin), values)
    return _mkl_sin(values), tangent -> (NeuralChainRules.NoTangent(), NeuralChainRules.unthunk(tangent) .* _mkl_cos(values))
end

function NeuralChainRules.rrule(::typeof(_mkl_cos), values)
    return _mkl_cos(values), tangent -> (NeuralChainRules.NoTangent(), NeuralChainRules.unthunk(tangent) .* (-_mkl_sin(values)))
end

function NeuralChainRules.rrule(::typeof(_mkl_tanh), values)
    result = _mkl_tanh(values)
    return result, tangent -> (NeuralChainRules.NoTangent(), NeuralChainRules.unthunk(tangent) .* (1f0 .- result.^2))
end

function NeuralChainRules.rrule(::typeof(_mkl_exp), values)
    result = _mkl_exp(values)
    return result, tangent -> (NeuralChainRules.NoTangent(), NeuralChainRules.unthunk(tangent) .* result)
end
