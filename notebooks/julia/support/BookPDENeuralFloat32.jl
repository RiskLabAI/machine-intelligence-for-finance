"""
Float32 adaptation of the preserved public RiskLabAI neural components for the book.

BSD 3-Clause License

Copyright (c) 2022, RiskLab AI (risklab.ai)
All rights reserved.

RiskLabAI includes codes and algorithms from Marcos López de Prado's books
"Advances in Financial Machine Learning" and
"Machine Learning for Asset Managers".

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
module BookPDENeuralFloat32

using RiskLabAI, Statistics, Zygote

include(joinpath(@__DIR__, "BookPDENeuralKernels.jl"))

_nse_count(value, name) = RiskLabAI.Pde._nse_count(value, name)
_nse_rng(seed) = RiskLabAI.Pde._nse_rng(seed)

abstract type AbstractPDENetwork end

function _pde_dense_parameters(rng, inputs, outputs; bias = true)
    bound = Float32(1 / sqrt(inputs))
    weight = Float32.((2f0 .* rand(rng, Float32, outputs, inputs) .- 1f0) .* bound)
    return bias ? (weight = weight, bias = Float32.((2f0 .* rand(rng, Float32, outputs) .- 1f0) .* bound)) : (weight = weight,)
end

function _pde_parameter_copy(supplied, template)
    isnothing(supplied) && return template
    if template isa NamedTuple
        supplied isa NamedTuple && keys(supplied) == keys(template) || throw(ArgumentError("Network parameters must match the named architecture"))
        return NamedTuple{keys(template)}(Tuple(_pde_parameter_copy(getproperty(supplied, name), getproperty(template, name)) for name in keys(template)))
    elseif template isa Tuple
        supplied isa Tuple && length(supplied) == length(template) || throw(ArgumentError("Network layer counts must match the architecture"))
        return map(_pde_parameter_copy, supplied, template)
    end
    supplied isa AbstractArray{Float32} && size(supplied) == size(template) && all(isfinite, supplied) || throw(ArgumentError("Network parameters must be finite Float32 arrays with the expected shapes"))
    return copy(supplied)
end

function _pde_check_array(value, dimensions, features)
    value isa AbstractArray{Float32} && ndims(value) == dimensions && !isempty(value) && size(value, dimensions) == features && all(isfinite, value) || throw(ArgumentError("Inputs must be finite nonempty Float32 arrays with $dimensions dimensions and $features features"))
    return nothing
end

function _pde_dense(parameters, inputs::AbstractMatrix)
    return haskey(parameters, :bias) ? _mkl_affine(inputs, parameters.weight, parameters.bias) : _mkl_multiply(inputs, parameters.weight')
end

function _pde_dense(parameters, inputs::AbstractArray{<:Real, 3})
    flattened = reshape(permutedims(inputs, (3, 1, 2)), size(inputs, 3), :)
    projected = parameters.weight * flattened
    projected = haskey(parameters, :bias) ? projected .+ parameters.bias : projected
    return permutedims(reshape(projected, size(projected, 1), size(inputs, 1), size(inputs, 2)), (2, 3, 1))
end

_pde_relu(value) = value > 0 ? value : zero(value)

function _pde_layer_sizes(sizes)
    sizes isa Union{Tuple, AbstractVector} && length(sizes) >= 2 || throw(ArgumentError("Supply at least input and output layer sizes"))
    return Tuple(_nse_count(value, "layer size") for value in sizes)
end

function _pde_feedforward_parameters(rng, sizes; bias = true)
    return (layers = Tuple(_pde_dense_parameters(rng, sizes[index], sizes[index+1]; bias) for index in 1:length(sizes)-1),)
end

"""Public linear-only PDE component; the unused Python batch-normalization module does not enter the native computation."""
mutable struct Net1 <: AbstractPDENetwork
    input_dim::Int
    output_dim::Int
    parameters::NamedTuple
    function Net1(input_dim, output_dim; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        template = (layer = _pde_dense_parameters(_nse_rng(seed), inputs, outputs),)
        new(inputs, outputs, _pde_parameter_copy(parameters, template))
    end
end

function (model::Net1)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, model.input_dim)
    end
    return _pde_dense(model.parameters.layer, inputs)
end

"""Six-layer public time network with widths 4,100,150,200,300,200,output_dim and tanh after every layer."""
mutable struct TimeNet <: AbstractPDENetwork
    output_dim::Int
    parameters::NamedTuple
    function TimeNet(output_dim; seed = 0, parameters = nothing)
        outputs = _nse_count(output_dim, "output_dim")
        template = _pde_feedforward_parameters(_nse_rng(seed), (4, 100, 150, 200, 300, 200, outputs))
        new(outputs, _pde_parameter_copy(parameters, template))
    end
end

function (model::TimeNet)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, 4)
    end
    return _pde_time_forward(model.parameters, inputs)
end

function _pde_time_forward(parameters, inputs)
    output = inputs
    for layer in parameters.layers
        output = _mkl_tanh(_pde_dense(layer, output))
    end
    return output
end

"""Feed-forward FBSNN with sine after every hidden dense layer and a linear final layer."""
mutable struct FBSNNNetwork <: AbstractPDENetwork
    layer_sizes::Tuple
    parameters::NamedTuple
    function FBSNNNetwork(layer_sizes; seed = 0, parameters = nothing)
        sizes = _pde_layer_sizes(layer_sizes)
        new(sizes, _pde_parameter_copy(parameters, _pde_feedforward_parameters(_nse_rng(seed), sizes)))
    end
end

"""Public bias-free Deep BSDE network with ReLU hidden layers and linear output; inactive batch normalization is omitted."""
mutable struct DeepBSDE <: AbstractPDENetwork
    layer_sizes::Tuple
    parameters::NamedTuple
    function DeepBSDE(layer_sizes; seed = 0, parameters = nothing)
        sizes = _pde_layer_sizes(layer_sizes)
        new(sizes, _pde_parameter_copy(parameters, _pde_feedforward_parameters(_nse_rng(seed), sizes; bias = false)))
    end
end

function _pde_hidden_forward(parameters, inputs, activation)
    count = length(parameters.layers)
    output = inputs
    for (index, layer) in enumerate(parameters.layers)
        output = _pde_dense(layer, output)
        output = index < count ? (activation === sin ? _mkl_sin(output) : activation.(output)) : output
    end
    return output
end

function (model::Union{FBSNNNetwork, DeepBSDE})(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, first(model.layer_sizes))
    end
    return _pde_hidden_forward(model.parameters, inputs, model isa FBSNNNetwork ? sin : _pde_relu)
end


function _pde_time_dependent_parameters(rng, inputs, hidden, outputs; monte_carlo = false)
    widths = (inputs, hidden...)
    return (layers = Tuple(_pde_dense_parameters(rng, widths[index], widths[index+1]) for index in 1:length(hidden)),
        time_layers = Tuple(_pde_feedforward_parameters(rng, (4, 100, 150, 200, 300, 200, width)) for width in (monte_carlo ? (widths[1:end-1]..., outputs) : widths[1:end-1])),
        linear = _pde_dense_parameters(rng, last(hidden), outputs),
        normalization = monte_carlo ? () : Tuple((scale = ones(Float32, width), offset = zeros(Float32, width)) for width in widths[1:end-1]))
end

function _pde_hidden_sizes(values)
    values isa Union{Tuple, AbstractVector} && !isempty(values) || throw(ArgumentError("Supply at least one hidden layer"))
    return Tuple(_nse_count(value, "hidden layer size") for value in values)
end

include(joinpath(@__DIR__, "BookPDEBatchNormalization.jl"))

_pde_batchnorm(parameters, running, inputs, training) = _neural_batch_norm(parameters, running, inputs, training)

"""
Time-dependent sine network with the public active batch normalization,
polynomial/exponential time features and multiplicative time weights.
`training=true` uses batch moments and updates running mean/unbiased variance
with momentum 0.01 and epsilon 1e-6. Set `model.training=false` for fixed
running-statistic evaluation. Unused normalization and output-time modules
from the Python constructor are omitted from the native parameter tree.
"""
mutable struct TimeDependentNetwork <: AbstractPDENetwork
    input_dim::Int
    hidden_sizes::Tuple
    output_dim::Int
    training::Bool
    parameters::NamedTuple
    state::Tuple
    function TimeDependentNetwork(input_dim, hidden_sizes, output_dim; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        hidden = _pde_hidden_sizes(hidden_sizes)
        template = _pde_time_dependent_parameters(_nse_rng(seed), inputs, hidden, outputs)
        state = Tuple((mean = zeros(Float32, width), variance = ones(Float32, width), batches = 0) for width in (inputs, hidden[1:end-1]...))
        new(inputs, hidden, outputs, true, _pde_parameter_copy(parameters, template), state)
    end
end

function _pde_dtnn_forward(model, parameters, time, inputs; update_state = true, strided_input = false)
    features, next_states = inputs, ()
    for index in eachindex(parameters.layers)
        time_features = hcat(time, time.^2, time.^3, _mkl_exp(time))
        normalized, next_state = strided_input && index == 1 && model.training ? _training_strided_batchnorm(parameters.normalization[index], model.state[index], features) : _pde_batchnorm(parameters.normalization[index], model.state[index], features, model.training)
        time_weights = _pde_time_forward(parameters.time_layers[index], time_features)
        projected = _pde_dense(parameters.layers[index], normalized .* (1 .+ time_weights))
        features = _mkl_sin(projected)
        next_states = (next_states..., next_state)
    end
    if update_state
        Zygote.ignore() do
            model.state = next_states
        end
    end
    return _pde_dense(parameters.linear, features)
end

function (model::TimeDependentNetwork)(time, inputs)
    Zygote.ignore() do
        _pde_check_array(time, 2, 1)
        _pde_check_array(inputs, 2, model.input_dim)
        size(time, 1) == size(inputs, 1) || throw(ArgumentError("Time and feature batches must agree"))
    end
    return _pde_dtnn_forward(model, model.parameters, time, inputs)
end

function _pde_sine_value_gradient(layers, inputs)
    projected = _pde_dense(first(layers), inputs)
    if length(layers) == 1
        return projected, _mkl_multiply(ones(Float32, size(projected)), first(layers).weight)
    end
    output, gradient = _pde_sine_value_gradient(Base.tail(layers), _mkl_sin(projected))
    return output, _mkl_multiply(gradient .* _mkl_cos(projected), first(layers).weight)
end


include(joinpath(@__DIR__, "BookPDEStridedNormalization.jl"))
include(joinpath(@__DIR__, "BookPDETraining.jl"))

end
