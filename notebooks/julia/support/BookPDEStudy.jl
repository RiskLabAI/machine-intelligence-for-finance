module BookPDEStudy

using RiskLabAI, Zygote, LinearAlgebra, CSV, DataFrames, TOML, SHA, Statistics
include(joinpath(@__DIR__, "BookPDEMathRuntime.jl"))
include(joinpath(@__DIR__, "BookPDENeuralFloat32.jl"))
include(joinpath(@__DIR__, "BookPDEFloat32.jl"))
const A = BookPDENeuralFloat32
const E = BookPDEFloat32
const directory = normpath(joinpath(@__DIR__, "..", "data", "chapter29_inputs"))
const input_manifest = TOML.parsefile(joinpath(directory, "training.toml"))
const primitive_manifest = TOML.parsefile(joinpath(directory, "primitives.toml"))

function raw_array(folder, file, shape)
    values = reinterpret(Float32, read(joinpath(directory, "training", folder, file)))
    return copy(reshape(values, Tuple(shape)))
end

function supplied(template, specification, folder, path="")
    if template isa NamedTuple
        return NamedTuple{keys(template)}(Tuple(supplied(getproperty(template, key), specification, folder, isempty(path) ? String(key) : path * "/" * String(key)) for key in keys(template)))
    elseif template isa Tuple
        return Tuple(supplied(value, specification, folder, path * "/" * string(index)) for (index, value) in enumerate(template))
    end
    record = only(filter(value -> value["native_path"] == path, specification))
    @assert Tuple(record["shape"]) == size(template)
    return raw_array(folder, record["file"], record["shape"])
end

function batch(record, step)
    count = step == 0 ? 128 : record["batch"]
    dimension, intervals = record["dimension"], record["intervals"]
    prefix = "step_" * lpad(string(step), 3, '0')
    return (paths=raw_array(record["group"], prefix * "_paths.bin", [count, dimension, intervals+1]),
        increments=raw_array(record["group"], prefix * "_increments.bin", [count, dimension, intervals]), time=ones(Float32, count, 1))
end

function fingerprint(values)
    @assert eltype(values) == Float32 && all(isfinite, values)
    return bytes2hex(sha256(reinterpret(UInt8, collect(vec(values)))))
end

function update_tree(parameters, gradient, states, record, step, hashes, path="")
    if parameters isa NamedTuple
        return NamedTuple{keys(parameters)}(Tuple(update_tree(getproperty(parameters, key), getproperty(gradient, key), states, record, step, hashes, isempty(path) ? String(key) : path * "/" * String(key)) for key in keys(parameters)))
    elseif parameters isa Tuple
        return Tuple(update_tree(value, gradient[index], states, record, step, hashes, path * "/" * string(index)) for (index, value) in enumerate(parameters))
    end
    index = only(findall(value -> value["native_path"] == path, record["parameters"]))
    rate = path == "value" ? 0.1 : path == "gradient" ? 0.001 : record["learning_rate"]
    betas = record["method"] == "FBSNN" ? (0.99, 0.999) : (0.9, 0.99)
    updated, state, moment, variance = A._training_adam(parameters, gradient, get(states, path, nothing), rate, betas)
    states[path] = state
    push!(hashes, (step=step, parameter=index, gradient=fingerprint(gradient), value=fingerprint(updated), moment=fingerprint(moment), variance=fingerprint(variance), optimizer_step=step))
    return updated
end

function run_case(record, trace_root)
    name, method = record["identifier"], record["method"]
    d, w, intervals = record["dimension"], record["width"], record["intervals"]
    equation_type = Dict("hjb"=>HJBLQ, "quadratic"=>BlackScholesBarenblatt, "default"=>PricingDefaultRisk, "different"=>PricingDiffRate, "maxmin"=>BlackScholesBarenblatt, "basket"=>BlackScholesBarenblatt, "best"=>BlackScholesBarenblatt)[record["family"]]
    equation = equation_type(d, record["horizon"], intervals)
    record["family"] in ("maxmin", "basket", "best") && (equation = A._TrainingPayoffEquation(equation, record["family"]))
    model = method == "DTNN" ? A.TimeDependentNetwork(d, [w, w], d) : method == "FBSNN" ? A.FBSNNNetwork([d+1, w, w, 1]) : nothing
    template = method == "DeepBSDE" ? (networks=Tuple(A.DeepBSDE([d, w, w, d]).parameters for _ in 2:intervals), value=zeros(Float32, 1), gradient=zeros(Float32, 1, d)) : (network=model.parameters,)
    parameters = supplied(template, record["parameters"], name)
    validation = batch(record, 0)
    optimizer_states = Dict{String, Any}()
    objective(p, b) = method == "FBSNN" ? A._training_fbsnn_loss(equation, p, b.paths, b.increments, b.time) : A._training_bsde_loss(equation, method, model, p, b.paths, b.increments, b.time, record["initial_value"])
    histories = zeros(Float32, record["iterations"], 3)
    hashes, states = NamedTuple[], NamedTuple[]
    started = time()
    for step in 1:record["iterations"]
        training = batch(record, step)
        result, pullback = Zygote.pullback(p -> objective(p, training), parameters)
        value = method == "FBSNN" ? result : first(result)
        gradient = only(pullback(method == "FBSNN" ? 1f0 : (1f0, nothing, nothing, nothing)))
        parameters = update_tree(parameters, gradient, optimizer_states, record, step, hashes)
        result = objective(parameters, validation)
        validation_loss = method == "FBSNN" ? result : first(result)
        estimate = if method == "DeepBSDE"
            only(parameters.value)
        elseif method == "FBSNN"
            A._training_mean(A._pde_hidden_forward(parameters.network, hcat(zeros(Float32, 128, 1), validation.paths[:, :, 1]), sin))
        else
            A._training_mean((result[4] .- result[3]) ./ result[2])
        end
        @assert value isa Float32 && validation_loss isa Float32 && estimate isa Float32 && all(isfinite, (value, validation_loss, estimate))
        histories[step, :] = [value, validation_loss, estimate]
        if method == "DTNN"
            for (index, state) in enumerate(model.state)
                push!(states, (step=step, layer=index, mean=fingerprint(state.mean), variance=fingerprint(state.variance), batches=state.batches))
            end
        end
    end
    CSV.write(joinpath(trace_root, name * "_hashes.csv"), DataFrame(hashes))
    !isempty(states) && CSV.write(joinpath(trace_root, name * "_states.csv"), DataFrame(states))
    return (losses=Float64.(histories[:,2]), estimates=Float64.(histories[:,3]), training_losses=Float64.(histories[:,1]),
        seconds=time()-started, parameters=source_parameter_count(record["method"],d,w,intervals), method=method,
        seed=record["label_seed"], family=record["family"], identifier=name, study=record["study"], dimension=d, intervals=intervals)
end

function source_parameter_count(method, d, w, intervals)
    dense(sizes) = sum(sizes[index+1] * (sizes[index]+1) for index in 1:length(sizes)-1)
    method == "FBSNN" && return dense([d+1,w,w,1])
    method == "DeepBSDE" && return intervals * (2d*w+w*w+4d+4w)
    return w*(d+3)+w*(w+3)+sum(dense([4,100,150,200,300,200,width]) for width in (d,w,d))+w*d+d+4d+4w
end

function records(study)
    return filter(record -> record["study"] == study, input_manifest["cases"])
end

function equation(family, dimension, horizon, intervals)
    kind = Dict("default"=>PricingDefaultRisk,"hjb"=>HJBLQ,"quadratic"=>BlackScholesBarenblatt,"different"=>PricingDiffRate,"maxmin"=>BlackScholesBarenblatt,"basket"=>BlackScholesBarenblatt,"best"=>BlackScholesBarenblatt)[family]
    value = kind(dimension,horizon,intervals)
    return family in ("maxmin","basket","best") ? A._TrainingPayoffEquation(value,family) : value
end

function equation_sample(name)
    record = primitive_manifest["equations"][name]
    raw = copy(reshape(reinterpret(Float64,read(joinpath(directory,record["file"]))),record["count"],4,record["steps"]))
    value = equation(record["family"],4,record["horizon"],record["steps"])
    increments,paths = pde_sample(value,record["count"];normal_draws=raw)
    return value,increments,paths
end

function control_input(family)
    return copy(reshape(reinterpret(Float32,read(joinpath(directory,"generator_"*family*"_control.bin"))),64,4))
end

function numpy_sum64(values)
    count = length(values)
    count < 8 && return foldl(+,values;init=-0.0)
    if count <= 128
        lanes = Float64[values[index] for index in 1:8]
        index = 9
        while index+7 <= count
            for lane in 1:8
                lanes[lane] += values[index+lane-1]
            end
            index += 8
        end
        total = ((lanes[1]+lanes[2])+(lanes[3]+lanes[4]))+((lanes[5]+lanes[6])+(lanes[7]+lanes[8]))
        while index <= count
            total += values[index]
            index += 1
        end
        return total
    end
    middle = div(div(count,2),8)*8
    return numpy_sum64(view(values,1:middle))+numpy_sum64(view(values,middle+1:count))
end

numpy_rows64(values) = Float64[numpy_sum64(view(values,row,:)) for row in axes(values,1)]

function monte_carlo_reference(name, kind, horizon, sigma; rate=0.0, initial=nothing)
    record = primitive_manifest["monte_carlo"][name]
    count,dimension,chunk = record["count"],record["dimension"],record["chunk"]
    normals = copy(reshape(reinterpret(Float64,read(joinpath(directory,record["file"]))),count,dimension))
    total,total_square,observed = 0.0,0.0,0
    for first in 1:chunk:count
        last = min(first+chunk-1,count)
        shocks = normals[first:last,:]
        if kind == "hjb"
            terminal = sqrt(2horizon) .* shocks
            transformed = 2.0 ./ (1.0 .+ numpy_rows64(terminal.^2))
        else
            terminal = initial' .* exp.((rate-0.5sigma^2)*horizon .+ (sigma*sqrt(horizon)) .* shocks)
            payoff = kind == "maxmin" ? vec(maximum(terminal;dims=2)-minimum(terminal;dims=2)) : kind == "best" ? vec(maximum(terminal;dims=2)) : numpy_rows64(terminal.^2)
            transformed = exp(-rate*horizon) .* payoff
        end
        total += numpy_sum64(transformed)
        total_square += numpy_sum64(transformed.^2)
        observed += last-first+1
    end
    center = total/observed
    variance = (total_square-observed*center^2)/(observed-1)
    error = sqrt(variance/observed)
    return kind == "hjb" ? (-log(center),error/center) : (center,error)
end

function histogram_data(values, count; density=false)
    lower,upper = extrema(values)
    lower == upper && ((lower,upper)=(lower-0.5,upper+0.5))
    edges = collect(range(Float64(lower),Float64(upper);length=count+1))
    totals = zeros(Int,count)
    for value in values
        totals[clamp(searchsortedlast(edges,Float64(value)),1,count)] += 1
    end
    widths = diff(edges)
    return (centers=(edges[1:end-1]+edges[2:end])/2, widths=widths,
        values=density ? totals./(sum(totals).*widths) : totals)
end

export A,E,records,run_case,equation_sample,equation,control_input,numpy_sum64,numpy_rows64,
    monte_carlo_reference,histogram_data,source_parameter_count,primitive_manifest,directory
end
