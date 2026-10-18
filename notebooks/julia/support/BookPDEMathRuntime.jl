"""Explicit optional oneMKL binding for the book's released Float32 calculations."""
module BookPDEMathRuntime

using Libdl, TOML

const vml_mode = Int64(258)

struct Runtime
    library::Ptr{Cvoid}
    path::String
    version::String
    gemm::Ptr{Cvoid}
    unary::Dict{Symbol,Ptr{Cvoid}}
end

const selected_runtime = Ref{Union{Nothing,Runtime}}(nothing)
const runtime_lock = ReentrantLock()

function configured_library()
    project = Base.active_project()
    isnothing(project) && throw(ArgumentError("Activate the book Julia environment and configure its oneMKL runtime"))
    preferences = joinpath(dirname(project), "LocalPreferences.toml")
    isfile(preferences) || throw(ArgumentError("Set [RiskLabAI].book_pde_mkl_library in the active environment's LocalPreferences.toml to the externally supplied oneMKL runtime"))
    configuration = get(TOML.parsefile(preferences), "RiskLabAI", Dict())
    path = get(configuration, "book_pde_mkl_library", nothing)
    path isa AbstractString && !isempty(path) || throw(ArgumentError("Set [RiskLabAI].book_pde_mkl_library to the externally supplied oneMKL runtime library"))
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(preferences), path))
end

function required_symbol(library, names)
    for name in names
        pointer = Libdl.dlsym_e(library, name)
        pointer != C_NULL && return pointer
    end
    throw(ArgumentError("The supplied oneMKL runtime lacks a required documented symbol: " * join(string.(names), ", ")))
end

function Runtime(path::AbstractString)
    isfile(path) || throw(ArgumentError("The configured oneMKL runtime library does not exist"))
    library = Libdl.dlopen(abspath(path))
    interface = required_symbol(library, (:MKL_Set_Interface_Layer, :mkl_set_interface_layer))
    threading = required_symbol(library, (:MKL_Set_Threading_Layer, :mkl_set_threading_layer))
    ccall(interface, Cint, (Cint,), 0) == 0 || throw(ArgumentError("The supplied oneMKL runtime could not select LP64"))
    ccall(threading, Cint, (Cint,), 1) == 1 || throw(ArgumentError("The supplied oneMKL runtime could not select sequential execution"))
    version_pointer = required_symbol(library, (:MKL_Get_Version_String, :mkl_get_version_string, :MKL_GetVersionString))
    buffer = zeros(UInt8, 1024)
    ccall(version_pointer, Cvoid, (Ptr{UInt8}, Cint), buffer, length(buffer))
    version = GC.@preserve buffer unsafe_string(pointer(buffer))
    occursin("2026.1", version) && occursin("20260612", version) || throw(ArgumentError("The book's strict Float32 reference requires oneMKL 2026.1 build 20260612; the supplied runtime reports " * version))
    gemm = required_symbol(library, (:cblas_sgemm,))
    unary = Dict(name => required_symbol(library, (symbol,)) for (name, symbol) in ((:sin, :vmsSin), (:cos, :vmsCos), (:tanh, :vmsTanh), (:exp, :vmsExp), (:log, :vmsLn)))
    return Runtime(library, abspath(path), version, gemm, unary)
end

function runtime()
    return lock(runtime_lock) do
        if isnothing(selected_runtime[])
            selected_runtime[] = Runtime(configured_library())
        end
        selected_runtime[]::Runtime
    end
end

function unary(name::Symbol, values::AbstractArray{Float32})
    Base.get_zero_subnormals() && throw(ArgumentError("Strict Float32 calculations require gradual underflow"))
    input = collect(vec(values))
    output = similar(input)
    length(input) <= typemax(Int32) || throw(ArgumentError("The VML input exceeds the LP64 interface limit"))
    if !isempty(input)
        function_pointer = runtime().unary[name]
        ccall(function_pointer, Cvoid, (Int32, Ptr{Float32}, Ptr{Float32}, Int64), length(input), input, output, vml_mode)
    end
    Base.get_zero_subnormals() && error("The math runtime changed gradual-underflow handling")
    return reshape(output, size(values))
end

function information()
    active = runtime()
    return Dict("library" => active.path, "version" => active.version, "interface" => "LP64", "threading" => "sequential", "per_call_vml_mode" => vml_mode, "binary_bundled" => false)
end

end
