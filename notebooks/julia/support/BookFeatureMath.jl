module BookFeatureMath

using Libdl

const handle = Sys.iswindows() ? Libdl.dlopen("ucrtbase.dll") : nothing
const log_pointer = Sys.iswindows() ? Libdl.dlsym(handle, :log) : C_NULL
const log1p_pointer = Sys.iswindows() ? Libdl.dlsym(handle, :log1p) : C_NULL

"""Evaluate the existing Float64 logarithm formula using the Windows C runtime on Windows and Julia math on other platforms."""
runtime_log(value::Float64) = Sys.iswindows() ? ccall(log_pointer, Float64, (Float64,), value) : log(value)

"""Evaluate the existing Float64 log1p formula using the Windows C runtime on Windows and Julia math on other platforms."""
runtime_log1p(value::Float64) = Sys.iswindows() ? ccall(log1p_pointer, Float64, (Float64,), value) : log1p(value)

end
