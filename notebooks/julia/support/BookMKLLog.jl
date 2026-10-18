"""Float32 logarithms through the explicit optional book math runtime."""
module BookMKLLog

using ...BookPDEMathRuntime

version_text() = BookPDEMathRuntime.runtime().version
mode() = BookPDEMathRuntime.vml_mode
log_values(values::AbstractArray{Float32}) = BookPDEMathRuntime.unary(:log, values)

end
