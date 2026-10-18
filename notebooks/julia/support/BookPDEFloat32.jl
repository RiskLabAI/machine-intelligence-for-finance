module BookPDEFloat32

include(joinpath(@__DIR__, "BookMKLLog.jl"))

using RiskLabAI
const Pde = RiskLabAI.Pde

function row_sum(values)
    output = zeros(Float32, size(values, 1), 1)
    for row in axes(values, 1), column in axes(values, 2)
        output[row, 1] += values[row, column]
    end
    return output
end

relu(values) = max.(values, 0f0)
divide_scalar(values, divisor) = values ./ Float32(divisor)

terminal(eq::Pde.PricingDefaultRisk, values) = relu(minimum(values; dims=2))
terminal(eq::Pde.BlackScholesBarenblatt, values) = row_sum(values .* values)
function terminal(eq::Pde.HJBLQ, values)
    norms = sqrt.(row_sum(values .* values))
    return BookMKLLog.log_values(0.5f0 .* (1f0 .+ norms .* norms))
end
function terminal(eq::Pde.PricingDiffRate, values)
    largest = maximum(values; dims=2)
    return relu(largest .- 120f0) .- 2f0 .* relu(largest .- 150f0)
end

terminal_for_sample(eq, values) = terminal(eq, values)
terminal_for_sample(eq::Pde.PricingDiffRate, values) = relu(maximum(values; dims=2) .- 120f0)

driver(eq::Pde.HJBLQ, x, y, z) = zeros(Float32, size(x, 1), 1)
driver(eq::Pde.BlackScholesBarenblatt, x, y, z) = ones(Float32, size(x, 1), 1) .* Float32(eq.rate)
function driver(eq::Pde.PricingDefaultRisk, x, y, z)
    piecewise = relu(y .- Float32(eq.vh))
    piecewise = piecewise .* Float32(eq.slope)
    piecewise = piecewise .+ Float32(eq.gammah)
    piecewise = piecewise .- Float32(eq.gammal)
    piecewise = relu(piecewise) .+ Float32(eq.gammal)
    return Float32(1 - eq.delta) .* piecewise .+ Float32(eq.rate)
end
function driver(eq::Pde.PricingDiffRate, x, y, z)
    temporary = divide_scalar(row_sum(z), eq.sigma) .- y
    return ifelse.(temporary .> 0f0, Float32(eq.rb), Float32(eq.rl))
end

hamiltonian(eq::Pde.PricingDefaultRisk, x, y, z) = zeros(Float32, size(x, 1), 1)
hamiltonian(eq::Pde.HJBLQ, x, y, z) = divide_scalar(row_sum(z .* z), eq.sigma^2)
hamiltonian(eq::Pde.BlackScholesBarenblatt, x, y, z) = divide_scalar((-1f0 .* row_sum(z)) .* Float32(eq.rate), eq.sigma)
function hamiltonian(eq::Pde.PricingDiffRate, x, y, z)
    sum_z = row_sum(z)
    temporary = divide_scalar(sum_z, eq.sigma) .- y
    upper = divide_scalar(Float32(eq.mu_bar - eq.rb) .* sum_z, eq.sigma)
    lower = divide_scalar(Float32(eq.mu_bar - eq.rl) .* sum_z, eq.sigma)
    return ifelse.(temporary .> 0f0, upper, lower)
end

function pairwise_sum(values)
    count = length(values)
    count < 8 && return foldl(+, values; init=-0f0)
    if count <= 128
        lanes = Float32[values[index] for index in 1:8]
        index = 9
        while index + 7 <= count
            for lane in 1:8
                lanes[lane] += values[index + lane - 1]
            end
            index += 8
        end
        total = ((lanes[1] + lanes[2]) + (lanes[3] + lanes[4])) + ((lanes[5] + lanes[6]) + (lanes[7] + lanes[8]))
        while index <= count
            total += values[index]
            index += 1
        end
        return total
    end
    middle = (count ÷ 2) ÷ 8 * 8
    return pairwise_sum(view(values, 1:middle)) + pairwise_sum(view(values, middle+1:count))
end

numpy_mean(values) = pairwise_sum(values) / Float32(length(values))
function numpy_std(values)
    center = numpy_mean(values)
    deviations = (values .- center).^2
    return sqrt(pairwise_sum(deviations) / Float32(length(values) - 1))
end

end
