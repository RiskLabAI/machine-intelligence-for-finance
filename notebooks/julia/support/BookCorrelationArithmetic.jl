module BookCorrelationArithmeticCandidate

using LinearAlgebra

function pandas_correlation(values; reciprocal=true)
    all(isfinite, values) || throw(ArgumentError("This bounded correlation replay requires finite observations."))
    output = zeros(size(values, 2), size(values, 2))
    for right in axes(values, 2), left in 1:right
        mean_right = mean_left = square_right = square_left = cross = 0.0
        for row in axes(values, 1)
            value_right, value_left = values[row, right], values[row, left]
            delta_right, delta_left = value_right - mean_right, value_left - mean_left
            if reciprocal
                inverse = 1.0 / row
                mean_right += inverse * delta_right
                mean_left += inverse * delta_left
            else
                mean_right += delta_right / row
                mean_left += delta_left / row
            end
            square_right += (value_right - mean_right) * delta_right
            square_left += (value_left - mean_left) * delta_left
            cross += (value_right - mean_right) * delta_left
        end
        denominator = sqrt(square_right * square_left)
        value = denominator == 0 ? NaN : clamp(cross / denominator, -1.0, 1.0)
        output[right, left] = output[left, right] = value
    end
    return output
end

function numpy_covariance(values)
    all(isfinite, values) && size(values, 1) > 1 || throw(ArgumentError("At least two finite observations are required."))
    average = zeros(1, size(values, 2))
    for row in axes(values, 1), column in axes(values, 2)
        average[column] += values[row, column]
    end
    average ./= size(values, 1)
    packed = permutedims(values .- average)
    covariance = Matrix(Symmetric(BLAS.syrk('U', 'N', 1.0, packed), :U))
    covariance .*= 1.0 / (size(values, 1) - 1)
    return covariance
end

function block_covariance(n_columns, n_blocks, minimum_size, sigma, cuts, blocks)
    limit = n_columns - (minimum_size - 1) * n_blocks
    positions = vcat(sort(Int.(cuts)), limit)
    sizes = vcat(positions[1], diff(positions)) .- 1 .+ minimum_size
    length(blocks) == length(sizes) || throw(DimensionMismatch("Supply one primitive record per covariance block."))
    output = zeros(n_columns, n_columns)
    offset = 0
    for (index, width) in enumerate(sizes)
        record = blocks[index]
        values = reshape(record.common, :, 1) .+ sigma .* record.noise
        covariance = width == 1 ? ones(1, 1) : numpy_covariance(values)
        output[offset+1:offset+width, offset+1:offset+width] = covariance
        offset += width
    end
    return output
end

function random_block_correlation(n_columns, n_blocks; block_size_min=1, random_state=nothing, draws)
    first = block_covariance(n_columns, n_blocks, block_size_min, 0.5, draws.cuts, draws.blocks)
    second = block_covariance(n_columns, 1, n_columns, 1.0, Int[], [draws.market])
    covariance = first .+ second
    scales = sqrt.(diag(covariance))
    scales[scales .== 0] .= 1.0
    correlation = clamp.(covariance ./ (scales * scales'), -1.0, 1.0)
    correlation[diagind(correlation)] .= 1.0
    return correlation
end

function adjusted_rand(first, second)
    length(first) == length(second) || throw(DimensionMismatch("Partitions must align."))
    joint, rows, columns = Dict{Tuple{Int,Int},Int}(), Dict{Int,Int}(), Dict{Int,Int}()
    for (a, b) in zip(first, second)
        joint[(a, b)] = get(joint, (a, b), 0) + 1
        rows[a] = get(rows, a, 0) + 1
        columns[b] = get(columns, b, 0) + 1
    end
    true_positive = sum(Int128(value) * (value - 1) for value in values(joint))
    false_negative = sum(Int128(value) * (value - 1) for value in values(rows)) - true_positive
    false_positive = sum(Int128(value) * (value - 1) for value in values(columns)) - true_positive
    true_negative = Int128(length(first)) * (length(first) - 1) - true_positive - false_negative - false_positive
    false_positive == false_negative == 0 && return 1.0
    numerator = 2 * (true_positive * true_negative - false_negative * false_positive)
    denominator = (true_positive + false_negative) * (false_negative + true_negative) + (true_positive + false_positive) * (false_positive + true_negative)
    return Float64(numerator) / Float64(denominator)
end

end
