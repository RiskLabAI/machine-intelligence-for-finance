module BookKMeans

using LinearAlgebra, Statistics

struct KMeansFit
    labels::Vector{Int}
    centers::Matrix{Float64}
    initial_indices::Vector{Int}
    iterations::Int
    inertia::Float64
end

function sequential_cumsum(values)
    total = 0.0
    output = similar(values, Float64)
    for index in eachindex(values)
        total += values[index]
        output[index] = total
    end
    return output
end

function column_average(values)
    output = zeros(1, size(values, 2))
    for row in axes(values, 1), column in axes(values, 2)
        output[column] += values[row, column]
    end
    output ./= size(values, 1)
    return output
end

function pairwise_sum(values)
    count = length(values)
    count < 8 && return foldl(+, values; init=-0.0)
    if count <= 128
        accumulators = Float64[values[index] for index in 1:8]
        index = 9
        while index + 7 <= count
            for lane in 1:8
                accumulators[lane] += values[index + lane - 1]
            end
            index += 8
        end
        total = ((accumulators[1] + accumulators[2]) + (accumulators[3] + accumulators[4])) + ((accumulators[5] + accumulators[6]) + (accumulators[7] + accumulators[8]))
        while index <= count
            total += values[index]
            index += 1
        end
        return total
    end
    middle = (count ÷ 2) ÷ 8 * 8
    return pairwise_sum(view(values, 1:middle)) + pairwise_sum(view(values, middle+1:count))
end

function row_norms(values)
    output = zeros(size(values, 1))
    for row in axes(values, 1)
        low, high = 0.0, 0.0
        column = 1
        while column + 7 <= size(values, 2)
            for offset in (6, 4, 2, 0)
                a, b = values[row, column + offset], values[row, column + offset + 1]
                low += a * a
                high += b * b
            end
            column += 8
        end
        while column + 1 <= size(values, 2)
            a, b = values[row, column], values[row, column + 1]
            low += a * a
            high += b * b
            column += 2
        end
        column <= size(values, 2) && (low += values[row, column]^2)
        output[row] = low + high
    end
    return output
end

function squared_distances(x, centers)
    xp, cp = permutedims(x), permutedims(centers)
    product = size(centers, 1) == 1 ? reshape(BLAS.gemv('T', xp, vec(cp)), :, 1) : BLAS.gemm('T', 'N', xp, cp)
    product .*= -2.0
    product .+= reshape(row_norms(centers), 1, :)
    product .+= reshape(row_norms(x), :, 1)
    return permutedims(max.(product, 0.0))
end

function row_potentials(distances)
    size(distances, 1) == 1 && return [BLAS.dot(vec(distances), ones(size(distances, 2)))]
    return BLAS.gemv('T', permutedims(distances), ones(size(distances, 2)))
end

function initial_centers(x, k, draws)
    trials = 2 + floor(Int, log(k))
    length(draws) == 1 + (k - 1) * trials || throw(DimensionMismatch("supply one first-center draw and all greedy candidate uniforms"))
    all(value -> 0 <= value < 1, draws) || throw(ArgumentError("center uniforms must lie in [0,1)"))
    cumulative = sequential_cumsum(fill(1 / size(x, 1), size(x, 1)))
    cumulative ./= last(cumulative)
    initial = [min(searchsortedlast(cumulative, draws[1]) + 1, size(x, 1))]
    closest = vec(squared_distances(x, x[initial, :]))
    potential = only(row_potentials(reshape(closest, 1, :)))
    for center in 2:k
        cumulative = sequential_cumsum(closest)
        positions = [min(searchsortedfirst(cumulative, draws[1 + (center - 2) * trials + trial] * potential), size(x, 1)) for trial in 1:trials]
        distances = min.(squared_distances(x, x[positions, :]), transpose(closest))
        potentials = row_potentials(distances)
        selected = argmin(potentials)
        push!(initial, positions[selected])
        closest = distances[selected, :]
        potential = potentials[selected]
    end
    return x[initial, :], initial
end

function assign_labels(x, centers)
    scores = repeat(row_norms(centers), 1, size(x, 1))
    BLAS.gemm!('T', 'N', -2.0, permutedims(centers), permutedims(x), 1.0, scores)
    return [argmin(view(scores, :, row)) for row in axes(x, 1)]
end

function squared_shift(a, b)
    result = 0.0
    index = 1
    while index + 3 <= length(a)
        d0, d1, d2, d3 = a[index] - b[index], a[index+1] - b[index+1], a[index+2] - b[index+2], a[index+3] - b[index+3]
        result += ((d0 * d0 + d1 * d1) + d2 * d2) + d3 * d3
        index += 4
    end
    while index <= length(a)
        value = a[index] - b[index]
        result += value * value
        index += 1
    end
    return result
end

"""
Fit one finite unweighted Lloyd run using the public greedy k-means++ uniforms,
mean-centered Float64 data, 2+floor(log(k)) trials, the mean-variance-scaled
tolerance, strict label convergence and the final assignment pass. This bounded
book implementation requires every fitted cluster to remain nonempty.
"""
function fit_kmeans(values, k, draws; max_iter=300, tol=1e-4)
    all(isfinite, values) && 2 <= k < size(values, 1) || throw(ArgumentError("finite observations and 2<=k<n are required"))
    average = column_average(values)
    x = Float64.(values .- average)
    variances = vec(column_average(x .* x))
    tolerance = pairwise_sum(variances) / length(variances) * tol
    centers, initial = initial_centers(x, k, draws)
    previous = fill(0, size(x, 1))
    labels = copy(previous)
    iterations = 0
    strict = false
    for iteration in 1:max_iter
        labels = assign_labels(x, centers)
        updated = zeros(size(centers))
        counts = zeros(Int, k)
        for row in axes(x, 1)
            group = labels[row]
            counts[group] += 1
            for column in axes(x, 2)
                updated[group, column] += x[row, column]
            end
        end
        all(>(0), counts) || throw(ArgumentError("this candidate requires empty-cluster relocation"))
        for group in 1:k
            reciprocal = 1.0 / counts[group]
            for column in axes(x, 2)
                updated[group, column] *= reciprocal
            end
        end
        shifts = [sqrt(squared_shift(view(centers, group, :), view(updated, group, :)))^2 for group in 1:k]
        shift = pairwise_sum(shifts)
        centers = updated
        iterations = iteration
        if labels == previous
            strict = true
            break
        end
        shift <= tolerance && break
        previous = copy(labels)
    end
    strict || (labels = assign_labels(x, centers))
    inertia = 0.0
    for row in axes(x, 1)
        inertia += squared_shift(view(x, row, :), view(centers, labels[row], :))
    end
    return KMeansFit(labels .- 1, centers .+ average, initial, iterations, inertia)
end

export KMeansFit, fit_kmeans

end
