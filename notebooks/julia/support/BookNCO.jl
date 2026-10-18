module BookNCO

using LinearAlgebra, Statistics
using RiskLabAI.Cluster: cluster_k_means_base,covariance_to_correlation
using RiskLabAI.Optimization: get_optimal_portfolio_weights,get_optimal_portfolio_weights_nco
include(joinpath(@__DIR__,"BookKMeans.jl"))

export nco_components,cluster_components,block_covariance

function cluster_components(correlation,draws;maximum_clusters=4,trace=nothing)
    step = Ref(0)
    function provider(features,k,rng)
        step[] += 1
        count = 1+(k-1)*(2+floor(Int,log(k)))
        fitted = BookKMeans.fit_kmeans(features,k,draws[step[],2:count+1])
        if !isnothing(trace)
            trace(step[],k,features,fitted)
        end
        return fitted.labels
    end
    return cluster_k_means_base(correlation;max_clusters=maximum_clusters,iterations=10,random_state=0,label_provider=provider)
end

function nco_components(covariance,draws;mu=nothing,maximum_clusters=4,trace=nothing)
    correlation = covariance_to_correlation(covariance)
    sorted_correlation,clusters,silhouette = cluster_components(correlation,draws;maximum_clusters,trace)
    keys_sorted = sort(collect(keys(clusters)))
    within = zeros(size(covariance,1),length(keys_sorted))
    for (column,key) in enumerate(keys_sorted)
        members = clusters[key]
        local_mean = isnothing(mu) ? nothing : mu[members]
        within[members,column] = get_optimal_portfolio_weights(covariance[members,members];mu=local_mean)
    end
    reduced_covariance = within'*covariance*within
    reduced_mean = isnothing(mu) ? nothing : within'*mu
    between = get_optimal_portfolio_weights(reduced_covariance;mu=reduced_mean)
    final = within*between
    released = get_optimal_portfolio_weights_nco(covariance;mu,number_clusters=maximum_clusters,clusters)
    return (;correlation,sorted_correlation,clusters,silhouette,keys_sorted,within,reduced_covariance,between,final,released)
end

function block_covariance(blocks,block_size,within,between=0.03)
    n = blocks*block_size
    correlation = fill(Float64(between),n,n)
    for block in 0:blocks-1
        indices = block*block_size.+(1:block_size)
        correlation[indices,indices] .= within
    end
    correlation[diagind(correlation)] .= 1.0
    scales = 0.10 .+ collect(0:n-1).*((0.22-0.10)/(n-1))
    scales[end] = 0.22
    return correlation.*(scales*scales'),correlation
end

end
