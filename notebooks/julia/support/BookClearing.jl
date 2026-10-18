module BookClearing

using LinearAlgebra,Statistics
using RiskLabAI.Optimization: eisenberg_noe_clearing

export ring_matrix,complete_matrix,core_periphery_matrix,community_matrix,greatest_iteration,clearing_metrics,network_structures

function ring_matrix(n,internal_share=0.8)
    matrix = zeros(n,n)
    for debtor in 1:n
        matrix[debtor,mod1(debtor+1,n)] = internal_share
    end
    return matrix
end

function complete_matrix(n,internal_share=0.8)
    matrix = fill(internal_share/(n-1),n,n)
    matrix[diagind(matrix)] .= 0
    return matrix
end

function core_periphery_matrix(n,internal_share=0.8,core_size=4)
    matrix = zeros(n,n)
    core,periphery = collect(1:core_size),collect(core_size+1:n)
    for debtor in 1:n
        if debtor<=core_size
            other_core = filter(!=(debtor),core)
            matrix[debtor,other_core] .= 0.4/length(other_core)
            matrix[debtor,periphery] .= 0.4/length(periphery)
        else
            matrix[debtor,core] .= 0.64/length(core)
            others = filter(!=(debtor),periphery)
            matrix[debtor,others] .= 0.16/length(others)
        end
    end
    return matrix*(internal_share/0.8)
end

function community_matrix(n,internal_share=0.8)
    matrix = zeros(n,n)
    half = n÷2
    for debtor in 1:n
        own = debtor<=half ? collect(1:half) : collect(half+1:n)
        other = debtor<=half ? collect(half+1:n) : collect(1:half)
        own = filter(!=(debtor),own)
        matrix[debtor,own] .= 0.68/length(own)
        matrix[debtor,other] .= 0.12/length(other)
    end
    return matrix*(internal_share/0.8)
end

network_structures(n=12) = ["Ring"=>ring_matrix(n),"Complete"=>complete_matrix(n),"Core-periphery"=>core_periphery_matrix(n),"Two-community"=>community_matrix(n)]

function greatest_iteration(external,nominal,relative;tolerance=1e-12,maximum_steps=10000)
    payments = copy(nominal)
    history = [copy(payments)]
    for step in 1:maximum_steps
        updated = min.(nominal,external+relative'*payments)
        push!(history,copy(updated))
        maximum(abs.(updated-payments))<=tolerance && return updated,permutedims(reduce(hcat,history))
        payments = updated
    end
    error("Monotone iteration did not converge.")
end

function clearing_metrics(external,nominal,relative)
    payments = eisenberg_noe_clearing(external,nominal,relative)
    fixed = min.(nominal,external+relative'*payments)
    unpaid = max.(nominal-payments,0)
    return (;payments,unpaid,total_payments=sum(payments),total_unpaid=sum(unpaid),default_count=count(>(1e-8),unpaid),payment_ratio=sum(nominal)==0 ? 1.0 : sum(payments)/sum(nominal),fixed_point_residual=isempty(payments) ? 0.0 : maximum(abs.(payments-fixed)))
end

end
