module BookDecisionStudy

using LinearAlgebra,Statistics,Dates,Combinatorics,Zygote
using RiskLabAI.Optimization: spo_plus

export decision_design,train_decision_rule,predict_costs

function decision_design(values,dates)
    n = size(values,1)-60
    features = zeros(n*16,6)
    for t in 1:n,i in 1:16
        target = t+60
        for (column,window) in enumerate([5,20,60])
            observations = values[target-window:target-1,i]
            features[(t-1)*16+i,column] = mean(observations)
            features[(t-1)*16+i,column+3] = std(observations)
        end
    end
    training_count = count(<=(Date(2021,12,31)),dates[61:end])
    training = features[1:16training_count,:]
    center = vec(mean(training;dims=1))
    scale = vec(std(training;dims=1))
    design = hcat(ones(n*16),(features.-center')./scale')
    members = collect(combinations(1:16,4))
    vertices = zeros(length(members),16)
    for (row,indices) in enumerate(members)
        vertices[row,indices] .= 0.25
    end
    return (;features,design,center,scale,vertices,members,dates=dates[61:end],costs=-values[61:end,:],training_count)
end

predict_costs(design,beta) = permutedims(reshape(design*beta,16,:))

function train_decision_rule(design,costs,vertices,batches,loss_name;trace=nothing)
    beta,moment,second_moment = zeros(7),zeros(7),zeros(7)
    history = Float64[]
    for step in axes(batches,1)
        indices = Int.(batches[step,:])
        rows = reduce(vcat,[(index-1)*16 .+ (1:16) for index in indices])
        batch_design,truth = design[rows,:],costs[indices,:]
        objective = loss_name=="Squared error" ? b->mean(abs2,predict_costs(batch_design,b)-truth) : b->mean(spo_plus(predict_costs(batch_design,b),truth,vertices))
        loss,back = Zygote.pullback(objective,beta)
        gradient = only(back(1.0))
        prediction = predict_costs(batch_design,beta)
        moment .= moment.+(gradient.-moment).*(1-0.9)
        second_moment .= 0.999second_moment.+(1-0.999).*gradient.^2
        denominator = sqrt.(second_moment)./sqrt(1-0.999^step).+1e-8
        beta .-= (0.02/(1-0.9^step)).*moment./denominator
        push!(history,loss)
        if !isnothing(trace)
            trace(step,loss,gradient,beta,prediction,truth)
        end
    end
    return (;beta,history)
end

end
