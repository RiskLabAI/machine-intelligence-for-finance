module BookTailFit

using RiskLabAI.Backtest: generalized_pareto_tail_risk

export fit_public_pot, pareto_objective

"""Evaluate the zero-location Pareto likelihood with the public support penalty."""
function pareto_objective(parameters,excesses)
    shape,scale = parameters
    isfinite(shape) && isfinite(scale) && scale>0 || return Inf
    terms = Float64[]
    invalid = 0
    for excess in excesses
        value = excess/scale
        if value<0 || (shape<0 && value > -1/shape)
            invalid += 1
            continue
        end
        density = shape==0 ? -value : shape == -1 ? 0. : -(shape+1)*log1p(shape*value)/shape
        if isfinite(density)
            push!(terms,density)
        else
            invalid += 1
        end
    end
    return -sum(terms)+invalid*log(floatmax(Float64))*100+length(excesses)*log(scale)
end

"""Fit the public two-parameter simplex with its original initialization and tolerances."""
function public_simplex(excesses)
    vertices = [[1.,1.],[1.05,1.],[1.,1.05]]
    evaluations = 0
    function objective(point)
        evaluations>=400 && error("Pareto simplex exceeded its public evaluation limit")
        evaluations += 1
        return pareto_objective(point,excesses)
    end
    values = objective.(vertices)
    iterations = 1
    while iterations<400 && evaluations<400
        order = sortperm(values)
        vertices,values = vertices[order],values[order]
        if maximum(maximum(abs.(vertex.-vertices[1])) for vertex in vertices[2:3])<=1e-4 && maximum(abs.(values[2:3].-values[1]))<=1e-4
            return (;parameters=vertices[1],objective=values[1],iterations,evaluations)
        end
        center = (vertices[1].+vertices[2])./2
        reflected = 2 .*center.-vertices[3]
        reflection = objective(reflected)
        shrink = false
        if reflection<values[1]
            expanded = 3 .*center.-2 .*vertices[3]
            expansion = objective(expanded)
            vertices[3],values[3] = expansion<reflection ? (expanded,expansion) : (reflected,reflection)
        elseif reflection<values[2]
            vertices[3],values[3] = reflected,reflection
        elseif reflection<values[3]
            contracted = 1.5 .*center.-.5 .*vertices[3]
            contraction = objective(contracted)
            if contraction<=reflection
                vertices[3],values[3] = contracted,contraction
            else
                shrink = true
            end
        else
            contracted = .5 .*center.+.5 .*vertices[3]
            contraction = objective(contracted)
            if contraction<values[3]
                vertices[3],values[3] = contracted,contraction
            else
                shrink = true
            end
        end
        if shrink
            for index in 2:3
                vertices[index] = vertices[1].+.5 .*(vertices[index].-vertices[1])
                values[index] = objective(vertices[index])
            end
        end
        iterations += 1
    end
    error("Pareto simplex did not converge within its public iteration limits")
end

"""Use the public numerical fitting contract with native Pareto risk formulas."""
function fit_public_pot(losses;threshold,probability)
    all(isfinite,losses) && isfinite(threshold) || throw(ArgumentError("finite losses and threshold required"))
    excesses = Float64.(losses[losses.>threshold].-threshold)
    length(excesses)>=2 && maximum(excesses)>minimum(excesses) || throw(ArgumentError("two distinct positive excesses required"))
    result = public_simplex(excesses)
    shape,scale = result.parameters
    all(1 .+ shape.*excesses./scale .>0) || error("Pareto fit violates its support")
    rate = length(excesses)/length(losses)
    risk = generalized_pareto_tail_risk(threshold,rate,shape,scale,probability)
    return (;risk...,shape,scale,threshold,exceedance_count=length(excesses),sample_count=length(losses),exceedance_probability=rate,iterations=result.iterations,evaluations=result.evaluations,negative_log_likelihood=result.objective)
end

end
