module BookHedging

using RiskLabAI, CSV, DataFrames, Dates, Statistics, TOML, SHA

function public_volatility(path)
    frame = CSV.read(path,DataFrame;missingstring=[".",""],types=Dict(:NASDAQCOM=>Float64))
    frame.observation_date = Date.(frame.observation_date)
    design = frame[Date(1994,1,4).<=frame.observation_date.<=Date(2014,12,31),:]
    observed = dropmissing(design,:NASDAQCOM)
    prices = Float64.(observed.NASDAQCOM)
    returns = log.(prices[2:end]./prices[1:end-1])
    observed[!,"Log return"] = vcat(missing,returns)
    return frame,design,observed,returns,std(returns)*sqrt(252)
end

function primitive(root,name)
    manifest = TOML.parsefile(joinpath(root,"inputs.toml"))["files"]
    path = joinpath(root,name)
    bytes2hex(sha256(read(path)))==manifest[name]["sha256"] || error("The fixed hedging input hash differs")
    return path
end

function hedge_paths(root,seed,count,steps,sigma;maturity=30/365)
    path = primitive(root,"normal_$(seed)_$(count)_$(steps).csv")
    draws = Matrix{Float64}(CSV.read(path,DataFrame))
    return RiskLabAI.Backtest.gbm_hedging_paths(;spot=1.0,maturity,volatility=sigma,drift=0.0,steps,paths=count,seed,normal_draws=draws)
end

function read_parameters(path)
    content = TOML.parsefile(path)
    matrix(name) = permutedims(reduce(hcat,content[name]))
    return (weight1=matrix("weight1"),bias1=Float64.(content["bias1"]),weight2=matrix("weight2"),bias2=Float64.(content["bias2"]),weight3=matrix("weight3"),bias3=Float64.(content["bias3"]))
end

function initial_model(root,seed)
    return RiskLabAI.Backtest.DeepHedge(16;seed,parameters=read_parameters(primitive(root,"initial_$(seed).toml")))
end

function save_parameters(path,parameters)
    content = Dict(string(name)=>(ndims(value)==2 ? [collect(row) for row in eachrow(value)] : collect(value)) for (name,value) in pairs(parameters))
    open(path,"w") do stream
        TOML.print(stream,content)
    end
    return nothing
end

function cashflow_components(prices,positions,strike,cost_rate;premium=0.0)
    zero = zeros(size(positions,1),1)
    trades = diff(hcat(zero,positions,zero);dims=2)
    gains = vec(sum(positions.*diff(prices;dims=2);dims=2))
    turnover = vec(sum(prices.*abs.(trades);dims=2))
    costs = cost_rate.*turnover
    payoff = max.(prices[:,end].-strike,0.0)
    pnl = premium.+gains.-costs.-payoff
    return (;trades,gains,turnover,costs,payoff,pnl)
end

function metric_record(name,prices,positions,strike,cost_rate,risk_aversion)
    components = cashflow_components(prices,positions,strike,cost_rate)
    pnl = components.pnl
    cutoff = quantile(pnl,.05)
    return (Strategy=name,risk=RiskLabAI.Backtest.entropic_risk(pnl,risk_aversion),mean=mean(pnl),sd=std(pnl),fifth=cutoff,tail=-mean(pnl[pnl.<=cutoff]),turnover=mean(components.turnover),costs=mean(components.costs),payoff=mean(components.payoff))
end

const METRIC_COLUMNS = ["Strategy","Entropic risk","Mean cash flow","Cash flow standard deviation","Fifth percentile","Mean loss in worst five percent","Mean turnover","Mean trading cost","Mean payoff"]

export public_volatility,primitive,hedge_paths,initial_model,read_parameters,save_parameters,cashflow_components,metric_record,METRIC_COLUMNS

end
