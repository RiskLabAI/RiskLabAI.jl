using Distributions
using Statistics
using PyCall
using IterTools
using DataFrames, GLM
using LinearAlgebra
using PlotlyJS

"""----------------------------------------------------------------------
function: backtesting with synthetic data
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: p.175 snippet 13.1
----------------------------------------------------------------------"""

function Syntheticbacktesting(forcast, # long run price 
                              HalfLife, # half life of model 
                              σ; # standadrd deviation that use in model
                              maximumIteration =1e3, # maximum number of Iteration
                              maximumHoldingPeriod = 100, # maximum Holding Period
                              profitTakingRange = LinRange(0.5,10,20), # profit Taking Range
                              stopLossRange = LinRange(0.5,10,20), #  stop Loss Range
                              seed = 0) # starting price 

    ϕ = 2^(-1/HalfLife) # compute ρ coeficient frome halfLife
    output  = zeros((20,20)) # initial output with zero 
    standardNormalDistribution = Normal() # create object of Normal Distribution
    for (i_idx, i) in enumerate(profitTakingRange) 
        for (j_idx,j) in enumerate(stopLossRange)
            output2 = [] # create output2 for store Profit and loss in every iteration 
            for _ in 1:maximumIteration
                price ,HoldingPeriod = seed , 0 
                while true 
                    r = rand(standardNormalDistribution) # random sample from standardNormalDistribution
                    price = (1 - ϕ) * forcast + ϕ * price + σ * r # O_U process 
                    priceDifference = price -seed # diffrence between current price and first price 
                    HoldingPeriod += 1 # increase holdingperiod by 1 
                    if priceDifference > i* σ || priceDifference <-1*j* σ || HoldingPeriod > maximumHoldingPeriod # check stop condition and then append Profit to output2
                        append!(output2,priceDifference)
                        break
                    end
                end
            end
            mean , std = Statistics.mean(output2) , Statistics.std(output2) # compute mean and standadrd deviation for fixed stop loss and profit taking 
            output[i_idx,j_idx] = mean/std # compute sharp ratio and store it 
        end
    end
    return output
end

"""----------------------------------------------------------------------
function: fitting O-U process on data
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: p.173 
----------------------------------------------------------------------"""
function fit_O_U_process(price) # vector of stock price 
    data = DataFrame(Y = price[2:end] .- price[1:end-1] , X = price[1:end-1]) # creat dataframe with one columns Y(dependent variable) and X(independent variable)
    ols = lm(@formula(Y ~ X), data) # fit OLS 
    
    #get coeficient of OLS
    ρ = GLM.coef(ols)[2] + 1  
    future = GLM.coef(ols)[1]/(1-ρ)
    σ = std(data[!,:Y] .- GLM.coef(ols)[1] .- GLM.coef(ols)[2]*data[!,:X])

    return ρ,future,σ
end

"""----------------------------------------------------------------------
function: simulate O-U process on data
reference: -
methodology: -
----------------------------------------------------------------------"""

function simulate_O_U_process(ρ, #coeficient that related to halfLife 
                              future, # long run price 
                              σ, # standadrd deviation of model 
                              P_0, # starting price 
                              periodlength) # number of day that we want to simulate 

    price = zeros(periodlength) # creat array wtih periodlength zeros
    price[1] = P_0 # set first elements of array to P_0
    standardnormaldistribution = Normal() # creat an object of Normal Distribution
    for i in 2:periodlength
        r = rand(standardnormaldistribution) # random sample of Normal Distribution
        price[i] =(1 - ρ) * future + ρ * price[i-1] +σ* r # simulate O-U process
    end
    return price
end

"""----------------------------------------------------------------------
function: backtesting with synthetic data for specefice prices 
reference: -
methodology: -
----------------------------------------------------------------------"""

function Syntheticbacktesting(price; # vector of stock price 
                              maximumIteration =1e5, # maximum number of Iteration
                              maximumHoldingPeriod = 100, # maximum Holding Period
                              profitTakingRange = LinRange(0.5,10,20), # profit Taking Range
                              stopLossRange = LinRange(0.5,10,20), # stop Loss Range
                              seed = 0) # starting price 

    ρ,future,σ = fit_O_U_process(price) # fit O-U process on data 
    out = Syntheticbacktesting(future,-1.0/log2(ρ),σ;maximumHoldingPeriod = maximumHoldingPeriod , maximumIteration =maximumIteration,seed = seed) # backtesting with recent line parameters 
    Plots.display(heatmap(profitTakingRange,stopLossRange,transpose(out),c=cgrad([:black, :white]), xlabel = "Profit-Taking",ylabel = "Stop-Loss",title = "Forecast = $(round(future,digits = 3)) | H-L=$(round((-1.0/log2(ρ)),digits = 3)) | Sigma=$(round(σ,digits = 3))"))
    return out
end



