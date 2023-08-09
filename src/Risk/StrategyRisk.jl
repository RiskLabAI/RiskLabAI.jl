using Distributions
using Random
using Statistics
using SymPy

"""
Calculate the Sharpe ratio as a function of the number of bets.

Reference: De Prado, M. (2018) Advances in financial machine learning. Page 213, Snippet 15.1
"""
function sharpe_ratio_trials(p, n_run)
    result = []
    for i in 1:n_run
        b = Binomial(1, p)
        random = rand(b, 1)
        if random[1] == 1
            x = 1
        else 
            x = -1
        end
        append!(result, [x])
    end
    return (mean(result), std(result), mean(result) / std(result))
end 

"""
Use the SymPy library for symbolic operations.

Reference: De Prado, M. (2018) Advances in financial machine learning. Page 214, Snippet 15.2
"""
function target_sharpe_ratio_symbolic()
    p, u, d = symbols("p u d")
    m2 = p * u^2 + (1 - p) * d^2
    m1 = p * u + (1 - p) * d
    v = m2 - m1^2
    factor(v) 
end

"""
Compute implied precision.

Reference: De Prado, M. (2018) Advances in financial machine learning. Page 214, Snippet 15.3
"""
function implied_precision(stop_loss, profit_taking, freq, target_sharpe_ratio)
    a = (freq + target_sharpe_ratio^2) * (profit_taking - stop_loss)^2
    b = (2 * freq * stop_loss - target_sharpe_ratio^2 * (profit_taking - stop_loss)) * (profit_taking - stop_loss)
    c = freq * stop_loss^2
    precision = (-b + (b^2 - 4 * a * c)^0.5) / (2 * a)
    return precision
end

"""
Compute the number of bets per year needed to achieve a Sharpe ratio with a certain precision rate.

Reference: De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4
"""
function bin_frequency(stop_loss, profit_taking, precision, target_sharpe_ratio)
    freq = (target_sharpe_ratio * (profit_taking - stop_loss))^2 * precision * (1 - precision) / ((profit_taking - stop_loss) * precision + stop_loss)^2
    bin_sr(sl0, pt0, freq0, p0) = (((pt0 - sl0) * p0 + sl0) * freq0^0.5) / ((pt0 - sl0) * (p0 * (1 - p0))^0.5)
    if !isapprox(bin_sr(stop_loss, profit_taking, freq, precision), target_sharpe_ratio, atol = 0.5)
        return nothing
    end
    return freq
end

"""
Calculate the strategy risk in practice.

Reference: De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4
"""
function mix_gaussians(μ1, μ2, σ1, σ2, probability1, n_obs)
    return1 = rand(Normal(μ1, σ1), trunc(Int, n_obs * probability1))
    return2 = rand(Normal(μ2, σ2), trunc(Int, n_obs) - trunc(Int, n_obs * probability1))
    returns = append!(return1, return2)
    shuffle!(returns)
    return returns
end 

function failure_probability(returns, freq, target_sharpe_ratio)
    r_positive, r_negative = mean(returns[returns .> 0]), mean(returns[returns .<= 0])
    p = size(returns[returns .> 0], 1) / size(returns, 1)
	threshold_p = implied_precision(r_negative, r_positive, freq, target_sharpe_ratio)
	risk = cdf(Normal(p, p * (1 - p)), threshold_p)
	return risk
end

function calculate_strategy_risk(μ1, μ2, σ1, σ2, probability1, n_obs, freq, target_sharpe_ratio)
    returns = mix_gaussians(μ1, μ2, σ1, σ2, probability1, n_obs)
    probability_fail = failure_probability(returns, freq, target_sharpe_ratio)
    println("Probability strategy will fail: ", probability_fail)
    return probability_fail
end
