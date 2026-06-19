"""
Synthetic backtesting — native Julia port mirroring the Python
`RiskLabAI.backtest.backtest_synthetic_data` API (López de Prado, AFML Ch. 13):
backtest a grid of profit-taking / stop-loss levels on Ornstein–Uhlenbeck price
paths.

Deliberate divergence: the noise generator is Julia's `randn` with an optional
`rng` keyword (in place of Python's global `random.gauss`), so results are
reproducible but not bit-identical to the Python draw sequence. The OU recursion,
stop logic, and summary statistics match the Python implementation.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 13.
"""

"""
    synthetic_back_testing(forecast, half_life, sigma; n_iteration=100_000,
        maximum_holding_period=100, profit_taking_range=range(0.5, 10; length=20),
        stop_loss_range=range(0.5, 10; length=20), seed=0,
        rng=Random.default_rng()) -> Vector{NTuple{5,Float64}}

For each `(profit_taking, stop_loss)` pair, simulate `n_iteration` OU price paths
`Pₜ = (1-ρ)·F + ρ·Pₜ₋₁ + σ·Zₜ` (`ρ = 2^(-1/half_life)`), exiting on the first
barrier touch or after `maximum_holding_period` steps, and summarise the exit
gains. Returns tuples `(profit_taking, stop_loss, mean, std, sharpe)`. `seed` is
the initial price. Stochastic. Mirrors Python's `synthetic_back_testing`.
"""
function synthetic_back_testing(
    forecast::Real,
    half_life::Real,
    sigma::Real;
    n_iteration::Integer = 100_000,
    maximum_holding_period::Integer = 100,
    profit_taking_range = range(0.5, 10; length = 20),
    stop_loss_range = range(0.5, 10; length = 20),
    seed::Real = 0,
    rng::AbstractRNG = Random.default_rng(),
)
    rho = 2.0^(-1.0 / half_life)
    results = NTuple{5,Float64}[]
    for profit_taking in profit_taking_range, stop_loss in stop_loss_range
        stop_returns = Float64[]
        for _ = 1:n_iteration
            price = float(seed)
            holding_period = 0
            while true
                price = (1 - rho) * forecast + rho * price + sigma * randn(rng)
                gain = price - seed
                holding_period += 1
                if gain > profit_taking ||
                   gain < -stop_loss ||
                   holding_period > maximum_holding_period
                    push!(stop_returns, gain)
                    break
                end
            end
        end
        mean_return = mean(stop_returns)
        std_return = std(stop_returns; corrected = false)
        sharpe = std_return > 0 ? mean_return / std_return : 0.0
        push!(results, (profit_taking, stop_loss, mean_return, std_return, sharpe))
    end
    return results
end
