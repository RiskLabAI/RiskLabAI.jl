"""
Structural-break features — native Julia port mirroring the Python
`RiskLabAI.features.structural_breaks` API (López de Prado, AFML Ch. 17): the
ADF design-matrix construction and the (Backward) Supremum ADF explosiveness
test for detecting bubbles.

Representation note (deliberate divergence): pandas Series/DataFrames become
plain `Vector`s / `Matrix`es. `prepare_data` returns `(y, x, index)` with the
ADF regression design (lagged level first, then lagged deltas, then the
constant/trend terms), matching pandas' `diff`/`shift`/`dropna` alignment
exactly (verified in `test/runtests.jl`). OLS uses `LinearAlgebra` (LAPACK), so
the t-statistics match the NumPy/`statsmodels`-style implementation.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 17.
"""

using LinearAlgebra: inv, SingularException

"""
    lag_dataframe(data, lags) -> Matrix{Float64}

Lagged copies of `data` (a vector) as columns: an integer `lags` produces lags
`0:lags`, a vector produces those specific lags. Shifted-out leading entries are
`NaN`. Mirrors Python's `lag_dataframe`.
"""
function lag_dataframe(data::AbstractVector{<:Real}, lags::Union{Integer,AbstractVector{<:Integer}})
    lag_list = lags isa Integer ? collect(0:lags) : collect(lags)
    n = length(data)
    out = fill(NaN, n, length(lag_list))
    for (c, lag) in enumerate(lag_list)
        for i = (lag+1):n
            out[i, c] = data[i-lag]
        end
    end
    return out
end

"""
    prepare_data(log_price, constant, lags) -> (y, x, index)

Build the ADF regression matrices from a log-price vector: `y` is the change in
log price, and `x` has the lagged level (first column), `lags` lagged deltas, and
the requested deterministic terms (`"nc"`, `"c"`, `"ct"`, `"ctt"`). `index` is
the 1-based position in `log_price` of each retained row. Mirrors Python's
`prepare_data`.
"""
function prepare_data(log_price::AbstractVector{<:Real}, constant::AbstractString, lags::Integer)
    n = length(log_price)
    delta(k) = log_price[k] - log_price[k-1]          # d[k] = p[k] - p[k-1], k ≥ 2
    kept = collect((lags+2):n)
    m = length(kept)
    n_constant = constant == "nc" ? 0 : constant == "c" ? 1 : constant == "ct" ? 2 : 3
    n_cols = 1 + lags + n_constant
    y = Vector{Float64}(undef, m)
    x = Matrix{Float64}(undef, m, n_cols)
    for (r, k) in enumerate(kept)
        y[r] = delta(k)
        col = 1
        x[r, col] = log_price[k-1]                    # level_l1
        col += 1
        for i = 1:lags
            x[r, col] = delta(k - i)                  # delta_l{i}
            col += 1
        end
        if constant != "nc"
            x[r, col] = 1.0
            col += 1                                   # constant
        end
        if constant == "ct" || constant == "ctt"
            x[r, col] = k
            col += 1                                    # trend (1-based position)
        end
        if constant == "ctt"
            x[r, col] = k^2                             # trend squared
            col += 1
        end
    end
    return (y = y, x = x, index = kept)
end

"""
    compute_beta(y_window, x_window) -> (beta_mean, beta_variance)

OLS coefficients and their variance-covariance matrix. Returns `NaN`-filled
arrays for a singular design. Mirrors Python's `compute_beta`.
"""
function compute_beta(y_window::AbstractVecOrMat{<:Real}, x_window::AbstractMatrix{<:Real})
    p = size(x_window, 2)
    try
        xtx_inv = inv(x_window' * x_window)
        beta_mean = xtx_inv * (x_window' * y_window)
        residual = y_window - x_window * beta_mean
        variance_e = (residual' * residual) / (size(x_window, 1) - p)
        return (beta_mean, variance_e .* xtx_inv)
    catch err
        err isa SingularException || err isa LinearAlgebra.LAPACKException || rethrow()
        return (fill(NaN, p), fill(NaN, p, p))
    end
end

# t-statistic on the lagged-level coefficient (the ADF statistic).
function _adf_t_statistic(beta_mean, beta_variance)
    isnan(beta_variance[1, 1]) && return NaN
    std_level = sqrt(beta_variance[1, 1])
    std_level == 0 && return beta_mean[1] < 0 ? -Inf : Inf
    return beta_mean[1] / std_level
end

"""
    get_expanding_window_adf(log_price, min_sample_length, constant, lags) -> (index, statistics)

ADF t-statistic over an expanding window (from the start up to each point with at
least `min_sample_length` rows). Mirrors Python's `get_expanding_window_adf`.
"""
function get_expanding_window_adf(
    log_price::AbstractVector{<:Real},
    min_sample_length::Integer,
    constant::AbstractString,
    lags::Integer,
)
    y, x, index = prepare_data(log_price, constant, lags)
    n = length(y)
    statistics = Float64[]
    times = eltype(index)[]
    for i = min_sample_length:n
        beta_mean, beta_variance = compute_beta(view(y, 1:i), view(x, 1:i, :))
        push!(statistics, _adf_t_statistic(beta_mean, beta_variance))
        push!(times, index[i])
    end
    return (index = times, statistics = statistics)
end

"""
    get_bsadf_statistic(log_price, min_sample_length, constant, lags) -> (time, bsadf)

Backward Supremum ADF: the supremum of the ADF t-statistic over all backward
expanding windows ending at the last observation (bubble-origination test).
`time` is the last 1-based position of `log_price`. Mirrors Python's
`get_bsadf_statistic`.
"""
function get_bsadf_statistic(
    log_price::AbstractVector{<:Real},
    min_sample_length::Integer,
    constant::AbstractString,
    lags::Integer,
)
    y, x, _ = prepare_data(log_price, constant, lags)
    n = length(y)
    bsadf = -Inf
    for start = 1:(n-min_sample_length+1)
        beta_mean, beta_variance = compute_beta(view(y, start:n), view(x, start:n, :))
        isnan(beta_variance[1, 1]) && continue
        t_statistic = _adf_t_statistic(beta_mean, beta_variance)
        t_statistic > bsadf && (bsadf = t_statistic)
    end
    return (time = length(log_price), bsadf = bsadf)
end
