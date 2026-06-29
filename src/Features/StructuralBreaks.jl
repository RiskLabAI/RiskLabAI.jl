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
using Statistics: quantile
using Random: AbstractRNG, default_rng, randn

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

# --------------------------------------------------------------------------- #
# GSADF / BSADF (Phillips–Shi–Yu 2015): the generalized sup-ADF and its backward
# sequence, with date-stamping and a finite-sample critical-value simulator.
#
# de Prado (AFML ch.17) ships the single-window SADF: the forward-expanding ADF is
# anchored at the sample origin, so several explosive episodes dilute and mask one
# another and SADF collapses them to about one. GSADF varies *both* window endpoints
# and the backward BSADF sequence dates each origination, recovering and counting
# multiple bubbles where SADF sees one. Clean-room from Phillips–Shi–Yu (2015),
# building on the validated `get_bsadf_statistic` / `get_expanding_window_adf`;
# numeric parity asserted against the Python reference in `test/runtests.jl`.
# Admitted in Appraisal 05 (`library_extension/appraisals/05_verdict.md`).
# --------------------------------------------------------------------------- #

"""
    psy_minimum_window(sample_length) -> Int

Phillips–Shi–Yu minimum window length in observations: `round(r0·T)` with
`r0 = 0.01 + 1.8/√T`, floored at 3. Mirrors Python's `psy_minimum_window`.
"""
function psy_minimum_window(sample_length::Integer)
    r0 = 0.01 + 1.8 / sqrt(sample_length)
    return max(round(Int, r0 * sample_length), 3)
end

# Fast O(T²) prefix-sum kernel for the PSY-standard ADF specification (intercept,
# no lag augmentation: Δyₜ = α + β·yₜ₋₁ + eₜ). Returns `(sadf, bsadf)`, each length
# `length(y)`, indexed by window endpoint (1-based), `NaN` before the first valid
# endpoint. Mirrors Python's `_psy_sadf_bsadf_sequences`.
function _psy_sadf_bsadf_sequences(y::AbstractVector{<:Real}, min_sample_length::Integer)
    yv = Float64.(y)
    T = length(yv)
    sadf = fill(NaN, T)
    bsadf = fill(NaN, T)
    T < min_sample_length + 1 && return (sadf, bsadf)

    # Regression observation i uses level xᵢ = y[i] and Δᵢ = y[i+1] - y[i].
    xreg = yv[1:(T-1)]
    d = yv[2:T] .- yv[1:(T-1)]

    # Prefix sums with a leading 0 so any window's sufficient statistics are O(1).
    cx = [0.0; cumsum(xreg)]
    cxx = [0.0; cumsum(xreg .^ 2)]
    cd = [0.0; cumsum(d)]
    cxd = [0.0; cumsum(xreg .* d)]
    cdd = [0.0; cumsum(d .* d)]
    g(c, i) = @inbounds c[i+1]          # fetch the Python 0-based prefix entry c[i]

    nmin = min_sample_length
    for r2 = nmin:(T-1)                  # 0-based endpoint
        best = -Inf
        sadf_val = NaN
        for r1 = 0:(r2-nmin)            # 0-based window start
            n = r2 - r1                  # regression observations [r1, r2)
            sx = g(cx, r2) - g(cx, r1)
            sxx = g(cxx, r2) - g(cxx, r1)
            sd = g(cd, r2) - g(cd, r1)
            sxd = g(cxd, r2) - g(cxd, r1)
            sdd = g(cdd, r2) - g(cdd, r1)
            den = n * sxx - sx * sx
            valid = false
            t_stat = NaN
            if den > 0.0 && n >= 3
                beta = (n * sxd - sx * sd) / den
                alpha = (sd - beta * sx) / n
                ssr = max(sdd - alpha * sd - beta * sxd, 0.0)
                var_beta = (ssr / (n - 2)) * n / den
                if var_beta > 0.0
                    t_stat = beta / sqrt(var_beta)
                    valid = true
                end
            end
            tt = valid ? t_stat : -Inf
            tt > best && (best = tt)
            # r1 == 0 is the forward-expanding (origin-anchored) ADF = SADF.
            r1 == 0 && (sadf_val = valid ? t_stat : NaN)
        end
        bsadf[r2+1] = isfinite(best) ? best : NaN
        sadf[r2+1] = sadf_val
    end
    return (sadf, bsadf)
end

"""
    get_sadf_sequence(log_price, min_sample_length, constant="c", lags=0) -> (index, values)

Forward-expanding ADF (SADF) sequence indexed by window endpoint (1-based position
in the NaN-dropped series). The date-stamping sequence for de Prado's single-window
SADF. Mirrors Python's `get_sadf_sequence`.
"""
function get_sadf_sequence(
    log_price::AbstractVector{<:Real},
    min_sample_length::Integer,
    constant::AbstractString = "c",
    lags::Integer = 0,
)
    clean_idx = findall(!isnan, log_price)
    price = log_price[clean_idx]
    T = length(price)
    if constant == "c" && lags == 0
        sadf, _ = _psy_sadf_bsadf_sequences(price, min_sample_length)
        rng_idx = (min_sample_length+1):T
        return (index = clean_idx[rng_idx], values = sadf[rng_idx])
    end
    res = get_expanding_window_adf(price, min_sample_length, constant, lags)
    return (index = clean_idx[res.index], values = res.statistics)
end

"""
    get_bsadf_sequence(log_price, min_sample_length, constant="c", lags=0) -> (index, values)

Backward sup-ADF (BSADF) sequence indexed by window endpoint (1-based position in
the NaN-dropped series): for each endpoint the supremum of the ADF t-statistic over
all window starts, so a later episode is tested on its own sub-sample rather than
diluted by the earlier sample. For the standard intercept-only, no-lag spec this
uses the fast prefix-sum kernel; otherwise it reuses `get_bsadf_statistic` on
expanding prefixes. Mirrors Python's `get_bsadf_sequence`.
"""
function get_bsadf_sequence(
    log_price::AbstractVector{<:Real},
    min_sample_length::Integer,
    constant::AbstractString = "c",
    lags::Integer = 0,
)
    clean_idx = findall(!isnan, log_price)
    price = log_price[clean_idx]
    T = length(price)
    if constant == "c" && lags == 0
        _, bsadf = _psy_sadf_bsadf_sequences(price, min_sample_length)
        rng_idx = (min_sample_length+1):T
        return (index = clean_idx[rng_idx], values = bsadf[rng_idx])
    end
    values = Float64[]
    times = Int[]
    first_end = min_sample_length + lags + 1
    for e = first_end:T
        res = get_bsadf_statistic(view(price, 1:e), min_sample_length, constant, lags)
        push!(values, res.bsadf)
        push!(times, clean_idx[e])
    end
    return (index = times, values = values)
end

"""
    get_gsadf_statistic(log_price, min_sample_length, constant="c", lags=0) -> Float64

Generalized sup-ADF (GSADF) statistic: the supremum of the BSADF sequence over all
flexible windows. A series is flagged as containing at least one explosive episode
when the GSADF exceeds its finite-sample critical value
(`simulate_psy_critical_values`). Returns `NaN` if the series is too short.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer GSADF/BSADF over single-window SADF when a series may contain more than one
explosive episode (it recovers and counts each). For a single suspected bubble,
SADF is at least as good. Use seasonally-adjusted data and the simulated
finite-sample critical values; GSADF over-flags strongly trending or seasonal
series and is mildly oversized in large samples, so treat broad flags cautiously.

Reference: Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for multiple
bubbles. International Economic Review, 56(4), 1043–1078.
"""
function get_gsadf_statistic(
    log_price::AbstractVector{<:Real},
    min_sample_length::Integer,
    constant::AbstractString = "c",
    lags::Integer = 0,
)
    seq = get_bsadf_sequence(log_price, min_sample_length, constant, lags)
    finite = filter(isfinite, seq.values)
    isempty(finite) && return NaN
    return maximum(finite)
end

"""
    get_bubble_episodes(statistic_index, statistic_values, critical_value; min_duration=1)

Date-stamp explosive episodes from a sup-ADF sequence: a maximal run of endpoints
where the statistic exceeds its critical value (scalar or per-endpoint vector)
lasting at least `min_duration` periods. Returns a vector of
`(origination, collapse)` index labels. Mirrors Python's `get_bubble_episodes`.
"""
function get_bubble_episodes(
    statistic_index::AbstractVector,
    statistic_values::AbstractVector{<:Real},
    critical_value::Union{Real,AbstractVector{<:Real}};
    min_duration::Integer = 1,
)
    stat = Float64.(statistic_values)
    n = length(stat)
    cv =
        critical_value isa AbstractVector ? Float64.(critical_value) :
        fill(Float64(critical_value), n)
    above = [isfinite(stat[i]) && isfinite(cv[i]) && stat[i] > cv[i] for i = 1:n]
    episodes = Tuple{eltype(statistic_index),eltype(statistic_index)}[]
    t = 1
    while t <= n
        if above[t]
            start = t
            while t <= n && above[t]
                t += 1
            end
            stop = t                       # first endpoint back below (exclusive)
            if stop - start >= min_duration
                collapse = stop <= n ? statistic_index[stop] : statistic_index[n]
                push!(episodes, (statistic_index[start], collapse))
            end
        else
            t += 1
        end
    end
    return episodes
end

"""
    simulate_psy_critical_values(sample_length, min_sample_length=nothing; constant="c",
                                 lags=0, n_simulations=2000, level=0.95, rng=default_rng())

Finite-sample critical values for SADF and GSADF by simulating the random-walk null
(Phillips–Shi–Yu). Returns a `NamedTuple` with `sadf_global_cv`, `gsadf_global_cv`
(scalars used for detection) and `sadf_sequence_cv`, `bsadf_sequence_cv`
(per-endpoint arrays used for date-stamping), plus the inputs.

Deliberate divergence (behavioural): the random-walk paths use Julia's `randn`/`rng`
rather than NumPy's PCG64 stream, so the critical values are reproducible under a
given `rng` but not bit-identical to the Python reference (the ADF statistics are
pivotal under the null, so the values depend only on `T`, the window and the
specification). Mirrors Python's `simulate_psy_critical_values`.
"""
function simulate_psy_critical_values(
    sample_length::Integer,
    min_sample_length::Union{Integer,Nothing} = nothing;
    constant::AbstractString = "c",
    lags::Integer = 0,
    n_simulations::Integer = 2000,
    level::Real = 0.95,
    rng::AbstractRNG = default_rng(),
)
    nmin =
        min_sample_length === nothing ? psy_minimum_window(sample_length) : min_sample_length
    fast = constant == "c" && lags == 0
    sadf_paths = fill(NaN, n_simulations, sample_length)
    bsadf_paths = fill(NaN, n_simulations, sample_length)
    for s = 1:n_simulations
        y = cumsum(randn(rng, sample_length))           # random-walk null
        if fast
            sadf, bsadf = _psy_sadf_bsadf_sequences(y, nmin)
        else
            sadf = fill(NaN, sample_length)
            bsadf = fill(NaN, sample_length)
            sseq = get_sadf_sequence(y, nmin, constant, lags)
            bseq = get_bsadf_sequence(y, nmin, constant, lags)
            sadf[sseq.index] .= sseq.values
            bsadf[bseq.index] .= bseq.values
        end
        sadf_paths[s, :] = sadf
        bsadf_paths[s, :] = bsadf
    end

    _nanmax_row(row) = (f = filter(isfinite, row); isempty(f) ? NaN : maximum(f))
    _nanquant(col, q) = (f = filter(isfinite, col); isempty(f) ? NaN : quantile(f, q))

    global_sadf = [_nanmax_row(@view sadf_paths[s, :]) for s = 1:n_simulations]
    global_gsadf = [_nanmax_row(@view bsadf_paths[s, :]) for s = 1:n_simulations]
    cols = (nmin+1):sample_length
    sadf_seq_cv = [_nanquant(@view(sadf_paths[:, c]), level) for c in cols]
    bsadf_seq_cv = [_nanquant(@view(bsadf_paths[:, c]), level) for c in cols]

    return (
        sample_length = sample_length,
        min_sample_length = nmin,
        level = level,
        sadf_global_cv = _nanquant(global_sadf, level),
        gsadf_global_cv = _nanquant(global_gsadf, level),
        sadf_sequence_cv = sadf_seq_cv,
        bsadf_sequence_cv = bsadf_seq_cv,
    )
end
