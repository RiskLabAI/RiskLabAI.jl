using Test
using Dates
using DataFrames
using LinearAlgebra
using Random
using RiskLabAI

@testset "RiskLabAI smoke tests" begin
    # The package loads and the exported symbols actually exist
    # (the old exports referenced undefined, wrongly-cased names).
    @test isdefined(RiskLabAI, :probabilityOfBacktestOverfitting)
    @test isdefined(RiskLabAI, :generateSignal)

    # discreteSignal: rounds to step size and caps at +/- 1
    s = RiskLabAI.discreteSignal([0.26, 0.94, -1.4], 0.1)
    @test s ≈ [0.3, 0.9, -1.0]

    # selectRows: contiguous row blocks for the chosen partitions
    @test RiskLabAI.selectRows([1, 3], 2) == [1, 2, 5, 6]
end

@testset "Utils" begin
    # ewma matches the Python RiskLabAI.utils.ewma exactly (same denominator
    # 1 + (1-a) + (1-a)^2 + ...). Values computed from both implementations.
    result = RiskLabAI.ewma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    expected = [1.0, 1.6666666667, 2.4285714286, 3.2666666667, 4.1612903226]
    @test result ≈ expected atol = 1e-9

    # First element is always the seed value; output length matches input.
    @test result[1] == 1.0
    @test length(result) == 5

    # Constants are defined, with ASCII identifiers.
    @test RiskLabAI.Utils.CUMULATIVE_DOLLAR == "Cumulative Dollar Value"
    @test isascii(string(:CUMULATIVE_THETA))

    # The struct-field-inheritance macro is available.
    @test isdefined(RiskLabAI.Utils, Symbol("@field_inherit"))
end

@testset "Data.Structures — standard bars (parity with Python)" begin
    # Same fixture as the Python test_standard_bars.py.
    # Bar row layout: [date_time, idx, open, high, low, close, volume,
    #                  buy_vol, sell_vol, ticks, dollar, threshold]  (1-indexed)
    ticks = DataFrame(
        date_time = DateTime.([
            "2020-01-01T10:00:00",
            "2020-01-01T10:00:01",
            "2020-01-01T10:00:02",
            "2020-01-01T10:00:03",
            "2020-01-01T10:00:04",
            "2020-01-01T10:00:05",
            "2020-01-01T10:00:06",
        ]),
        price = [100, 101, 100, 101, 102, 103, 102],
        volume = [10, 5, 20, 10, 10, 10, 5],
    )

    # Tick bars, threshold 3
    tb =
        RiskLabAI.Data.StandardBars{RiskLabAI.Data.Tick}(bar_type = "tick", threshold = 3.0)
    bars = RiskLabAI.Data.construct_bars_from_data(tb; data = ticks)
    @test length(bars) == 2
    @test bars[1][1] == DateTime("2020-01-01T10:00:02")
    @test bars[1][3] == 100.0   # open
    @test bars[1][4] == 101.0   # high
    @test bars[1][5] == 100.0   # low
    @test bars[1][6] == 100.0   # close
    @test bars[1][10] == 3.0    # ticks
    @test bars[2][1] == DateTime("2020-01-01T10:00:05")
    @test bars[2][3] == 101.0
    @test bars[2][4] == 103.0
    @test bars[2][5] == 101.0
    @test bars[2][6] == 103.0
    @test bars[2][10] == 3.0

    # Volume bars, threshold 35
    vb = RiskLabAI.Data.StandardBars{RiskLabAI.Data.Volume}(
        bar_type = "volume",
        threshold = 35.0,
    )
    vbars = RiskLabAI.Data.construct_bars_from_data(vb; data = ticks)
    @test length(vbars) == 2
    @test vbars[1][7] == 35.0
    @test vbars[2][7] == 35.0

    # Dollar bars, threshold 3500
    db = RiskLabAI.Data.StandardBars{RiskLabAI.Data.Dollar}(
        bar_type = "dollar",
        threshold = 3500.0,
    )
    dbars = RiskLabAI.Data.construct_bars_from_data(db; data = ticks)
    @test length(dbars) == 2
    @test dbars[1][11] == 3505.0
    @test dbars[2][11] == 3570.0
end

@testset "Data.Structures — time bars (parity with Python)" begin
    # Same fixture as the Python test_time_bars.py (1-second bars).
    ticks = DataFrame(
        date_time = DateTime.([
            "2020-01-01T10:00:00.100",
            "2020-01-01T10:00:00.500",
            "2020-01-01T10:00:01.200",
            "2020-01-01T10:00:01.800",
            "2020-01-01T10:00:02.100",
            "2020-01-01T10:00:02.500",
        ]),
        price = [100, 101, 100, 101, 102, 103],
        volume = [10, 5, 20, 10, 10, 10],
    )

    tb = RiskLabAI.Data.TimeBars(resolution_type = "S", resolution_units = 1)
    bars = RiskLabAI.Data.construct_bars_from_data(tb; data = ticks)

    @test length(bars) == 2
    @test bars[1][1] == DateTime("2020-01-01T10:00:01")  # end time = bucket boundary
    @test bars[1][3] == 100.0   # open
    @test bars[1][4] == 101.0   # high
    @test bars[1][5] == 100.0   # low
    @test bars[1][6] == 101.0   # close (previous tick)
    @test bars[1][10] == 2.0    # ticks
    @test bars[2][1] == DateTime("2020-01-01T10:00:02")
    @test bars[2][3] == 100.0
    @test bars[2][4] == 101.0
    @test bars[2][5] == 100.0
    @test bars[2][6] == 101.0
    @test bars[2][10] == 2.0
end

@testset "Data.Structures — imbalance bars (parity with Python)" begin
    # Same fixture as Python test_imbalance_bars.py.
    # Prices 100..103..100 -> tick imbalances [0,1,1,1,-1,-1,-1].
    ticks = DataFrame(
        date_time = DateTime.([
            "2020-01-01T10:00:00",
            "2020-01-01T10:00:01",
            "2020-01-01T10:00:02",
            "2020-01-01T10:00:03",
            "2020-01-01T10:00:04",
            "2020-01-01T10:00:05",
            "2020-01-01T10:00:06",
        ]),
        price = [100, 101, 102, 103, 102, 101, 100],
        volume = [10, 10, 10, 10, 10, 10, 10],
    )

    # Fixed E[T] = 2  ->  2 bars with 3 and 4 ticks.
    fb = RiskLabAI.Data.FixedImbalanceBars{RiskLabAI.Data.Tick}(
        bar_type = "tick_imbalance",
        expected_imbalance_window = 10,
        initial_estimate_of_expected_n_ticks_in_bar = 2.0,
    )
    fbars = RiskLabAI.Data.construct_bars_from_data(fb; data = ticks)
    @test length(fbars) == 2
    @test fbars[1][10] == 3.0   # ticks in bar 1
    @test fbars[2][10] == 4.0   # ticks in bar 2

    # Expected E[T] (EWMA, init 2, window 10) -> 1 bar with 3 ticks.
    eb = RiskLabAI.Data.ExpectedImbalanceBars{RiskLabAI.Data.Tick}(
        bar_type = "tick_imbalance",
        window_size_for_expected_n_ticks_estimation = 10,
        expected_imbalance_window = 10,
        initial_estimate_of_expected_n_ticks_in_bar = 2.0,
    )
    ebars = RiskLabAI.Data.construct_bars_from_data(eb; data = ticks)
    @test length(ebars) == 1
    @test ebars[1][10] == 3.0
end

@testset "Data.Structures — run bars (parity with Python)" begin
    # Same fixture as Python test_run_bars.py.
    # Prices 100..103..100 -> buy run then sell run. The threshold stays Inf
    # until both buy- and sell-imbalance EWMAs warm up (>= E[T] obs each), so
    # the first (and only) bar forms at the last tick with all 7 ticks.
    ticks = DataFrame(
        date_time = DateTime.([
            "2020-01-01T10:00:00",
            "2020-01-01T10:00:01",
            "2020-01-01T10:00:02",
            "2020-01-01T10:00:03",
            "2020-01-01T10:00:04",
            "2020-01-01T10:00:05",
            "2020-01-01T10:00:06",
        ]),
        price = [100, 101, 102, 103, 102, 101, 100],
        volume = [10, 10, 10, 10, 10, 10, 10],
    )

    # Fixed E[T] = 3 -> 1 bar with 7 ticks.
    fb = RiskLabAI.Data.FixedRunBars{RiskLabAI.Data.Tick}(
        bar_type = "tick_run",
        expected_imbalance_window = 10,
        initial_estimate_of_expected_n_ticks_in_bar = 3.0,
    )
    fbars = RiskLabAI.Data.construct_bars_from_data(fb; data = ticks)
    @test length(fbars) == 1
    @test fbars[1][10] == 7.0

    # Expected E[T] (init 3) -> same single 7-tick bar on this fixture.
    eb = RiskLabAI.Data.ExpectedRunBars{RiskLabAI.Data.Tick}(
        bar_type = "tick_run",
        window_size_for_expected_n_ticks_estimation = 10,
        expected_imbalance_window = 10,
        initial_estimate_of_expected_n_ticks_in_bar = 3.0,
    )
    ebars = RiskLabAI.Data.construct_bars_from_data(eb; data = ticks)
    @test length(ebars) == 1
    @test ebars[1][10] == 7.0
end

@testset "Data.Differentiation — fractional differentiation (parity with Python)" begin
    D = RiskLabAI.Data
    x = [100.0, 101.5, 100.8, 102.3, 101.1, 103.0, 102.5, 104.1, 103.7, 105.0]

    # Weights (reversed, w0 last): exact match to Python calculate_weights_*.
    @test D.calculate_weights_std(0.5, 6) ≈
          [-0.02734375, -0.0390625, -0.0625, -0.125, -0.5, 1.0]
    @test D.calculate_weights_ffd(0.4, 0.1) ≈ [-0.12, -0.4, 1.0]

    # Fixed-width FFD: width 3 -> first 2 entries NaN, then 8 values.
    ffd = D.fractional_difference_fixed(x, 0.4; threshold = 0.1)
    @test length(ffd) == 10
    @test all(isnan, ffd[1:2])
    @test ffd[3:end] ≈ [48.2, 49.8, 48.084, 50.284, 49.168, 50.74, 49.76, 51.028]

    # Standard (expanding-window): skip 1 -> first entry NaN, then 9 values.
    fds = D.fractional_difference_std(x, 0.5; threshold = 0.01)
    @test length(fds) == 10
    @test isnan(fds[1])
    @test fds[2:end] ≈ [
        51.5,
        37.55,
        32.9625,
        27.1,
        26.66328125,
        23.205078125,
        23.2110351562,
        20.6416259766,
        20.8013458252,
    ]

    # ADF-based finders (behavioural: ADF impl differs from statsmodels).
    prices = 100.0 .+ 10.0 .* sin.(0.1 .* (1:200))
    res = D.find_optimal_ffd(prices)
    @test length(res.d) == 11
    @test all(0.0 .<= res.d .<= 1.0)
    @test length(res.p_value) == length(res.d)

    fd = D.fractionally_differentiated_log_price(prices)
    @test length(fd) == 200
    @test any(!isnan, fd)
end

@testset "Data.Weights — sample weighting (parity with Python)" begin
    D = RiskLabAI.Data
    close_index = collect(1:10)
    event_start = [1, 3, 6]
    event_end = [4, 7, 9]
    molecule = [1, 3, 6]
    price = [100.0, 101, 103, 102, 104, 103, 105, 106, 104, 107]

    # Concurrency over the affected span.
    exp = D.expand_label_for_meta_labeling(close_index, event_start, event_end, molecule)
    @test exp.index == collect(1:9)
    @test exp.concurrency ≈ [1.0, 1, 2, 2, 1, 2, 2, 1, 1]

    # Average uniqueness from the indicator matrix (events 1-4, 3-7, 6-9).
    index_matrix = zeros(10, 3)
    index_matrix[1:4, 1] .= 1
    index_matrix[3:7, 2] .= 1
    index_matrix[6:9, 3] .= 1
    @test D.calculate_average_uniqueness(index_matrix) ≈ [0.75, 0.6, 0.75]

    # Absolute-return sample weights (normalised to N=3; NaN first return skipped).
    w = D.sample_weight_absolute_return_meta_labeling(
        event_start,
        event_end,
        close_index,
        price,
        molecule,
    )
    @test w ≈ [0.6362107498, 1.2538702552, 1.1099189949]
    @test sum(w) ≈ 3.0

    # Linear time decay (weights assumed chronological).
    @test D.calculate_time_decay([1.0, 1.0, 1.0]; clf_last_weight = 0.5) ≈
          [0.6666666667, 0.8333333333, 1.0]
    @test D.calculate_time_decay([1.0, 1.0, 1.0]; clf_last_weight = 1.0) ≈ [1.0, 1.0, 1.0]
end

@testset "Data.Denoise — RMT denoising (parity with Python)" begin
    D = RiskLabAI.Data
    cov = [4.0 2.0 0.6; 2.0 3.0 0.5; 0.6 0.5 1.0]

    # cov <-> corr round trip (exact parity).
    corr = D.cov_to_corr(cov)
    @test corr ≈ [
        1.0 0.5773502692 0.3
        0.5773502692 1.0 0.2886751346
        0.3 0.2886751346 1.0
    ]
    @test D.corr_to_cov(corr, sqrt.([4.0, 3.0, 1.0])) ≈ cov

    # PCA eigenvalues, descending (exact parity).
    evals, evecs = D.pca(corr)
    @test evals ≈ [1.7952446957, 0.7822555891, 0.4224997152]

    # Denoised correlation keeping the top factor (sign-invariant reconstruction).
    @test D.denoised_corr(evals, evecs, 1) ≈ [
        1.0 0.437303437 0.362034825
        0.437303437 1.0 0.3607926887
        0.362034825 0.3607926887 1.0
    ]

    # Marcenko–Pastur PDF on a small grid (exact parity).
    grid, pdf = D.marcenko_pastur_pdf(0.5, 10; num_points = 5)
    @test grid ≈ [0.233772234, 0.391886117, 0.55, 0.708113883, 0.866227766]
    @test pdf ≈ [0.0, 2.2244409457, 1.8301531674, 1.2310555486, 0.0]

    # GMV portfolio weights (exact parity).
    @test D.optimal_portfolio(cov) ≈ [0.0320924262, 0.1463414634, 0.8215661104]

    # denoise_cov is behavioural (KDE fit differs from sklearn): valid output.
    denoised = D.denoise_cov(cov, 10.0)
    @test size(denoised) == (3, 3)
    @test denoised ≈ denoised'                  # symmetric
    @test diag(denoised) ≈ diag(cov)            # variances preserved
    @test all(isfinite, denoised)
end

@testset "Data.Labeling — CUSUM / barriers / meta-labeling (parity with Python)" begin
    D = RiskLabAI.Data
    dates = DateTime(2020, 1, 1) .+ Day.(0:19)
    close = [
        100.0,
        102,
        101,
        103,
        105,
        104,
        106,
        103,
        101,
        99,
        100,
        102,
        104,
        103,
        105,
        107,
        106,
        108,
        110,
        109,
    ]

    # Symmetric CUSUM events.
    events = D.symmetric_cusum_filter(dates, close, 3.0)
    @test events ==
          DateTime.(["2020-01-05", "2020-01-09", "2020-01-13", "2020-01-16", "2020-01-19"])

    # Daily volatility (pandas debiased EWM std; first value NaN).
    vol = D.daily_volatility_with_log_returns(dates, close; span = 5)
    @test length(vol.index) == 18
    @test vol.index[1] == DateTime("2020-01-03")
    @test isnan(vol.volatility[1])
    @test vol.volatility[2] ≈ 0.000137289 atol = 1e-9
    @test vol.volatility[3] ≈ 0.018224119 atol = 1e-9

    # Vertical barriers (events past the series end are dropped).
    vb = D.vertical_barrier(dates, events, 3)
    @test vb.event == DateTime.(["2020-01-05", "2020-01-09", "2020-01-13", "2020-01-16"])
    @test vb.barrier == DateTime.(["2020-01-08", "2020-01-12", "2020-01-16", "2020-01-19"])

    # Triple-barrier meta-events.
    target = Dict(vol.index[i] => vol.volatility[i] for i in eachindex(vol.index))
    vbdict = Dict(vb.event[i] => vb.barrier[i] for i in eachindex(vb.event))
    ev = D.meta_events(
        dates,
        close,
        events,
        (1.0, 1.0),
        target,
        0.0;
        vertical_barriers = vbdict,
    )
    @test ev.event_start == events
    @test ev.base_width ≈
          [0.018224119, 0.0322455982, 0.034249469, 0.0235399705, 0.0182630643]
    @test ev.end_time[1:4] ==
          DateTime.(["2020-01-08", "2020-01-12", "2020-01-16", "2020-01-19"])
    @test ismissing(ev.end_time[5])

    # Meta-labeling (event 5 dropped: no barrier touch).
    ml = D.meta_labeling(ev, dates, close)
    @test ml.ret ≈ [-0.0192313619, 0.0098522964, 0.0284379353, 0.0276515313]
    @test ml.label == [-1.0, 1.0, 1.0, 1.0]

    # t-value of an OLS fit (matches scipy linregress slope/stderr).
    @test D.calculate_t_value_linear_regression([100.0, 102, 101, 103, 105]) ≈ 3.6666666667 atol =
        1e-9

    # Trend scanning (behavioural shape check).
    ts = D.find_trend_using_trend_scanning([dates[1]], dates, close, (3, 6))
    @test nrow(ts) == 1
    @test ts.end_time[1] == DateTime("2020-01-06")
    @test ts.trend[1] in (-1.0, 0.0, 1.0)
end

@testset "Data.Distance — information-theoretic metrics (parity with Python)" begin
    D = RiskLabAI.Data
    x = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2.0, 1, 4, 3, 6, 5, 8, 7, 10, 9]

    # Optimal bin counts (univariate / bivariate).
    @test D.calculate_number_of_bins(10) == 3
    @test D.calculate_number_of_bins(100; correlation = 0.5) == 5

    # Variation of information (fixed bins).
    @test D.calculate_variation_of_information(x, y, 3) ≈ 0.763817002
    @test D.calculate_variation_of_information(x, y, 3; norm = true) ≈ 0.5193177863

    # Mutual information (optimal bins).
    @test D.calculate_mutual_information(x, y) ≈ 1.6094379124
    @test D.calculate_mutual_information(x, y; norm = true) ≈ 1.0

    # KL divergence and cross-entropy.
    p = [0.2, 0.3, 0.5]
    q = [0.1, 0.4, 0.5]
    @test D.calculate_kullback_leibler_divergence(p, q) ≈ 0.0523248144
    @test D.calculate_cross_entropy(p, q) ≈ 1.0819778284

    # Angular distance matrices.
    dep = [1.0 0.5; 0.5 1.0]
    @test D.calculate_distance(dep) ≈ [0.0 0.5; 0.5 0.0]
    @test D.calculate_distance(dep; metric = "absolute_angular") ≈ [0.0 0.5; 0.5 0.0]
end

@testset "Backtest — statistics (parity with Python)" begin
    B = RiskLabAI.Backtest

    # sharpe_ratio uses the population std (numpy ddof=0).
    returns = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.005]
    @test B.sharpe_ratio(returns) ≈ 0.36291502734548026
    @test B.sharpe_ratio(returns; risk_free_rate = 0.001) ≈ 0.29839680026183946
    @test B.sharpe_ratio([0.5, 0.5, 0.5]) == 0.0

    # bet_timing: position closes / flips, plus the final timestamp.
    idx = DateTime.([
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
    ])
    positions = [0.0, 1.0, 1.0, 0.0, -1.0, 0.0]
    @test B.bet_timing(idx, positions) == DateTime.(["2020-01-04", "2020-01-06"])

    # average holding period.
    hp = B.calculate_holding_period(idx, positions)
    @test nrow(hp.holding_periods) == 2
    @test hp.holding_periods.dT ≈ [2.0, 1.0]
    @test hp.holding_periods.w ≈ [1.0, 1.0]
    @test hp.mean_holding_period ≈ 1.5

    # Herfindahl–Hirschman index (can exceed 1 for signed returns).
    bet_returns = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.005, 0.04, -0.03]
    @test B.calculate_hhi(bet_returns) ≈ 1.6060606060606066
    @test B.calculate_hhi(bet_returns[bet_returns .>= 0]) ≈ 0.07083333333333337
    @test B.calculate_hhi(bet_returns[bet_returns .< 0]) ≈ 0.11637080867850098
    @test isnan(B.calculate_hhi([0.1, 0.2]))   # <= 2 observations

    # HHI concentration (positive, negative, monthly counts [4, 4, 2]).
    weekly = DateTime.([
        "2020-01-05",
        "2020-01-12",
        "2020-01-19",
        "2020-01-26",
        "2020-02-02",
        "2020-02-09",
        "2020-02-16",
        "2020-02-23",
        "2020-03-01",
        "2020-03-08",
    ])
    conc = B.calculate_hhi_concentration(weekly, bet_returns)
    @test conc.positive ≈ 0.07083333333333337
    @test conc.negative ≈ 0.11637080867850098
    @test conc.time ≈ 0.04000000000000017

    # drawdowns and time under water.
    pidx = DateTime.([
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    ])
    pnl = [100.0, 102.0, 101.0, 103.0, 99.0, 98.0, 105.0, 104.0]

    dd = B.compute_drawdowns_time_under_water(pidx, pnl; dollars = true)
    @test dd.start == DateTime.(["2020-01-02", "2020-01-04", "2020-01-07"])
    @test dd.drawdown ≈ [1.0, 5.0, 1.0]
    @test dd.time_under_water ≈ [1 / 365.25, 2 / 365.25, 1 / 365.25]

    ddp = B.compute_drawdowns_time_under_water(pidx, pnl; dollars = false)
    @test ddp.drawdown ≈ [1 - 101 / 102, 1 - 98 / 103, 1 - 104 / 105]
    @test ddp.time_under_water ≈ [1 / 365.25, 2 / 365.25, 1 / 365.25]
end

@testset "Backtest — PSR & test-set overfitting (parity with Python)" begin
    B = RiskLabAI.Backtest

    # Probabilistic Sharpe ratio (Φ of the Z-statistic).
    psr_sat = B.probabilistic_sharpe_ratio(
        2.5,
        1.0,
        252;
        skewness_of_returns = -0.5,
        kurtosis_of_returns = 4.0,
    )
    @test psr_sat ≈ 1.0
    @test B.probabilistic_sharpe_ratio(1.5, 1.0, 100) ≈ 0.9996784793766524
    psr_z = B.probabilistic_sharpe_ratio(1.5, 1.0, 100; return_test_statistic = true)
    @test psr_z ≈ 3.4127787539671264
    # Non-positive denominator: PSR -> 0, Z -> -Inf.
    @test B.probabilistic_sharpe_ratio(3.0, 1.0, 100; skewness_of_returns = 2.0) == 0.0
    psr_inf = B.probabilistic_sharpe_ratio(
        3.0,
        1.0,
        100;
        skewness_of_returns = 2.0,
        return_test_statistic = true,
    )
    @test psr_inf == -Inf

    # Benchmark (expected-max) Sharpe ratio.
    @test B.benchmark_sharpe_ratio([0.5, 1.2, -0.3, 0.8, 1.5, 0.9]) ≈ 0.7418275166314174
    @test B.benchmark_sharpe_ratio([0.7]) == 0.7
    @test B.benchmark_sharpe_ratio(Float64[]) == 0.0

    # Expected maximum Sharpe ratio (truncated Euler constant).
    @test B.expected_max_sharpe_ratio(10, 0.0, 1.0) ≈ 1.5745983013449716
    @test B.expected_max_sharpe_ratio(50, 0.5, 1.2) ≈ 3.231563712103708
    @test B.expected_max_sharpe_ratio(1, 0.33, 1.0) == 0.33
    @test B.expected_max_sharpe_ratio(0, 0.33, 1.0) == 0.0

    # Sharpe-ratio Z-statistic.
    @test B.estimated_sharpe_ratio_z_statistics(1.5, 100) ≈ 10.238336261901377
    z_skew = B.estimated_sharpe_ratio_z_statistics(
        1.2,
        52;
        true_sharpe_ratio = 0.5,
        skew = -0.3,
        kurt = 4,
    )
    @test z_skew ≈ 3.200281749891488
    @test isnan(B.estimated_sharpe_ratio_z_statistics(3.0, 100; skew = 2.0))

    # Multiple-testing type-1 / type-2 errors.
    @test B.strategy_type1_error_probability(1.96) ≈ 0.024997895148220373
    alpha_k = B.strategy_type1_error_probability(1.96; k = 10)
    @test alpha_k ≈ 0.22365361940347483
    theta = B.theta_for_type2_error(1.2, 52, 1.0; skew = -0.3, kurt = 4)
    @test theta ≈ 4.571831071273555
    @test isnan(B.theta_for_type2_error(3.0, 100, 1.0; skew = 2.0))
    @test B.strategy_type2_error_probability(alpha_k, 10, theta) ≈ 0.004502937106211302

    # Monte-Carlo helpers are stochastic — check structure with a seeded RNG.
    sims = B.generate_max_sharpe_ratios(20, [5, 10], 1.0, 0.0; rng = MersenneTwister(1))
    @test names(sims) == ["max_SR", "n_trials"]
    @test nrow(sims) == 40
    @test all(isfinite, sims.max_SR)

    err = B.mean_std_error(20, 3, [5, 10]; rng = MersenneTwister(2))
    @test names(err) == ["n_trials", "meanErr", "stdErr"]
    @test nrow(err) == 2
    @test all(isfinite, err.meanErr)
end

@testset "Backtest — strategy risk (parity with Python)" begin
    B = RiskLabAI.Backtest

    # Closed-form binomial variance (Python returns the SymPy expression).
    @test B.target_sharpe_ratio_symbolic(0.3, 1.0, -1.0) ≈ 0.84

    # Implied precision: real root in (0, 1), and NaN when the discriminant < 0.
    @test B.implied_precision(0.001, 0.01, 2, 1.0) ≈ 0.11111111111111126
    @test isnan(B.implied_precision(0.02, 0.04, 260, 1.5))

    # Implied bets-per-year frequency.
    @test B.bin_frequency(0.02, 0.04, 0.7, 1.5) ≈ 0.16349480968858132
    @test B.bin_frequency(0.02, 0.04, 0.0, 1.5) == Inf
    @test B.bin_frequency(0.02, 0.04, 1.0, 1.5) == Inf

    # Annualised binomial Sharpe ratio (signed Inf when dispersion is zero).
    @test B.binomial_sharpe_ratio(-0.02, 0.04, 260, 0.7) ≈ 12.90174509339827
    @test B.binomial_sharpe_ratio(-0.02, 0.04, 260, 0.0) == -Inf
    @test B.binomial_sharpe_ratio(-0.02, 0.04, 260, 1.0) == Inf

    # Failure probability (normal-CDF Z-score on observed vs required precision).
    rets = [0.01, 0.012, 0.008, -0.001, -0.0012, -0.0008]
    @test B.failure_probability(rets, 2, 1.0) ≈ 0.9716202768069931
    # No losing returns -> 0.0.
    @test B.failure_probability([0.01, 0.02, 0.03], 260, 1.0) == 0.0
    # Target unachievable (required precision is NaN) -> 1.0.
    @test B.failure_probability([0.04, -0.02, 0.03, -0.01], 260, 1.5) == 1.0

    # Monte-Carlo helpers are stochastic — check structure with a seeded RNG.
    trials = B.sharpe_ratio_trials(0.6, 200; rng = MersenneTwister(3))
    @test length(trials) == 3
    @test all(isfinite, trials)
    mixed = B.mix_gaussians(0.05, -0.02, 0.01, 0.01, 0.6, 100; rng = MersenneTwister(4))
    @test length(mixed) == 100
    risk = B.calculate_strategy_risk(
        0.05,
        -0.02,
        0.01,
        0.01,
        0.6,
        100,
        260,
        1.0;
        rng = MersenneTwister(5),
    )
    @test 0.0 <= risk <= 1.0
end

@testset "Backtest — PBO & synthetic backtesting (parity with Python)" begin
    B = RiskLabAI.Backtest

    # Deterministic fixture: rows = observations, cols = strategies.
    perf = [sin(0.7t + 1.3i) for t = 1:12, i = 1:4]
    metric = (r, rf) -> B.sharpe_ratio(r; risk_free_rate = rf)

    pe = B.performance_evaluation(perf[1:6, :], perf[7:12, :], 4, metric, 0.0)
    @test pe[1] == true
    @test pe[2] ≈ -0.4054651081081643

    pbo, logits = B.probability_of_backtest_overfitting(perf; n_partitions = 4)
    @test pbo ≈ 1.0
    @test logits ≈ [
        -0.4054651081081643,
        -0.4054651081081643,
        -1.3862943611198906,
        -1.3862943611198906,
        -0.4054651081081643,
        -0.4054651081081643,
    ]
    @test_throws ArgumentError B.probability_of_backtest_overfitting(perf; n_partitions = 3)

    # Synthetic backtesting is stochastic — check structure with tiny, seeded params.
    results = B.synthetic_back_testing(
        0.0,
        5.0,
        1.0;
        n_iteration = 50,
        maximum_holding_period = 20,
        profit_taking_range = [1.0, 2.0],
        stop_loss_range = [1.0, 2.0],
        seed = 0,
        rng = MersenneTwister(7),
    )
    @test length(results) == 4
    @test all(t -> length(t) == 5, results)
end

@testset "Backtest — bet sizing (parity with Python)" begin
    B = RiskLabAI.Backtest

    # Probability bet size: side · (2·Φ(p) - 1).
    pbs = B.probability_bet_size([0.55, 0.6, 0.45], [1.0, -1.0, 1.0])
    @test pbs ≈ [0.417680626423, -0.4514937645, 0.347289559424]

    # Concurrent average bet size over numeric dates.
    abs_out = B.average_bet_sizes([0, 1, 2, 3, 4], [0.0, 2.0], [2.0, 4.0], [0.5, -0.3])
    @test abs_out ≈ [0.5, 0.5, 0.1, -0.3, -0.3]

    # Discretisation (round to step, cap at ±1).
    @test B.discrete_signal([0.26, 0.94, -1.4, 0.05], 0.1) ≈ [0.3, 0.9, -1.0, 0.0]

    # Sigmoid position-sizing family.
    @test B.bet_size_sigmoid(2.0, 1.0) ≈ 0.5773502691896258
    @test B.target_position(2.0, 105.0, 100.0, 10) == 9
    @test B.inverse_price(100.0, 2.0, 0.5) ≈ 99.18350341907228
    @test B.inverse_price(100.0, 2.0, 1.0) == 100.0
    @test B.limit_price(5, 2, 100.0, 2.0, 10) ≈ 99.37384680974242
    @test B.compute_sigmoid_width(5.0, 0.5) ≈ 75.0
    @test B.compute_sigmoid_width(5.0, 0.0) == Inf

    # Concurrent active-signal averaging.
    starts = [1.0, 2.0, 3.0]
    ends = [3.0, 4.0, 5.0]
    tp, avg = B.avg_active_signals(starts, ends, [1.0, -0.5, 0.8])
    @test tp == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test avg ≈ [1.0, 0.25, 0.15, 0.8, 0.0]

    # A missing end never closes its signal.
    avg_missing =
        B.mp_avg_active_signals(starts, [missing, 4.0, 5.0], [1.0, -0.5, 0.8], [5.0])
    @test avg_missing ≈ [1.0]

    # End-to-end signal generation (no meta-label side).
    gtp, gsig =
        B.generate_signal(starts, ends, nothing, [0.6, 0.55, 0.7], [1.0, -1.0, 1.0], 2, 0.1)
    @test gtp == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test gsig ≈ [0.2, 0.0, 0.1, 0.3, 0.0]
end

@testset "Features — entropy (parity with Python)" begin
    F = RiskLabAI.Features
    m = "11100010011110100"

    @test F.shannon_entropy(m) ≈ 0.9975025463691152
    @test F.shannon_entropy("1111") == 0.0
    @test F.shannon_entropy("") == 0.0

    @test F.lempel_ziv_entropy(m) ≈ 0.47058823529411764

    @test F.probability_mass_function(m, 2) ==
          Dict("11" => 0.3125, "10" => 0.25, "00" => 0.25, "01" => 0.1875)

    @test F.plug_in_entropy_estimator(m, 1) ≈ 0.9975025463691152
    @test F.plug_in_entropy_estimator(m, 2) ≈ 0.9886085007312413

    @test F.longest_match_length(m, 5, 5) == (2, "0")
    # Kontoyiannis H_k is the *averaged* Σ log2(nᵢ)/Lᵢ (de Prado's formula).
    @test F.kontoyiannis_entropy(m) ≈ 0.9847738922739608
    @test F.kontoyiannis_entropy(m; window = 5) ≈ 0.8868475362417008
end

@testset "Features — microstructural (parity with Python)" begin
    F = RiskLabAI.Features
    high = [10.0, 11, 12, 11, 13, 12, 14, 13, 15, 14]
    low = [9.0, 9.5, 11, 10, 12, 11, 13, 12, 14, 13]

    # Corwin–Schultz β: NaN for the first 3 (rolling-2 then rolling-3 warm-up).
    beta = F.beta_estimates(high, low, 3)
    @test all(isnan, beta[1:3])
    @test beta[4:end] ≈ [
        0.026103995125,
        0.020403144632,
        0.015374563434,
        0.014177217155,
        0.012979870876,
        0.012042883073,
        0.01110589527,
    ]

    # γ: NaN only for the first point (rolling-2).
    gamma = F.gamma_estimates(high, low)
    @test isnan(gamma[1])
    @test gamma[2:end] ≈ [
        0.040268728017,
        0.054575898693,
        0.033241150072,
        0.06883500727,
        0.027907067203,
        0.058159137648,
        0.023762432091,
        0.049793044493,
        0.020477851451,
    ]

    # α and spread are floored at 0 for this fixture.
    spread = F.corwin_schultz_estimator(high, low, 3)
    @test all(isnan, spread[1:3])
    @test spread[4:end] == zeros(7)

    # Bekker–Parkinson volatility.
    bp = F.bekker_parkinson_volatility_estimates(high, low, 3)
    @test all(isnan, bp[1:3])
    @test bp[4:end] ≈ [
        0.665889352092,
        0.741771939318,
        0.552081947665,
        0.652306085438,
        0.508261603535,
        0.602526273704,
        0.470915692879,
    ]
end

@testset "Features — structural breaks (parity with Python)" begin
    F = RiskLabAI.Features

    # lag_dataframe: columns are lags 0..2 with NaN warm-up.
    lagged = F.lag_dataframe([10.0, 11, 12, 13], 2)
    @test size(lagged) == (4, 3)
    @test lagged[:, 1] == [10.0, 11.0, 12.0, 13.0]
    @test all(isnan, lagged[1:1, 2])
    @test lagged[2:4, 2] == [10.0, 11.0, 12.0]
    @test all(isnan, lagged[1:2, 3])
    @test lagged[3:4, 3] == [10.0, 11.0]

    # prepare_data ADF design (constant="ct", lags=2): exact diff/shift alignment.
    periodic = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.35, 1.5, 1.45, 1.6]
    y, x, index = F.prepare_data(periodic, "ct", 2)
    @test index == [4, 5, 6, 7, 8, 9, 10, 11, 12]
    @test y ≈ [0.15, -0.05, 0.15, -0.05, 0.15, -0.05, 0.15, -0.05, 0.15]
    @test size(x) == (9, 5)                                  # level, Δl1, Δl2, const, trend
    @test x[:, 1] ≈ [1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.35, 1.5, 1.45]   # lagged level
    @test x[:, 4] == ones(9)                                  # constant
    @test x[:, 5] == [4.0, 5, 6, 7, 8, 9, 10, 11, 12]         # trend

    # OLS β and the ADF t-statistic on a well-conditioned (constant="c") design.
    noisy = [1.0, 1.03, 1.01, 1.06, 1.1, 1.08, 1.15, 1.13, 1.2, 1.26, 1.24, 1.33]
    yc, xc, _ = F.prepare_data(noisy, "c", 1)
    beta_mean, beta_variance = F.compute_beta(yc, xc)
    @test beta_mean ≈ [0.14542507467, -0.831800466234, -0.113785422889]
    @test beta_mean[1] / sqrt(beta_variance[1, 1]) ≈ 1.007102910535

    # Backward Supremum ADF and the expanding-window ADF path.
    bsadf = F.get_bsadf_statistic(noisy, 6, "c", 1)
    @test bsadf.bsadf ≈ 1.007102910535
    adf = F.get_expanding_window_adf(noisy, 6, "c", 1)
    @test adf.statistics ≈
          [0.16894098733, 0.437811843619, 1.415912133027, 0.543280756955, 1.007102910535]
end

@testset "Optimization — HRP & hedging (parity with Python)" begin
    O = RiskLabAI.Optimization
    cov = [0.04 0.006 0.0 0.0; 0.006 0.09 0.0 0.0; 0.0 0.0 0.16 0.012; 0.0 0.0 0.012 0.25]

    # Inverse-variance weights.
    @test O.inverse_variance_weights(cov) ≈
          [0.53924505692, 0.239664469742, 0.13481126423, 0.086279209107]

    # Cluster variance (inverse-variance weighted).
    @test O.cluster_variance(cov, [1, 2]) ≈ 0.03024852071
    @test O.cluster_variance(cov, [3, 4]) ≈ 0.103271861987

    # Quasi-diagonal ordering from a SciPy-format linkage matrix (-> 1-based).
    link = [0.0 1 0.5 2; 2 3 0.6 2; 4 5 0.9 4]
    @test O.quasi_diagonal(link) == [1, 2, 3, 4]

    # Recursive-bisection HRP weights.
    @test O.recursive_bisection(cov, [1, 2, 3, 4]) ≈
          [0.535468091151, 0.237985818289, 0.138137860097, 0.088408230462]

    # Correlation distance.
    @test O.distance_corr([1.0 0.5; 0.5 1.0]) ≈ [0.0 0.5; 0.5 0.0]

    # PCA hedging: sign-free invariant wᵀ C w = risk_target² · Σρ.
    w_min = O.pca_weights(cov)
    @test length(w_min) == 4
    @test w_min' * cov * w_min ≈ 1.0
    rd = [0.1, 0.2, 0.3, 0.4]
    w_custom = O.pca_weights(cov; risk_distribution = rd, risk_target = 2.0)
    @test w_custom' * cov * w_custom ≈ 2.0^2 * sum(rd)
end

@testset "Cluster — ONC, silhouette & generators" begin
    C = RiskLabAI.Cluster

    # silhouette_samples matches scikit-learn exactly (precomputed distance).
    x = [0.0, 0.1, 0.2, 1.0, 1.1, 1.3]
    dist = [abs(x[i] - x[j]) for i = 1:6, j = 1:6]
    sil_two = C.silhouette_samples(dist, [1, 1, 1, 2, 2, 2])
    expected_two = [
        0.867647058824,
        0.903225806452,
        0.839285714286,
        0.777777777778,
        0.85,
        0.791666666667,
    ]
    @test sil_two ≈ expected_two atol = 1e-9

    # A singleton cluster yields silhouette 0 for that point.
    sil_one = C.silhouette_samples(dist, [1, 1, 1, 1, 1, 2])
    expected_one = [
        0.538461538462,
        0.5625,
        0.545454545455,
        -0.571428571429,
        -0.741935483871,
        0.0,
    ]
    @test sil_one ≈ expected_one atol = 1e-9
    @test sil_one[6] == 0.0

    # covariance_to_correlation delegates to Data.cov_to_corr.
    cov2 = [4.0 1.0; 1.0 9.0]
    @test C.covariance_to_correlation(cov2) ≈ [1.0 1/6; 1/6 1.0]

    # Two clearly separated blocks built programmatically (formatter-safe).
    corr = fill(0.05, 6, 6)
    corr[1:3, 1:3] .= 0.9
    corr[4:6, 4:6] .= 0.85
    corr[diagind(corr)] .= 1.0

    # k-means base returns a valid partition over all items.
    _, clusters, sil = C.cluster_k_means_base(corr; max_clusters = 5, random_state = 1)
    @test sort(reduce(vcat, collect(values(clusters)))) == collect(1:6)
    @test length(sil) == 6
    @test all(!isempty(v) for v in values(clusters))

    # ONC top-level also yields a valid partition over all items.
    _, clusters_top, sil_top = C.cluster_k_means_top(corr; random_state = 1)
    @test sort(reduce(vcat, collect(values(clusters_top)))) == collect(1:6)
    @test length(sil_top) == 6

    # Random block correlation: square, symmetric, unit diagonal.
    rbc = C.random_block_correlation(8, 2; random_state = 7)
    @test size(rbc) == (8, 8)
    @test rbc ≈ rbc'
    @test all(abs.(diag(rbc) .- 1.0) .< 1e-9)
end

@testset "Optimization — hrp() & NCO (parity with Python)" begin
    O = RiskLabAI.Optimization
    cov = [1.0 0.8 0.0 0.0; 0.8 1.0 0.0 0.0; 0.0 0.0 1.0 0.5; 0.0 0.0 0.5 1.0]

    # Markowitz GMV weights (exact).
    gmv = O.get_optimal_portfolio_weights(cov)
    @test gmv ≈ [0.227272727273, 0.227272727273, 0.272727272727, 0.272727272727]
    @test sum(gmv) ≈ 1.0

    # Markowitz MVO weights (exact); highest-return asset is overweighted.
    mu = [0.1, 0.2, 0.05, 0.1]
    mvo = O.get_optimal_portfolio_weights(cov; mu = mu)
    @test mvo ≈ [-0.625, 1.25, 0.0, 0.375]
    @test sum(mvo) ≈ 1.0
    @test mvo[2] > mvo[1]

    # NCO weights: valid allocation summing to one (behavioural; stochastic k-means).
    nco = O.get_optimal_portfolio_weights_nco(cov; number_clusters = 2)
    @test length(nco) == 4
    @test sum(nco) ≈ 1.0

    # HRP weights: long-only allocation in original asset order (behavioural order).
    corr = RiskLabAI.Cluster.covariance_to_correlation(cov)
    w_hrp = O.hrp(cov, corr)
    @test length(w_hrp) == 4
    @test sum(w_hrp) ≈ 1.0
    @test all(w_hrp .> 0)
end

@testset "Validation — cross-validators (parity with Python)" begin
    V = RiskLabAI.Validation
    # 10 observations; each label's info range spans two steps ahead.
    starts = collect(0:9)
    ends = starts .+ 2

    # Standard K-Fold (no shuffle): contiguous test folds, complementary train.
    kfold = V.cv_split(V.KFoldCV(5), 10)
    @test kfold[1] == ([3, 4, 5, 6, 7, 8, 9, 10], [1, 2])
    @test kfold[3] == ([1, 2, 3, 4, 7, 8, 9, 10], [5, 6])
    @test length(kfold) == 5

    # Purged K-Fold with embargo = 0.1 (matches pandas reference exactly).
    purged = V.cv_split(V.PurgedKFoldCV(5, starts, ends; embargo = 0.1))
    @test purged[1] == ([6, 7, 8, 9, 10], [1, 2])
    @test purged[2] == ([8, 9, 10], [3, 4])
    @test purged[3] == ([1, 2, 10], [5, 6])
    @test purged[5] == ([1, 2, 3, 4, 5, 6], [9, 10])

    # CPCV: C(6,2) = 15 splits; first split purged against the combination.
    cpcv = V.CombinatorialPurgedCV(6, 2, starts, ends)
    @test V.get_n_splits(cpcv) == 15
    splits = V.cv_split(cpcv)
    @test length(splits) == 15
    @test splits[1] == ([7, 8, 9, 10], [1, 2, 3, 4])
    # Number of backtest paths = n_test_groups · C / n_splits = 2·15/6 = 5.
    @test length(V.backtest_paths(cpcv)) == 5

    # Walk-Forward with gap = 1: growing train window, walking test fold.
    walk = V.cv_split(V.WalkForwardCV(5; gap = 1), 10)
    @test walk[1] == (Int[], [1, 2])
    @test walk[2] == ([1], [3, 4])
    @test walk[3] == ([1, 2, 3], [5, 6])
    @test walk[5] == ([1, 2, 3, 4, 5, 6, 7], [9, 10])
end

@testset "Data.SyntheticData (parity with Python)" begin
    S = RiskLabAI.Data

    # form_block_matrix: exact block-diagonal correlation.
    fbm = S.form_block_matrix(2, 2, 0.5)
    @test fbm == [1.0 0.5 0.0 0.0; 0.5 1.0 0.0 0.0; 0.0 0.0 1.0 0.5; 0.0 0.0 0.5 1.0]

    # drift_volatility_burst: exact drift/vol profiles (midpoint clamp + NaN fill).
    drifts, vols = S.drift_volatility_burst(5, 1.0, 2.0, 0.5, 1.0, 0.5, 0.5)
    @test drifts ≈ [1.414213562373, 2.0, 0.0, 4.0, 2.828427124746]
    @test vols ≈ [0.707106781187, 1.0, 1.0, 2.0, 1.414213562373]

    # compute_log_returns: exact Heston–Merton step given fixed increments.
    lr = S.compute_log_returns(
        3,
        [0.1, 0.1, 0.2],
        [1.0, 1.0, 1.0],
        [0.04, 0.04, 0.05],
        [0.2, 0.2, 0.2],
        [0.5, -0.3, 0.1],
        [0.2, 0.1, -0.2],
        [0.0, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        0.01,
        0.1,
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [false, false, true],
    )
    @test lr ≈ [0.010775, 0.294711297037, 0.003961067977]

    # align_params_length: scalars broadcast, short vectors padded with last value.
    aligned, max_len = S.align_params_length(
        Dict("mu" => 0.1, "kappa" => [1.0, 2.0], "theta" => 0.04),
    )
    @test max_len == 2
    @test aligned["mu"] == [0.1, 0.1]
    @test aligned["kappa"] == [1.0, 2.0]
    @test aligned["theta"] == [0.04, 0.04]

    # Stochastic generators: structural checks (shape, symmetry, positivity).
    rc = S.random_cov(5, 3; rng = MersenneTwister(1))
    @test size(rc) == (5, 5)
    @test rc ≈ rc'

    mu0, cov0 = S.form_true_matrix(2, 3, 0.5; rng = MersenneTwister(2))
    @test length(mu0) == 6
    @test size(cov0) == (6, 6)

    mu1, cov1 = S.simulates_cov_mu(mu0, cov0, 500; rng = MersenneTwister(3))
    @test length(mu1) == 6
    @test size(cov1) == (6, 6)

    regimes = Dict(
        "calm" => Dict(
            "mu" => 0.05,
            "kappa" => 1.0,
            "theta" => 0.04,
            "xi" => 0.2,
            "rho" => -0.5,
            "lam" => 0.1,
            "m" => 0.0,
            "v" => 0.1,
        ),
        "turbulent" => Dict(
            "mu" => -0.1,
            "kappa" => 2.0,
            "theta" => 0.1,
            "xi" => 0.5,
            "rho" => -0.7,
            "lam" => 1.0,
            "m" => -0.02,
            "v" => 0.2,
        ),
    )
    transition = [0.9 0.1; 0.2 0.8]
    prices, regime_path =
        S.generate_prices_from_regimes(regimes, transition, 1.0, 50; random_state = 1)
    @test length(prices) == 50
    @test all(prices .> 0)
    @test length(regime_path) == 50

    all_prices, all_regimes =
        S.parallel_generate_prices(3, regimes, transition, 1.0, 20; random_state = 1)
    @test size(all_prices) == (20, 3)
    @test size(all_regimes) == (20, 3)
end

@testset "Features — feature importance (orthogonal & weighted-τ)" begin
    F = RiskLabAI.Features
    x = [1.0 2.0 3.0; 2.0 1.0 0.0; 3.0 3.0 1.0; 0.0 2.0 2.0; 1.0 0.0 4.0; 2.0 2.0 2.0]

    # orthogonal_features: eigenvalues + cumulative variance exact; columns orthogonal.
    orth, eigenvalues, _, cumulative = F.orthogonal_features(x)
    @test eigenvalues ≈ [9.4191800225, 3.2941004976, 2.2867194799]
    @test cumulative ≈ [0.6279453348, 0.8475520347, 1.0]
    @test size(orth) == (6, 3)
    gram = orth' * orth
    @test maximum(abs.(gram - Diagonal(diag(gram)))) < 1e-8   # uncorrelated components

    # variance_threshold trims the retained component count.
    orth2, eig2, _, _ = F.orthogonal_features(x; variance_threshold = 0.80)
    @test length(eig2) == 2
    @test size(orth2) == (6, 2)

    # weighted Kendall-τ (wrapper compares importances to 1/ranks, flipping signs).
    @test F.calculate_weighted_tau([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) ≈ -1.0
    @test F.calculate_weighted_tau([4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]) ≈ 1.0
    @test F.calculate_weighted_tau([0.5, 0.2, 0.8, 0.1], [1.0, 2.0, 3.0, 4.0]) ≈ 0.2
end

@testset "Features — tree feature importance (DecisionTree.jl)" begin
    F = RiskLabAI.Features

    # Separable dataset: column 1 informative, column 2 pure noise.
    rng = MersenneTwister(42)
    n = 200
    y = rand(rng, 0:1, n)
    x = hcat(Float64.(y) .+ 0.3 .* randn(rng, n), randn(rng, n))

    # MDI / MDA / SFI must all rank the informative feature above the noise one.
    mdi = F.feature_importance_mdi(x, y; n_trees = 40, random_state = 1)
    @test length(mdi.mean) == 2
    @test mdi.mean[1] > mdi.mean[2]

    mda = F.feature_importance_mda(x, y; n_splits = 5, n_trees = 40, random_state = 1)
    @test length(mda.mean) == 2
    @test mda.mean[1] > mda.mean[2]

    sfi = F.feature_importance_sfi(x, y; n_splits = 5, n_trees = 40)
    @test length(sfi.mean) == 2
    @test sfi.mean[1] > sfi.mean[2]

    # get_test_dataset: shapes and informative/redundant/noise column naming.
    xd, yd, names = F.get_test_dataset(;
        n_features = 12,
        n_informative = 4,
        n_redundant = 3,
        n_samples = 60,
        random_state = 0,
    )
    @test size(xd) == (60, 12)
    @test length(yd) == 60
    @test length(names) == 12
    @test count(startswith("I_"), names) == 4
    @test count(startswith("R_"), names) == 3
end

@testset "Ensemble & CV scoring (DecisionTree.jl)" begin
    E = RiskLabAI.Ensemble
    V = RiskLabAI.Validation

    # Theoretical bagging accuracy: exact (binomial survival function).
    @test E.bagging_classifier_accuracy(11, 0.6) ≈ 0.75349813248
    @test E.bagging_classifier_accuracy(101, 0.55) ≈ 0.843755399638
    @test E.bagging_classifier_accuracy(3, 0.7) ≈ 0.784
    @test E.bagging_classifier_accuracy(7, 0.51) ≈ 0.5218662521
    @test_throws ArgumentError E.bagging_classifier_accuracy(10, 0.6)

    # Separable dataset for the behavioural pieces.
    rng = MersenneTwister(7)
    n = 200
    y = rand(rng, 0:1, n)
    x = hcat(3.0 .* y .+ randn(rng, n), randn(rng, n))
    train = 1:150
    test = 151:200

    schemes = E.bagging_evaluate_schemes(
        x[train, :], y[train], x[test, :], y[test];
        n_estimators = 40, max_samples = 60, random_state = 1,
    )
    @test Set(keys(schemes)) == Set(["uniform", "c_i", "variance"])
    @test all(0.0 <= v <= 1.0 for v in values(schemes))
    @test schemes["uniform"] > 0.6   # informative signal is learnable

    trees, classes = E.fit_bagging(
        x[train, :], y[train]; n_estimators = 40, max_samples = 60, random_state = 1,
    )
    values_boot, mean_boot, std_boot =
        E.calculate_bootstrap_accuracy(trees, classes, x[test, :], y[test]; n_bootstraps = 50)
    @test length(values_boot) == 50
    @test 0.0 <= mean_boot <= 1.0

    # cross_val_score over a purged K-Fold and a plain K-Fold.
    scores = V.cross_val_score(V.KFoldCV(5), x, y; n_trees = 30, random_state = 1)
    @test length(scores) == 5
    @test sum(scores) / length(scores) > 0.7

    starts = collect(1:n)
    purged = V.PurgedKFoldCV(5, starts, starts; embargo = 0.0)
    pscores = V.cross_val_score(purged, x, y; n_trees = 30, random_state = 1)
    @test length(pscores) == 5
end

@testset "Hyper-parameter tuning (DecisionTree.jl)" begin
    V = RiskLabAI.Validation
    rng = MersenneTwister(11)
    n = 200
    y = rand(rng, 0:1, n)
    x = hcat(3.0 .* y .+ randn(rng, n), randn(rng, n))

    grid = Dict(:n_trees => [10, 30], :max_depth => [2, 4])

    # Grid search: every combination scored; best is recovered and a model refit.
    gs = V.grid_search_cv(V.KFoldCV(4), x, y, grid; random_state = 1)
    @test length(gs.results) == 4
    @test haskey(gs.best_params, :n_trees)
    @test gs.best_score > 0.7
    @test gs.best_score == maximum(score for (_, score) in gs.results)

    # Randomised search: n_iter configurations; best score reported.
    rs = V.random_search_cv(V.KFoldCV(4), x, y, grid; n_iter = 3, random_state = 1)
    @test length(rs.results) == 3
    @test rs.best_score > 0.7
end
