using Test
using Dates
using DataFrames
using LinearAlgebra
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
            "2020-01-01T10:00:00", "2020-01-01T10:00:01", "2020-01-01T10:00:02",
            "2020-01-01T10:00:03", "2020-01-01T10:00:04", "2020-01-01T10:00:05",
            "2020-01-01T10:00:06",
        ]),
        price = [100, 101, 100, 101, 102, 103, 102],
        volume = [10, 5, 20, 10, 10, 10, 5],
    )

    # Tick bars, threshold 3
    tb = RiskLabAI.Data.StandardBars{RiskLabAI.Data.Tick}(bar_type = "tick", threshold = 3.0)
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
    vb = RiskLabAI.Data.StandardBars{RiskLabAI.Data.Volume}(bar_type = "volume", threshold = 35.0)
    vbars = RiskLabAI.Data.construct_bars_from_data(vb; data = ticks)
    @test length(vbars) == 2
    @test vbars[1][7] == 35.0
    @test vbars[2][7] == 35.0

    # Dollar bars, threshold 3500
    db = RiskLabAI.Data.StandardBars{RiskLabAI.Data.Dollar}(bar_type = "dollar", threshold = 3500.0)
    dbars = RiskLabAI.Data.construct_bars_from_data(db; data = ticks)
    @test length(dbars) == 2
    @test dbars[1][11] == 3505.0
    @test dbars[2][11] == 3570.0
end

@testset "Data.Structures — time bars (parity with Python)" begin
    # Same fixture as the Python test_time_bars.py (1-second bars).
    ticks = DataFrame(
        date_time = DateTime.([
            "2020-01-01T10:00:00.100", "2020-01-01T10:00:00.500",
            "2020-01-01T10:00:01.200", "2020-01-01T10:00:01.800",
            "2020-01-01T10:00:02.100", "2020-01-01T10:00:02.500",
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
            "2020-01-01T10:00:00", "2020-01-01T10:00:01", "2020-01-01T10:00:02",
            "2020-01-01T10:00:03", "2020-01-01T10:00:04", "2020-01-01T10:00:05",
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
            "2020-01-01T10:00:00", "2020-01-01T10:00:01", "2020-01-01T10:00:02",
            "2020-01-01T10:00:03", "2020-01-01T10:00:04", "2020-01-01T10:00:05",
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
    @test fds[2:end] ≈ [51.5, 37.55, 32.9625, 27.1, 26.66328125,
        23.205078125, 23.2110351562, 20.6416259766, 20.8013458252]

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
        event_start, event_end, close_index, price, molecule
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
