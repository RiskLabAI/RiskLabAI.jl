using Test
using TOML
using LinearAlgebra
using RiskLabAI

const BetSizingContracts = RiskLabAI.Backtest
const bet_sizing_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "bet_sizing_python_320.toml"))

@testset "Kelly and pessimistic-box sizing against Python 3.2.0" begin
    for row in bet_sizing_fixtures["kelly"]
        result = BetSizingContracts.kelly_bet_fraction(row["probabilities"]; payout_ratio = row["payout"], max_fraction = row["cap"])
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        @test BetSizingContracts.kelly_bet_fraction(row["probabilities"], row["payout"], row["cap"]) == result
        @test size(BetSizingContracts.kelly_bet_fraction(reshape(row["probabilities"], 2, :); payout_ratio = row["payout"], max_fraction = row["cap"])) == (2, 4)
        @test all(0 .<= result .<= row["cap"])
    end
    for row in bet_sizing_fixtures["robust"]
        result = BetSizingContracts.distributionally_robust_kelly_fraction(row["probabilities"], row["count"]; payout_ratio = row["payout"], max_fraction = row["cap"], confidence_z = row["z"])
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        @test BetSizingContracts.distributionally_robust_kelly_fraction(row["probabilities"], row["count"], row["payout"], row["cap"], row["z"]) == result
        @test all(0 .<= result .<= row["cap"])
    end
    @test BetSizingContracts.kelly_bet_fraction(0.6) ≈ 0.2
    @test BetSizingContracts.distributionally_robust_kelly_fraction(0.6, 100_000) <= 0.2
    @test BetSizingContracts.distributionally_robust_kelly_fraction(0.6, -2) == BetSizingContracts.distributionally_robust_kelly_fraction(0.6, 1)
    @test BetSizingContracts.distributionally_robust_kelly_fraction(0.6, big(10)^30) ≈ BetSizingContracts.kelly_bet_fraction(0.6) rtol = 1e-10 atol = 1e-12
    @test BetSizingContracts.distributionally_robust_kelly_fraction([0.0, 1.0], 100; confidence_z = 0) == BetSizingContracts.kelly_bet_fraction([1e-4, 1 - 1e-4])
    for payout in (0, -1, NaN, Inf, true)
        @test_throws ArgumentError BetSizingContracts.kelly_bet_fraction(0.5; payout_ratio = payout)
    end
    for cap in (-1, NaN, Inf, true)
        @test_throws ArgumentError BetSizingContracts.kelly_bet_fraction(0.5; max_fraction = cap)
    end
    for count in (true, 1.5, Inf)
        @test_throws ArgumentError BetSizingContracts.distributionally_robust_kelly_fraction(0.5, count)
    end
    @test_throws ArgumentError BetSizingContracts.distributionally_robust_kelly_fraction(0.5, 10; confidence_z = -1)
    @test_throws ArgumentError BetSizingContracts.kelly_bet_fraction([NaN])
end

@testset "Public right-closed calibration error bins" begin
    for row in bet_sizing_fixtures["ece"]
        result = BetSizingContracts.expected_calibration_error(row["probabilities"], row["outcomes"]; n_bins = row["bins"])
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        @test BetSizingContracts.expected_calibration_error(row["probabilities"], row["outcomes"], row["bins"]) == result
    end
    @test BetSizingContracts.expected_calibration_error([0.0, 1.0], [0, 1]) == 0
    @test BetSizingContracts.expected_calibration_error([0.0, 1.0], [1, 0]) == 1
    @test BetSizingContracts.expected_calibration_error([0.3, nextfloat(0.3)], [0, 1]; n_bins = 10) ≈ 0.2
    for bins in (0, -1, 1.5, true)
        @test_throws ArgumentError BetSizingContracts.expected_calibration_error([0.5], [1]; n_bins = bins)
    end
    for (probabilities, outcomes) in ((Float64[], Int[]), ([0.5], [0, 1]), ([0.5], [2]), ([NaN], [1]), ([1.1], [1]), ([0.5], [NaN]), (ones(1, 1), [1]))
        @test_throws ArgumentError BetSizingContracts.expected_calibration_error(probabilities, outcomes)
    end
end

function independent_platt_objective(parameters, probabilities, outcomes)
    clipped = clamp.(probabilities, 1e-4, 1 - 1e-4)
    logits = log.(clipped ./ (1 .- clipped))
    score = parameters[1] * logits .+ parameters[2]
    losses = [outcome == 1 ? log1p(exp(-abs(value))) + max(-value, 0) : log1p(exp(-abs(value))) + max(value, 0) for (value, outcome) in zip(score, outcomes)]
    return sum(losses) / length(losses) + 0.5e-6 / length(losses) * parameters[1]^2
end

@testset "Fixed-objective Platt calibration and identity behavior" begin
    for row in bet_sizing_fixtures["platt"]
        calibrator = BetSizingContracts.PlattCalibrator()
        @test BetSizingContracts.transform(calibrator, row["query"]) == row["query"]
        @test BetSizingContracts.fit!(calibrator, row["probabilities"], row["outcomes"]) === calibrator
        result = BetSizingContracts.transform(calibrator, row["query"])
        if row["identity"]
            @test isnothing(calibrator.coefficients)
            @test result == row["expected"]
        else
            parameters = collect(calibrator.coefficients)
            @test parameters ≈ row["tight_parameters"] rtol = 1e-5 atol = 1e-5
            @test result ≈ row["tight_predictions"] rtol = 1e-5 atol = 1e-5
            @test norm(parameters - row["default_parameters"]) <= row["coefficient_error_bound"] + 1e-5
            @test all(abs.(result - row["expected"]) .<= row["prediction_error_bounds"] .+ 1e-5)
            loss = independent_platt_objective(parameters, row["probabilities"], row["outcomes"])
            @test loss ≈ row["tight_loss"] rtol = 1e-9 atol = 1e-11
            @test loss <= row["reference_loss"] + 1e-11
            score = [(independent_platt_objective(parameters + step, row["probabilities"], row["outcomes"]) - independent_platt_objective(parameters - step, row["probabilities"], row["outcomes"])) / 2e-5 for step in ([1e-5, 0.0], [0.0, 1e-5])]
            @test norm(score) < 1e-8
            matrix_query = [0.01 0.2; 0.8 0.99]
            @test BetSizingContracts.transform(calibrator, matrix_query) == BetSizingContracts.transform(calibrator, [0.01, 0.2, 0.8, 0.99])
            @test BetSizingContracts.transform(calibrator, 0.5) == BetSizingContracts.transform(calibrator, [0.5])
        end
        @test all(0 .<= result .<= 1)
        @test BetSizingContracts.fit!(calibrator, [0.1, 0.9], [1, 1]) === calibrator
        @test isnothing(calibrator.coefficients)
        matrix_query = [0.0 0.3; 0.8 1.0]
        @test BetSizingContracts.transform(calibrator, matrix_query) == matrix_query
        @test size(BetSizingContracts.transform(calibrator, matrix_query)) == size(matrix_query)
    end
    calibrator = BetSizingContracts.PlattCalibrator()
    @test BetSizingContracts.transform(calibrator, 0.5) == 0.5
    for (probabilities, outcomes) in ((Float64[], Int[]), ([0.2], [0, 1]), ([0.2, 0.8], [0, 2]), ([NaN, 0.8], [0, 1]), ([0.2, 0.8], [0, NaN]))
        @test_throws ArgumentError BetSizingContracts.fit!(calibrator, probabilities, outcomes)
    end
    @test_throws ArgumentError BetSizingContracts.transform(calibrator, [Inf])
    probabilities = [0.1 0.2; 0.8 0.9]
    first_fit = BetSizingContracts.fit!(BetSizingContracts.PlattCalibrator(), probabilities, [0, 1, 0, 1])
    second_fit = BetSizingContracts.fit!(BetSizingContracts.PlattCalibrator(), [0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1])
    @test first_fit.coefficients == second_fit.coefficients
end
