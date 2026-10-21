using Test
using TOML
using LinearAlgebra
using RiskLabAI

const DensityContracts = RiskLabAI.Data
const TailContracts = RiskLabAI.Backtest
const density_tail_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "density_tail_python_320.toml"))
density_tail_matrix(rows) = reduce(vcat, permutedims.(rows))

@testset "Fitted univariate density against Python 3.2.0" begin
    for row in density_tail_fixtures["density"]
        observations = copy(row["observations"])
        estimate = DensityContracts.fit_kde(observations; bandwidth = row["bandwidth"], kernel = row["kernel"])
        result = DensityContracts.score_samples(estimate, row["query"])
        @test all(isapprox.(result, row["expected"]; rtol = 1e-10, atol = 1e-12))
        @test DensityContracts.score_samples(estimate, reshape(row["query"], :, 1)) == result
        @test DensityContracts.score_samples(DensityContracts.fit_kde(observations, row["bandwidth"], row["kernel"]), row["query"]) == result
        observations .= 999.0
        @test DensityContracts.score_samples(estimate, row["query"]) == result
        @test estimate.observations isa Tuple
    end
    @test DensityContracts.score_samples(DensityContracts.fit_kde([0.0]), [0.0]) ≈ [-log(0.01 * sqrt(2pi))]
    @test DensityContracts.score_samples(DensityContracts.fit_kde([-1e308]; bandwidth = 1e308), [1e308]) ≈ [-2 - log(1e308) - log(2pi) / 2]
    for kernel in ("tophat", "epanechnikov", "linear", "cosine")
        @test DensityContracts.score_samples(DensityContracts.fit_kde([0.0]; bandwidth = 1, kernel), [-1.0, 1.0]) == [-Inf, -Inf]
    end
    @test DensityContracts.score_samples(DensityContracts.fit_kde([1.0 2.0; 3.0 4.0]; bandwidth = 1), [0.0, 1.0]) ≈ DensityContracts.score_samples(DensityContracts.fit_kde([1.0, 2.0, 3.0, 4.0]; bandwidth = 1), [0.0, 1.0])
    for observations in (Float64[], [NaN], [Inf], [1im], ["1"], 1.0)
        @test_throws ArgumentError DensityContracts.fit_kde(observations)
    end
    for bandwidth in (0, -1, Inf, NaN, "scott")
        @test_throws ArgumentError DensityContracts.fit_kde([0.0]; bandwidth)
    end
    @test_throws ArgumentError DensityContracts.fit_kde([0.0]; kernel = "quartic")
    estimate = DensityContracts.fit_kde([0.0])
    for query in (Float64[], zeros(1, 2), [NaN], [Inf], [1im], 1.0)
        @test_throws ArgumentError DensityContracts.score_samples(estimate, query)
    end
end

@testset "Targeted eigenvector shrinkage against Python 3.2.0" begin
    for row in density_tail_fixtures["shrinkage"]
        values, vectors = density_tail_matrix(row["values"]), density_tail_matrix(row["vectors"])
        result = DensityContracts.denoised_corr2(values, vectors, row["count"]; alpha = row["alpha"])
        @test result ≈ density_tail_matrix(row["expected"]) rtol = 1e-10 atol = 1e-12
        @test DensityContracts.denoised_corr2(values, vectors, row["count"], row["alpha"]) == result
        signal_vectors, noise_vectors = vectors[:, 1:row["count"]], vectors[:, row["count"]+1:end]
        signal = signal_vectors * values[1:row["count"], 1:row["count"]] * signal_vectors'
        noise = noise_vectors * values[row["count"]+1:end, row["count"]+1:end] * noise_vectors'
        @test diag(result) ≈ diag(signal + noise) rtol = 1e-10 atol = 1e-12
        @test values == density_tail_matrix(row["values"]) && vectors == density_tail_matrix(row["vectors"])
    end
    values, vectors = [2.0 0.0; 0.0 0.5], [0.6 -0.8; 0.8 0.6]
    @test DensityContracts.denoised_corr2(values, vectors, 0) ≈ Diagonal(diag(vectors * values * vectors'))
    @test DensityContracts.denoised_corr2(values, vectors, 2) ≈ vectors * values * vectors'
    for invalid in (zeros(0, 0), ones(2, 3), [1.0, 2.0], [1.0 2e-12; 0.0 2.0], [1.0 0.0; 0.0 NaN], [true false; false true], complex.(values))
        @test_throws ArgumentError DensityContracts.denoised_corr2(invalid, vectors, 1)
    end
    for invalid in (ones(3, 3), [1.0 0.0; 0.0 Inf], [true false; false true], complex.(vectors))
        @test_throws ArgumentError DensityContracts.denoised_corr2(values, invalid, 1)
    end
    for count in (-1, 3, true, 1.0)
        @test_throws ArgumentError DensityContracts.denoised_corr2(values, vectors, count)
    end
    for alpha in (-0.1, 1.1, NaN, Inf, true, 1im)
        @test_throws ArgumentError DensityContracts.denoised_corr2(values, vectors, 1; alpha)
    end
end

@testset "Empirical and Pareto tail risk against Python 3.2.0" begin
    for row in density_tail_fixtures["empirical"]
        result = TailContracts.expected_shortfall(row["returns"], 0.0123; confidence_level = row["level"])
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        @test TailContracts.expected_shortfall(row["returns"], 100.0, row["level"]) == result
    end
    @test TailContracts.expected_shortfall([-0.3, 0.2], 0.0) == -0.3
    for row in density_tail_fixtures["pareto"]
        result = TailContracts.generalized_pareto_tail_risk(row["threshold"], row["rate"], row["shape"], row["scale"], row["probability"])
        for key in (:value_at_risk, :expected_shortfall)
            @test getproperty(result, key) ≈ row["expected"][string(key)] rtol = 1e-10 atol = 1e-12
        end
        @test result.expected_shortfall >= result.value_at_risk >= row["threshold"]
        if row["shape"] < 0
            @test result.expected_shortfall <= row["threshold"] - row["scale"] / row["shape"] + 1e-12
        end
    end
    exponential = TailContracts.generalized_pareto_tail_risk(2, 0.1, 0, 3, 0.99)
    @test exponential.value_at_risk ≈ 2 - 3log(0.1)
    @test exponential.expected_shortfall ≈ exponential.value_at_risk + 3
    for values in (Float64[], [NaN], [Inf], [true], [1im], ones(2, 1))
        @test_throws ArgumentError TailContracts.expected_shortfall(values, 0.0)
    end
    for level in (-0.1, 1.1, true, NaN)
        @test_throws ArgumentError TailContracts.expected_shortfall([0.0], 0; confidence_level = level)
    end
    for parameters in ((0, 0, 0, 1, 0.99), (0, 1.1, 0, 1, 0.99), (0, 0.1, 0, 1, 0.9), (0, 0.1, 0, 1, 1), (0, 0.1, 0, 0, 0.99), (0, 0.1, NaN, 1, 0.99), (Inf, 0.1, 0, 1, 0.99), (true, 0.1, 0, 1, 0.99), (0, 0.1, 1000, 1, 0.999))
        @test_throws ArgumentError TailContracts.generalized_pareto_tail_risk(parameters...)
    end
end

@testset "Quantile exceptions against exhaustive Python 3.2.0 fixtures" begin
    for row in density_tail_fixtures["exceptions"]
        result = TailContracts.quantile_exception_tests(row["hits"]; exception_probability = row["probability"])
        for name in propertynames(result)
            expected = row["expected"][string(name)]
            actual = getproperty(result, name)
            if expected == "undefined"
                @test isnothing(actual)
            elseif name == :transitions
                @test actual == density_tail_matrix(expected)
            elseif name in (:exception_count, :sample_count)
                @test actual == expected
            else
                @test actual ≈ expected rtol = 1e-10 atol = 1e-12
            end
        end
        @test sum(result.transitions) == length(row["hits"]) - 1
    end
    @test TailContracts.quantile_exception_tests([true, false, true]; exception_probability = 0.1) == TailContracts.quantile_exception_tests([1.0, 0.0, 1.0]; exception_probability = 0.1)
    for hits in (Int[], [0, 2], [-1, 1], [0.0, NaN], [0.0, Inf], [0.5], ["0"], ones(2, 1), [0im])
        @test_throws ArgumentError TailContracts.quantile_exception_tests(hits; exception_probability = 0.1)
    end
    for probability in (0, 1, -0.1, 1.1, true, NaN, Inf)
        @test_throws ArgumentError TailContracts.quantile_exception_tests([0, 1]; exception_probability = probability)
    end
end

function independent_pareto_loss(shape, scale, excesses)
    return length(excesses) * log(scale) + sum(shape == 0 ? value / scale : (1 + inv(shape)) * log1p(shape * value / scale) for value in excesses)
end

@testset "Supplied-threshold Pareto likelihood fitting" begin
    for row in density_tail_fixtures["fit"]
        result = TailContracts.fit_pot_tail_risk(row["losses"]; threshold = row["threshold"], probability = row["probability"])
        expected = row["expected"]
        @test result.shape ≈ expected["shape"] rtol = 5e-4 atol = 5e-4
        @test result.scale ≈ expected["scale"] rtol = 5e-4 atol = 5e-4
        @test result.shape ≈ row["tight_shape"] rtol = 1e-5 atol = 1e-5
        @test result.scale ≈ row["tight_scale"] rtol = 1e-5 atol = 1e-5
        for name in (:threshold, :exceedance_count, :sample_count, :exceedance_probability)
            @test getproperty(result, name) == expected[string(name)]
        end
        for name in (:value_at_risk, :expected_shortfall)
            @test getproperty(result, name) ≈ expected[string(name)] rtol = 5e-4 atol = 5e-4
        end
        excesses = (row["losses"][row["losses"] .> row["threshold"]] .- row["threshold"]) / row["unit"]
        objective = independent_pareto_loss(result.shape, result.scale / row["unit"], excesses)
        @test abs(objective - row["normalized_objective"]) / length(excesses) < 1e-7
        @test all(1 .+ result.shape * excesses / (result.scale / row["unit"]) .> 0)
        @test isfinite(objective)
        if result.shape >= 1
            @test result.expected_shortfall == Inf
        end
    end
    row = first(density_tail_fixtures["fit"])
    base = TailContracts.fit_pot_tail_risk(row["losses"]; threshold = row["threshold"], probability = row["probability"])
    for factor in (1e-50, 1e50)
        scaled = TailContracts.fit_pot_tail_risk(factor * row["losses"]; threshold = factor * row["threshold"], probability = row["probability"])
        @test scaled.shape ≈ base.shape rtol = 1e-5 atol = 1e-5
        @test scaled.scale / factor ≈ base.scale rtol = 1e-5 atol = 1e-5
        @test scaled.value_at_risk / factor ≈ base.value_at_risk rtol = 1e-5 atol = 1e-5
        @test scaled.expected_shortfall / factor ≈ base.expected_shortfall rtol = 1e-5 atol = 1e-5
    end
    for losses in (Float64[], [0.0, 0.5], [0.0, 2.0], [2.0, 2.0], [NaN, 2.0, 3.0], [1.0, Inf], [true, false])
        @test_throws ArgumentError TailContracts.fit_pot_tail_risk(losses; threshold = 1.0, probability = 0.99)
    end
    @test_throws ArgumentError TailContracts.fit_pot_tail_risk([0.0, 2.0, 3.0]; threshold = NaN, probability = 0.99)
    @test_throws ArgumentError TailContracts.fit_pot_tail_risk(row["losses"]; threshold = row["threshold"], probability = 1.0)
end
