using Test
using TOML
using LinearAlgebra
using Zygote
using RiskLabAI

const ConvexUnderTest = RiskLabAI.Optimization
convex_fixture_matrix(rows) = reduce(vcat, permutedims.(rows))
const convex_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "convex_python_320.toml"))

@testset "Greatest clearing payments against Python 3.2.0" begin
    for row in convex_fixtures["clearing"]
        assets, liabilities = row["external_assets"], row["nominal_liabilities"]
        relative = convex_fixture_matrix(row["relative_liabilities"])
        original = (copy(assets), copy(liabilities), copy(relative))
        result = ConvexUnderTest.eisenberg_noe_clearing(assets, liabilities, relative)
        scale = max(maximum(liabilities), floatmin(Float64))
        @test maximum(abs.(result - row["expected"])) / scale <= 1e-8
        @test maximum(abs.(result - row["independent"])) / scale <= 1e-8
        @test maximum(abs.(result - min.(liabilities, assets + relative' * result))) / scale <= 1e-8
        @test all(0 .<= result .<= liabilities)
        @test original == (assets, liabilities, relative)
    end
    @test ConvexUnderTest.eisenberg_noe_clearing([0.0, 0.0], [1.0, 1.0], [0.0 1.0; 1.0 0.0]) ≈ [1, 1] atol = 1e-8
    for (assets, liabilities, relative) in (([], [], zeros(0, 0)), ([0.0], [1.0, 2.0], zeros(1, 1)), ([-1.0], [1.0], zeros(1, 1)), ([0.0], [NaN], zeros(1, 1)), ([0.0], [1.0], fill(1.01, 1, 1)), ([true], [1.0], zeros(1, 1)), ([0.0], [1.0], fill(-0.1, 1, 1)))
        @test_throws ArgumentError ConvexUnderTest.eisenberg_noe_clearing(assets, liabilities, relative)
    end
    for tolerance in (0, 1, true, NaN)
        @test_throws ArgumentError ConvexUnderTest.eisenberg_noe_clearing([0.0], [1.0], zeros(1, 1); atol = tolerance)
    end
end

@testset "Public-book clearing payment and unpaid-balance accuracy" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "clearing_accuracy_book_cases.toml"))
    for row in fixture["cases"]
        assets, liabilities = row["external_assets"], row["nominal_liabilities"]
        relative = convex_fixture_matrix(row["relative_liabilities"])
        payments = ConvexUnderTest.eisenberg_noe_clearing(assets, liabilities, relative)
        unpaid = max.(liabilities - payments, 0)
        @test all(isapprox.(payments, row["expected_payments"]; rtol = 1e-9, atol = 1e-11))
        @test all(isapprox.(unpaid, row["expected_unpaid"]; rtol = 1e-9, atol = 1e-11))
        @test (unpaid .> 1e-8) == row["expected_defaults"]
        @test all(0 .<= payments .<= liabilities)
        @test maximum(abs.(payments - min.(liabilities, assets + relative' * payments))) / maximum(liabilities) <= 1e-12
        @test sum(unpaid) ≈ sum(row["expected_unpaid"]) rtol = 1e-9 atol = 1e-11
    end
end

@testset "Discrete execution against Python 3.2.0 and first-order equations" begin
    for row in convex_fixtures["execution"]
        kwargs = (; (Symbol(name) => row[name] for name in ("volatility", "temporary_impact", "permanent_impact", "risk_aversion"))...)
        result = ConvexUnderTest.almgren_chriss_execution(row["inventory"], row["horizon"], row["intervals"]; kwargs...)
        for name in (:times, :inventory, :trades, :expected_cost, :cost_variance)
            @test getproperty(result, name) ≈ row["expected_" * string(name)] rtol = 1e-10 atol = 1e-12
        end
        @test result.inventory ≈ row["independent_inventory"] rtol = 1e-10 atol = 1e-12
        @test all(result.trades .>= 0)
        @test result.inventory[1] == row["inventory"] && result.inventory[end] == 0
        @test sum(result.trades) ≈ row["inventory"] rtol = 1e-12
        tau = row["horizon"] / row["intervals"]
        eta = row["temporary_impact"] - row["permanent_impact"] * tau / 2
        @test result.expected_cost ≈ row["permanent_impact"] * row["inventory"]^2 / 2 + eta / tau * sum(abs2, result.trades) rtol = 1e-12
        @test result.cost_variance ≈ row["volatility"]^2 * tau * sum(abs2, result.inventory[2:end]) rtol = 1e-12
    end
    @test ConvexUnderTest.almgren_chriss_execution(1.0, 1.0, 1; volatility = 2.0, temporary_impact = 1.0).cost_variance == 0
    @test ConvexUnderTest.almgren_chriss_execution(0.0, 1.0, 3; volatility = 2.0, temporary_impact = 1.0).trades == zeros(3)
    @test ConvexUnderTest.almgren_chriss_execution(3.0, 1.0, 3; volatility = 0.0, temporary_impact = 1.0, risk_aversion = 1.0).trades ≈ ones(3)
    for count in (0, -1, true, 2.0)
        @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(1.0, 1.0, count; volatility = 0.2, temporary_impact = 0.1)
    end
    @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(1.0, 0.0, 2; volatility = 0.2, temporary_impact = 0.1)
    @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(1.0, 1.0, 2; volatility = 0.2, temporary_impact = 0.1, permanent_impact = 0.4)
    @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(1e200, 1.0, 2; volatility = 0.2, temporary_impact = 0.1)
    @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(true, 1.0, 2; volatility = 0.2, temporary_impact = 0.1)
    @test_throws ArgumentError ConvexUnderTest.almgren_chriss_execution(-1, 1.0, 2; volatility = 0.2, temporary_impact = 0.1)
end

@testset "Finite-hull SPO+ values, ties and native gradients" begin
    for row in convex_fixtures["spo"]
        predicted, truth = row["predicted_cost"], row["true_cost"]
        vertices = convex_fixture_matrix(row["vertices"])
        result = ConvexUnderTest.spo_plus_loss(predicted, truth, vertices)
        @test result.loss == row["expected_loss"]
        @test result.subgradient == row["expected_subgradient"]
        @test result.true_vertex == row["expected_true_vertex"] + 1
        @test result.transformed_vertex == row["expected_transformed_vertex"] + 1
        @test ConvexUnderTest.spo_plus(predicted, truth, vertices) == result.loss
        gradients = Zygote.gradient((p, t, v) -> ConvexUnderTest.spo_plus(p, t, v), predicted, truth, vertices)
        @test gradients[1] == result.subgradient
        @test isnothing(gradients[2]) && isnothing(gradients[3])
        batch_prediction = permutedims(hcat(predicted, -predicted))
        batch_truth = permutedims(hcat(truth, truth))
        batch = ConvexUnderTest.spo_plus(batch_prediction, batch_truth, vertices)
        expected_batch = [ConvexUnderTest.spo_plus_loss(vec(p), vec(t), vertices) for (p, t) in zip(eachrow(batch_prediction), eachrow(batch_truth))]
        @test batch == [result.loss for result in expected_batch]
        batch_gradient = only(Zygote.gradient(p -> sum(ConvexUnderTest.spo_plus(p, batch_truth, vertices)), batch_prediction))
        @test batch_gradient == reduce(vcat, permutedims.([result.subgradient for result in expected_batch]))
    end
    @test_throws ArgumentError ConvexUnderTest.spo_plus_loss([1.0], [1.0, 2.0], ones(2, 1))
    @test_throws ArgumentError ConvexUnderTest.spo_plus_loss([NaN], [1.0], ones(1, 1))
    @test_throws ArgumentError ConvexUnderTest.spo_plus_loss([1e308], [-1e308], fill(2.0, 1, 1))
    @test_throws ArgumentError ConvexUnderTest.spo_plus([1], [1], ones(1, 1))
    @test_throws ArgumentError ConvexUnderTest.spo_plus(Float32[1], [1.0], ones(1, 1))
    @test_throws ArgumentError ConvexUnderTest.spo_plus([1.0], [1.0], zeros(0, 1))
end

function check_birkhoff(matrix)
    original = copy(matrix)
    weights, permutations = ConvexUnderTest.birkhoff_von_neumann_decomposition(matrix)
    n = size(matrix, 1)
    reconstruction = zeros(n, n)
    @test all(weights .> 0)
    @test abs(sum(weights) - 1) <= 1e-10
    for (weight, permutation) in zip(weights, eachrow(permutations))
        @test sort(permutation) == collect(1:n)
        for i in 1:n
            reconstruction[i, permutation[i]] += weight
        end
    end
    @test maximum(abs.(reconstruction - matrix)) <= 1e-10
    @test matrix == original
end

@testset "Majorization witnesses and Birkhoff reconstructions" begin
    for row in convex_fixtures["majorization"]
        target, source = row["target"], row["source"]
        @test ConvexUnderTest.is_majorized(target, source) == row["expected"]
        if row["expected"]
            witness = ConvexUnderTest.majorization_matrix(target, source)
            @test ConvexUnderTest.is_doubly_stochastic(witness)
            @test maximum(abs.(witness * source - target)) <= 1e-10
            check_birkhoff(witness)
        else
            @test_throws ArgumentError ConvexUnderTest.majorization_matrix(target, source)
        end
    end
    for row in convex_fixtures["doubly_stochastic"]
        matrix = convex_fixture_matrix(row["matrix"])
        @test ConvexUnderTest.is_doubly_stochastic(matrix) == row["expected"]
        check_birkhoff(matrix)
    end
    @test ConvexUnderTest.is_doubly_stochastic([true false; false true])
    for matrix in (zeros(0, 0), ones(2, 3), ones(2, 2), [1.0 0.0; NaN 1.0], [1.1 -0.1; -0.1 1.1], ones(ComplexF64, 2, 2))
        @test !ConvexUnderTest.is_doubly_stochastic(matrix)
        @test_throws ArgumentError ConvexUnderTest.birkhoff_von_neumann_decomposition(matrix)
    end
    for tolerance in (0, 1, true, NaN)
        @test_throws ArgumentError ConvexUnderTest.is_doubly_stochastic(ones(1, 1); atol = tolerance)
        @test_throws ArgumentError ConvexUnderTest.is_majorized([1.0], [1.0]; atol = tolerance)
    end
    for (target, source) in (([], []), ([1.0], [1.0, 2.0]), ([NaN], [1.0]), ([1.0im], [1.0]))
        @test_throws ArgumentError ConvexUnderTest.is_majorized(target, source)
    end
end
