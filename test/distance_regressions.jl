using Test
using TOML
using Statistics
using RiskLabAI

distance_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-10, atol = 1e-12)
distance_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(distance_equal(a, b) for (a, b) in zip(left, right))
distance_matrix(rows) = permutedims(reduce(hcat, Float64.(row) for row in rows))

@testset "Shared floating histogram edges and automatic information bins" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "distance_python_320.toml"))
    for case in fixture["bins"]
        correlation = case["automatic"] ? nothing : case["correlation"]
        @test D.calculate_number_of_bins(case["observations"]; correlation = correlation) == case["output"]
    end
    for case in fixture["histograms"]
        x, y = Float64.(case["x"]), Float64.(case["y"])
        @test D._histogram2d(x, y, case["bins"]) == distance_matrix(case["histogram"])
        @test distance_equal(D.calculate_variation_of_information(x, y, case["bins"]), case["variation"])
        @test distance_equal(D.calculate_variation_of_information(x, y, case["bins"]; norm = true), case["normalized"])
        @test distance_equal(D.calculate_variation_of_information(y, x, case["bins"]), case["variation"])
    end
    for case in fixture["automatic"]
        x, y = Float64.(case["x"]), Float64.(case["y"])
        @test distance_equal(D.calculate_mutual_information(x, y), case["mi"])
        @test distance_equal(D.calculate_mutual_information(x, y; norm = true), case["normalized_mi"])
        @test distance_equal(D.calculate_variation_of_information_extended(x, y), case["vi"])
        @test distance_equal(D.calculate_variation_of_information_extended(x, y; norm = true), case["normalized_vi"])
    end
    @test_throws ArgumentError D.calculate_variation_of_information(Float64[], Float64[], 2)
    @test_throws ArgumentError D.calculate_variation_of_information([1., NaN], [1., 2.], 2)
    @test_throws ArgumentError D.calculate_variation_of_information([1., 2.], [1., 2.], 0)
    @test_throws DimensionMismatch D.calculate_variation_of_information([1.], [1., 2.], 2)
end

@testset "Shared angular distances and centered distance correlation" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "distance_python_320.toml"))
    for case in fixture["distances"]
        @test distance_equal(D.calculate_distance(distance_matrix(case["dependence"]); metric = case["metric"]), distance_matrix(case["output"]))
    end
    for case in fixture["dcor"]
        x, y = Float64[case["x"]...], Float64[case["y"]...]
        original = deepcopy((x, y))
        @test distance_equal(D.distance_correlation(x, y), case["output"])
        @test distance_equal(D.distance_correlation(y, x), case["output"])
        @test isequal((x, y), original)
        @test distance_equal(D.distance_correlation(-3 .* x, 2 .* y), case["output"])
    end
    @test_throws ArgumentError D.calculate_distance(ones(2, 2); metric = "unknown")
end

@testset "Shared KSG jitter and integer neighbor counts" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "distance_python_320.toml"))
    for case in fixture["ksg"]
        x, y = Float64[case["x"]...], Float64[case["y"]...]
        draws = (Float64[case["x_noise"]...], Float64[case["y_noise"]...])
        original = deepcopy((x, y, draws))
        actual = D.ksg_mutual_information(x, y; k = case["k"], jitter_draws = draws)
        @test distance_equal(actual, case["output"])
        @test isequal((x, y, draws), original)
        if length(x) >= 2
            jitter(values, noise) = values .+ noise .* (std(values; corrected = false) == 0 ? 1. : std(values; corrected = false)) .* 1e-10
            xj, yj = jitter(x, draws[1]), jitter(y, draws[2])
            k = min(case["k"], length(x)-1)
            for index in eachindex(x)
                radius = sort(max.(abs.(xj .- xj[index]), abs.(yj .- yj[index])))[k+1]*(1-1e-10)
                @test count(value -> abs(value-xj[index]) <= radius, xj)-1 == case["nx"][index]
                @test count(value -> abs(value-yj[index]) <= radius, yj)-1 == case["ny"][index]
            end
        end
    end
    @test_throws DimensionMismatch D.ksg_mutual_information([1.], [1., 2.])
    @test_throws DimensionMismatch D.ksg_mutual_information([1., 2.], [1., 2.]; jitter_draws = ([1.], [1., 2.]))
    @test_throws ArgumentError D.ksg_mutual_information([1., 2.], [1., 2.]; jitter_draws = ([1., NaN], [1., 2.]))
    @test_throws ArgumentError D.ksg_mutual_information([1., 2.], [1., 2.]; k = 0)
end
