using Test
using TOML
using Dates
using DataFrames
using RiskLabAI

trend_weight_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 2e-8, atol = 2e-9)
trend_weight_equal(left::AbstractVector, right::AbstractVector) = length(left) == length(right) && all(trend_weight_equal(a, b) for (a, b) in zip(left, right))

@testset "Shared trend regression and exact forward-window selection" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "trend_weights_python_320.toml"))
    for case in fixture["regression"]
        @test trend_weight_equal(D.calculate_t_value_linear_regression(Float64[case["values"]...]), case["output"])
    end
    date(offset) = DateTime(2020, 1, 1)+Minute(offset)
    for case in fixture["scanning"]
        values = Float64.(case["values"])
        index, molecule = date.(case["offsets"]), date.(case["molecule"])
        original = deepcopy((values, index, molecule))
        result = D.find_trend_using_trend_scanning(molecule, index, values, Tuple(Int.(case["span"])))
        @test result.event_start == date.(Int[case["starts"]...])
        @test result.end_time == date.(Int[case["ends"]...])
        @test trend_weight_equal(result.t_value, case["t"])
        @test result.trend == case["trend"]
        @test all(!isnan(value) for value in result.trend)
        @test isequal((values, index, molecule), original)
    end
end

@testset "Shared exhaustive uniqueness and chronological time-decay weights" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "trend_weights_python_320.toml"))
    for case in fixture["uniqueness"]
        matrix = zeros(case["rows"], case["columns"])
        for row in axes(matrix, 1), column in axes(matrix, 2)
            matrix[row, column] = case["matrix"][row][column]
        end
        original = copy(matrix)
        @test trend_weight_equal(D.calculate_average_uniqueness(matrix), case["output"])
        @test isequal(matrix, original)
    end
    for case in fixture["decay"]
        values = Float64[case["values"]...]
        original = copy(values)
        @test trend_weight_equal(D.calculate_time_decay(values; clf_last_weight = case["last_weight"]), case["output"])
        @test isequal(values, original)
    end
    @test_throws ArgumentError D.calculate_time_decay([1., 2.]; clf_last_weight = -.01)
    @test_throws ArgumentError D.calculate_time_decay([1., 2.]; clf_last_weight = 1.01)
end
