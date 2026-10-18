using Test
using TOML
using Random
using RiskLabAI
using RiskLabAI.Pde

function pde_equation_array(values)
    dimensions = Int[]
    cursor = values
    while cursor isa AbstractVector
        push!(dimensions, length(cursor))
        cursor = first(cursor)
    end
    flatten(value::Number) = [Float64(value)]
    flatten(value::AbstractVector) = reduce(vcat, flatten.(value))
    return permutedims(reshape(flatten(values), reverse(dimensions)...), reverse(Tuple(1:length(dimensions))))
end

@testset "Public PDE equation formulas and supplied-normal forward paths" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pde_equations_python_320.toml"))
    for case in fixture["equations"]
        config = case["configuration"]
        equation = getproperty(RiskLabAI.Pde, Symbol(case["kind"]))(config["dim"], config["total_time"], config["num_time_interval"])
        @test equation isa Equation
        @test equation.delta_t == config["total_time"] / config["num_time_interval"]
        @test equation.sqrt_delta_t == sqrt(equation.delta_t)
        normal = pde_equation_array(case["normal_draws"])
        original = copy(normal)
        increments, paths = pde_sample(equation, size(normal, 1); normal_draws = normal)
        @test isapprox(increments, pde_equation_array(case["increments"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(paths, pde_equation_array(case["paths"]); rtol = 1e-10, atol = 1e-12)
        @test normal == original
        x, y, z = (pde_equation_array(case[name]) for name in ("x", "y", "z"))
        @test isapprox(pde_driver(equation, 0.37, x, y, z), pde_equation_array(case["rate"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(pde_hamiltonian(equation, 0.37, x, y, z), pde_equation_array(case["hamiltonian"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(pde_terminal(equation, equation.total_time, x), pde_equation_array(case["terminal"]); rtol = 1e-10, atol = 1e-12)
        sigma = case["sigma"] isa Number ? case["sigma"] : pde_equation_array(case["sigma"])
        @test isapprox(pde_sigma(equation, x), sigma; rtol = 1e-10, atol = 1e-12)
        @test isapprox(pde_terminal_for_sample(equation, pde_equation_array(case["sample_states"])), pde_equation_array(case["sample_terminal"]); rtol = 1e-10, atol = 1e-12)
        rng = MersenneTwister(734)
        expected = rand(rng, 5)
        rng = MersenneTwister(734)
        pde_sample(equation, size(normal, 1); rng, normal_draws = normal)
        @test rand(rng, 5) == expected
    end
    @test isabstracttype(Equation)
    @test_throws MethodError Equation()
end

@testset "PDE equation domains and separate terminal helper contracts" begin
    for constructor in (HJBLQ, BlackScholesBarenblatt, PricingDefaultRisk, PricingDiffRate)
        @test_throws ArgumentError constructor(0, 1., 3)
        @test_throws ArgumentError constructor(2, -1., 3)
        @test_throws ArgumentError constructor(2, Inf, 3)
        @test_throws ArgumentError constructor(2, 1., 0)
        @test_throws ArgumentError constructor(true, 1., 3)
        equation = constructor(2, 0., 3)
        dw, path = pde_sample(equation, 2; normal_draws = ones(2, 2, 3))
        @test all(iszero, dw)
        @test all(path[:, :, index] == path[:, :, 1] for index in 2:4)
    end
    equation = PricingDiffRate(1, 1., 2)
    @test pde_terminal(equation, 1., fill(175., 1, 1)) == fill(5., 1, 1)
    @test pde_terminal_for_sample(equation, fill(175., 1, 1, 1)) == fill(55., 1, 1, 1)
    @test_throws ArgumentError pde_sample(equation, 0)
    @test_throws ArgumentError pde_sample(equation, true)
    @test_throws ArgumentError pde_sample(equation, 2; normal_draws = ones(2, 1, 3))
    @test_throws ArgumentError pde_sample(equation, 2; normal_draws = ones(Float32, 2, 1, 2))
    @test_throws ArgumentError pde_sample(equation, 2; normal_draws = fill(NaN, 2, 1, 2))
    @test_throws ArgumentError pde_terminal_for_sample(equation, ones(2, 2, 2))
end
