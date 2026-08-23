@testset "specification diagnostics" begin
    fork = fork_population_diagnostics()
    @test fork.unconditioned.coefficients == (0.0, 0.5)
    @test fork.conditioned.coefficients == (0.0, 0.0, 1.0)
    @test fork.unconditioned.r_squared == 0.25
    @test fork.conditioned.r_squared == 0.5

    collider = collider_population_diagnostics()
    @test collider.unconditioned.coefficients == (0.0, 0.0)
    @test collider.conditioned.coefficients == (0.0, -0.5, 0.5)
    @test collider.conditioned.r_squared == 0.5

    mediator = confounded_mediator_population_diagnostics()
    @test mediator.unconditioned.coefficients == (0.0, 1.0)
    @test mediator.conditioned.coefficients == (0.0, -0.5, 1.5)
    @test mediator.unconditioned.r_squared == 1.0 / 7.0
    @test mediator.conditioned.r_squared == 11.0 / 14.0

    fork_sample = fork_specification_experiment()
    @test fork_sample.unconditioned.coefficients[2] ≈ 0.5 atol = 0.06
    @test fork_sample.conditioned.coefficients[2] ≈ 0.0 atol = 0.06
    @test fork_sample.conditioned.coefficients[3] ≈ 1.0 atol = 0.06

    collider_sample = collider_specification_experiment()
    @test collider_sample.unconditioned.coefficients[2] ≈ 0.0 atol = 0.06
    @test collider_sample.conditioned.coefficients[2] ≈ -0.5 atol = 0.06
    @test collider_sample.conditioned.coefficients[3] ≈ 0.5 atol = 0.06

    mediator_sample = confounded_mediator_specification_experiment()
    @test mediator_sample.unconditioned.coefficients[2] ≈ 1.0 atol = 0.08
    @test mediator_sample.conditioned.coefficients[2] ≈ -0.5 atol = 0.08
    @test mediator_sample.conditioned.coefficients[3] ≈ 1.5 atol = 0.08

    @test fork_specification_experiment(200, 17) ==
          fork_specification_experiment(200, 17)
    @test fork_specification_experiment(200, 17) !=
          fork_specification_experiment(200, 18)
    @test_throws ArgumentError fork_specification_experiment(3, 0)
    @test_throws ArgumentError fork_specification_experiment(10, -1)
end
