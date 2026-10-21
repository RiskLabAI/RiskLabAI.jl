using Test, TOML, RiskLabAI

@testset "OU saved stopping boundaries and independent moments" begin
    B = RiskLabAI.Backtest
    reference = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ou_boundary_reference.toml"))["cases"]
    contacts = Dict("python_fit"=>0, "julia_fit"=>0)
    for row in reference
        actual = B.optimal_ou_trading_rule(row["theta"], row["sigma"], row["entry_gap"]; cost=row["cost"], bounds=(row["lower"],row["upper"]))
        for (key, expected) in row["expected"]
            @test isapprox(getproperty(actual,Symbol(key)), expected; rtol=1e-9, atol=2e-11)
        end
        at_upper = abs(actual.stop_loss-row["upper"]) <= 1e-4 + 1e-5 * abs(row["upper"])
        @test at_upper
        startswith(row["case"], "bound_") && (contacts[row["variant"]] += at_upper)
    end
    @test contacts["python_fit"] == 4
    @test contacts["julia_fit"] == 4
    @test isnan(B.mean_exit_time(1e16,.1,.1,.1,1.))
end
