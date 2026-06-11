using Test
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
