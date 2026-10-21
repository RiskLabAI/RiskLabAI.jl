using Test, TOML, LinearAlgebra

@testset "MP analytic gradient and preserved failure rules" begin
    D=RiskLabAI.Data
    fixture=TOML.parsefile(joinpath(@__DIR__,"fixtures","mp_gradient_reference.toml"))
    for case in fixture["case"]
        values=Float64.(case["eigenvalues"])
        actual=D._mp_pdf_fit_gradient(case["variance"],case["q"],values,case["bandwidth"])
        @test actual ≈ case["gradient"] rtol=fixture["gradient_rtol"] atol=fixture["gradient_atol"]
        @test D._mp_pdf_fit_error(case["variance"],case["q"],values,case["bandwidth"]) ≈ case["objective"] rtol=2e-10 atol=2e-11
    end
    for (objective,derivative,expected) in [(v->(v-.23)^2,v->2*(v-.23),.23),(v->(v+2)^2,v->2*(v+2),1e-5),(v->(v-2)^2,v->2*(v-2),1-1e-5),(v->3.,v->0.,.5)]
        result=D._mp_bounded_local_min(objective;derivative)
        @test result.converged
        @test result.variance ≈ expected atol=2e-7
    end
    @test !D._mp_bounded_local_min(v->(v-.23)^2;derivative=v->2*(v-.23),max_iter=0).converged
    @test !D._mp_bounded_local_min(v->(v-.23)^2;derivative=v->2*(v-.23),max_evals=1).converged
    @test !D._mp_bounded_local_min(v->NaN;derivative=v->0.).converged
    cutoff,variance=D.find_max_eval([1.,2.],1.,nextfloat(0.))
    @test variance==1.
    @test cutoff==4.
    case=only(filter(row->row["name"]=="chapter23_0.42385584339082083",fixture["case"]))
    values=Float64.(case["eigenvalues"])
    cutoff,variance=D.find_max_eval(values,case["q"],case["bandwidth"])
    @test variance ≈ .4238558433793075 rtol=1e-8 atol=2e-9
    @test count(>(cutoff),values)==5
    @test D.find_max_eval(Matrix(Diagonal(values)),case["q"],case["bandwidth"])==(cutoff,variance)
end
