using Test
using TOML
using Distributions: Normal, cdf, pdf
using RiskLabAI

const SearchFDR = RiskLabAI.CausalFactorAnalysis
const search_fdr_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "search_fdr_python_320.toml"))

search_fdr_model(row) = SearchFDR.GaussianTrialMixture(row["trial_null_probability"], row["null_standard_deviation"], row["alternative_mean"], row["alternative_standard_deviation"])

function check_search_fields(result, expected)
    for name in propertynames(result)
        actual, reference = getproperty(result, name), expected[string(name)]
        if actual isa Bool || actual isa Integer
            @test actual == reference
        else
            @test actual ≈ reference rtol = 1e-10 atol = 1e-12
        end
    end
end

@testset "Single and family search posteriors against Python 3.2.0" begin
    for row in search_fdr_fixtures["single"]
        @test SearchFDR.single_trial_false_discovery_rate(row["alpha"], row["beta"], row["probability"]) ≈ row["expected"] rtol = 1e-10 atol = 1e-12
    end
    for row in search_fdr_fixtures["family"]
        @test SearchFDR.family_level_false_discovery_rate(row["alpha"], row["beta"], row["probability"], row["trials"]) ≈ row["expected"] rtol = 1e-10 atol = 1e-12
    end
    for row in search_fdr_fixtures["errors"]
        result = SearchFDR.max_selection_family_errors(row["alpha"], row["beta"], row["probability"], row["trials"])
        @test result isa SearchFDR.MaxSelectionFamilyErrors
        check_search_fields(result, row["expected"])
    end
    for row in search_fdr_fixtures["comparison"]
        result = SearchFDR.compare_single_and_family_fdr(row["alpha"], row["beta"], row["probability"], row["trials"]; family_type_i_error = row["family_alpha"], family_type_ii_error = row["family_beta"])
        @test result isa SearchFDR.FDRComparisonEvidence
        check_search_fields(result, row["expected"])
    end
    for row in search_fdr_fixtures["mixture"]
        @test SearchFDR.maximum_mixture_cdf(row["null_cdf"], row["alternative_cdf"], row["probability"], row["trials"]) ≈ row["expected"] rtol = 1e-10 atol = 1e-12
    end
    for row in search_fdr_fixtures["conditional"]
        @test SearchFDR.conditional_upper_tail_probability(row["value"], row["threshold"]) ≈ row["expected"] rtol = 1e-10 atol = 1e-12
    end
    @test SearchFDR.single_trial_false_discovery_rate(1e-300, 0, 0.5) ≈ 1e-300 rtol = 1e-12 atol = 0.0
    @test SearchFDR.family_level_false_discovery_rate(0.1, 1, 0.1, 1000) == 1.0
end

@testset "Gaussian selected maxima and nonidentification witness" begin
    for row in search_fdr_fixtures["gaussian"]
        model = search_fdr_model(row["model"])
        @test !ismutabletype(typeof(model))
        check_search_fields(model, row["model"])
        for (index, observation) in enumerate(row["observations"])
            @test SearchFDR.gaussian_trial_mixture_cdf(model, observation) ≈ row["expected"]["trial_cdf"][index] rtol = 1e-10 atol = 1e-12
            @test SearchFDR.gaussian_max_selection_cdf(model, observation, row["trials"]) ≈ row["expected"]["maximum_cdf"][index] rtol = 1e-10 atol = 1e-12
            @test SearchFDR.gaussian_max_selection_log_density(model, observation, row["trials"]) ≈ row["expected"]["log_density"][index] rtol = 1e-10 atol = 1e-12
        end
        @test SearchFDR.gaussian_max_selection_log_likelihood(model, row["observations"], row["trials"]) ≈ row["expected"]["log_likelihood"] rtol = 1e-10 atol = 1e-12
    end
    for row in search_fdr_fixtures["aggregate"]
        model, thresholds = search_fdr_model(row["model"]), copy(row["thresholds"])
        result = SearchFDR.gaussian_search_adjusted_false_discovery_rate(model, thresholds, row["trials"])
        @test result isa SearchFDR.GaussianSearchAdjustedFDR
        check_search_fields(result, row["expected"])
        thresholds .= 999.0
        @test collect(result.thresholds) == row["thresholds"]
        @test result.thresholds isa Vector{Float64}
    end
    for row in search_fdr_fixtures["witness"]
        result = SearchFDR.fdr_nonidentification_witness(row["observable"], row["target"])
        @test result isa SearchFDR.FDRNonIdentificationWitness
        check_search_fields(result, row["expected"])
        @test result.family_false_discovery_rate ≈ row["target"] rtol = 1e-10 atol = 1e-12
        @test result.reconstructed_observable_cdf_at_threshold ≈ row["observable"] rtol = 1e-10 atol = 1e-12
    end
end

@testset "Selected-winner quadrature and independent identities" begin
    if all(isdefined(SearchFDR, name) for name in (:_SELECTION_KRONROD_WEIGHTS, :_SELECTION_KRONROD_POINTS, :_SELECTION_GAUSS_WEIGHTS))
        for degree in 0:22
            value = sum(SearchFDR._SELECTION_KRONROD_WEIGHTS[index] * (SearchFDR._SELECTION_KRONROD_POINTS[index]^degree + (-SearchFDR._SELECTION_KRONROD_POINTS[index])^degree) for index in 1:7) + SearchFDR._SELECTION_KRONROD_WEIGHTS[8] * SearchFDR._SELECTION_KRONROD_POINTS[8]^degree
            @test value ≈ (iseven(degree) ? 2 / (degree + 1) : 0.0) rtol = 2e-14 atol = 2e-14
        end
        for degree in 0:13
            value = sum(SearchFDR._SELECTION_GAUSS_WEIGHTS[index] * (SearchFDR._SELECTION_KRONROD_POINTS[2index]^degree + (-SearchFDR._SELECTION_KRONROD_POINTS[2index])^degree) for index in 1:3) + SearchFDR._SELECTION_GAUSS_WEIGHTS[4] * SearchFDR._SELECTION_KRONROD_POINTS[8]^degree
            @test value ≈ (iseven(degree) ? 2 / (degree + 1) : 0.0) rtol = 2e-14 atol = 2e-14
        end
    end
    for row in search_fdr_fixtures["selection"]
        model = search_fdr_model(row["model"])
        result = SearchFDR.max_selection_null_probability(row["threshold"], model.trial_null_probability, row["trials"];
            null_density = value -> pdf(Normal(0, model.null_standard_deviation), value),
            mixture_cdf = value -> SearchFDR.gaussian_trial_mixture_cdf(model, value))
        @test result isa SearchFDR.SelectionLevelProbabilityEvidence
        for name in (:joint_null_and_selection_probability, :selection_probability, :conditional_null_probability)
            @test getproperty(result, name) ≈ row["expected"][string(name)] rtol = 1e-10 atol = 1e-12
        end
        @test result.joint_null_and_selection_probability ≈ row["independent_joint"] rtol = 1e-8 atol = 1e-10
        @test 0 <= result.conditional_null_probability <= 1
        @test 0 <= result.quadrature_absolute_error <= max(1e-12 * row["trials"] * model.trial_null_probability, 1e-10 * result.joint_null_and_selection_probability) * (1 + 1e-12)
    end
    for probability in (0.0, 0.1, 0.9, 1.0), trials in (1, 5, 100)
        result = SearchFDR.max_selection_null_probability(0.0, probability, trials; null_density = value -> pdf(Normal(), value), mixture_cdf = value -> cdf(Normal(), value))
        @test result.conditional_null_probability ≈ probability rtol = 1e-10 atol = 1e-12
        @test result.selection_probability ≈ 1 - 0.5^trials rtol = 1e-10 atol = 1e-12
    end
    skipped = SearchFDR.max_selection_null_probability(0.0, 0.0, 2; null_density = value -> error("density must not be evaluated"), mixture_cdf = value -> 0.5)
    @test skipped.joint_null_and_selection_probability == skipped.quadrature_absolute_error == 0
    if isdefined(SearchFDR, :_SELECTION_KRONROD_WEIGHTS)
        @test_throws ArgumentError SearchFDR.max_selection_null_probability(-2.0, 0.5, 10; null_density = value -> pdf(Normal(), value), mixture_cdf = value -> cdf(Normal(), value), absolute_tolerance = 1e-14, relative_tolerance = 1e-14, max_subintervals = 1)
    end
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0.0, 1.0, 1; null_density = value -> 10pdf(Normal(), value), mixture_cdf = value -> cdf(Normal(), value))
end

@testset "Selected-winner reported quadrature diagnostics" begin
    model = SearchFDR.GaussianTrialMixture(0.95, 1.0, 0.3, 1.0)
    expected_errors = [3.444780539307629e-14, 2.425603797050804e-13, 7.234731622609446e-13, 2.7334009146924937e-12, 2.1909677950044404e-11]
    expected_probabilities = [0.907421416202264, 0.9071719137893014, 0.9068565103644982, 0.9062157995070059, 0.9042505451539194]
    for (index, trials) in enumerate([1, 5, 10, 20, 50])
        result = SearchFDR.max_selection_null_probability(1.96, 0.95, trials;
            null_density = value -> exp(-0.5 * value^2) / sqrt(2pi),
            mixture_cdf = value -> SearchFDR.gaussian_trial_mixture_cdf(model, value))
        if isdefined(SearchFDR, :_SELECTION_KRONROD_WEIGHTS)
            @test result.quadrature_absolute_error ≈ expected_errors[index] rtol = 1e-9 atol = 1e-11
        else
            @test 0 <= result.quadrature_absolute_error <= 1e-9
        end
        @test result.conditional_null_probability ≈ expected_probabilities[index] rtol = 1e-10 atol = 1e-12
    end
end

@testset "Search probability domains and undefined posteriors" begin
    for invalid in (-0.1, 1.1, true, NaN, Inf)
        @test_throws ArgumentError SearchFDR.single_trial_false_discovery_rate(invalid, 0.2, 0.5)
        @test_throws ArgumentError SearchFDR.single_trial_false_discovery_rate(0.1, invalid, 0.5)
        @test_throws ArgumentError SearchFDR.single_trial_false_discovery_rate(0.1, 0.2, invalid)
        @test_throws ArgumentError SearchFDR.GaussianTrialMixture(invalid, 1, 0, 1)
    end
    @test_throws ArgumentError SearchFDR.single_trial_false_discovery_rate(0, 1, 0.5)
    @test_throws ArgumentError SearchFDR.family_level_false_discovery_rate(0, 1, 0.5, 5)
    @test_throws ArgumentError SearchFDR.single_trial_false_discovery_rate(0, 0, 1)
    for invalid in (0, -1, true, NaN, Inf)
        @test_throws ArgumentError SearchFDR.GaussianTrialMixture(0.5, invalid, 0, 1)
        @test_throws ArgumentError SearchFDR.GaussianTrialMixture(0.5, 1, 0, invalid)
    end
    @test_throws ArgumentError SearchFDR.GaussianTrialMixture(0.5, 1, NaN, 1)
    for trials in (0, -1, true, 1.0)
        @test_throws ArgumentError SearchFDR.family_level_false_discovery_rate(0.1, 0.2, 0.5, trials)
        @test_throws ArgumentError SearchFDR.max_selection_family_errors(0.1, 0.2, 0.5, trials)
        @test_throws ArgumentError SearchFDR.maximum_mixture_cdf(0.1, 0.2, 0.5, trials)
    end
    for probability in (0, 1)
        @test_throws ArgumentError SearchFDR.max_selection_family_errors(0.1, 0.2, probability, 2)
        @test_throws ArgumentError SearchFDR.fdr_nonidentification_witness(probability, 0.5)
        @test_throws ArgumentError SearchFDR.fdr_nonidentification_witness(0.5, probability)
    end
    @test_throws ArgumentError SearchFDR.conditional_upper_tail_probability(0.4, 0.5)
    @test_throws ArgumentError SearchFDR.conditional_upper_tail_probability(1, 1)
    @test_throws ArgumentError SearchFDR.compare_single_and_family_fdr(0, 0.2, 0.5, 2; family_type_i_error = 0.2, family_type_ii_error = 0.1)
    @test_throws ArgumentError SearchFDR.compare_single_and_family_fdr(0.1, 0.2, 0.5, 2; family_type_i_error = 0.2, family_type_ii_error = 0.1, relative_tolerance = -1)
    model = SearchFDR.GaussianTrialMixture(0.5, 1, 1, 1)
    for values in (Float64[], [true], [NaN], [Inf], [1im], ones(1, 2))
        @test_throws ArgumentError SearchFDR.gaussian_max_selection_log_likelihood(model, values, 2)
        @test_throws ArgumentError SearchFDR.gaussian_search_adjusted_false_discovery_rate(model, values, 2)
    end
    @test SearchFDR.gaussian_max_selection_log_likelihood(model, [-1e200], 2) == -Inf
    @test_throws ArgumentError SearchFDR.gaussian_trial_mixture_cdf(model, NaN)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = 1, mixture_cdf = value -> 0.5)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> -1, mixture_cdf = value -> 0.5)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> NaN, mixture_cdf = value -> 0.5)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> 1, mixture_cdf = value -> 1.1)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> 1, mixture_cdf = value -> 1.0)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> 1, mixture_cdf = value -> 0.5, max_subintervals = 0)
    @test_throws ArgumentError SearchFDR.max_selection_null_probability(0, 0.5, 2; null_density = value -> 1, mixture_cdf = value -> 0.5, absolute_tolerance = 0)
end
