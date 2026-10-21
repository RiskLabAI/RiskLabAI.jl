using Test
using TOML
using Dates
using DataFrames
using RiskLabAI

@testset "Infrastructure: work partitions and submitted-order mapping" begin
    P = RiskLabAI.ParallelWork
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "infrastructure_python_320.toml"))
    for case in fixture["partitions"]
        @test P.linear_partitions(case["count"], case["workers"]) == case["linear"]
        @test P.nested_partitions(case["count"], case["workers"]) == case["nested"]
        @test P.nested_partitions(case["count"], case["workers"]; upper_triangle = true) == case["upper"]
        boundaries = case["linear"]
        flattened = reduce(vcat, [collect(boundaries[index]+1:boundaries[index+1]) for index in 1:length(boundaries)-1]; init = Int[])
        @test flattened == collect(1:case["count"])
    end
    for case in fixture["parallel"]
        values = Vector{Int}(case["values"])
        callback = case["chunked"] ? (positions -> [[index, values[index]^2] for index in positions]) : (value -> value^2)
        result = P.parallel_run(callback, values; num_cpus = case["workers"], lin_partition = case["chunked"])
        @test result == case["output"]
    end
    @test P.parallel_run((value; scale) -> value*scale, (index for index in 1:5); num_cpus = 2, scale = 3) == [3,6,9,12,15]
    @test_throws ArgumentError P.linear_partitions(5, 0)
    @test_throws ArgumentError P.nested_partitions(-1, 2)
    @test_throws ArgumentError P.parallel_run(identity, 1:3; num_cpus = 0)
end

@testset "Infrastructure: completion collection and explicit table indices" begin
    P = RiskLabAI.ParallelWork
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "infrastructure_python_320.toml"))
    for case in fixture["indexed"]
        lookup = Dict(zip(case["indices"], case["values"]))
        callback = function (; molecule, values)
            selected = reverse(molecule)
            return DataFrame(index = selected, value = [values[index]^2 for index in selected])
        end
        result = P.parallel_indexed(callback, case["indices"]; num_threads = 2,
            mp_batches = case["batches"], linear_partition = case["linear"], values = lookup)
        @test result.index == case["output_indices"]
        @test result.value == case["output_values"]
    end
    callback = (; key, delay, value) -> begin
        sleep(delay)
        (key, value^2)
    end
    jobs = [(func = callback, key = index, delay = index == 1 ? .035 : 0., value = index+1) for index in 1:6]
    expected = [(index, (index+1)^2) for index in 1:6]
    @test P.process_jobs_sequential(jobs) == expected
    @test P.process_jobs(jobs; num_threads = 2, ordered = true) == expected
    completion = P.process_jobs(jobs; num_threads = 2)
    @test sort(completion; by = first) == expected
    @test length(unique(first.(completion))) == length(jobs)
    @test all(job.func === callback for job in jobs)
    dictionaries = [Dict("func" => ((; value) -> value^2), "value" => index) for index in 1:4]
    @test P.process_jobs(dictionaries; num_threads = 2, ordered = true) == [1,4,9,16]
    @test all(haskey(job, "func") for job in dictionaries)
    @test isempty(P.process_jobs(NamedTuple[]; num_threads = 2))
    @test_throws Exception P.process_jobs([(func = (; value) -> error("job failed"), value = 1)]; num_threads = 2)
end

@testset "Infrastructure: registry aliases, replacement and lazy factories" begin
    C = RiskLabAI.Components
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "infrastructure_python_320.toml"))
    registry = C.ComponentRegistry("local contract")
    scalar = (; factor = 1.) -> 4*factor
    for case in fixture["registry"]
        plan = case["plan"]
        action, key = plan["action"], plan["key"]
        if action in ("register", "override")
            C.register_component!(registry, key, scalar; aliases = plan["aliases"], metadata = plan["metadata"], override = action == "override", accepted_keywords = [:factor])
            @test C.component_metadata(registry, key) == plan["metadata"]
            copied = C.component_metadata(registry, key)
            copied["family"] = "changed locally"
            @test C.component_metadata(registry, key) == plan["metadata"]
        elseif action == "create"
            @test C.create_component(registry, key; factor = plan["factor"], ignored = 99, filter_unknown_kwargs = plan["filter"]) == plan["output"]
        elseif action == "duplicate"
            @test_throws KeyError C.register_component!(registry, key, scalar)
        else
            C.unregister_component!(registry, key)
        end
        @test C.available_components(registry) == case["state"]["available"]
        @test C.component_aliases(registry) == case["state"]["aliases"]
        @test length(registry) == case["state"]["length"]
        for name in C.available_components(registry)
            @test uppercase(name) in registry
            @test registry[name] === scalar
        end
    end
    loads = Ref(0)
    C.register_lazy!(registry, "counter", () -> begin
        loads[] += 1
        text -> Dict(character => count(==(character), text) for character in unique(text))
    end)
    @test C.is_lazy(registry, "counter")
    @test C.create_component(registry, "COUNTER", "abracadabra")['a'] == 5
    @test !C.is_lazy(registry, "counter")
    @test C.create_component(registry, "counter", "abaca")['a'] == 3
    @test loads[] == 1
    @test !(42 in registry)
    @test_throws KeyError C.create_component(registry, "absent")
    @test_throws ArgumentError C.register_component!(registry, "", scalar)
    @test C.list_components() == fixture["catalogue"]
    for (family, names) in fixture["catalogue"]
        for name in names
            @test C.component_metadata(C.get_registry(family), name) == fixture["catalogue_metadata"][family][name]
        end
    end
    @test C.create_component(C.CROSS_VALIDATORS, "KFOLD", 3) isa RiskLabAI.Validation.KFoldCV
    @test C.create_component(C.FEATURE_IMPORTANCE, "mdi").function_value === RiskLabAI.Features.feature_importance_mdi
    @test C.create_component(C.BARS, "standard_bars"; metric = :tick, threshold = 7.) isa RiskLabAI.Data.StandardBars{RiskLabAI.Data.Tick}
    @test C.create_component(C.BARS, "time_bars"; resolution_type = "S", resolution_units = 1) isa RiskLabAI.Data.TimeBars
end

struct InfrastructureDirectionLabeler <: RiskLabAI.Components.AbstractLabeler
    horizon::Int
end

function RiskLabAI.Components.label(model::InfrastructureDirectionLabeler, prices)
    forward = [index+model.horizon <= length(prices) ? prices[index+model.horizon]/value-1 : NaN for (index, value) in enumerate(prices)]
    return (forward = forward, direction = [isnan(value) ? 0 : Int(sign(value)) for value in forward])
end

@testset "Infrastructure: labeler dispatch and complete bar-input status" begin
    C = RiskLabAI.Components
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "infrastructure_python_320.toml"))
    for case in fixture["labels"]
        actual = C.label(InfrastructureDirectionLabeler(case["horizon"]), case["prices"])
        @test isequal(actual.forward, case["forward"])
        @test actual.direction == case["direction"]
    end
    @test C.BAR_COLUMNS == fixture["bar_columns"]
    @test length(unique(C.bar_methods())) == 16
    for case in fixture["bars"]
        rows = case["rows"]
        data = Matrix{Any}(undef, length(rows), 3)
        for (index, row) in enumerate(rows)
            data[index, :] = [DateTime(1970)+Millisecond(Int(row[1])), row[2], row[3]]
        end
        result = C.process_bars(C.BarController(), case["method"], case["arguments"], data; batch_size = case["batch"])
        @test result.completed_input
        @test result.processed_rows == length(rows)
        @test result.error === nothing
        @test names(result.bars) == case["columns"]
        @test nrow(result.bars) == length(case["output"])
        for (actual, expected) in zip(eachrow(result.bars), case["output"])
            @test Dates.value(actual[1]-DateTime(1970)) == expected[1]
            @test collect(Float64, actual[2:end]) ≈ Vector{Float64}(expected[2:end]) rtol = 1e-10 atol = 2e-10 nans = true
        end
    end
    failed = C.process_bars(C.BarController(), "unknown", Dict(), zeros(3,3))
    @test !failed.completed_input && failed.processed_rows == 0
    @test failed.error !== nothing && nrow(failed.bars) == 0
    @test names(failed.bars) == C.BAR_COLUMNS
    empty = C.process_bars(C.BarController(), "tick_standard_bars", Dict(), Matrix{Any}(undef,0,3))
    @test empty.completed_input && empty.processed_rows == 0 && nrow(empty.bars) == 0
    @test_throws ArgumentError C.process_bars(C.BarController(), "tick_standard_bars", Dict(), zeros(3,3); batch_size = 0)
end
