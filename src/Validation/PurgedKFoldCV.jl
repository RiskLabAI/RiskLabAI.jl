using TimeSeries
using DataFrames
using Statistics
using MLJ
using MLDataUtils

"""
Function to purge test observations in the training set.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 106, snippet 7.1
"""
function purged_train_times(data::TimeArray, test::TimeArray)::TimeArray
    train_times = deepcopy(data)

    for (start_time, end_time) in zip(timestamp(test), values(test)[:, 1])
        start_within_test_times = timestamp(train_times)[(timestamp(train_times) .>= start_time) .* (timestamp(train_times) .<= end_time)]
        end_within_test_times = timestamp(train_times)[(values(train_times)[:, 1] .>= start_time) .* (values(train_times)[:, 1] .<= end_time)]
        envelope_test_times = timestamp(train_times)[(timestamp(train_times) .<= start_time) .* (values(train_times)[:, 1] .>= end_time)]
        filtered_times = setdiff(timestamp(train_times), union(start_within_test_times, end_within_test_times, envelope_test_times))
        train_times = train_times[filtered_times]
    end

    return train_times
end

"""
Function to get embargo time for each bar.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 108, snippet 7.2
"""
function embargo_times(times::Array, percent_embargo::Float64)::TimeArray
    step = round(Int, length(times) * percent_embargo)

    if step == 0
        embargo = TimeArray((Times = times, timestamp = times), timestamp = :timestamp)
    else
        embargo = TimeArray((Times = times[step + 1:end], timestamp = times[1:end - step]), timestamp = :timestamp)
        tail_times = TimeArray((Times = repeat([times[end]], step), timestamp = times[end - step + 1:end]), timestamp = :timestamp)
        embargo = [embargo; tail_times]
    end

    return embargo
end

"""
Custom struct for cross validation with purging of overlapping observations.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 109, snippet 7.3
"""
mutable struct PurgedKFold
    n_splits::Int64
    times::TimeArray
    percent_embargo::Float64

    function PurgedKFold(n_splits::Int = 3, times::TimeArray = nothing, percent_embargo::Float64 = 0.)
        times isa TimeArray ? new(n_splits, times, percent_embargo) : error("The times parameter should be a TimeArray.")
    end
end

"""
Function to split data when observations overlap.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 109, snippet 7.3
"""
function purged_kfold_split(self::PurgedKFold, data::TimeArray)
    if timestamp(data) != timestamp(self.times)
        error("data and ThruDateValues must have the same index.")
    end

    indices = collect(1:length(data))
    embargo = round(Int64, length(data) * self.percent_embargo)
    final_test = []
    final_train = []

    test_ranges = [(i[1], i[end]) for i in [kfolds(collect(1:length(self.times)), k = self.n_splits)[i][2] for i in collect(1:self.n_splits)]]

    for (start_index, end_index) in test_ranges
        first_test_index = timestamp(self.times)[start_index]
        test_indices = indices[start_index:end_index]
        max_test_index = searchsortedfirst(timestamp(self.times), maximum(values(self.times)[test_indices, 1]))
        search_times(y) = findfirst(x -> x == y, timestamp(self.times))
        train_indices = search_times.(timestamp(self.times)[values(self.times)[:, 1] .<= first_test_index])

        if max_test_index + embargo <= length(values(data))
            append!(train_indices, indices[max_test_index + embargo:end])
        end

        append!(final_test, [Int64.(test_indices)])
        append!(final_train, [Int64.(train_indices)])
    end

    return Tuple(zip(final_train, final_test))
end

"""
Function to calculate cross-validation scores with purging.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 110, snippet 7.4
"""
function cross_validation_score(classifier, data::TimeArray, labels::TimeArray, sample_weights::Array, scoring::String = "Log Loss", times::TimeArray = nothing, cross_validation_generator::PurgedKFold = nothing, n_splits::Int = nothing, percent_embargo::Float64 = 0.0)
    if scoring ∉ ["Log Loss", "Accuracy"]
        error("Wrong scoring method.")
    end

    if isnothing(times)
        times = TimeArray(DataFrame(startTime = timestamp(data), endTime = append!(timestamp(data)[2:end], [timestamp(data)[end] + (timestamp(data)[2] - timestamp(data)[1])])), timestamp = :startTime)
    end

    if isnothing(cross_validation_generator)
        cross_validation_generator = PurgedKFold(n_splits, times, percent_embargo)
    end

    name = colnames(labels)[1]
    data, labels = DataFrame(data), DataFrame(labels)
    select!(data, Not(:timestamp))
    select!(labels, Not(:timestamp))

    sample = hcat(data, labels)
    for column in names(sample)
        if eltype(sample[!, column]) == Float64 || eltype(sample[!, column]) == Int
            sample = coerce(sample, Symbol(column) => Continuous)
        else
            sample = coerce(sample, Symbol(column) => Multiclass)
        end
    end
    labels, data = unpack(sample, ==(name), colname -> true)

    machine = MLJ.MLJBase.machine(classifier, data, labels)
    scores = []

    for (train, test) in purged_kfold_split(cross_validation_generator, times)
        fit!(machine, rows = train)

        if scoring == "Log Loss"
            predictions = MLJ.MLJBase.predict(machine, data[test, :])
            score = log_loss(predictions, labels[test]) |> mean
        else
            predictions = MLJ.MLJBase.predict_mode(machine, data[test, :])
            score = accuracy(predictions, labels[test], sample_weights[test]) |> mean
        end

        append!(scores, [score])
    end

    return scores
end
