mutable struct PurgedKFold
    n_splits::Int64
    sample_info_sets::Union{TimeArray, Missing}
    pct_embargo::Float64
end 

function PurgedKFold(;samples_info_sets::Union{TimeArray, Missing} = missing)
    if !(samples_info_sets isa TimeArray)
        throw(ArgumentError("The samples_info_sets param must be a TimeArray"))
    else
        PurgedKFold(3, samples_info_sets, 0.)
    end
end

function PurgedKFold(samples_info_sets::TimeArray)
    PurgedKFold(;samples_info_sets = samples_info_sets)
end

function split(self::PurgedKFold, X::TimeArray; y::Union{TimeArray, Missing} = missing, groups=missing)

    if length(X) != length(self.samples_info_sets)
        throw(ArgumentError("X and the 'samples_info_sets' series param must be the same length"))
    end

    indices = collect(1:length(X))
    embargo = Int(floor(length(X) * self.pct_embargo))

    test_ranges = [(ix[1], ix[end]) for ix ∈ array_split(collect(1:length(X)), self.n_splits)]
    out_vals = []
    for (start_ix, end_ix) ∈ test_ranges
        test_indices = indices[start_ix:end_ix]

        if end_ix < length(X)
            end_ix += embargo
        end

        test_times = TimeArray([values(self.sample_info_sets)[start_ix]],
                                [maximum(values(self.sample_info_sets[start_ix:end_ix]))])
        train_times = ml_get_train_times(self.sample_info_sets, test_times)

        train_indices = []
        for train_ix ∈ timestamp(train_times)
           append!(train_indices, timestamp(self.sample_info_sets)[timestamp(self.sample_info_sets) .== train_ix])
        end
        append!(out_vals, [(train_indices, test_indices)])
    end
    return out_vals
end

function ml_cross_val_score(
        classifier,
        X::TimeArray,
        y::TimeArray,
        cv_gen::PurgedKFold,
        scoring::String = "log_loss";
        sample_weight_train::Union{Array, Missing} = missing,
        sample_weight_score::Union{Array, Missing} = missing)

       # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if ismissing(sample_weight_train)
        sample_weight_train = ones(length(X))
    end

    if ismissing(sample_weight_score)
        sample_weight_score = ones(length(X))
    end

    if (scoring != "log_loss") & (scoring != "accuracy")
        throw(ArgumentError("Wrong scoring method."))
    end
    scores = []
    for (train, test) ∈ Kfold(length(X), cv_gen.n_splits)
        fit = classifier(values(X[train]), values(y[train]))
        pred = fit.predict(values(X[test]))
        if scoring == "log_loss"
            score_ = binary_accuracy(pred, values(y[test]))
        else
            score_ = binary_accuracy(pred, values(y[test]))
        end
        append!(scores, score_)
    end
    return scores
end


mutable struct StackedPurgedKFold
    n_splits::Int64
    sample_info_sets::Union{Dict, Missing}
    pct_embargo::Float64
end 

function StackedPurgedKFold(;samples_info_sets::Union{Dict, Missing} = missing)
    if !(samples_info_sets isa TimeArray)
        throw(ArgumentError("The samples_info_sets param must be a TimeArray"))
    else
        StackedPurgedKFold(3, samples_info_sets, 0.)
    end
end

function StackedPurgedKFold(samples_info_sets::Dict)
    StackedPurgedKFold(;samples_info_sets = samples_info_sets)
end

function split(self::StackedPurgedKFold, X::Dict; y::Union{Dict, Missing} = missing, groups=missing)
    first_asset = collect(keys(X))
    for asset ∈ keys(X)
        if length(X[asset]) != length(self.samples_info_sets[asset])
            throw(ArgumentError("X and the 'samples_info_sets' series param must be the same length"))
        end
    end
    test_ranges = Dict()
    for asset ∈ keys(X)
        push!(test_ranges, (asset => [(ix[1], ix[end]) for ix ∈ array_split(collect(1:length(X[asset])), self.n_splits)]))
    end

    for i ∈ 1:length(test_ranges[first_asset])
        train_indices_dict = Dict()  # Dictionary of asset: [train indices]
        test_indices_dict = Dict()
        for (asset, X_asset) ∈ X
            info_sets = self.sample_info_sets[asset]
            indices = collect(1:length(X_asset))
            embargo = Int(floor(length(X_asset) * self.pct_embargo))

            (start_ix, end_ix) = test_ranges[asset][i]
            test_indices = indices[start_ix:end_ix]

            if end_ix < length(X_asset)
                end_ix += embargo
            end

            test_times = TimeArray([values(info_sets)[start_ix]],
                                [maximum(values(info_sets[start_ix:end_ix]))])
            train_times = ml_get_train_times(info_sets, test_times)

            train_indices = []
            for train_ix ∈ timestamp(train_times)
                append!(train_indices, timestamp(info_sets)[timestamp(info_sets) .== train_ix])
            end

            push!(train_indices_dict, (asset => train_indices))
            push!(test_indices_dict, (asset => test_indices))
        end
        append!(out_vals, [(train_indices_dict, test_indices_dict)])
    end 
    return out_vals
end


function stacked_ml_cross_val_score(
    classifier,
    X_dict::Dict,
    y_dict::Dict,
    cv_gen::StackedPurgedKFold,
    scoring::String = "log_loss";
    sample_weight_train::Union{Dict, Missing} = missing,
    sample_weight_score::Union{Dict, Missing} = missing)

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if ismissing(sample_weight_train)
        sample_weight_train = Dict()
        for asset ∈ keys(X_dict)
            push!(sample_weight_train, (asset => ones(length(X_dict[asset]))))
        end
    end

    if ismissing(sample_weight_score)
        sample_weight_score = Dict()
        for asset ∈ keys(X_dict)
            push!(sample_weight_score, (asset => ones(length(X_dict[asset]))))
        end
    end

    if (scoring != "log_loss") & (scoring != "accuracy")
        throw(ArgumentError("Wrong scoring method."))
    end
    scores = Dict()
    for asset ∈ keys(X_dict)
        asset_score = []
        for (train, test) ∈ Kfold(length(X_dict[asset]), cv_gen.n_splits)
            fit = classifier(values(X_dict[asset][train]), values(y_dict[asset][train]))
            pred = fit.predict(values(X_dict[asset][test]))
            if scoring == "log_loss"
                score_ = binary_accuracy(pred, values(y_dict[asset][test]))
            else
                score_ = binary_accuracy(pred, values(y_dict[asset][test]))
            end
            append!(asset_score, score_)
        end
        push!(scores, (asset => asset_score))
    end

    return scores
end