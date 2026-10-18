function _differencing_wavelet_filters(wavelet)
    wavelet isa AbstractString || throw(ArgumentError("Supply a supported wavelet name"))
    if wavelet in ("haar", "db1")
        scale = inv(sqrt(2.0))
        return [scale, scale], [-scale, scale]
    end
    wavelet == "db2" || throw(ArgumentError("Native wavelet Hurst supports db2, haar and db1"))
    root, denominator = sqrt(3.0), 4 * sqrt(2.0)
    low = [1 - root, 3 - root, 3 + root, 1 + root] ./ denominator
    return low, reverse(low) .* [-1.0, 1.0, -1.0, 1.0]
end

function _differencing_symmetric_index(position, count)
    folded = mod(position, 2count)
    return folded < count ? folded + 1 : 2count - folded
end

function _differencing_wavelet_step(values, low, high)
    count = fld(length(values) + length(low) - 1, 2)
    approximation, detail = zeros(count), zeros(count)
    for index in 0:count-1
        for offset in 0:length(low)-1
            value = values[_differencing_symmetric_index(2index + 1 - offset, length(values))]
            approximation[index+1] += low[offset+1] * value
            detail[index+1] += high[offset+1] * value
        end
    end
    return approximation, detail
end

function _differencing_wavelet_details(series, wavelet)
    series isa AbstractVector{<:Real} && !isempty(series) || throw(ArgumentError("Wavelet Hurst requires a nonempty real vector"))
    low, high = _differencing_wavelet_filters(wavelet)
    count = length(series)
    maximum_level = max(0, floor(Int, log2(count / (length(low) - 1))))
    level = max(3, min(maximum_level, floor(Int, log2(count)) - 2))
    values = Float64.(series) .- mean(series)
    details = Vector{Float64}[]
    for _ in 1:level
        values, detail = _differencing_wavelet_step(values, low, high)
        push!(details, detail)
    end
    return reverse(details)
end

function _differencing_wavelet_hurst(series, wavelet, skip_fine)
    details = _differencing_wavelet_details(series, wavelet)
    octaves = collect(length(details):-1:1)
    keep = octaves .>= 1 + skip_fine
    count(keep) >= 2 || return NaN
    powers = [mean(detail.^2) for detail in details][keep]
    adjusted = ifelse.(powers .<= 0, 1e-12, powers)
    return (_polyfit_slope(Float64.(octaves[keep]), log2.(adjusted)) + 1) / 2
end
