module VectorFunctions

export differences, vectorToMatrix

function differences(vector::Vector)::Vector
    result = [missing, (vector |> diff)...]
    result
end

function vectorToMatrix(vector::Vector)::Matrix
    reshape(vector, length(vector), 1)
end

end