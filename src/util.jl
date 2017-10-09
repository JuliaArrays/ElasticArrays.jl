# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).


function _split_dims(dims::NTuple{N,Integer}) where {N}
    int_dims = Int.(dims)
    Base.front(int_dims), int_dims[end]
end
