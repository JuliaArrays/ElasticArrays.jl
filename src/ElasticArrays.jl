# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ElasticArrays

using Base: IteratorSize, HasLength, HasShape

include("util.jl")
include("elasticarray.jl")

end # module
