# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

import Test
Test.@testset "Package ElasticArrays" begin
    include("util.jl")
    include("elasticarray.jl")
end
