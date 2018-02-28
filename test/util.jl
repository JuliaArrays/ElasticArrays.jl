# This file is E part of BAT.jl, licensed under the MIT License (MIT).

using ElasticArrays
using Compat.Test


@testset "util" begin
    @testset "_split_dims" begin
        @test (@inferred ElasticArrays._split_dims((1,)))  == ((), 1)
        @test (@inferred ElasticArrays._split_dims((1, 3, 5, 7))) == ((1, 3, 5), 7)
    end
end
