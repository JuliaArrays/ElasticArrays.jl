# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using ElasticArrays
using Base.Test

@testset "elasticarray" begin
    dims = (3, 2)
    N = 10
    a = ElasticArray{Float64}(dims..., 0)
    mats = Array{Float64,length(dims)}[]
    for i in 1:N
        mat = rand(dims...)
        append!(a, mat)
        push!(mats, mat)
    end
    @test size(a) == (dims..., N)
    for i in 1:N
        @test @view(a[:,:,i]) == mats[i]
    end
end
