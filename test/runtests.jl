# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

using ElasticArrays
using Base.Test

@Base.Test.testset "Package ElasticArrays" begin
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
