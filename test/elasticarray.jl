# This file is E part of BAT.jl, licensed under the MIT License (MIT).

using ElasticArrays
using Compat.Test


@testset "elasticarray" begin
    test_dims = (3, 2, 4)
    test_kernel_dims = Base.front(test_dims)

    function test_A(test_code)
        A = rand!(Array{Int}(test_dims...), 0:99)
        test_code(A)
    end

    function test_E(test_code)
        E = rand!(ElasticArray{Int}(test_dims...), 0:99)
        test_code(E)
    end

    function test_E_A(test_code)
        E = rand!(ElasticArray{Int}(test_dims...), 0:99)
        A = rand!(Array{Int}(test_dims...), 0:99)
        test_code(E, A)
    end

    function test_E1_E2(test_code)
        E1 = rand!(ElasticArray{Int}(test_dims...), 0:99)
        E2 = rand!(ElasticArray{Int}(test_dims...), 0:99)
        test_code(E1, E2)
    end

    function test_E_V(test_code)
        E = ElasticArray{Float64}(test_kernel_dims..., 0)
        V = Vector{Array{Float64,length(test_kernel_dims)}}()
        test_code(E, V)
    end

    Base.@pure filltuple(x, n::Integer) = ((x for i in 1:n)...)

    lastdim_slice_idxs(A::AbstractArray{T,N}, i::Integer) where {T,N} = (filltuple(:, N - 1)..., i)

    test_comp(E::ElasticArray, V::Vector{<:Array}) =
        all(i -> @view(E[lastdim_slice_idxs(E, i)...]) == V[i], eachindex(V))


    @testset "size, length and index style" begin
        @test @inferred size(@inferred ElasticArray{Int}(4)) == (4,)
        @test @inferred size(@inferred ElasticArray{Int}(2,3,4)) == (2,3,4)

        test_E() do E
            @test length(E) == prod(size(E))
            @test IndexStyle(E) == IndexLinear()
            @test linearindices(E) == linearindices(parent(E))
            @test eachindex(E) == eachindex(parent(E))
        end
    end


    @testset "getindex and setindex!" begin
        test_E_A() do E, A
            for i in eachindex(E, A)
                E[i] = A[i]
            end
            @test parent(E) == A[:]
            @test all(i -> E[i] == A[i], eachindex(E, A))
            @test all(i -> E[i] == A[i], CartesianRange(size(A)))
        end

        @test all(x -> x == 42, @inferred fill!(ElasticArray{Int}(2,3,4), 42))
    end


    @testset "equality" begin
        test_E() do E
            @test E == @inferred deepcopy(E)
        end

        test_E1_E2() do E1, E2
            @test E1 != E2
        end

        @test fill!(ElasticArray{Int}(2, 3, 4), 0) == fill!(ElasticArray{Int}(2, 3, 4), 0)
        @test fill!(ElasticArray{Int}(2, 3, 4), 0) != fill!(ElasticArray{Int}(2, 4, 3), 0)
    end


    @testset "copy!, convert and similar" begin
        test_E_A() do E, A
            @test E === @inferred copy!(E, A)
            @test E == A
        end

        test_E_A() do E, A
            A2 = Array(deepcopy(E))
            @test E === @inferred copy!(E, 3, A, 5, 7)
            copy!(A2, 3, A, 5, 7)
            @test E == A2
        end

        test_E_A() do E, A
            @test A === @inferred copy!(A, E)
            @test A == E
        end

        test_E_A() do E, A
            A2 = deepcopy(A)
            @test A === @inferred copy!(A, 3, E, 5, 7)
            copy!(A2, 3, Array(deepcopy(E)), 5, 7)
            @test A == A2
        end

        test_E1_E2() do E1, E2
            @test E1 === @inferred copy!(E1, E2)
            @test E1 == E2
        end

        test_E1_E2() do E1, E2
            A1 = Array(deepcopy(E1))
            A2 = Array(deepcopy(E2))
            @test E1 === @inferred copy!(E1, 3, E2, 5, 7)
            copy!(A1, 3, A2, 5, 7)
            @test E1 == A1
        end

        test_A() do A
            E = @inferred convert(ElasticArray{Float64}, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float64
        end

        test_A() do A
            E = @inferred convert(ElasticArray, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == eltype(A)
        end

        @test typeof(@inferred similar(ElasticArray{Int}, (2,3,4))) == ElasticArray{Int,3,2}
        @test size(@inferred similar(ElasticArray{Int}, (2,3,4))) == (2,3,4)
    end


    @testset "pointer and unsafe_convert" begin
        test_E() do E
            @test pointer(E) == pointer(parent(E))
            @test pointer(E, 4) == pointer(parent(E), 4)
        end

        test_E() do E
            @test Base.unsafe_convert(Ptr{eltype(E)}, E) == Base.unsafe_convert(Ptr{eltype(E)}, parent(E))
        end
    end


    @testset "resize!" begin
        function resize_test(delta::Integer)
            test_E() do E
                A = Array(deepcopy(E))
                new_size = (Base.front(size(E))..., size(E, ndims(E)) + delta)
                cmp_idxs = (Base.front(indices(E))..., 1:(last(size(E)) + min(0, delta)))
                @test E === @inferred sizehint!(E, new_size...)
                @test E === @inferred resize!(E, new_size...)
                @test size(E) == new_size
                @test E[cmp_idxs...] == A[cmp_idxs...]
            end
        end

        resize_test(0)
        resize_test(2)
        resize_test(-2)
    end


    @testset "append! and prepend!" begin
       test_A() do A
            E = @inferred convert(ElasticArray{Float64}, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float64
        end


        test_E_V() do E, V
            dims = Base.front(size(E))
            for i in 1:4
                push!(V, rand(dims...))
                @inferred append!(E, last(V))
            end
            @test size(E) == (dims..., length(V))
            @test test_comp(E, V)
        end

        test_E_V() do E, V
            dims = Base.front(size(E))
            for i in 1:4
                unshift!(V, rand(dims...))
                @inferred prepend!(E, first(V))
            end
            @test size(E) == (dims..., length(V))
            @test test_comp(E, V)
        end
    end
end
