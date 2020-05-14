# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

using ElasticArrays
using Test

using Random

@testset "elasticarray" begin
    test_dims = (3, 2, 4)
    test_kernel_dims = Base.front(test_dims)

    function test_A(test_code)
        A = rand!(Array{Int}(undef, test_dims...), 0:99)
        test_code(A)
    end

    function test_E(test_code)
        E = rand!(ElasticArray{Int}(undef, test_dims...), 0:99)
        test_code(E)
    end

    function test_E_A(test_code)
        E = rand!(ElasticArray{Int}(undef, test_dims...), 0:99)
        A = rand!(Array{Int}(undef, test_dims...), 0:99)
        test_code(E, A)
    end

    function test_E1_E2(test_code)
        E1 = rand!(ElasticArray{Int}(undef, test_dims...), 0:99)
        E2 = rand!(ElasticArray{Int}(undef, test_dims...), 0:99)
        test_code(E1, E2)
    end

    function test_E_V(test_code)
        E = ElasticArray{Float64}(undef, test_kernel_dims..., 0)
        V = Vector{Array{Float64,length(test_kernel_dims)}}()
        test_code(E, V)
    end

    lastdim_slice_idxs(A::AbstractArray{T,N}, i::Integer) where {T,N} = (ntuple(_ -> :, Val(N - 1))..., i)

    test_comp(E::ElasticArray, V::Vector{<:Array}) =
        all(i -> @view(E[lastdim_slice_idxs(E, i)...]) == V[i], eachindex(V))

    @testset "ctors" begin
        @test (@inferred ElasticArray{Int}(undef, 2, 3, 4)).kernel_size == (2, 3)
        @test ElasticArray{Int}(undef, 2, 3, 4).kernel_length == Base.MultiplicativeInverses.SignedMultiplicativeInverse(2 * 3)
        @test size(ElasticArray{Int}(undef, 2, 3, 4).data) == (2 * 3 * 4,)

        @test fill!((@inferred ElasticArray{Int}(undef, 2, 3, 4)), 42) == fill!(ElasticArray{Int}(undef, 2, 3, 4), 42)
    end

    @testset "size, length and index style" begin
        @test (4,) == @inferred size(@inferred ElasticArray{Int}(undef, 4))
        @test (2,3,4) == @inferred size(@inferred ElasticArray{Int}(undef, 2,3,4))

        test_E() do E
            @test length(E) == prod(size(E))
            @test IndexStyle(E) == IndexLinear()
            @test eachindex(E) == eachindex(parent(E))
            @test sizeof(E) == sizeof(E.data)
        end
    end


    @testset "getindex and setindex!" begin
        test_E_A() do E, A
            for i in eachindex(E, A)
                E[i] = A[i]
            end
            @test parent(E) == A
            @test all(i -> E[i] == A[i], eachindex(E, A))
            @test all(i -> E[i] == A[i], CartesianIndices(size(A)))
        end

        @test all(x -> x == 42, @inferred fill!(ElasticArray{Int}(undef, 2,3,4), 42))
    end


    @testset "equality" begin
        test_E() do E
            @test E == @inferred deepcopy(E)
        end

        test_E1_E2() do E1, E2
            @test E1 != E2
        end

        @test fill!(ElasticArray{Int}(undef, 2, 3, 4), 0) == fill!(ElasticArray{Int}(undef, 2, 3, 4), 0)
        @test fill!(ElasticArray{Int}(undef, 2, 3, 4), 0) != fill!(ElasticArray{Int}(undef, 2, 4, 3), 0)
    end


    @testset "mightalias and dataids" begin
        E1 = ElasticArray{Int}(undef, 10, 5)
        E2 = ElasticArray{Int}(undef, 10, 5)
        @test Base.dataids(parent(E1)) == @inferred Base.dataids(E1)
        @test @inferred !Base.mightalias(E1, E2)
        @test @inferred !Base.mightalias(view(E1, 2:3, 1:2), view(E1, 4:5, 1:2))
        @test @inferred Base.mightalias(view(E1, 2:4, 1:2), view(E1, 3:5, 1:2))
        @test @inferred !Base.mightalias(view(E1, 2:4, 1:2), view(E2, 3:5, 1:2))
    end


    @testset "copyto!, conversion ctor, convert and similar" begin
        test_E_A() do E, A
            @test E === @inferred copyto!(E, A)
            @test E == A
        end

        test_E_A() do E, A
            A2 = Array(deepcopy(E))
            @test E === @inferred copyto!(E, 3, A, 5, 7)
            copyto!(A2, 3, A, 5, 7)
            @test E == A2
        end

        test_E_A() do E, A
            @test A === @inferred copyto!(A, E)
            @test A == E
        end

        test_E_A() do E, A
            A2 = deepcopy(A)
            @test A === @inferred copyto!(A, 3, E, 5, 7)
            copyto!(A2, 3, Array(deepcopy(E)), 5, 7)
            @test A == A2
        end

        test_E1_E2() do E1, E2
            @test E1 === @inferred copyto!(E1, E2)
            @test E1 == E2
        end

        test_E1_E2() do E1, E2
            A1 = Array(deepcopy(E1))
            A2 = Array(deepcopy(E2))
            @test E1 === @inferred copyto!(E1, 3, E2, 5, 7)
            copyto!(A1, 3, A2, 5, 7)
            @test E1 == A1
        end

        test_A() do A
            E = @inferred ElasticArray{Float64}(A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float64
        end

        test_A() do A
            E = @inferred ElasticArray{Float32}(A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float32
        end

        test_A() do A
            E = @inferred ElasticArray{Float64,3}(A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float64
        end

        test_A() do A
            E = @inferred ElasticArray{Float32,3}(A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float32
        end

        test_A() do A
            E = @inferred ElasticArray(A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == eltype(A)
        end

        test_A() do A
            E = @inferred convert(ElasticArray{Float64,3}, A)
            @test E isa ElasticArray{Float64,3,2}
            @test E == A
            @test eltype(E) == Float64

            @test convert(ElasticArray{Float64,3,2}, E) === E
            @test convert(ElasticArray{Float64,3}, E) === E
            @test convert(ElasticArray{Float64}, E) === E
            @test convert(ElasticArray, E) === E

            @test E == @inferred convert(ElasticArray{Float64,3,2}, A)
            @test_throws ArgumentError convert(ElasticArray{Float64,3,1}, A)
        end

        test_A() do A
            E = @inferred convert(ElasticArray{Float32,3}, A)
            @test E isa ElasticArray{Float32,3,2}
            @test E == A
            @test eltype(E) == Float32

            @test convert(ElasticArray{Float32,3,2}, E) === E
            @test convert(ElasticArray{Float32,3}, E) === E
            @test convert(ElasticArray{Float32}, E) === E
            @test convert(ElasticArray, E) === E

            @test E == @inferred convert(ElasticArray{Float32,3,2}, A)
            @test_throws ArgumentError convert(ElasticArray{Float32,3,4}, A)
        end

        test_A() do A
            E = @inferred convert(ElasticArray{Float64}, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float64
        end

        test_A() do A
            E = @inferred convert(ElasticArray{Float32}, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == Float32
        end

        test_A() do A
            E = @inferred convert(ElasticArray, A)
            @test E isa ElasticArray
            @test E == A
            @test eltype(E) == eltype(A)
        end

        @test typeof(@inferred similar(ElasticArray{Int}, (2,3,4))) == ElasticArray{Int,3,2,Vector{Int}}
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
                cmp_idxs = (Base.front(axes(E))..., 1:(last(size(E)) + min(0, delta)))
                @test E === @inferred sizehint!(E, new_size...)
                @test E === @inferred resize!(E, new_size...)
                @test size(E) == new_size
                @test E[cmp_idxs...] == A[cmp_idxs...]
            end
        end

        resize_test(0)
        resize_test(2)
        resize_test(-2)


        function resize_lastdim_test(delta::Integer)
            test_E() do E
                A = Array(deepcopy(E))
                new_size = (Base.front(size(E))..., size(E, ndims(E)) + delta)
                cmp_idxs = (Base.front(axes(E))..., 1:(last(size(E)) + min(0, delta)))
                @test E === @inferred ElasticArrays.sizehint_lastdim!(E, size(E, ndims(E)) + delta)
                @test E === @inferred ElasticArrays.resize_lastdim!(E, size(E, ndims(E)) + delta)
                @test size(E) == new_size
                @test E[cmp_idxs...] == A[cmp_idxs...]
            end
        end

        resize_lastdim_test(0)
        resize_lastdim_test(2)
        resize_lastdim_test(-2)
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
                pushfirst!(V, rand(dims...))
                @inferred prepend!(E, first(V))
            end
            @test size(E) == (dims..., length(V))
            @test test_comp(E, V)
        end

        test_E() do E
            kernel_size = Base.front(size(E))
            kernel_length = last(size(E))
            V = rand(Int, prod(kernel_size))
            @inferred append!(E, (el for el in V))
            @test size(E) == (kernel_size..., kernel_length + 1)
            @test vec(E[:, :, kernel_length + 1]) == V
        end

        test_E() do E
            kernel_size = Base.front(size(E))
            kernel_length = last(size(E))
            V = rand(Int, prod(kernel_size))
            @inferred prepend!(E, (el for el in V))
            @test size(E) == (kernel_size..., kernel_length + 1)
            @test vec(E[:, :, 1]) == V
        end
    end


    @testset "basic math" begin
        T = Float64
        E1 = rand!(ElasticArray{T}(undef, 9, 9))
        E2 = rand!(ElasticArray{T}(undef, 9, 9))
        E3 = rand!(ElasticArray{T}(undef, 9, 7))

        A1 = Array(E1)
        A2 = Array(E2)
        A3 = Array(E3)

        @test @inferred(2 * E1) isa ElasticArray{T,2}
        @test 2 * E1 == 2 * A1

        @test @inferred(E1 .+ 2) isa ElasticArray{T,2}
        @test E1 .+ 2 == A1 .+ 2

        @test @inferred(E1 + E2) isa ElasticArray{T,2}
        @test E1 + E2 == A1 + A2

        @test @inferred(E1 * E2) isa ElasticArray{T,2}
        @test E1 * E2 == A1 * A2
        @test E1 * E3 == A1 * A3

        @test E1^3 == A1^3
        @test inv(E1) == inv(A1)
        @test inv(E1) isa ElasticArray{T,2}
    end
end
