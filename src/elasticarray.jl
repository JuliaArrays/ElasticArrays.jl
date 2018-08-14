# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

using Base: @propagate_inbounds
using Base.MultiplicativeInverses: SignedMultiplicativeInverse


"""
    ElasticArray{T,N,M} <: DenseArray{T,N}

An `ElasticArray` can grow/shrink in its last dimension. `N` is the total
number of dimensions, `M == N - 1` the number of non-resizable dimensions.

Constructors:

    ElasticArray{T}(dims::Integer...)
    convert(ElasticArray, A::AbstractArray)
"""
struct ElasticArray{T,N,M} <: DenseArray{T,N}
    kernel_size::NTuple{M,Int}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::Vector{T}

    function ElasticArray{T}(::UndefInitializer, dims::Integer...) where {T}
        kernel_size, size_lastdim = _split_dims(dims)
        kernel_length = prod(kernel_size)
        data = Vector{T}(undef, kernel_length * size_lastdim)
        new{T,length(dims),length(kernel_size)}(
            kernel_size,
            SignedMultiplicativeInverse{Int}(kernel_length),
            data
        )
    end
end

export ElasticArray


ElasticArray{T,N}(A::AbstractArray{U,N}) where {T,N,U} = copyto!(ElasticArray{T}(undef, size(A)...), A)
ElasticArray{T}(A::AbstractArray{U,N}) where {T,N,U} = ElasticArray{T,N}(A)
ElasticArray(A::AbstractArray{T,N}) where {T,N} = ElasticArray{T,N}(A)

Base.convert(::Type{ElasticArray{T,N}}, A::AbstractArray) where {T,N} = ElasticArray{T,N}(A)
Base.convert(::Type{ElasticArray{T}}, A::AbstractArray) where {T} = ElasticArray{T}(A)
Base.convert(::Type{ElasticArray}, A::AbstractArray) = ElasticArray(A)


@static if VERSION < v"0.7.0-DEV.2552"
    @inline ElasticArray{T}(dims::Integer...) where {T} = ElasticArray{T}(undef, dims...)
elseif VERSION < v"1.0.0-"
    Base.@deprecate(ElasticArray{T}(dims::Integer...) where {T}, ElasticArray{T}(undef, dims...))
end



function _split_resize_dims(A::ElasticArray, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticArray"))
    kernel_size, size_lastdim
end


import Base.==
(==)(A::ElasticArray, B::ElasticArray) =
    ndims(A) == ndims(B) && A.kernel_size == B.kernel_size && A.data == B.data


Base.parent(A::ElasticArray) = A.data

Base.size(A::ElasticArray) = (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))
@propagate_inbounds Base.getindex(A::ElasticArray, i::Integer) = getindex(A.data, i)
@propagate_inbounds Base.setindex!(A::ElasticArray, x, i::Integer) = setindex!(A.data, x, i)
@inline Base.IndexStyle(A::ElasticArray) = IndexStyle(A.data)

Base.length(A::ElasticArray) = length(A.data)

@static if VERSION < v"0.7.0-beta.250"
    Base._length(A::ElasticArray) = Base._length(A.data)
end

@static if VERSION < v"0.7.0-DEV.2791"
    Base.repremptyarray(io::IO, X::ElasticArray{T}) where {T} = print(io, "ElasticArray{$T}(", join(size(X),','), ')')
end

@static if VERSION >= v"0.7.0-DEV.4404"
    Base.dataids(A::ElasticArray) = Base.dataids(A.data)
end

@inline function Base.resize!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


@inline function Base.sizehint!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


function Base.append!(dest::ElasticArray, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    append!(dest.data, src)
    dest
end


function Base.prepend!(dest::ElasticArray, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't prepend, length of source array is incompatible"))
    prepend!(dest.data, src)
    dest
end


@inline function _copyto_impl!(dest::ElasticArray, args...)
    copyto!(dest.data, args...)
    dest
end

@inline Compat.copyto!(dest::ElasticArray, doffs::Integer, src::AbstractArray, args::Integer...) = _copyto_impl!(dest, doffs, src, args...)
@inline Compat.copyto!(dest::ElasticArray, src::AbstractArray) = _copyto_impl!(dest, src)

@inline Compat.copyto!(dest::ElasticArray, doffs::Integer, src::ElasticArray, args::Integer...) = _copyto_impl!(dest, doffs, src, args...)
@inline Compat.copyto!(dest::ElasticArray, src::ElasticArray) = _copyto_impl!(dest, src)

@inline Compat.copyto!(dest::AbstractArray, doffs::Integer, src::ElasticArray, args::Integer...) = copyto!(dest, doffs, src.data, args...)
@inline Compat.copyto!(dest::AbstractArray, src::ElasticArray) = copyto!(dest, src.data)


Base.similar(::Type{ElasticArray{T}}, dims::Dims{N}) where {T,N} = ElasticArray{T}(undef, dims...)


Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticArray{T}) where T = Base.unsafe_convert(Ptr{T}, A.data)

Base.pointer(A::ElasticArray, i::Integer) = pointer(A.data, i)
