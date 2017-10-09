# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

using Base: @propagate_inbounds
using Base.MultiplicativeInverses: SignedMultiplicativeInverse


"""
    ElasticArray{T,N,M} <: DenseArray{T,N}

An `ElasticArray` can grow in its last dimension. `N` is the total
number of dimensions, `N == M + 1` the number of non-resizable dimensions.

Constructors:

    ElasticArray{T}(dims::Integer...)
"""
struct ElasticArray{T,N,M} <: DenseArray{T,N}
    kernel_size::NTuple{M,Int}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::Vector{T}

    function ElasticArray{T}(dims::Integer...) where {T}
        kernel_size, size_lastdim = _split_dims(dims)
        kernel_length = prod(kernel_size)
        data = Vector{T}(kernel_length * size_lastdim)
        new{T,length(dims),length(kernel_size)}(
            kernel_size,
            SignedMultiplicativeInverse{Int}(kernel_length),
            data
        )
    end
end

export ElasticArray


function _split_resize_dims(A::ElasticArray, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticArray"))
    kernel_size, size_lastdim
end


import Base.==
(==)(A::ElasticArray, B::ElasticArray) =
    ndims(A) == ndims(B) && A.kernel_size == B.kernel_size && A.data == B.data


Base.parent(A::ElasticArray) = A.data

Base.size(A::ElasticArray) = (A.kernel_size..., div(length(linearindices(A.data)), A.kernel_length))
@propagate_inbounds Base.getindex(A::ElasticArray, i::Integer) = getindex(A.data, i)
@propagate_inbounds Base.setindex!(A::ElasticArray, x, i::Integer) = setindex!(A.data, x, i)
@inline Base.IndexStyle(A::ElasticArray) = IndexStyle(A.data)

Base.length(A::ElasticArray) = length(A.data)
Base.linearindices(A::ElasticArray) = linearindices(A.data)

Base.repremptyarray(io::IO, X::ElasticArray{T}) where {T} = print(io, "ElasticArray{$T}(", join(size(X),','), ')')

function Base.resize!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


function Base.sizehint!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


function Base.append!(dest::ElasticArray, src::AbstractArray)
    rem(length(linearindices(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    append!(dest.data, src)
    dest
end


function Base.prepend!(dest::ElasticArray, src::AbstractArray)
    rem(length(linearindices(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't prepend, length of source array is incompatible"))
    prepend!(dest.data, src)
    dest
end


function _copy_impl!(dest::ElasticArray, args...)
    copy!(dest.data, args...)
    dest
end

Base.copy!(dest::ElasticArray, doffs::Integer, src::AbstractArray, args::Integer...) = _copy_impl!(dest, doffs, src, args...)
Base.copy!(dest::ElasticArray, src::AbstractArray) = _copy_impl!(dest, src)

Base.copy!(dest::ElasticArray, doffs::Integer, src::ElasticArray, args::Integer...) = _copy_impl!(dest, doffs, src, args...)
Base.copy!(dest::ElasticArray, src::ElasticArray) = _copy_impl!(dest, src)

Base.copy!(dest::AbstractArray, doffs::Integer, src::ElasticArray, args::Integer...) = copy!(dest, doffs, src.data, args...)
Base.copy!(dest::AbstractArray, src::ElasticArray) = copy!(dest, src.data)

Base.convert(::Type{ElasticArray{T}}, A::AbstractArray) where {T} =
    copy!(ElasticArray{T}(size(A)...), A)

Base.convert(::Type{ElasticArray}, A::AbstractArray{T}) where {T} =
    convert(ElasticArray{T}, A)
