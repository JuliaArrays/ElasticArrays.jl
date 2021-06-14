# This file is a part of ElasticArrays.jl, licensed under the MIT License (MIT).

using Adapt
using Base: @propagate_inbounds
using Base.MultiplicativeInverses: SignedMultiplicativeInverse
using Base.Broadcast: Broadcast, ArrayStyle, Broadcasted

export ElasticArray, ElasticVector, ElasticMatrix


"""
    ElasticArray{T,N,M} <: DenseArray{T,N}

An `ElasticArray` can grow/shrink in its last dimension. `N` is the total
number of dimensions, `M == N - 1` the number of non-resizable dimensions.

Constructors:

    ElasticArray{T}(dims::Integer...)
    convert(ElasticArray, A::AbstractArray)
"""
struct ElasticArray{T,N,M,V<:DenseVector{T}} <: DenseArray{T,N}
    kernel_size::Dims{M}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::V
    function ElasticArray{T,N,M,V}(kernel_size, kernel_length, data) where {T,N,M,V}
        if M::Int != N::Int - 1
            throw(ArgumentError("ElasticArray parameter M does not satisfy requirement M == N - 1"))
        end
        if rem(length(eachindex(data)), kernel_length) != 0
            throw(ArgumentError("length(data) must be integer multiple of prod(kernel_size)"))
        end
        new(kernel_size, kernel_length, data)
    end
end

function ElasticArray{T,N,M}(kernel_size, kernel_length, data) where {T,N,M}
    ElasticArray{T,N,M,typeof(data)}(kernel_size, kernel_length, data)
end
function ElasticArray{T,N}(kernel_size, kernel_length, data) where {T,N,M}
    ElasticArray{T,N,N-1}(kernel_size, kernel_length, data)
end

# Need to support ElasticArrays with zero-sized kernel, e.g. for LinearAlgebra.eigvals
_smi_kernel_size(kernel_length::Integer) = SignedMultiplicativeInverse{Int}(kernel_length != 0 ? kernel_length : -1)

function ElasticArray{T,N,M,V}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M,V}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_length = prod(kernel_size)
    kernel_length == 0 && size_lastdim != 0 && throw(ArgumentError("ElasticArray with empty kernel must have 0-sized last dimension"))
    data = similar(V, kernel_length * size_lastdim)
    ElasticArray{T,N,M,V}(kernel_size, _smi_kernel_size(kernel_length), data)
end
ElasticArray{T,N,M}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M} = ElasticArray{T,N,M,Vector{T}}(undef, dims)
ElasticArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} = ElasticArray{T,N,N-1}(undef, dims)
ElasticArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} = ElasticArray{T,N}(undef, dims)

ElasticArray{T,N,M,V}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M,V} = ElasticArray{T,N,M,V}(undef, dims)
ElasticArray{T,N,M}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M,V} = ElasticArray{T,N,M}(undef, dims)
ElasticArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M,V} = ElasticArray{T,N}(undef, dims)
ElasticArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M,V} = ElasticArray{T,N}(undef, dims)

ElasticArray{T,N,M,V}(A::AbstractArray{<:Any,N}) where {T,N,M,V} = copyto!(ElasticArray{T,N,M,V}(undef, size(A)), A)
ElasticArray{T,N,M}(A::AbstractArray{<:Any,N}) where {T,N,M} = copyto!(ElasticArray{T,N,M}(undef, size(A)), A)
ElasticArray{T,N}(A::AbstractArray{<:Any,N}) where {T,N} = copyto!(ElasticArray{T,N}(undef, size(A)), A)
ElasticArray{T}(A::AbstractArray{<:Any,N}) where {T,N} = copyto!(ElasticArray{T,N}(undef, size(A)), A)
ElasticArray(A::AbstractArray{T,N}) where {T,N} = copyto!(ElasticArray{T,N}(undef, size(A)), A)

Base.convert(::Type{T}, A::AbstractArray) where {T<:ElasticArray} = A isa T ? A : T(A)


@inline function Base.similar(A::ElasticArray, ::Type{T}, dims::Dims{N}) where {T,N}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_length = prod(kernel_size)
    data = similar(A.data, T, prod(dims))
    ElasticArray{T,N}(kernel_size, _smi_kernel_size(kernel_length), data)
end

@inline Base.similar(A::ElasticArray, ::Type{T}, dims::Dims{0}) where {T} = similar(A.data, T, dims)


@inline function Base.:(==)(A::ElasticArray, B::ElasticArray)
    return ndims(A) == ndims(B) && A.kernel_size == B.kernel_size && A.data == B.data
end


@inline Base.vec(A::ElasticArray) = A.data

@inline Base.size(A::ElasticArray) = (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))

@inline Base.length(A::ElasticArray) = length(eachindex(A.data))

@inline Base.sizeof(A::ElasticArray) = sizeof(A.data)

@propagate_inbounds Base.getindex(A::ElasticArray, i::Int) = getindex(A.data, i)

@propagate_inbounds Base.setindex!(A::ElasticArray, x, i::Int) = setindex!(A.data, x, i)

Base.IndexStyle(::Type{<:ElasticArray}) = IndexLinear()


Base.elsize(::Type{ElasticArray{T,N,M,V}}) where {T,N,M,V} = Base.elsize(V)


@inline Base.resize!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N} = resize!(A, dims)

@inline function Base.resize!(A::ElasticArray{T,N}, dims::NTuple{N,Integer}) where {T,N}
    _, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end

@inline function resize_lastdim!(A::ElasticArray, size_lastdim::Integer)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end


@inline Base.sizehint!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N} = sizehint!(A, dims)

@inline function Base.sizehint!(A::ElasticArray{T,N}, dims::NTuple{N,Integer}) where {T,N}
    _, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end

@inline function sizehint_lastdim!(A::ElasticArray, size_lastdim::Integer)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end


function _split_resize_dims(A::ElasticArray, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticArray"))
    return kernel_size, size_lastdim
end


Base.append!(A::ElasticArray, iter) = _append!(A, IteratorSize(iter), iter)

function _append!(A::ElasticArray, ::Union{HasLength,HasShape}, iter)
    _check_size(A, iter)
    append!(A.data, iter)
    return A
end

function _append!(A::ElasticArray, ::IteratorSize, iter)
    for item in iter
        append!(A, item)
    end
    return A
end


Base.prepend!(A::ElasticArray, iter) = _prepend!(A, IteratorSize(iter), iter)

function _prepend!(A::ElasticArray, ::Union{HasLength,HasShape}, iter)
    _check_size(A, iter)
    prepend!(A.data, iter)
    return A
end

function _prepend!(A::ElasticArray, ::IteratorSize, iter)
    for item in iter
        prepend!(A, item)
    end
    return A
end


@inline function _check_size(A::ElasticArray, iter)
    n = length(iter)
    if rem(n, A.kernel_length) != 0 || A.kernel_length.divisor <= 0 && n > 0
        throw(DimensionMismatch("Length of source array is incompatible"))
    end
    return nothing
end


@inline function Base.copyto!(
    dest::ElasticArray,
    doffs::Integer,
    src::AbstractArray,
    soffs::Integer,
    N::Integer,
)
    copyto!(dest.data, doffs, src, soffs, N)
    return dest
end
@inline function Base.copyto!(
    dest::AbstractArray,
    doffs::Integer,
    src::ElasticArray,
    soffs::Integer,
    N::Integer,
)
    copyto!(dest, doffs, src.data, soffs, N)
end

@inline Base.copyto!(dest::ElasticArray, src::AbstractArray) = (copyto!(dest.data, src); dest)
@inline Base.copyto!(dest::AbstractArray, src::ElasticArray) = copyto!(dest, src.data)

@inline function Base.copyto!(
    dest::ElasticArray,
    doffs::Integer,
    src::ElasticArray,
    soffs::Integer,
    N::Integer,
)
    copyto!(dest.data, doffs, src.data, soffs, N)
    return dest
end
@inline function Base.copyto!(dest::ElasticArray, src::ElasticArray)
    copyto!(dest.data, src.data)
    return dest
end


@inline Base.dataids(A::ElasticArray) = Base.dataids(A.data)

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticArray{T}) where T = Base.unsafe_convert(Ptr{T}, A.data)

@inline Base.pointer(A::ElasticArray, i::Integer) = pointer(A.data, i)


Broadcast.BroadcastStyle(::Type{<:ElasticArray}) = ArrayStyle{ElasticArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{ElasticArray}}, ::Type{T}) where {T}
    similar(ElasticArray{T}, axes(bc))
end


function Adapt.adapt_structure(to, A::ElasticArray{<:Any,N,M}) where {N,M}
    data = adapt(to, A.data)
    ElasticArray{eltype(data),N,M,typeof(data)}(A.kernel_size, A.kernel_length, data)
end


"""
    const ElasticVector{T,V<:DenseVector{T}} = ElasticArray{T,1,0,V}

Type alias for 1D `ElasticArray`.
"""
const ElasticVector{T,V<:DenseVector{T}} = ElasticArray{T,1,0,V}

"""
    ElasticVector(A::AbstractVector{T}) where {T}

Construct an `ElasticVector` from an `AbstractVector`.
"""
ElasticVector(A::AbstractVector{T}) where {T} = ElasticArray(A)


"""
    const ElasticMatrix{T,V<:DenseVector{T}} = ElasticArray{T,2,1,V}

Type alias for 2D `ElasticArray`.
"""
const ElasticMatrix{T,V<:DenseVector{T}} = ElasticArray{T,2,1,V}

"""
    ElasticMatrix(A::AbstractMatrix{T}) where {T}

Construct an `ElasticMatrix` from an `AbstractMatrix`.
"""
ElasticMatrix(A::AbstractMatrix{T}) where {T} = ElasticArray(A)
