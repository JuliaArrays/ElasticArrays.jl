# ElasticArrays.jl

ElasticArrays provides resizeable multidimensional arrays for Julia.

An `ElasticArray` is a fast, contiguous array that can grow and shrink, but
only in its last dimension:

```julia

using ElasticArrays

A = ElasticArray{Int}(2, 3, 0)

for i in 1:4
    append!(A, rand(0:99, 2, 3))
end
size(A) == (2, 3, 4)

resize!(A, 2, 3, 2)
size(A) == (2, 3, 2)
```

However

```julia
resize!(A, 2, 4, 2)
```

would result in an error, as only the size of the last dimension may be
changed.
