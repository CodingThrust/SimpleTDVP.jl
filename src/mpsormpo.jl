const MPSOrMPO{T} = Union{MPS{T}, MPO{T}}

function Base.show(io::IO, tn::MPSOrMPO{T}) where T
    println(io, "MPS{$T}($(nsite(tn)) sites, $(nflavor(tn)) flavors)")
    for (i, tensor) in enumerate(tensors(tn))
        print(io, i, ". ", size(tensor))
        if i == canonical_center(tn)
            print(io, " (canonical center)")
        end
        i < nsite(tn) && println(io)
    end
end
Base.show(io::IO, ::MIME"text/plain", tn::MPSOrMPO) = show(io, tn)

function right_move!(tn::MPSOrMPO; Dmax=typemax(Int), atol=1e-10)
    @assert tn.canonical_center > 0 "Canonical center is not set"
    @assert tn.canonical_center < nsite(tn) "Canonical center is already at the right edge"
    A, B = tn.tensors[tn.canonical_center], tn.tensors[tn.canonical_center+1]
    shapeA, shapeB = size(A), size(B)
    AB = reshape(A, prod(shapeA[1:end-1]), shapeA[end]) * reshape(B, shapeB[1], prod(shapeB[2:end]))
    u, s, v = truncated_svd(AB, Dmax, atol)
    tn.tensors[tn.canonical_center] = reshape(u, shapeA[1:end-1]..., size(u, 2))
    tn.tensors[tn.canonical_center+1] = reshape(s .* v, size(u, 2), shapeB[2:end]...)
    tn.canonical_center += 1
    return tn
end
function left_move!(tn::MPSOrMPO; Dmax=typemax(Int), atol=1e-10)
    @assert tn.canonical_center > 0 "Canonical center is not set"
    @assert tn.canonical_center > 1 "Canonical center is already at the left edge"
    A, B = tn.tensors[tn.canonical_center-1], tn.tensors[tn.canonical_center]
    shapeA, shapeB = size(A), size(B)
    AB = reshape(A, prod(shapeA[1:end-1]), shapeA[end]) * reshape(B, shapeB[1], prod(shapeB[2:end]))
    u, s, v = truncated_svd(AB, Dmax, atol)
    tn.tensors[tn.canonical_center-1] = reshape(u .* s', shapeA[1:end-1]..., size(u, 2))
    tn.tensors[tn.canonical_center] = reshape(v, size(v, 1), shapeB[2:end]...)
    tn.canonical_center -= 1
    return tn
end

function canonical_move!(tn::MPSOrMPO, direction::Int)
    for _ = 1:abs(direction)
        direction > 0 ? right_move!(tn) : left_move!(tn)
    end
    return tn
end

function to_right_canonical!(tn::MPSOrMPO; Dmax=typemax(Int), atol=1e-10)
    if canonical_center(tn) == 0
        tn.canonical_center = nsite(tn)
    end
    canonical_move!(tn, 1-canonical_center(tn))
    return tn
end
function to_left_canonical!(tn::MPSOrMPO; Dmax=typemax(Int), atol=1e-10)
    if canonical_center(tn) == 0
        tn.canonical_center = 1
    end
    canonical_move!(tn, nsite(tn)-canonical_center(tn))
    return tn
end

# check the validity of the canonical form
function check_canonical(tn::MPSOrMPO{T}; atol=1e-10) where T
    tn.canonical_center == 0 && return true
    valid = true
    for i = 1:canonical_center(tn)-1
        tensor = tn.tensors[i]
        shape = size(tensor)
        U = reshape(tensor, prod(shape[1:end-1]), shape[end])
        n = size(U, 2)
        valid &= isapprox(U' * U, Matrix{T}(I, n, n); atol)
    end
    for i = canonical_center(tn)+1:nsite(tn)
        tensor = tn.tensors[i]
        shape = size(tensor)
        U = reshape(tensor, shape[1], prod(shape[2:end]))
        n = size(U, 1)
        valid &= isapprox(U * U', Matrix{T}(I, n, n); atol)
    end
    return valid
end

Base.:+(tn1::T, tns::T...) where T <: MPSOrMPO = add(tn1, tns...)
function add(tn1::T, tns::T...; reduce_end=true) where T<:MPSOrMPO
    tns = (tn1, tns...)
    @assert allequal(nsite.(tns)) "MPS/MPO have different number of sites"
    @assert allequal(nflavor.(tns)) "MPS/MPO have different number of flavors"
    cc = !reduce_end && allequal(canonical_center.(tns)) ? canonical_center(tns[1]) : 0
    lastdim = nphysicalsites(tn1)+2
    tensors = [cat((tn.tensors[i] for tn in tns)...; dims=(reduce_end && i == 1 ? (lastdim,) : (reduce_end && i==nsite(tn1) ? (1,) : (1, lastdim)))) for i = 1:nsite(tn1)]
    return similar(tn1, tensors, cc)
end

function compress!(tn::MPSOrMPO; niter::Int=3, Dmax=typemax(Int), atol=1e-10)
    tn.canonical_center = 1
    for _ = 1:niter
        for _ = 1:nsite(tn)-1
            right_move!(tn; Dmax, atol)
        end
        for _ = 1:nsite(tn)-1
            left_move!(tn; Dmax, atol)
        end
    end
    return tn
end