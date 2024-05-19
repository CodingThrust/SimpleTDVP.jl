mutable struct MPS{T, AT<:AbstractArray{T,3}} <: AbstractTensorNetwork{T}
    const tensors::Vector{AT}
    canonical_center::Int  # if canonical_center == 0, the MPS is not canonicalized
    function MPS(tensors::Vector{AT}, canonical_center::Int=0) where {T, AT<:AbstractArray{T,3}}
        n = length(tensors)
        nflavor = size(tensors[1], 2)
        @assert length(tensors) > 0
        @assert canonical_center > 0 && canonical_center <= n || canonical_center == 0
        @assert all(i->size(tensors[i], 2) == nflavor && size(tensors[i], 3) == size(tensors[mod1(i+1, n)], 1), 1:n)
        new{T, AT}(tensors, canonical_center)
    end
end
nsite(mps::MPS) = length(mps.tensors)
nflavor(mps::MPS) = size(mps.tensors[1], 2)
canonical_center(mps::MPS) = mps.canonical_center
is_canonicalized(mps::MPS) = mps.canonical_center > 0
openended(mps::MPS) = size(mps.tensors[1], 1) == 1
Base.conj!(mps::MPS) = (conj!.(mps.tensors); mps)
Base.copy(mps::MPS) = MPS(copy(mps.tensors), mps.canonical_center)
Base.similar(::MPS, tensors::Vector, canonical_center::Int) = MPS(tensors, canonical_center)
tensors(mps::MPS) = mps.tensors
nphysicalsites(mpo::MPS) = 1
function rand_mps(::Type{T}, bondims::Vector{Int}; d::Int=2) where T
    tensors = [randn(T, 1, d, bondims[1])]
    for i = 2:length(bondims)
        push!(tensors, randn(T, bondims[i-1], d, bondims[i]))
    end
    push!(tensors, randn(T, bondims[end], d, 1))
    return MPS(tensors)
end
rand_mps(bondims::Vector{Int}; d::Int=2) = rand_mps(ComplexF64, bondims; d)

# convert a vector to an MPS
function vec2mps(v::AbstractVector; d=2, Dmax=typemax(Int), atol=1e-10)
    state = reshape(v, 1, length(v))
    tensors = typeof(reshape(state, 1, length(v), 1))[]
    nsite = round(Int, log2(length(v)) ÷ log2(d))
    @assert d^nsite == length(v) "Vector length is not a power of the physical dimension ($d), got: $(length(v))"
    for _ = 1:nsite-1
        state = reshape(state, (d * size(state, 1), size(state, 2) ÷ d))
        u, s, v, err = truncated_svd(state, Dmax, atol)
        push!(tensors, reshape(u, size(u, 1) ÷ d, d, size(u, 2)))
        state = s .* v
    end
    push!(tensors, reshape(state, size(state, 1), d, 1))
    return MPS(tensors, length(tensors))
end

# convert an MPS to a vector
function code_mps2vec(mps::MPS; optimizer=GreedyMethod())
    store = IndexStore()
    ixs = Vector{Int}[]
    iy = Int[]
    firstidx = newindex!(store)
    previdx = firstidx
    for k = 1:nsite(mps)
        physical = newindex!(store)
        nextidx = k == nsite(mps) ? firstidx : newindex!(store)
        push!(ixs, [previdx, physical, nextidx])
        push!(iy, physical)
        previdx = nextidx
    end
    size_dict = OMEinsum.get_size_dict(ixs, mps.tensors)
    return optimize_code(DynamicEinCode(ixs, iy), size_dict, optimizer)
end

function Base.vec(mps::MPS)
    code = code_mps2vec(mps)
    return vec(code(mps.tensors...))
end


# the energy function
# ... - a - C - f - D - k - ...
#           |       |
#           d       i       
#           |       |
# ... - c - A - h - B - m - ...
function code_dot(bra::MPS, ket::MPS; optimizer=GreedyMethod())
    store = IndexStore()
    ixs_bra = Vector{Int}[]
    ixs_ket = Vector{Int}[]
    firstidx_bra = newindex!(store)
    previdx_bra = firstidx_bra
    firstidx_ket = newindex!(store)
    previdx_ket = firstidx_ket
    for k = 1:nsite(bra)
        physical = newindex!(store)
        nextidx_bra = k == nsite(bra) ? firstidx_bra : newindex!(store)
        nextidx_ket = k == nsite(ket) ? firstidx_ket : newindex!(store)
        push!(ixs_bra, [previdx_bra, physical, nextidx_bra])
        push!(ixs_ket, [previdx_ket, physical, nextidx_ket])
        previdx_bra = nextidx_bra
        previdx_ket = nextidx_ket
    end
    ixs = [ixs_bra..., ixs_ket...]
    size_dict = OMEinsum.get_size_dict(ixs, [bra.tensors..., ket.tensors...])
    return optimize_code(DynamicEinCode(ixs, Int[]), size_dict, optimizer)
end

function LinearAlgebra.dot(mps1::MPS, mps2::MPS)
    code = code_dot(mps1, mps2)
    return code(conj.(mps1.tensors)..., mps2.tensors...)[]
end

function product_mps(states::AbstractVector...)
    return MPS([reshape(state, 1, size(state, 1), 1) for state in states])
end

function ghz_mps(::Type{T}, n::Int) where T
    @assert n >= 2
    state1 = zeros(T, 1, 2, 2)
    state1[1, 1, 1] = state1[1, 2, 2] = 1/sqrt(2)
    state2 = zeros(T, 2, 2, 2)
    state2[1, 1, 1] = state2[2, 2, 2] = 1
    state3 = zeros(T, 2, 2, 1)
    state3[1, 1, 1] = state3[2, 2, 1] = 1
    return MPS([state1, fill(state2, n-2)..., state3])
end

function aklt_mps(::Type{T}, n::Int) where T
    @assert n >= 2
    state1 = zeros(T, 1, 3, 3)
    state1[1, 1, 1] = state1[1, 2, 2] = state1[1, 3, 3] = 1/sqrt(3)
    state2 = zeros(T, 3, 3, 3)
    state2[1, 1, 1] = state2[2, 2, 2] = state2[3, 3, 3] = 1
    state3 = zeros(T, 3, 3, 1)
    state3[1, 1, 1] = state3[2, 2, 1] = state3[3, 3, 1] = 1
    return MPS([state1, fill(state2, n-2)..., state3])
end