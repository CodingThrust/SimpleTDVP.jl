mutable struct MPO{T, AT<:AbstractArray{T,4}} <: AbstractTensorNetwork{T}
    tensors::Vector{AT}
    canonical_center::Int  # if canonical_center == 0, the MPO is not canonicalized
    function MPO(tensors::Vector{AT}, canonical_center::Int=0) where {T, AT<:AbstractArray{T,4}}
        n = length(tensors)
        nflavor = size(tensors[1], 2)
        @assert n > 0
        @assert canonical_center > 0 && canonical_center <= n || canonical_center == 0
        @assert all(i->size(tensors[i], 2) == size(tensors[i], 3) == nflavor && size(tensors[i], 4) == size(tensors[mod1(i+1, n)], 1), 1:n)
        new{T, AT}(tensors, canonical_center)
    end
end
nsite(mpo::MPO) = length(mpo.tensors)
nflavor(mpo::MPO) = size(mpo.tensors[1], 2)
openended(mpo::MPO) = size(mpo.tensors[1], 1) == 1
Base.conj!(mpo::MPO) = (conj!.(mpo.tensors); mpo)
Base.copy(mpo::MPO) = MPO(copy(mpo.tensors))
Base.similar(::MPO, tensors::Vector, canonical_center::Int) = MPO(tensors, canonical_center)
tensors(mpo::MPO) = mpo.tensors
nphysicalsites(mpo::MPO) = 2
Base.adjoint(mpo::MPO) = MPO([ein"ijkl->ikjl"(conj.(tensor)) for tensor in mpo.tensors], mpo.canonical_center)

canonical_center(mpo::MPO) = mpo.canonical_center
is_canonicalized(mpo::MPO) = mpo.canonical_center > 0

function rand_mpo(::Type{T}, bondims::Vector{Int}; d::Int=2) where T
    tensors = [make_hermitian(randn(T, 1, d, d, bondims[1]))]
    for i = 2:length(bondims)
        push!(tensors, make_hermitian(randn(T, bondims[i-1], d, d, bondims[i])))
    end
    push!(tensors, make_hermitian(randn(T, bondims[end], d, d, 1)))
    return MPO(tensors)
end
make_hermitian(t::AbstractArray{T, 4}) where T = (t .+= permutedims(conj.(t), (1, 3, 2, 4)); t)
rand_mpo(bondims::Vector{Int}; d::Int=2) = rand_mpo(ComplexF64, bondims; d)

# convert a matrix to an MPO
function mat2mpo(m::AbstractMatrix; d=2, Dmax=typemax(Int), atol=1e-10)
    nsite = round(Int, log2(length(m)) ÷ log2(d) ÷ 2)
    @assert d^(2nsite) == length(m) "Matrix length is not a power of the physical dimension ($d), got: $(size(m))"
    n = d^nsite
    state = reshape(m, 1, n, 1, n)
    tensors = typeof(state)[]
    for _ = 1:nsite-1
        n ÷= d
        state = permutedims(reshape(state, (size(state, 1) * d, n, d, n)), (1, 3, 2, 4))
        u, s, v, err = truncated_svd(reshape(state, :, n^2), Dmax, atol)
        push!(tensors, reshape(u, size(u, 1) ÷ (d^2), d, d, size(u, 2)))
        state = reshape(s .* v, size(v, 1), n, 1, n)
    end
    push!(tensors, reshape(state, size(state, 1), d, d, 1))
    return MPO(tensors)
end

function mat(mpo::MPO)
    code = code_mpo2mat(mpo)
    n = nflavor(mpo)^nsite(mpo)
    return reshape(code(mpo.tensors...), n, n)
end
# the code for converting an MPO to a matrix
function code_mpo2mat(mpo::MPO; optimizer=GreedyMethod())
    store = IndexStore()
    ixs = Vector{Int}[]
    iy1 = Int[]
    iy2 = Int[]
    firstidx = newindex!(store)
    previdx = firstidx
    for k = 1:nsite(mpo)
        physical1, physical2 = newindex!(store), newindex!(store)
        nextidx = k == nsite(mpo) ? firstidx : newindex!(store)
        push!(ixs, [previdx, physical1, physical2, nextidx])
        push!(iy1, physical1)
        push!(iy2, physical2)
        previdx = nextidx
    end
    size_dict = OMEinsum.get_size_dict(ixs, mpo.tensors)
    return optimize_code(DynamicEinCode(ixs, [iy1..., iy2...]), size_dict, optimizer)
end

function code_sandwich(bra::MPS, op::MPO, ket::MPS; optimizer=GreedyMethod())
    store = IndexStore()
    ixs_bra = Vector{Int}[]
    ixs_op = Vector{Int}[]
    ixs_ket = Vector{Int}[]
    firstidx_bra = newindex!(store)
    previdx_bra = firstidx_bra
    firstidx_op = newindex!(store)
    previdx_op = firstidx_op
    firstidx_ket = newindex!(store)
    previdx_ket = firstidx_ket
    for k = 1:nsite(bra)
        physical_bra = newindex!(store)
        physical_ket = newindex!(store)
        nextidx_bra = k == nsite(bra) ? firstidx_bra : newindex!(store)
        nextidx_op = k == nsite(ket) ? firstidx_op : newindex!(store)
        nextidx_ket = k == nsite(ket) ? firstidx_ket : newindex!(store)
        push!(ixs_bra, [previdx_bra, physical_bra, nextidx_bra])
        push!(ixs_op, [previdx_op, physical_bra, physical_ket, nextidx_op])
        push!(ixs_ket, [previdx_ket, physical_ket, nextidx_ket])
        previdx_bra = nextidx_bra
        previdx_op = nextidx_op
        previdx_ket = nextidx_ket
    end
    ixs = [ixs_bra..., ixs_op..., ixs_ket...]
    size_dict = OMEinsum.get_size_dict(ixs, [bra.tensors..., op.tensors..., ket.tensors...])
    return optimize_code(DynamicEinCode(ixs, Int[]), size_dict, optimizer)
end

function sandwich(bra::MPS, op::MPO, ket::MPS)
    code = code_sandwich(bra, op, ket)
    return code(conj.(bra.tensors)..., op.tensors..., ket.tensors...)[]
end

function heisenberg_mpo(::Type{T}, n::Int) where T
    @assert n > 1
    tensor1 = zeros(T, 1, 2, 2, 5)
    tensor2 = zeros(T, 5, 2, 2, 5)
    tensor3 = zeros(T, 5, 2, 2, 1)
    tensor1[1, :, :, 1] = tensor2[1, :, :, 1] = tensor2[5, :, :, 5] = tensor3[5, :, :, 1] = Matrix{T}(I2)
    tensor1[1, :, :, 2] = tensor2[2, :, :, 5] = tensor2[1, :, :, 2] = tensor3[2, :, :, 1] = Matrix{T}(X)
    tensor1[1, :, :, 3] = tensor2[3, :, :, 5] = tensor2[1, :, :, 3] = tensor3[3, :, :, 1] = Matrix{T}(Y)
    tensor1[1, :, :, 4] = tensor2[4, :, :, 5] = tensor2[1, :, :, 4] = tensor3[4, :, :, 1] = Matrix{T}(Z)
    MPO([tensor1, fill(tensor2, n-2)..., tensor3])
end

function transverse_ising_mpo(::Type{T}, n::Int, h::Real) where T
    @assert n > 1
    tensor1 = zeros(T, 1, 2, 2, 3)
    tensor2 = zeros(T, 3, 2, 2, 3)
    tensor3 = zeros(T, 3, 2, 2, 1)
    tensor1[1, :, :, 1] = tensor2[1, :, :, 1] = tensor2[3, :, :, 3] = tensor3[3, :, :, 1] = Matrix{T}(I2)
    tensor1[1, :, :, 2] = tensor2[2, :, :, 3] = tensor2[1, :, :, 2] = tensor3[2, :, :, 1] = Matrix{T}(Z)
    tensor1[1, :, :, 3] = tensor2[1, :, :, 3] = tensor3[1, :, :, 1] = Matrix{T}(X) .* h
    MPO([tensor1, fill(tensor2, n-2)..., tensor3])
end