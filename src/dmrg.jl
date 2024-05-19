struct DMRGResult{T}
    energy::T
    mps::MPS{Complex{T}}
    truncation_error::T
    convergence_error::T
end

function dmrg(mps::MPS{T}, mpo::MPO{T}; nsweeps, atol=1e-10, Dmax=typemax(Int)) where T
    dmrg!(deepcopy(mps), mpo; nsweeps=nsweeps, atol=atol, Dmax=Dmax)
end
function dmrg!(mps::MPS{T}, mpo::MPO{T}; nsweeps, atol=1e-10, Dmax=typemax(Int)) where T
    @assert nsite(mps) == nsite(mpo) "The number of sites in MPS and MPO are different"
    @assert nflavor(mps) == nflavor(mpo) "The number of flavors in MPS and MPO are different"
    to_right_canonical!(mps)
    left_env = empty(mps.tensors)
    right_env = [fill!(similar(mps.tensors[end], 1, 1, 1), one(T))]
    for i=nsite(mps):-1:3   # update the right environment 3:nsite(mps)
        push!(right_env, update_right_env(right_env[end], mps.tensors[i], mpo.tensors[i]))
    end
    convergence_error = eng = eng_prev = real(T)(Inf)
    truncation_error = zero(real(T))
    for k = 1:nsweeps
        for i = 1:nsite(mps)-1  # NOTE: site `1` is unneseccary to update, but is convenient to have
            # update site i, i+1
            if i == 1
                L = fill!(similar(mps.tensors[1], 1, 1, 1), one(T))
            else
                L = update_left_env(left_env[end], mps.tensors[i-1], mpo.tensors[i-1])
            end
            push!(left_env, L)
            r = DMRGRuntime(mps.tensors[i], mps.tensors[i+1], mpo.tensors[i], mpo.tensors[i+1], L, pop!(right_env))
            eng, A, S, B, err = update_sites(r; Dmax, atol)
            B .*= reshape(S, :, 1, 1)
            truncation_error += err
            @info "Sweep = $k (right moving), site = ($i, $(i+1)), energy = $eng, bond size = $(length(S)), error = $err"
            mps.tensors[i], mps.tensors[i+1] = A, B
            right_move!(mps, Dmax=Dmax, atol=atol)
        end
        #push!(right_env, fill!(similar(mps.tensors[end], 1, 1, 1), one(T)))
        for i = nsite(mps):-1:2  # NOTE: site `nsite(mps)` is unneseccary to update, but is convenient to have
            # update site i-1, i
            if i == nsite(mps)
                R = fill!(similar(mps.tensors[end], 1, 1, 1), one(T))
            else
                R = update_right_env(right_env[end], mps.tensors[i+1], mpo.tensors[i+1])
            end
            push!(right_env, R)
            r = DMRGRuntime(mps.tensors[i-1], mps.tensors[i], mpo.tensors[i-1], mpo.tensors[i], pop!(left_env), R)
            eng, A, S, B, err = update_sites(r; Dmax, atol)
            A .*= reshape(S, 1, 1, :)
            truncation_error += err
            @info "Sweep = $k (left moving), site = ($(i-1), $i), energy = $eng, bond size = $(length(S)), error = $err"
            mps.tensors[i-1], mps.tensors[i] = A, B
            left_move!(mps, Dmax=Dmax, atol=atol)
        end
        convergence_error = eng_prev - eng
        eng_prev = eng
    end
    return DMRGResult(eng, mps, truncation_error, convergence_error)
end

function idmrg!(mps::MPS{T}, mpo::MPO{T}; nsweeps, atol=1e-10, Dmax=typemax(Int)) where T
    @assert nsite(mps) == nsite(mpo) "The number of sites in MPS and MPO are different"
    @assert nflavor(mps) == nflavor(mpo) "The number of flavors in MPS and MPO are different"
    to_right_canonical!(mps)
    left_env = empty(mps.tensors)
    right_env = [fill!(similar(mps.tensors[end], 1, 1, 1), one(T))]
    for i=nsite(mps):-1:3   # update the right environment 3:nsite(mps)
        push!(right_env, update_right_env(right_env[end], mps.tensors[i], mpo.tensors[i]))
    end
    convergence_error = eng = eng_prev = real(T)(Inf)
    truncation_error = zero(real(T))
    for k = 1:nsweeps
        for i = 1:nsite(mps)-1  # NOTE: site `1` is unneseccary to update, but is convenient to have
            # update site i, i+1
            if i == 1
                L = fill!(similar(mps.tensors[1], 1, 1, 1), one(T))
            else
                L = update_left_env(left_env[end], mps.tensors[i-1], mpo.tensors[i-1])
            end
            push!(left_env, L)
            r = DMRGRuntime(mps.tensors[i], mps.tensors[i+1], mpo.tensors[i], mpo.tensors[i+1], L, pop!(right_env))
            eng, A, S, B, err = update_sites(r; Dmax, atol)
            B .*= reshape(S, :, 1, 1)
            truncation_error += err
            @info "Sweep = $k (right moving), site = ($i, $(i+1)), energy = $eng, bond size = $(length(S)), error = $err"
            mps.tensors[i], mps.tensors[i+1] = A, B
            right_move!(mps, Dmax=Dmax, atol=atol)
        end
        #push!(right_env, fill!(similar(mps.tensors[end], 1, 1, 1), one(T)))
        for i = nsite(mps):-1:2  # NOTE: site `nsite(mps)` is unneseccary to update, but is convenient to have
            # update site i-1, i
            if i == nsite(mps)
                R = fill!(similar(mps.tensors[end], 1, 1, 1), one(T))
            else
                R = update_right_env(right_env[end], mps.tensors[i+1], mpo.tensors[i+1])
            end
            push!(right_env, R)
            r = DMRGRuntime(mps.tensors[i-1], mps.tensors[i], mpo.tensors[i-1], mpo.tensors[i], pop!(left_env), R)
            eng, A, S, B, err = update_sites(r; Dmax, atol)
            A .*= reshape(S, 1, 1, :)
            truncation_error += err
            @info "Sweep = $k (left moving), site = ($(i-1), $i), energy = $eng, bond size = $(length(S)), error = $err"
            mps.tensors[i-1], mps.tensors[i] = A, B
            left_move!(mps, Dmax=Dmax, atol=atol)
        end
        convergence_error = eng_prev - eng
        eng_prev = eng
    end
    return DMRGResult(eng, mps, truncation_error, convergence_error)
end


struct DMRGRuntime{T, ST<:AbstractArray{T, 3}, OT<:AbstractArray{T, 4}, TL<:AbstractArray{T, 3}, TR<:AbstractArray{T, 3}}
    A::ST
    B::ST
    M::OT
    N::OT
    left_env::TL
    right_env::TR
end

function update_sites(r::DMRGRuntime{T}; Dmax::Int, atol::Real) where T
    # TODO: add GPU support
    op = linearoperator(r)
    x0 = get_x0(r)
    # TODO: support multiple target-energy-level
    evals, evecs = eigsolve(x->op*x, x0, 1, :SR; ishermitian=true, tol=atol)
    AB = reshape(evecs[1], size(r.A, 1) * size(r.A, 2), size(r.B, 2) * size(r.B, 3))
    U, S, V, err = truncated_svd(AB, Dmax, atol)
    A = reshape(U, size(r.A, 1), size(r.A, 2), size(U, 2))
    B = reshape(V, size(V, 1), size(r.B, 2), size(r.B, 3))
    return evals[1], A, S, B, err
end

function linearoperator(r::DMRGRuntime{T}) where T
    nrow = ncol = size(r.A, 1) * size(r.A, 2) * size(r.B, 2) * size(r.B, 3)
    return LinearOperator(T, nrow, ncol, true, true, (y, x, alpha, beta)->matmul!(y, r, x, alpha, beta))
end

# update the left environment tensor
# |---|- a - A - f -
# |   |      d      
# | L |- b - M - g -
# |   |      e
# |---|- c - A - h -
update_left_env(L0::AbstractArray{T, 3}, A::AbstractArray{T, 3}, M::AbstractArray{T, 4}) where T = ein"((abc,adf),bdeg),ceh->fgh"(L0, conj.(A), M, A)

# update the right environment tensor
# - f - B - k - |---|
#       i       |   |
# - g - N - l - | R |
#       j       |   |
# - h - B - m - |---|
update_right_env(R0::AbstractArray{T, 3}, B::AbstractArray{T, 3}, N::AbstractArray{T, 4}) where T = ein"((klm,fik),gijl),hjm->fgh"(R0, conj.(B), N, B)

# the energy function
# |---|- a - A - f - B - k - |---|
# |   |      d       i       |   |
# | L |- b - M - g - N - l - | R |
# |   |      e       j       |   |
# |---|- c - A - h - B - m - |---|
function energy(r::DMRGRuntime)
    return ein"((((((abc,adf),bdeg),ceh),fik),gijl),hjm),klm->"(r.left_env, conj.(r.A), r.M, r.A, conj.(r.B), r.N, r.B, r.right_env)[]
end

# the matrix multiplication function
# |---|- a -           - k - |---|
# |   |      d       i       |   |
# | L |- b - M - g - N - l - | R |
# |   |      e       j       |   |
# |---|- c - A - - - B - m - |---|
function matmul!(y::AbstractVector{T}, r::DMRGRuntime{T}, x::AbstractVector{T}, alpha=true, beta=false) where T
    code = ein"(((abc,cejm),bdeg),gijl),klm->adik"
    tensors = (r.left_env, reshape(x, size(r.A, 1), size(r.A, 2), size(r.B, 2), size(r.B, 3)), r.M, r.N, r.right_env)
    size_dict = OMEinsum.get_size_dict(getixsv(code), tensors)
    return einsum!(code, tensors, y, alpha, beta, size_dict)
end

# the state vector
#       e      j
# - c - A - h - B - m
function get_x0(r::DMRGRuntime{T}) where T
    return vec(ein"ceh,hjm->cejm"(r.A, r.B))
end