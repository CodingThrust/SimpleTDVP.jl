struct IMPS{T}
    Al::Array{Complex{T}, 3}
    S::Array{T, 1}
    Ar::Array{Complex{T}, 3}
    function IMPS(left::Array{Complex{T}, 3}, center::Array{T, 1}, right::Array{Complex{T}, 3}) where T
        @assert size(left, 1) == size(center, 1) == size(right, 1) == size(left, 3) == size(right, 3)
        @assert size(left, 2) == size(right, 2)
        new{T}(left, center, right)
    end
end
nflavor(mps::IMPS) = size(mps.left, 2)

struct IDMRGResult{T}
    energy::T
    mps::IMPS{T}
    truncation_error::T
    convergence_error::T
end

function idmrg(W::AbstractArray{T, 4}; nsteps::Int, atol=1e-10, Dmax=typemax(Int)) where T
    nflavor = size(W, 2)
    @assert nflavor == size(W, 3) "The number of flavors in MPS and MPO are different"
    truncation_error = zero(real(T))
    local S, eng
    Al = randn!(similar(W, 1, nflavor, 1))
    Ar = randn!(similar(W, 1, nflavor, 1))
    left_env = fill!(similar(Al, 1, size(W, 1), 1), one(T))
    right_env = fill!(similar(Ar, 1, size(W, 1), 1), one(T))
    convergence_error = eng_prev = Inf
    for k = 1:nsteps
        # update Al and Ar
        @show size(Al), size(Ar), size(W), size(W), size(left_env), size(right_env)
        r = DMRGRuntime(Al, Ar, W, W, left_env, right_env)
        eng, L, S, R, err = update_sites(r; Dmax, atol)
        truncation_error += err
        @info "Sweep = $k, energy = $eng, bond size = $(length(S)), error = $err"
        # update left and right env
        left_env = ein"((abc,aid),bije),cjf->def"(left_env, L, W, conj.(L))
        right_env = ein"((abc,dia),eijb),fjc->def"(right_env, R, W, conj.(R))
        # update Al and Ar
        Al = ein"aid,d->dia"(L, S)
        Ar = ein"dia,d->aid"(R, S)
        convergence_error = eng_prev - eng
        eng_prev = eng
    end
    return IDMRGResult(eng, IMPS(Al, S, Ar), truncation_error, convergence_error)
end


