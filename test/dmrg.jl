using SimpleTDVP, Test, SimpleTDVP.OMEinsum, LinearAlgebra, Random

@testset "runtime" begin
    mpo = rand_mpo([1,4,6,4,1])
    mps = rand_mps([1, 2, 4, 2, 1])
    to_right_canonical!(mps)
    T = ComplexF64
    L = fill!(similar(mps.tensors[1], 1, 1, 1), one(T))
    right_env = [fill!(similar(mps.tensors[end], 1, 1, 1), one(T))]
    for i=nsite(mps):-1:3   # update the right environment 3:nsite(mps)
        push!(right_env, SimpleTDVP.update_right_env(right_env[end], mps.tensors[i], mpo.tensors[i]))
    end
    i = 1
    r = SimpleTDVP.DMRGRuntime(mps.tensors[i], mps.tensors[i+1], mpo.tensors[i], mpo.tensors[i+1], L, pop!(right_env))
    e1 = SimpleTDVP.energy(r)
    e2 = sandwich(mps, mpo, mps)
    @test e1 ≈ e2
    # matmul!
    x0 = SimpleTDVP.get_x0(r)
    y = similar(x0)
    SimpleTDVP.matmul!(y, r, x0)
    @test x0' * y ≈ e1
    # linear operator
    op = SimpleTDVP.linearoperator(r)
    @test op * x0 ≈ y
    # update site
    eng, A, S, B, err = SimpleTDVP.update_sites(r; Dmax=10, atol=1e-10)
    x0 = vec(ein"ceh,hjm->cejm"(A, B))
    @show eng ≈ x0' * op * x0
    @test eng <= real(e1)
end

@testset "dmrg" begin
    Random.seed!(42)
    mps = rand_mps([1,2,4,6,6,4,2,1])
    @test nsite(mps) == 9
    mpo = rand_mpo([1,4,6,4,1])
    @test nsite(mpo) == 6
    @test mat(mpo) ≈ mat(mpo)'

    mps2 = rand_mps([1, 2, 4, 2, 1])
    dmrgres = dmrg(mps2, mpo; nsweeps=10)
    res = eigen(mat(mpo))
    emin, vmin = res.values[1], res.vectors[:, 1]
    @test abs(dot(vec(dmrgres.mps), vmin)) ≈ 1
    @test dmrgres.energy ≈ emin
    @test dmrgres.truncation_error > 0
    @test abs(dmrgres.convergence_error) < 1e-10
end