using Test, SimpleTDVP

@testset "MPO conversion" begin
    m = randn(ComplexF64, 2^5, 2^5)
    mpo = mat2mpo(m)
    @test mat(mpo) ≈ m

    mpo2 = copy(mpo)
    mpoa = mpo + mpo2
    @test mat(mpoa) ≈ 2m
end

@testset "sandwich" begin
    m = randn(ComplexF64, 2^5, 2^5)
    mpo = mat2mpo(m)
    vket = randn(ComplexF64, 2^5)
    ket = vec2mps(vket)
    vbra = randn(ComplexF64, 2^5)
    bra = vec2mps(vbra)
    @test sandwich(bra, mpo, ket) ≈ dot(vbra, m * vket)
end

@testset "compress" begin
    Random.seed!(5)
    χ = 10
    tensors = [randn(ComplexF64, 1, 2, 2, χ)]
    for i=1:3
        push!(tensors, randn(ComplexF64, χ, 2, 2, χ))
    end
    push!(tensors, randn(ComplexF64, χ, 2, 2, 1))
    mpo = MPO(tensors)
    @test num_of_elements(mpo) == 3*2*2*χ^2 + χ*2*2*2
    mpo2 = compress!(copy(mpo); niter=3, Dmax=χ)
    # (1, 2, 2, 4), (4, 2, 2, 10), (10, 2, 2, 10), (10, 2, 2, 4), (4, 2, 2, 1)
    @test num_of_elements(mpo2) == 1*2*2*4 + 4*2*2*10 + 10*2*2*10 + 10*2*2*4 + 4*2*2*1
    @test mat(mpo2) ≈ mat(mpo)
    @test SimpleTDVP.check_canonical(mpo2)
end