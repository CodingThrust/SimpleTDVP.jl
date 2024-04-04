using Test, SimpleTDVP, LinearAlgebra, Random

@testset "MPS conversion" begin
    v = randn(ComplexF64, 2^6)
    mps = vec2mps(v)
    @test vec(mps) ≈ v
    canonical_move!(mps, -3)
    @test SimpleTDVP.check_canonical(mps)
    @test vec(mps) ≈ v
    canonical_move!(mps, 2)
    @test vec(mps) ≈ v
end

@testset "canonical_move!" begin
    v = randn(ComplexF64, 2^6)
    mps = vec2mps(v)
    mps2 = copy(mps)
    canonical_move!(mps2, -2)
    mpsa = mps + mps2
    @test !is_canonicalized(mpsa)
    @test SimpleTDVP.check_canonical(mpsa)
    @test SimpleTDVP.check_canonical(mps2)
    @test SimpleTDVP.check_canonical(mps + mps)
    @test SimpleTDVP.check_canonical(SimpleTDVP.add(mps, mps2; reduce_end=false))
    @test vec(mpsa) ≈ 2v
    @test vec(SimpleTDVP.add(mps, mps2; reduce_end=false)) ≈ 2v

    mpsb = to_left_canonical!(copy(mpsa))
    @test canonical_center(mpsb) == nsite(mpsb)
    @test SimpleTDVP.check_canonical(mpsb)
    @test vec(mpsb) ≈ 2v

    mpsb = to_right_canonical!(mpsa)
    @test canonical_center(mpsb) == 1
    @test SimpleTDVP.check_canonical(mpsb)
    @test vec(mpsb) ≈ 2v
end

@testset "dot" begin
    v = randn(ComplexF64, 2^6)
    mps = vec2mps(v)
    @test dot(mps, mps) ≈ dot(v, v)
end

@testset "compress" begin
    Random.seed!(5)
    χ = 7
    tensors = [randn(ComplexF64, 1, 2, χ)]
    for i=1:5
        push!(tensors, randn(ComplexF64, χ, 2, χ))
    end
    push!(tensors, randn(ComplexF64, χ, 2, 1))
    mps = MPS(tensors)
    @test num_of_elements(mps) == 5*2*χ^2 + χ*2*2
    mps2 = compress!(copy(mps); niter=3, Dmax=χ)
    # (1, 2, 2), (2, 2, 4), (4, 2, 7), (7, 2, 7), (7, 2, 4), (4, 2, 2), (2, 2, 1)
    @test num_of_elements(mps2) == 1*2*2 + 2*2*4 + 4*2*7 + 7*2*7 + 7*2*4 + 4*2*2 + 2*2*1
    @test vec(mps2) ≈ vec(mps)
    @test SimpleTDVP.check_canonical(mps2)
end