using SimpleTDVP, Test, LinearAlgebra, OMEinsum

@testset "rotate" begin
    L = randn(ComplexF64, 5, 5)
    A = randn(ComplexF64, 5, 2, 5)
    A2, L2, λ, δ = SimpleTDVP.rotate(L, A)
    # A is left canonical
    @test ein"ijk,ijl->kl"(A2, conj.(A2)) ≈ I
    # AL = LA * λ
    @test norm(ein"ij,jkl->ikl"(L, A) - ein"ijk,kl->ijl"(A2, L2*λ)) < 1e-10
end

@testset "left/righteigen" begin
    A = randn(ComplexF64, 5, 2, 5)
    tm = SimpleTDVP.TransferMatrix(A)
    mA = Matrix(tm)
    vl = randn(ComplexF64, size(mA, 1))
    vr = randn(ComplexF64, size(mA, 2))
    @test mA * vr ≈ SimpleTDVP.right_mul(tm, vr)
    @test vec(transpose(vl) * mA) ≈ SimpleTDVP.left_mul(tm, vl)
    λl, LLl = SimpleTDVP.lefteigen(tm)
    λr, LLr = SimpleTDVP.righteigen(tm)
    @test abs(λl - λr) < 1e-10
    # abs(transpose(LLl) * LLr) ≈ 1? this is not true

    # eigen by QR
    λl, Ll = SimpleTDVP.lefthalfeigen_rotate(tm)
end