using SimpleTDVP, Test

@testset "idmrg" begin
    W = heisenberg_mpo(ComplexF64, 6).tensors[2]
    @show idmrg(W; nsteps=10, Dmax=64, atol=1e-10)
end