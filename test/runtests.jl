using SimpleTDVP
using Test

@testset "mps" begin
    include("mps.jl")
end

@testset "mpo" begin
    include("mpo.jl")
end

@testset "dmrg" begin
    include("dmrg.jl")
end