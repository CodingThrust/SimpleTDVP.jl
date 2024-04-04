using SimpleTDVP, Test
using LuxorGraphPlot.Luxor

@testset "visualize" begin
    mps = SimpleTDVP.rand_mps([1,2,4,6,6,4,2,1])
    @test SimpleTDVP.default_draw(mps) isa Luxor.Drawing
    mpo = SimpleTDVP.rand_mpo([1,4,6,4,1])
    @test SimpleTDVP.default_draw(mpo) isa Luxor.Drawing
end