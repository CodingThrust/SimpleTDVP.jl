using SimpleTDVP, Test
using LuxorGraphPlot.Luxor

@testset "visualize" begin
    mps = SimpleTDVP.rand_mps([1,2,4,6,6,4,2,1])
    display(mps)
end