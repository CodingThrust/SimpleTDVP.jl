struct TensorLayout{LT}
    nodes::Vector{Pair{Vector{LT}, Node}}         # labels => nodes
    mask::Vector{Bool}
    ignored_labels::Vector{LT}
end
function TensorLayout(ixsv::AbstractVector, locs; ignored_labels=Int[], kwargs...)
    @assert length(ixsv) == length(locs)
    tvg = TensorLayout{Int}(Pair{Vector{Int}, Node}[], Bool[], ignored_labels)
    for (ix, loc) in zip(ixsv, locs)
        addnode!(tvg, loc..., ix; kwargs...)
    end
    return tvg
end
get_width(tvg::TensorLayout) = maximum([node.loc[1] for (_, node) in tvg.nodes]) + 50
get_height(tvg::TensorLayout) = maximum([node.loc[2] for (_, node) in tvg.nodes]) + 50
function addnode!(tvg::TensorLayout, x::Real, y::Real, ix::AbstractVector; kwargs...)
    loc = (x, y)
    node = if length(ix) == 0
        Node(:dot, loc; kwargs...)
    elseif length(ix) == 1
        Node(:circle, loc; radius=7, kwargs...)
    elseif length(ix) == 2
        Node(:polygon, loc; relpath=[[-1, 0], [0, -1], [1, 0], [0, 1]], kwargs...)
    elseif length(ix) == 3
        Node(:circle, loc; radius=12, kwargs...)
    elseif length(ix) == 4
        Node(:box, loc; width=20, height=20, kwargs...)
    else
        Node(:circle, loc; radius=12, kwargs...)
    end
    push!(tvg.nodes, ix => node)
    push!(tvg.mask, true)
    tvg
end

function draw_einsum(gl::TensorLayout)
    Luxor.@layer begin
        for (_, node) in gl.nodes[gl.mask]
            Luxor.sethue("white")
            LuxorGraphPlot.fill(node)
            Luxor.sethue("black")
            LuxorGraphPlot.stroke(node)
        end
        Luxor.sethue("black")
        for i in 1:length(gl.nodes), j in i+1:length(gl.nodes)
            ix1, node1 = gl.nodes[i]
            ix2, node2 = gl.nodes[j]
            if (gl.mask[i] || gl.mask[j]) && !isempty(setdiff(ix1 ∩ ix2, gl.ignored_labels))
                LuxorGraphPlot.stroke(Connection(node1, node2))
            end
        end
    end
end

function default_draw(f, gl::TensorLayout)
    return Luxor.@drawsvg begin
        Luxor.origin(0, 0)
        Luxor.background("white")
        draw_einsum(gl)
        f(gl)
        Luxor.finish()
    end get_width(gl) get_height(gl)
end
default_draw(gl::TensorLayout) = default_draw(gl->nothing, gl)
function default_draw(mps::MPS)
    n = nsite(mps)
    code = code_mps2vec(mps)
    ixs = OMEinsum.getixsv(OMEinsum.flatten(code))
    phantoms = [[ix[2]] for ix in ixs]
    layout = TensorLayout([ixs..., phantoms...],
            [[(i*50, 90) for i in 1:n]...  # real tensors
            [(i*50, 50) for i in 1:n]...];  # phantoms
            ignored_labels=[ixs[1][1]]  # ignore the long-ancilla label
        )
    layout.mask[n+1:end] .= false
    return default_draw(layout) do layout
        Luxor.fontsize(14)
        for i = 1:n
            if i < n
                node = LuxorGraphPlot.midpoint(layout.nodes[i].second, layout.nodes[i+1].second)
                Luxor.text("$(size(mps.tensors[i], 3))", node.loc + Luxor.Point(0, -10); valign=:middle, halign=:center)
            end
            if canonical_center(mps) == i
                Luxor.sethue("gray")
                LuxorGraphPlot.fill(layout.nodes[i].second)
            end
        end
    end
end

function default_draw(mpo::MPO)
    n = nsite(mpo)
    code = code_mpo2mat(mpo)
    ixs = OMEinsum.getixsv(OMEinsum.flatten(code))
    phantoms = [[[ix[2]] for ix in ixs]..., [[ix[3]] for ix in ixs]...]
    layout = TensorLayout([ixs..., phantoms...],
            [[(i*50, 90) for i in 1:n]...
             [(i*50, 50) for i in 1:n]...
             [(i*50, 130) for i in 1:n]...];
            ignored_labels = [ixs[1][1]]  # ignore the long-ancilla label
        )
    layout.mask[n+1:end] .= false
    return default_draw(layout) do layout
        Luxor.fontsize(14)
        for i = 1:n
            if i < n
                node = LuxorGraphPlot.midpoint(layout.nodes[i].second, layout.nodes[i+1].second)
                Luxor.text("$(size(mpo.tensors[i], 4))", node.loc + Luxor.Point(0, -10); valign=:middle, halign=:center)
            end
            if canonical_center(mpo) == i
                Luxor.sethue("gray")
                LuxorGraphPlot.fill(layout.nodes[i+n].second)
            end
        end
    end
end

for FORMAT in [MIME"text/html", MIME"image/svg+xml"]
    for OBJ in [MPS, MPO, TensorLayout]
        @eval function Base.show(io::IO, ::$FORMAT, obj::$OBJ)
            d = default_draw(obj)
            show(io, $FORMAT(), d)
        end
    end
end