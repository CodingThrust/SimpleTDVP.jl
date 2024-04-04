struct IndexStore{LT}
    indices::Vector{LT}
end
IndexStore() = IndexStore(Int[])
Base.length(store::IndexStore) = length(store.indices)
function newindex!(store::IndexStore{LT}) where LT <: Integer
    index = length(store) == 0 ? 1 : store.indices[end] + 1
    push!(store.indices, index)
    return index
end

function truncated_svd(tmat, dmax::Int, atol::Real)
    u, s, v = LinearAlgebra.svd(tmat)
    dmax = min(searchsortedfirst(s, atol, rev=true), dmax, length(s))
    return u[:, 1:dmax], s[1:dmax], v'[1:dmax, :], sum(s[dmax+1:end])
end

