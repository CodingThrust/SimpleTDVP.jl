# required interfaces: tensors
abstract type AbstractTensorNetwork{T} end
Base.eltype(::Type{<:AbstractTensorNetwork{T}}) where T = T
Base.eltype(::AbstractTensorNetwork{T}) where T = T

function num_of_elements(tn::AbstractTensorNetwork)
    return sum([length(tensor) for tensor in tensors(tn)])
end