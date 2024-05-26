# L: initial guess for left environment L'L
function left_orthonomalize!(A::AbstractArray{T, 3}, L::AbstractMatrix{T}, η) where T
    L ./= norm(L)  # normalize L
    L_old = L
    Al, L, λ = left_rotate(L, A)
    δ = norm(L - L_old)
    while δ > η
        # compute the fixed point
        _, evecs = eigsolve(x0, 1, :LR; ishermitian=false, tol=δ/10) do x
            m = reshape(x, size(Al, 1), size(Al, 1))
            vec(ein"(ij,iak),jal->kl"(m, Al, conj.(Al)))
        end

        # rotate
        L_old = L
        Al, L, λ, δ = left_rotate(L, Al)
        δ = norm(L - L_old)
    end
end

# rotate the left environment L'L and the tensor A
function left_rotate(L::AbstractArray{T, 2}, A::AbstractArray{T, 3}) where T
    @assert size(L, 1) == size(L, 2) == size(A, 1) == size(A, 3) "size mismatch"
    n, d = size(L, 1), size(A, 2)
    LA = ein"ij,jkl->ikl"(L, A)
    A2_, L2 = qr(reshape(LA, :, n))
    A2 = reshape(A2_[:, 1:n], n, d, n)
    λ = norm(L2)
    L2 ./= λ
    return A2, L2, λ
end

# transfer matrix 
# --- A ---
#     |
# --- Ā ---
struct TransferMatrix{T, AT<:AbstractArray{T, 3}}
    A::AT
end

function LinearAlgebra.Matrix(tm::TransferMatrix)
    res = ein"aic,bid->abcd"(tm.A, conj.(tm.A))
    return reshape(res, size(res, 1)^2, size(res, 3)^2)
end

function left_mul(tm::TransferMatrix{T}, x::AbstractVector{T}) where T
    @assert length(x) == size(tm.A, 1)^2
    X = reshape(x, size(tm.A, 1), size(tm.A, 1))
    res = ein"(aα,aib),αiβ->bβ"(X, tm.A, conj.(tm.A))
    return vec(res)
end

function right_mul(tm::TransferMatrix{T}, x::AbstractVector{T}) where T
    @assert length(x) == size(tm.A, 3)^2
    X = reshape(x, size(tm.A, 3), size(tm.A, 3))
    res = ein"(bβ,aib),αiβ->aα"(X, tm.A, conj.(tm.A))
    return vec(res)
end

function lefteigen(tm::TransferMatrix{T}, x0::AbstractVector{T}=randn(T, size(tm.A, 1)^2); tol=1e-10) where T
    evals, evecs = eigsolve(x->left_mul(tm, x), x0, 1, :LR; ishermitian=false, tol)
    return evals[1], evecs[1]
end

function righteigen(tm::TransferMatrix{T}, x0::AbstractVector{T}=randn(T, size(tm.A, 3)^2); tol=1e-10) where T
    evals, evecs = eigsolve(x->right_mul(tm, x), x0, 1, :LR; ishermitian=false, tol)
    return evals[1], evecs[1]
end

function lefthalfeigen_rotate(tm::TransferMatrix{T}, L::AbstractMatrix{T}=randn(T, size(tm.A, 1), size(tm.A, 1)); tol=1e-10) where T
    @assert size(L, 1) == size(L, 2) == size(tm.A, 1) == size(tm.A, 3) "size mismatch"
    L_old = L
    A = tm.A
    local λ
    while norm(L - L_old) > tol
        A, L, λ = left_rotate(L_old, A)
        L_old = L
    end
    return λ, L
end