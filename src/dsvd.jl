# Derivative of full SVD.
#  small-size fallback & reference.
#
using LinearAlgebra

SupportedMap{T} = Union{Matrix{T}, LinearMap{T}}

∂svd(A::Matrix{T},
     dA::Matrix{T}) where {T} = begin
    # SVD for values of A.
    U, s, V = svd(A)

    ∂svd(A, dA, U, s, V)
end

∂svd(A ::SupportedMap{T},
     dA::SupportedMap{T},
     U::AbstractMatrix{T},
     s::AbstractVector{T},
     V::AbstractMatrix{T}) where {T} = begin
    # bond dimensions
    χo, χi = size(A)
    # 'rank' of matrix
    r = length(s)
    size(U) == (χo, r) || throw(DimensionMismatch("U has incorrect size."))
    size(V) == (χi, r) || throw(DimensionMismatch("V has incorrect size."))

    # F matrix.
    S = Diagonal(s)
    F = [if i == j
             0.0
         else
             safercp(s[j]^2 - s[i]^2, 1.0, 1e-1, 1e-2)
         end for i=1:r, j=1:r]

    dAV = Array(dA * V)
    dAU = Array(dA'* U)
    UdAV = U'* dAV
    ds = diag(UdAV)
    dU = U * (F .* (UdAV * S + S * UdAV')) + (dAV - U * UdAV) * Diagonal(safercp.(s))
    dV = V * (F .* (S * UdAV + UdAV' * S)) + (dAU - V * UdAV')* Diagonal(safercp.(s))

    U, dU, s, ds, V, dV
end

safercp(x, c=0.0, tol=1e-10, brd=1e-12) = begin
    # In fact this operation will sometimes cause big error.
    if abs(x) < tol
        return x / (x^2 + brd) + c
    else
        return 1. / x
    end
end

