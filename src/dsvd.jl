# Derivative of full SVD.
#  small-size fallback & reference.
#
using LinearAlgebra

∂svd(A::Matrix{T},
     dA::Matrix{T}) where {T} = begin
    # SVD for values of A.
    U, s, V = svd(A)

    ∂svd(A, dA, U, s, V)
end

∂svd(A::Matrix{T},
     dA::Matrix{T},
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
    Scol = repeat(s, outer=(1, r))
    F = safercp.(Scol'.^2 - Scol.^2)

    ds = diag(U' * dA * V)
    dU = U * (F .* (U' * dA * V * S + S * V' * dA' * U)) + (I - U * U') * dA * V * safercp.(S)
    dV = V * (F .* (S * U' * dA * V + V' * dA' * U * S)) + (I - V * V') * dA'* U * safercp.(S)

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

