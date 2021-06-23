# Derivative of partial SVD.
#  will be used to refactor derivative.jl.
#
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Arpack

SupportedMap{T} = Union{Matrix{T}, LinearMap{T}}

∂svds(A ::SupportedMap{ElType},
      ∂A::SupportedMap{ElType}, χc) where {ElType<:Number} = begin
    # Compute partial SVD before differentiating.
    (U, Sx, V), = svds(A, nsv=χc)

    ∂svds(A, ∂A, U, Sx, V)
end

∂svds(A ::SupportedMap{ElType},
      ∂A::SupportedMap{ElType},
      U ::AbstractMatrix{ElType},
      Sx::AbstractVector{ElType},
      V ::AbstractMatrix{ElType}) where {ElType<:Number} = begin
    local χc = length(Sx)
    local χo, χi = size(A)
    size(U) == (χo, χc) || throw(DimensionMismatch("U has incorrect size."))
    size(V) == (χi, χc) || throw(DimensionMismatch("V has incorrect size."))

    # SpGE→GE multiplication
    spmm(A, B) = Array(A * B)

    local UdAV = U'spmm(∂A, V)
    local Fij
    local UdAVS, US2, ∂LHS_ΓU, ∂ΓU
    local SUdAV, VS2, ∂LHS_ΠV, ∂ΠV
    ∂Sx = diag(UdAV)
    Fij = [if i == j
               0.0
           else
               1.0 / (Sx[j]^2 - Sx[i]^2)
           end for i=1:χc, j=1:χc]
    UdAVS = UdAV * Diagonal(Sx)
    SUdAV = Diagonal(Sx) * UdAV
    ∂U = U * (Fij .*(UdAVS + UdAVS'))
    # ∂V = V * (Fij .*(SUdAV + SUdAV'))
    US2 = U * Diagonal(Sx.^2)
    VS2 = V * Diagonal(Sx.^2)

    # U off-diagonal part.
    # LHS
    ∂LHS_ΓU = spmm(∂A, V) * Diagonal(Sx) + spmm(A, spmm(∂A', U))
    ∂LHS_ΓU -= U * (UdAVS + UdAVS')
    # Trial: solution at U⊥'dAV⊥ = 0.
    ∂ΓU = spmm(∂A, V) * Diagonal(1.0 ./Sx) - U * UdAV * Diagonal(1.0 ./Sx)
    _, convUx = cg!(vec(∂ΓU),
                    LinearMap{ElType}(∂v -> begin
                                          ∂Γ = reshape(∂v, (χo, χc))
                                          vec(∂Γ*Diagonal(Sx.^2) - spmm(A, spmm(A', ∂Γ)) + US2*U'∂Γ)
                                      end, nothing, χo*χc, χo*χc),
                    vec(∂LHS_ΓU), log=true)
    @show convUx
    ∂U += ∂ΓU

    # V from ∂U computed.
    ∂V = (spmm(∂A', U) + spmm(A', ∂U) - V * Diagonal(∂Sx)) * Diagonal(Sx.^(-1))
    # V off-diagonal part.
    # ∂LHS_ΠV = spmm(∂A', U) * Diagonal(Sx) + spmm(A', spmm(∂A, V))
    # ∂LHS_ΠV -= V * (SUdAV + SUdAV')
    # ∂ΠV = spmm(∂A', U) * Diagonal(1.0 ./Sx) - V * UdAV'* Diagonal(1.0 ./Sx)
    # _, convVx = cg!(vec(∂ΠV),
    #                 LinearMap{ElType}(∂w -> begin
    #                                       ∂Π = reshape(∂w, (χi, χc))
    #                                       vec(∂Π*Diagonal(Sx.^2) - spmm(A', spmm(A, ∂Π)) + VS2*V'∂Π)
    #                                   end, nothing, χi*χc, χi*χc),
    #                 vec(∂LHS_ΠV), log=true)
    # @show convVx
    # ∂V += ∂ΠV

    U, ∂U, Sx, ∂Sx, V, ∂V
end

∂svd_(A, ∂A, U, S, V; info=missing) = begin
    χ1, χ2 = size(A)
    χs = length(S)
    local χc

    if ismissing(info)
        χc = min(χ1, χ2, 2*χs)
        (U, S, V), = svds(A, nsv=χc)
    else
        U, S, V = info
        χc = length(S)
    end
    _, dU, _, dS, _, dV = ∂svd(A, ∂A, U, S, V)
    (U[:, 1:χs], dU[:, 1:χs],
     S[1:χs], dS[1:χs],
     V[:, 1:χs], dV[:, 1:χs])
end

