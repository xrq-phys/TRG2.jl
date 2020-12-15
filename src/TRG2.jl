module TRG2

using ForwardDiff
using LinearAlgebra
using TensorOperations
using BliContractor
using LinearMaps
using IterativeSolvers
using Arpack

full_svd = false

inv_exp(x, kscal, tol=1e-8) = x^kscal / (x^(2*kscal) + tol)

fix_sign!(U::Matrix{ElType},
          S::Vector{ElType},
          V) where {ElType<:Real} = begin
    χo, χc = size(U)
    for k=1:χc
        # Fix sign against largest component.
        if max(U[:, k]...) < 0.0
            U[:, k] *= -1.0
            V[:, k] *= -1.0
        end
    end
end

bond_scale!(Ux0::Array{ElType, 3},
            Ux1::AbstractArray{ElType, 3},
            Uy0::Array{ElType, 3},
            Uy1::AbstractArray{ElType, 3},
            Sx::Vector{ElType},
            Sy::Vector{ElType},
            kscal::Real) where {ElType<:Real} = begin
    χo, _, χc = size(Ux0)

    rmul!(reshape(Ux0, (χo^2, χc)), Diagonal(Sx.^((1+kscal)/2)))
    rmul!(reshape(Ux1, (χo^2, χc)), Diagonal(Sx.^((1+kscal)/2)))
    rmul!(reshape(Uy0, (χo^2, χc)), Diagonal(Sy.^((1+kscal)/2)))
    rmul!(reshape(Uy1, (χo^2, χc)), Diagonal(Sy.^((1+kscal)/2)))
    Sx .= inv_exp.(Sx, kscal)
    Sy .= inv_exp.(Sy, kscal)

    Ux0, Ux1, Uy0, Uy1, Sx, Sy
end

bond_merge!(Ux0::Array{ElType, 3},
            Ux1::AbstractArray{ElType, 3},
            Uy0::Array{ElType, 3},
            Uy1::AbstractArray{ElType, 3},
            Sx_inner::Vector{ElType},
            Sy_inner::Vector{ElType}) where {ElType<:Real} = begin

    # Merge S.
    ((U, S) -> begin
         χy, χx, χk = size(U)
         U = reshape(U, (χy, χx*χk))
         if ndims(S) == 2
             mul!(U, S, copy(U))
         elseif ndims(S) == 1
             lmul!(Diagonal(S), U)
         end
         nothing
     end).([Ux0, Uy0, Ux1, Uy1], [Sy_inner, Sx_inner, Sy_inner, Sx_inner])
    Ux0, Ux1, Uy0, Uy1
end

"bTRG from 4-leg tensor."
bond_trg(T::Array{ElType, 4},
         Sx_outer::Vector{ElType},
         Sy_outer::Vector{ElType},
         χcut::Integer) where {ElType<:Real} = begin
    χd, χr, χu, χl = size(T)
    # Partition function (Way 1).
    @tensor Zcll = T[d, r, u, l] *
        Diagonal(Sx_outer)[l, r] *
        Diagonal(Sy_outer)[u, d]

    # Select SVD implementation.
    svd_(M::AbstractMatrix{ElType}, χ) =
        if full_svd || size(M)[1] ≤ χ + 2
            Uful, Sful, Vful = svd(M)
            χ_ = min(χ, length(Sful))
            return Uful[:, 1:χ_], Sful[1:χ_], Vful[:, 1:χ_], Sful
        else
            (Us, Ss, Vs), = svds(M, nsv=χ)
            return Us, Ss, Vs, missing
        end

    Ux0_2, Sx_2, Ux1_2, Sref = svd_(reshape(T, (χd*χr, χu*χl)), χcut)
    Uy0_2, Sy_2, Uy1_2, = svd_(reshape(permutedims(T, (4, 1, 2, 3)), (χl*χd, χr*χu)),
                               χcut)
    χcut = length(Sx_2)

    # (Way 2)
    Zcur = Sx_2[1]
    Sx_2 ./= Zcur
    Sy_2 ./= Zcur

    Ux0_2 = reshape(Ux0_2, (χd, χr, χcut))
    Ux1_2 = reshape(Ux1_2, (χu, χl, χcut))
    Uy0_2 = reshape(Uy0_2, (χl, χd, χcut))
    Uy1_2 = reshape(Uy1_2, (χr, χu, χcut))

    Zcll, Zcur, Ux0_2, Ux1_2, Uy0_2, Uy1_2, Sx_outer, Sy_outer, Sx_2, Sy_2, Sref
end

"bTRG from tensor ring."
bond_trg(Ux0::Array{ElType, 3},
         Ux1::Array{ElType, 3},
         Uy0::Array{ElType, 3},
         Uy1::Array{ElType, 3},
         Sx_outer::Vector{ElType},
         Sy_outer::Vector{ElType},
         χcut::Integer) where {ElType<:Real} = begin

    if full_svd || size(Ux0)[3]^2 ≤ χcut + 2
        # TODO: conj missing.
        # NOTE: conj is considered and guaranteed by TensorKit.jl in `z2-tensor` branch.
        @tensor T[d, r, u, l] :=
            Uy1[bl, bu, d] * Ux1[bd, bl, r] *
            Uy0[br, bd, u] * Ux0[bu, br, l]
        return bond_trg(T, Sx_outer, Sy_outer, χcut)
    end

    # Compute partition function - way 1.
    χy, χx, χk = size(Ux0)
    Ux0_sc = reshape(reshape(Ux0, (χy*χx, χk)) * Diagonal(Sx_outer), (χy, χx, χk))
    Uy0_sc = reshape(reshape(Uy0, (χy*χx, χk)) * Diagonal(Sy_outer), (χy, χx, χk))
    @tensor Zcll =
        (Ux0_sc[U, L, x] * Ux1[D, R, x]) *
        (Uy0_sc[L, D, y] * Uy1[R, U, y])

    # Sparse case.
    begin
        _, _, χd = size(Uy1)
        _, _, χr = size(Ux1)
        _, _, χu = size(Uy0)
        _, _, χl = size(Ux0)
        Utmp = zeros(size(Ux0)...)
        Q = zeros(size(Ux0)[1:2]...)

        # TODO: conj missing.
        (Ux0_2, Sx_2, Ux1_2), = svds(
            LinearMap{ElType}(v -> begin
                                  M = reshape(v, (χu, χl))
                                  @tensor M[d, r] := Uy1[bl, bu, d] *(Ux1[bd, bl, r] *
                                                                      (Uy0[br, bd, u] *
                                                                       (Ux0[bu, br, l] * M[u, l])))
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χd, χr))
                                  @tensor M[u, l] := Uy1[bl, bu, d] *(Ux1[bd, bl, r] *
                                                                      (Uy0[br, bd, u] *
                                                                       (Ux0[bu, br, l] * M[d, r])))
                                  vec(M)
                              end, χd*χr, χu*χl),
            nsv=χcut, tol=2e-4)
        (Uy0_2, Sy_2, Uy1_2), = svds(
            LinearMap{ElType}(v -> begin
                                  M = reshape(v, (χr, χu))
                                  @tensor M[l, d] := Uy1[bl, bu, d] *(Ux1[bd, bl, r] *
                                                                      (Uy0[br, bd, u] *
                                                                       (Ux0[bu, br, l] * M[r, u])))
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χl, χd))
                                  @tensor M[r, u] := Uy1[bl, bu, d] *(Ux1[bd, bl, r] *
                                                                      (Uy0[br, bd, u] *
                                                                       (Ux0[bu, br, l] * M[l, d])))
                                  vec(M)
                              end, χl*χd, χr*χu),
            nsv=χcut, tol=2e-4)
        # fix_sign!(Ux0_2, Sx_2, Ux1_2)
        # fix_sign!(Uy0_2, Sy_2, Uy1_2)
    end
    # Do not compute partition function - way 2.
    Zcur = Sx_2[1]
    Sx_2 ./= Zcur
    Sy_2 ./= Zcur

    Ux0_2 = reshape(Ux0_2, (χd, χr, χcut))
    Ux1_2 = reshape(Ux1_2, (χu, χl, χcut))
    Uy0_2 = reshape(Uy0_2, (χl, χd, χcut))
    Uy1_2 = reshape(Uy1_2, (χr, χu, χcut))

    Zcll, Zcur, Ux0_2, Ux1_2, Uy0_2, Uy1_2, Sx_outer, Sy_outer, Sx_2, Sy_2, missing

end

gauge_u(Ucur::Array{ElType, 3},
        Uref::Array{ElType, 3}) where {ElType<:Real} = begin
    
    # Vertical direction.
    @tensor Uyenv[i, I] := Ucur[i, j, k] * conj(Uref[I, j, k])
    Uy, sy, Vy = svd(Uyenv)
    Uytrans = Vy * Uy'

    # Horizontal direction.
    @tensor Uxenv[j, J] := Ucur[i, j, k] * conj(Uref[i, J, k])
    Ux, sx, Vx = svd(Uxenv)
    Uxtrans = Vx * Ux'

    Uytrans, Uxtrans
end

include("derivative.jl")
include("ising.jl")

end

