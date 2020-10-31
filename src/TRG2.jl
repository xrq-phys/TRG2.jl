module TRG2

using ForwardDiff
using LinearAlgebra
using TensorOperations
using BliContractor
using LinearMaps
using Arpack

full_svd = false

bond_trg(Ux0::Array{ElType, 3},
         Ux1::Array{ElType, 3},
         Uy0::Array{ElType, 3},
         Uy1::Array{ElType, 3},
         Sx_inner::Array{ElType},
         Sy_inner::Array{ElType},
         Sx_outer::Vector{ElType},
         Sy_outer::Vector{ElType},
         χcut::Integer,
         kscal::Real) where {ElType<:Real} = begin
    
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
    # S already applied. Make consistent input changes.
    (S -> begin
         if ndims(S) == 2
             S .= one(S)
         elseif ndims(S) == 1
             S .= 1
         end
     end).([Sy_inner, Sx_inner])

    # Compute partition function - way 1.
    χy, χx, χk = size(Ux0)
    Ux0_sc = reshape(reshape(Ux0, (χy*χx, χk)) * Diagonal(Sx_outer), (χy, χx, χk))
    Uy0_sc = reshape(reshape(Uy0, (χy*χx, χk)) * Diagonal(Sy_outer), (χy, χx, χk))
    Zcll = @tensor (Ux0_sc[U, L, x] * Ux1[D, R, x]) * (Uy0_sc[L, D, y] * Uy1[R, U, y])

    if full_svd || size(Ux0)[3]^2 ≤ χcut + 2
        # TODO: conj missing.
        @tensor T[d, r, u, l] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l]
        χd, χr, χu, χl = size(T)
        Ux0_2, Sx_2, Ux1_2 = svd(reshape(T, (χd*χr, χu*χl)))
        Uy0_2, Sy_2, Uy1_2 = svd(reshape(permutedims(T, (4, 1, 2, 3)), (χl*χd, χr*χu)))

        # Truncate
        if χd*χr > χcut
            Ux0_2 = Ux0_2[:, 1:χcut]
            Ux1_2 = Ux1_2[:, 1:χcut]
            Uy0_2 = Uy0_2[:, 1:χcut]
            Uy1_2 = Uy1_2[:, 1:χcut]

            # Reference S.
            Sref = Sx_2
            Sx_2 = Sx_2[1:χcut]
            Sy_2 = Sy_2[1:χcut]
        else
            Sref = Sx_2
            χcut = χd*χr
        end
    else
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
                                  @tensor M[bu, bd] := Uy0[br, bd, u] * (Ux0[bu, br, l] * M[u, l])
                                  @tensor M[d, r] := Uy1[bl, bu, d] * (Ux1[bd, bl, r] * M[bu, bd])
                                  # contract!(Ux0, "URl", M, "ul", Utmp, "URu")
                                  # contract!(Uy0, "RDu", Utmp, "URu", Q, "UD")
                                  # contract!(Ux1, "DLr", Q, "UD", Utmp, "LUr")
                                  # contract!(Uy1, "LUd", Utmp, "LUr", M, "dr")
                                  # @tensor M[d, r] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[u, l]
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χd, χr))
                                  @tensor M[bu, bd] := Uy1[bl, bu, d] * (Ux1[bd, bl, r] * M[d, r])
                                  @tensor M[u, l] := Uy0[br, bd, u] * (Ux0[bu, br, l] * M[bu, bd])
                                  # contract!(Ux1, "DLr", M, "dr", Utmp, "DLd")
                                  # contract!(Uy1, "LUd", Utmp, "DLd", Q, "UD")
                                  # contract!(Ux0, "URl", Q, "UD", Utmp, "RDl")
                                  # contract!(Uy0, "RDu", Utmp, "RDl", M, "ul")
                                  # @tensor M[u, l] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[d, r]
                                  vec(M)
                              end, χd*χr, χu*χl),
            nsv=χcut, tol=2e-4)
        (Uy0_2, Sy_2, Uy1_2), = svds(
            LinearMap{ElType}(v -> begin
                                  M = reshape(v, (χr, χu))
                                  @tensor M[bl, br] := Ux1[bd, bl, r] * (Uy0[br, bd, u] * M[r, u])
                                  @tensor M[l, d] := Uy1[bl, bu, d] * (Ux0[bu, br, l] * M[bl, br])
                                  # contract!(Uy0, "RDu", M, "ru", Utmp, "RDr")
                                  # contract!(Ux1, "DLr", Utmp, "RDr", Q, "LR")
                                  # contract!(Ux0, "URl", Q, "LR", Utmp, "LUl")
                                  # contract!(Uy1, "LUd", Utmp, "LUl", M, "ld")
                                  # @tensor M[l, d] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[r, u]
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χl, χd))
                                  @tensor M[bl, br] := Uy1[bl, bu, d] * (Ux0[bu, br, l] * M[l, d])
                                  @tensor M[r, u] := Ux1[bd, bl, r] * (Uy0[br, bd, u] * M[bl, br])
                                  # contract!(Ux0, "URl", M, "ld", Utmp, "URd")
                                  # contract!(Uy1, "LUd", Utmp, "URd", Q, "LR")
                                  # contract!(Uy0, "RDu", Q, "LR", Utmp, "DLu")
                                  # contract!(Ux1, "DLr", Utmp, "DLu", M, "ru")
                                  # @tensor M[r, u] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[l, d]
                                  vec(M)
                              end, χl*χd, χr*χu),
            nsv=χcut, tol=2e-4)

        Sref = [Sx_2; zeros(χl*χd)]
    end
    # Do not compute partition function - way 2.
    Zcur = Sx_2[1]
    Sx_2 ./= Zcur
    Sy_2 ./= Zcur

    rmul!(Ux0_2, Diagonal(Sx_2.^((1+kscal)/2)))
    rmul!(Ux1_2, Diagonal(Sx_2.^((1+kscal)/2)))
    rmul!(Uy0_2, Diagonal(Sy_2.^((1+kscal)/2)))
    rmul!(Uy1_2, Diagonal(Sy_2.^((1+kscal)/2)))
    Ux0_2 = reshape(Ux0_2, (χd, χr, χcut))
    Ux1_2 = reshape(Ux1_2, (χu, χl, χcut))
    Uy0_2 = reshape(Uy0_2, (χl, χd, χcut))
    Uy1_2 = reshape(Uy1_2, (χr, χu, χcut))
    Sx_2 .= (x -> x^kscal / (x^(2*kscal) + 1e-8)).(Sx_2)
    Sy_2 .= (x -> x^kscal / (x^(2*kscal) + 1e-8)).(Sy_2)

    Zcll, Zcur, Ux0_2, Ux1_2, Uy0_2, Uy1_2, Sx_outer, Sy_outer, Sx_2, Sy_2, Sref

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

include("ising.jl")

end

