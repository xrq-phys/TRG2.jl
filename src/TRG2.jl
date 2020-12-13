module TRG2

using ForwardDiff
using LinearAlgebra
using TensorOperations
# using BliContractor
# using LinearMaps
using TensorKit
using IterativeSolvers
using Arpack

full_svd = true

inv_exp(x, kscal, tol=1e-8) = x^kscal / (x^(2*kscal) + tol)

telem_opr(f, T::TensorMap) = TensorMap(f.(convert(Array, T)), domain(T), codomain(T))

telem_exp(T::TensorMap, k) = telem_opr(x -> x^k, T)

tdiag(T::TensorMap) = diag(convert(Array, T))

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

bond_scale(Ux0::AbstractTensor,
           Ux1::AbstractTensor,
           Uy0::AbstractTensor,
           Uy1::AbstractTensor,
           Sx::AbstractTensorMap,
           Sy::AbstractTensorMap,
           kscal::Real) = begin

    # SV to be obsorbed.
    Sx_int = telem_exp(Sx, (1+kscal)/2)
    Sy_int = telem_exp(Sy, (1+kscal)/2)
    # SV to be left outside.
    Sx_ext = telem_opr(x -> inv_exp(x, kscal), Sx)
    Sy_ext = telem_opr(x -> inv_exp(x, kscal), Sy)

    @tensor Ux0[o1, o2, x] := Ux0[o1, o2, X] * Sx_int[X, x]
    @tensor Ux1[o1, o2, x] := Ux1[o1, o2, X] * Sx_int[X, x]
    @tensor Uy0[o1, o2, x] := Uy0[o1, o2, X] * Sy_int[X, x]
    @tensor Uy1[o1, o2, x] := Uy1[o1, o2, X] * Sy_int[X, x]

    # Return values are necessary now.
    Ux0, Ux1, Uy0, Uy1, Sx, Sy
end

bond_merge(Ux0::AbstractTensor,
           Ux1::AbstractTensor,
           Uy0::AbstractTensor,
           Uy1::AbstractTensor,
           Sx_inner::AbstractTensorMap,
           Sy_inner::AbstractTensorMap) = begin

    # Merge S.
    @tensor Ux0[o1, o2, x] := Sy_inner[o1, O1] * Ux0[O1, o2, x]
    @tensor Uy0[o1, o2, x] := Sx_inner[O1, o1] * Uy0[O1, o2, x]
    @tensor Ux1[o1, o2, x] := Sy_inner[o1, O1] * Ux1[O1, o2, x]
    @tensor Uy1[o1, o2, x] := Sx_inner[O1, o1] * Uy1[O1, o2, x]

    # Return all.
    Ux0, Ux1, Uy0, Uy1
end

bond_trg(Ux0::AbstractTensor,
         Ux1::AbstractTensor,
         Uy0::AbstractTensor,
         Uy1::AbstractTensor,
         Sx_outer::AbstractTensorMap,
         Sy_outer::AbstractTensorMap,
         χcut::Integer) = begin

    # Compute partition function - way 1.
    @tensor Zcll =
        Uy1'[bl, bu, d] * Uy0[br, bd, u] * Sy_outer[u, d] *
        Ux1'[bd, bl, r] * Ux0[bu, br, l] * Sx_outer[l, r]

    # if full_svd || size(Ux0)[3]^2 ≤ χcut + 2
        @tensor T[d, r, u, l] :=
            Uy1'[bl, bu, d] * Uy0[br, bd, u] *
            Ux1'[bd, bl, r] * Ux0[bu, br, l]
        Ux0_2, Sx_2, Ux1t_2 = tsvd(T, (1, 2), (3, 4), trunc=truncdim(χcut))
        Uy0_2, Sy_2, Uy1t_2 = tsvd(T, (4, 1), (2, 3), trunc=truncdim(χcut))
        # Operate on codomain.
        # This is more intuitive for TRG applications.
        Ux0_2 = permute(Ux0_2, (1, 2, 3))
        Uy0_2 = permute(Uy0_2, (1, 2, 3))
        Ux1_2 = permute(Ux1t_2', (1, 2, 3))
        Uy1_2 = permute(Uy1t_2', (1, 2, 3))

    #= else
        error("Sparse method unsupported by upstream at the moment. Waiting for Jutho's new masterpiece.")
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
                                  @tensor M2[bu, bd] := Uy0[br, bd, u] * (Ux0[bu, br, l] * M[u, l])
                                  @tensor M[d, r] = Uy1[bl, bu, d] * (Ux1[bd, bl, r] * M2[bu, bd])
                                  # contract!(Ux0, "URl", M, "ul", Utmp, "URu")
                                  # contract!(Uy0, "RDu", Utmp, "URu", Q, "UD")
                                  # contract!(Ux1, "DLr", Q, "UD", Utmp, "LUr")
                                  # contract!(Uy1, "LUd", Utmp, "LUr", M, "dr")
                                  # @tensor M[d, r] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[u, l]
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χd, χr))
                                  @tensor M2[bu, bd] := Uy1[bl, bu, d] * (Ux1[bd, bl, r] * M[d, r])
                                  @tensor M[u, l] = Uy0[br, bd, u] * (Ux0[bu, br, l] * M2[bu, bd])
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
                                  @tensor M2[bl, br] := Ux1[bd, bl, r] * (Uy0[br, bd, u] * M[r, u])
                                  @tensor M[l, d] = Uy1[bl, bu, d] * (Ux0[bu, br, l] * M2[bl, br])
                                  # contract!(Uy0, "RDu", M, "ru", Utmp, "RDr")
                                  # contract!(Ux1, "DLr", Utmp, "RDr", Q, "LR")
                                  # contract!(Ux0, "URl", Q, "LR", Utmp, "LUl")
                                  # contract!(Uy1, "LUd", Utmp, "LUl", M, "ld")
                                  # @tensor M[l, d] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[r, u]
                                  vec(M)
                              end,
                              w -> begin
                                  M = reshape(w, (χl, χd))
                                  @tensor M2[bl, br] := Uy1[bl, bu, d] * (Ux0[bu, br, l] * M[l, d])
                                  @tensor M[r, u] = Ux1[bd, bl, r] * (Uy0[br, bd, u] * M2[bl, br])
                                  # contract!(Ux0, "URl", M, "ld", Utmp, "URd")
                                  # contract!(Uy1, "LUd", Utmp, "URd", Q, "LR")
                                  # contract!(Uy0, "RDu", Q, "LR", Utmp, "DLu")
                                  # contract!(Ux1, "DLr", Utmp, "DLu", M, "ru")
                                  # @tensor M[r, u] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l] * M[l, d]
                                  vec(M)
                              end, χl*χd, χr*χu),
            nsv=χcut, tol=2e-4)
        # fix_sign!(Ux0_2, Sx_2, Ux1_2)
        # fix_sign!(Uy0_2, Sy_2, Uy1_2)

    end =#
    # Do not compute partition function - way 2.
    Zcur = convert(Array, Sx_2)[1]
    Sx_2 = Sx_2 / Zcur
    Sy_2 = Sy_2 / Zcur

    Zcll, Zcur, Ux0_2, Ux1_2, Uy0_2, Uy1_2, Sx_outer, Sy_outer, Sx_2, Sy_2

end

# include("derivative.jl")
include("ising.jl")

end

