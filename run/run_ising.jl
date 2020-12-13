using TRG2
using TRG2: inv_exp
using TRG2: telem_exp, telem_opr, tdiag
using LinearAlgebra
using TensorOperations
using Arpack
using TensorKit
using DelimitedFiles

# global βc = 0.6
# @info "Low-temperature phase."

global χc = 40
# [ -5.0153000e-8, -5.0153001e-8 ]
# [ -5.6165040e-8, -5.6165050e-8 ]
global βc = log(1+sqrt(2))/2 - 5.6165040e-8;

# global χc = 64
# global βc = log(1+sqrt(2))/2;

Zi = TRG2.zi_2Dising(1, βc);
Uℤ = [ 1/2^0.5 -1/2^0.5;
       1/2^0.5  1/2^0.5 ];
@tensor Zℤ[u, l, d, r] := Zi[U, L, D, R] * Uℤ[U, u] * Uℤ[L, l] * Uℤ[D, d] * Uℤ[R, r];
𝕍₂ = ℤ₂Space(0=>1, 1=>1)
Zℤ = TensorMap(Zℤ, 𝕍₂ ⊗ 𝕍₂, 𝕍₂ ⊗ 𝕍₂)

global Ul, S0, Urt, = tsvd(Zℤ, (1, 2), (3, 4));
global Uu, S0, Udt, = tsvd(Zℤ, (4, 1), (2, 3));
# Operate all in codomain.
# This is more intuitive for TRG.
global Ul = permute(Ul, (1, 2, 3))
global Uu = permute(Uu, (1, 2, 3))
global Ur = permute(Urt', (1, 2, 3))
global Ud = permute(Udt', (1, 2, 3))
global kscal = 0.5
global nisoev = 20

S0i = telem_exp(S0, (kscal+1)/2)
@tensor Ur[o1, o2, x] := Ur[o1, o2, X] * S0i[X, x]
@tensor Ul[o1, o2, x] := Ul[o1, o2, X] * S0i[X, x]
@tensor Uu[o1, o2, x] := Uu[o1, o2, X] * S0i[X, x]
@tensor Ud[o1, o2, x] := Ud[o1, o2, X] * S0i[X, x]
global S0x = telem_opr(x -> inv_exp(x, kscal), S0)
global S0y = copy(S0x)
@tensor Tᵣ0[d, r, u, l] :=
    Ud'[bl, bu, d] * Uu[br, bd, u] *
    Ur'[bd, bl, r] * Ul[bu, br, l]
writedlm("T_Z2.0.dat", convert(Array, Tᵣ0))

begin
    global (Zcll, Zcur,
            Ux0, Ux1,
            Uy0, Uy1,
            Sx_in, Sy_in,
            Sx, Sy) =
        TRG2.bond_trg(Ul, Ur,
                      Uu, Ud,
                      S0x,
                      S0y,
                      χc);
    Ux0, Ux1, Uy0, Uy1 =
        TRG2.bond_merge(Ux0, Ux1,
                        Uy0, Uy1,
                        Sx_in,
                        Sy_in);
    @show (0, Zcll, Zcur)
end

global logSx = [tdiag(Sx); zeros(χc - dim(codomain(Sx)))];
global logSy = [tdiag(Sy); zeros(χc - dim(codomain(Sx)))];
global logUx = zeros(1000); # vec(Ux0[1:4, 1:4, 1:8]);
global logFb = [0.0];

# Other loggings.
global logZ = log(Zcur) / 2;
global logZll = [Zcll];

for i = 1:1
    #= Loop body: "invariants"
      Ux0, Ux1, Uy0, Uy1, Sx, Sx =>
      (S)  (S)
       \   /
        U-U
        | |
        U-U
       /   \
      (S)  (S)
     =#
    calc_critical_from_∂_simp = false
    calc_critical_from_∂_trad = i ≥ 0 && i < 20

    # Rescale weighted external bonds.
    Ux0, Ux1, Uy0, Uy1, Sx, Sy =
        TRG2.bond_scale(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);

    @tensor Tᵣ[d, r, u, l] :=
        Uy1'[bl, bu, d] * Uy0[br, bd, u] *
        Ux1'[bd, bl, r] * Ux0[bu, br, l]
    writedlm("T_Z2.$i.dat", convert(Array, Tᵣ))
    # RG forward.
    global (Zcll, Zcur,
            Ux0_2, Ux1_2,
            Uy0_2, Uy1_2,
            Sx_in, Sy_in,
            Sx_2, Sy_2) =
        TRG2.bond_trg(Ux0, Ux1,
                      Uy0, Uy1,
                      Sx,
                      Sy,
                      χc);

    # Isometric marks.
    umark = copy(vec(convert(Array, Ux0_2)[1:10, 1:10, 1:10]))

    # Weighted external bonds becomes inner now.
    # Merge.
    if calc_critical_from_∂_simp
        global Ux0_2_STEP3 = copy(Ux0_2)
        global Ux1_2_STEP3 = copy(Ux1_2)
        global Uy0_2_STEP3 = copy(Uy0_2)
        global Uy1_2_STEP3 = copy(Uy1_2)
        # Rigorously Sx_in should also be counted
        #  but skip for now.
        # Sx_2 is final output.
        #  not logging here.
    end
    Ux0_2, Ux1_2, Uy0_2, Uy1_2 =
        TRG2.bond_merge(Ux0_2, Ux1_2,
                        Uy0_2, Uy1_2,
                        Sx_in,
                        Sy_in);

    #= Cast sign mask.
    if size(Ux0_2) == size(Ux0) && i == 30
        Ux0_2 .*= sign.(Ux0_2) .* sign.(Ux0)
        Ux1_2 .*= sign.(Ux1_2) .* sign.(Ux1)
        Uy0_2 .*= sign.(Uy0_2) .* sign.(Uy0)
        Uy1_2 .*= sign.(Uy1_2) .* sign.(Uy1)
    end =#

    # For lines that might violate invariant rule, put them into blocks
    #  s.t. the whole block itself does not violate.
    begin
        global Ux0, Ux1 = Ux0_2, Ux1_2
        global Uy0, Uy1 = Uy0_2, Uy1_2
        global Sx, Sy = Sx_2, Sy_2
    end
    #= End of Loop body =#

    Sx_real = [tdiag(Sx); zeros(χc - dim(codomain(Sx)))]
    Sy_real = [tdiag(Sy); zeros(χc - dim(codomain(Sx)))]
    global logSx = [logSx Sx_real];
    global logSy = [logSy Sy_real];
    global logUx = [logUx umark];
    global logFb = [logFb; sqrt(umark'umark)];

    global logZ += log(Zcur) / 2^(i+1);
    global logZll = [logZll; Zcll];

    #= Rotate-"back"
    if i % 2 == 0
        global Uy1, Ux1, Uy0, Ux0 = Ux1, Uy0, Ux0, Uy1
        global Sx, Sy = Sy, Sx
        # S_in already merged. Not included in loop variables.
        # global Sx_in, Sy_in = Sy_in, Sx_in
    end =#

    # Compute 1-D transfer matrix.
    begin
        @tensor M1[r, l] :=
            Uy1'[bl, bu, d] * Uy0[br, bd, u] * Sy[u, d] *
            Ux1'[bd, bl, r] * Ux0[bu, br, l]

        # TRG-style √S-scaling.
        Sx_TRG = telem_exp(Sx * Zcur, 0.5)
        @tensor M1[r, L] := Sx_TRG[r, R] * (M1[R, l] * Sx_TRG[l, L])

        SM1, _ = eigs(convert(Array, M1), nev=9, ritzvec=false)
        @show real.(SM1)
        @show imag.(SM1)

        #=
        if calc_critical_from_∂_trad
            # TODO: Use LinearMap.
            @tensor T1[u, l, d, r] := Uy0[br, bd, u] * Ux0[bu, br, l] * Uy1[bl, bu, d] * Ux1[bd, bl, r]
            @tensor M2[l, L, r, R] := T1[u, l, d, r] * Diagonal(Sy*Zcur)[u, U] * T1[D, L, U, R] * Diagonal(Sy*Zcur)[d, D]
            @tensor M2[l, L, r, R] := M2[l0, L0, r0, R0] * SSx[l0, l] * SSx[L0, L] * SSx[r0, r] * SSx[R0, R]

            SM2, _ = eigs(reshape(M2, (χxc^2, χxc^2)), nev=9, ritzvec=false)
            @show real.(SM2)
            @show imag.(SM2)
        end
        =#
    end

    # Print
    @show (i, Zcll, Zcur, exp(logZ))
    flush(stdout)
    flush(stderr)
end

