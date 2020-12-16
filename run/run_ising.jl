using TRG2
using TRG2: inv_exp
using LinearAlgebra
using TensorOperations
using LinearMaps
using Arpack
using DelimitedFiles
# include("run_disometry.jl")

global χc = 40
# [ -5.0153000e-8, -5.0153001e-8 ]
# [ -5.6165040e-8, -5.6165050e-8 ]
global βc = log(1+sqrt(2))/2 - 5.6165040e-8;
# @info "High-temperature phase."

# global χc = 64
# global βc = log(1+sqrt(2))/2;

# global χc = 16
# global βc = log(1+sqrt(2))/2 - 1.8e-5;

Zi = TRG2.zi_2Dising(1, βc);
Uℤ = [ 1/2^0.5 -1/2^0.5;
       1/2^0.5  1/2^0.5 ];
@tensor Zℤ[u, l, d, r] := Zi[U, L, D, R] * Uℤ[U, u] * Uℤ[L, l] * Uℤ[D, d] * Uℤ[R, r];
kill_zeros!(T) = broadcast!(x -> if abs(x)>1e-13
                                return x
                            else return 0.0
                            end, T, T)

global Ul, S0, Ur = svd(reshape(Zℤ, (4, 4)));
global kscal = 0.5
global nisoev = 9
kill_zeros!(S0)

rmul!(Ur, Diagonal(S0.^((kscal+1)/2)));
rmul!(Ul, Diagonal(S0.^((kscal+1)/2)));
global S0b = S0.^(-kscal);
global Ul = reshape(Ul, (2, 2, 4));
global Ur = reshape(Ur, (2, 2, 4));

begin
    global (Zcll, Zcur,
            Ux0, Ux1,
            Uy0, Uy1,
            Sx_in, Sy_in,
            Sx, Sy,
            So) = TRG2.bond_trg(Ul, Array(Ur),
                                Ul, Array(Ur),
                                S0b,
                                S0b,
                                χc);
    kill_zeros!(Sx)
    kill_zeros!(Sy)
    TRG2.bond_merge!(Ux0, Ux1,
                     Uy0, Uy1,
                     Sx_in,
                     Sy_in);
end

global logSx = [Sx; zeros(χc - length(Sx))];
global logSy = [Sy; zeros(χc - length(Sx))];
global logUx = zeros(1000); # vec(Ux0[1:4, 1:4, 1:8]);
global logFb = [0.0];

# Other loggings.
# global logSo = [So; zeros(χc^2 - length(So))];
global logZ = log(Zcur) / 2;
global logZll = [Zcll];

for i = 1:40
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
    calc_critical_from_∂_simp = i ≥ 3 && i < 8
    calc_critical_from_∂ = i ≥ 4 && i < 8
    calc_critical_from_∂_trad = i ≥ 0 && i < 20

    # Rescale weighted external bonds.
    TRG2.bond_scale!(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);
    if calc_critical_from_∂ || calc_critical_from_∂_simp
        global T_STEP2
        @tensor T_STEP2[d, r, u, l] := Uy1[bl, bu, d] * Ux1[bd, bl, r] * Uy0[br, bd, u] * Ux0[bu, br, l]
        @tensor T_1[d, r, u, l] := T_STEP2[D, R, U, L] *
            Diagonal(Sy)[D, d] * Diagonal(Sx)[R, r] *
            Diagonal(Sy)[U, u] * Diagonal(Sx)[L, l]
        writedlm("T.$i.dat", T_1)
    end

    # RG forward.
    global (Zcll, Zcur,
            Ux0_2, Ux1_2,
            Uy0_2, Uy1_2,
            Sx_in, Sy_in,
            Sx_2, Sy_2,
            So,
            infox, infoy) = TRG2.bond_trg(Ux0, Array(Ux1),
                                          Uy0, Array(Uy1),
                                          Sx,
                                          Sy,
                                          χc);

    # Isometric marks.
    umark = copy(vec(Ux0_2[1:10, 1:10, 1:10]))

    # Weighted external bonds becomes inner now.
    # Merge.
    if calc_critical_from_∂ || calc_critical_from_∂_simp
        global Ux0_2_STEP3 = copy(Ux0_2)
        global Ux1_2_STEP3 = copy(Ux1_2)
        global Uy0_2_STEP3 = copy(Uy0_2)
        global Uy1_2_STEP3 = copy(Uy1_2)
        # Rigorously Sx_in should also be counted
        #  but skip for now.
        # Sx_2 is final output.
        #  not logging here.
    end
    TRG2.bond_merge!(Ux0_2, Ux1_2,
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

    # Simplified computation of scaling dimension
    if calc_critical_from_∂_simp
        local Scriti
        local diffIsometry
        local T_2
        # Scale `U`s first.
        local Ux0_E, Ux1_E
        local Uy0_E, Uy1_E
        @tensor Ux0_E[o1, o2, x] := Ux0_2_STEP3[o1, o2, X] * Diagonal(Sx_2.^((1+kscal)/2))[X, x]
        @tensor Ux1_E[o1, o2, x] := Ux1_2_STEP3[o1, o2, X] * Diagonal(Sx_2.^((1+kscal)/2))[X, x]
        @tensor Uy0_E[o1, o2, x] := Uy0_2_STEP3[o1, o2, X] * Diagonal(Sy_2.^((1+kscal)/2))[X, x]
        @tensor Uy1_E[o1, o2, x] := Uy1_2_STEP3[o1, o2, X] * Diagonal(Sy_2.^((1+kscal)/2))[X, x]

        local Ux0ᵀQ, Ux1ᵀQ #<< This `ᵀ` indicates its legs arranged in CW instead of CCW.
        local Uy0ᵀQ, Uy1ᵀQ
        @tensor Ux0ᵀQ[o1, o2, x] := Ux0_2_STEP3[o1, o2, X] * Diagonal(inv_exp.(Sx_2, (1-kscal)/2))[X, x]
        @tensor Ux1ᵀQ[o1, o2, x] := Ux1_2_STEP3[o1, o2, X] * Diagonal(inv_exp.(Sx_2, (1-kscal)/2))[X, x]
        @tensor Uy0ᵀQ[o1, o2, x] := Uy0_2_STEP3[o1, o2, X] * Diagonal(inv_exp.(Sy_2, (1-kscal)/2))[X, x]
        @tensor Uy1ᵀQ[o1, o2, x] := Uy1_2_STEP3[o1, o2, X] * Diagonal(inv_exp.(Sy_2, (1-kscal)/2))[X, x]

        # Absorb inner bond weights.
        TRG2.bond_merge!(Ux0_E, Ux1_E,
                         Uy0_E, Uy1_E,
                         Sx_in,
                         Sy_in)

        local iiter = 0
        diffIsometry = LinearMap{Float64}(v -> begin
                                              dT = reshape(v, (χc, χc, χc, χc))
                                              @tensor dT[d, r, u, l] := dT[D, R, U, L] *
                                                  Diagonal(inv_exp.(Sx_in, 1))[d, D] *
                                                  Diagonal(inv_exp.(Sx_in, 1))[u, U] *
                                                  Diagonal(inv_exp.(Sy_in, 1))[r, R] *
                                                  Diagonal(inv_exp.(Sy_in, 1))[l, L]
                                              # TODO: Add conj.
                                              @tensor dUx0_E[Br, Bd, El] := dT[Br, Bd, u, l] * Ux1ᵀQ[u, l, El]
                                              @tensor dUx1_E[Bl, Bu, Er] := dT[d, r, Bl, Bu] * Ux0ᵀQ[d, r, Er]
                                              @tensor dUy0_E[Bd, Bl, Eu] := dT[Bl, r, u, Bd] * Uy1ᵀQ[r, u, Eu]
                                              @tensor dUy1_E[Bu, Br, Ed] := dT[d, Bu, Br, l] * Uy0ᵀQ[l, d, Ed]

                                              TRG2.bond_merge!(dUx0_E, dUx1_E,
                                                               dUy0_E, dUy1_E,
                                                               Sx_in,
                                                               Sy_in)
                                              @tensor dT2[d, r, u, l] :=
                                                  (( Uy1_E[Bu, Br, d] *dUx1_E[Bl, Bu, r]) *
                                                   ( Uy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l])) +
                                                  (( Uy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                   (dUy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l]))
                                                  # (( Uy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                  #  ( Uy0_E[Bd, Bl, u] *dUx0_E[Br, Bd, l])) +
                                                  # ((dUy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                  #  ( Uy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l]))
                                              @tensor dT2[d, r, u, l] := dT2[D, R, U, L] *
                                                  Diagonal(inv_exp.(Sx_2, kscal))[d, D] *
                                                  Diagonal(inv_exp.(Sx_2, kscal))[u, U] *
                                                  Diagonal(inv_exp.(Sy_2, kscal))[r, R] *
                                                  Diagonal(inv_exp.(Sy_2, kscal))[l, L] * Zcur^(-1)

                                              iiter += 1
                                              iiter % 10 != 0 || (@info "ARPACK step $iiter.")
                                              vec(dT2)
                                          end,
                                          nothing,
                                          χc^4,
                                          χc^4)
        @tensor T_1[d, r, u, l] := T_STEP2[D, R, U, L] *
            Diagonal(Sy_in)[D, d] * Diagonal(Sx_in)[R, r] *
            Diagonal(Sy_in)[U, u] * Diagonal(Sx_in)[L, l]
        T_2 = Array(diffIsometry * vec(T_1))
        writedlm("T2.$i.dat", reshape(T_2, size(T_STEP2)))
        Scriti, _ = eigs(LinearMap{Float64}(v -> begin
                                                w = Array(diffIsometry * v)
                                                w .* sign.(T_2) .* sign.(vec(T_STEP2))
                                            end, nothing, χc^4, χc^4), nev=nisoev, ritzvec=false, tol=1e-3, maxiter=30);
        @show Scriti
        @show abs.(Scriti)
    end

    if calc_critical_from_∂
        local ScritiAD
        local diffIsometry
        local Ux0_E = copy(Ux0_2_STEP3)
        local Ux1_E = copy(Ux1_2_STEP3)
        local Uy0_E = copy(Uy0_2_STEP3)
        local Uy1_E = copy(Uy1_2_STEP3)

        # Absorb inner bond weights into **unscaled** `U`s.
        TRG2.bond_merge!(Ux0_E, Ux1_E,
                         Uy0_E, Uy1_E,
                         Sx_in,
                         Sy_in)
        # Value of central tensor.
        @tensor T2[d, r, u, l] :=
            ((Uy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
             (Uy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l]))
        local iiter = 0
        diffIsometry = LinearMap{Float64}(v -> begin
                                              dT = reshape(v, (χc, χc, χc, χc))
                                              @tensor dT[d, r, u, l] := dT[D, R, U, L] *
                                                  Diagonal(inv_exp.(Sx_in, 1))[d, D] *
                                                  Diagonal(inv_exp.(Sx_in, 1))[u, U] *
                                                  Diagonal(inv_exp.(Sy_in, 1))[r, R] *
                                                  Diagonal(inv_exp.(Sy_in, 1))[l, L]

                                              # Compute derivative.
                                              (∂Ux0_2, ∂Ux1_2,
                                               ∂Uy0_2, ∂Uy1_2,
                                               ∂Sx_2,
                                               ∂Sy_2) = TRG2.bond_trg_derivative(# T without S_outer.
                                                                                 T_STEP2,
                                                                                 dT,
                                                                                 # Output of separations.
                                                                                 Ux0_2_STEP3,
                                                                                 Ux1_2_STEP3,
                                                                                 Uy0_2_STEP3,
                                                                                 Uy1_2_STEP3,
                                                                                 Sx_2,
                                                                                 Sy_2,
                                                                                 # This info is always necessary.
                                                                                 infox=infox,
                                                                                 infoy=infoy)
                                              # Merge into unscaled `∂U`s.
                                              TRG2.bond_merge!(∂Ux0_2, ∂Ux1_2,
                                                               ∂Uy0_2, ∂Uy1_2,
                                                               Sx_in,
                                                               Sy_in)
                                              @tensor dT2[d, r, u, l] :=
                                                  (( Uy1_E[Bu, Br, d] *∂Ux1_2[Bl, Bu, r]) *
                                                   ( Uy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l])) +
                                                  (( Uy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                   (∂Uy0_2[Bd, Bl, u] * Ux0_E[Br, Bd, l])) +
                                                  (( Uy1_E[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                   ( Uy0_E[Bd, Bl, u] *∂Ux0_2[Br, Bd, l])) +
                                                  ((∂Uy1_2[Bu, Br, d] * Ux1_E[Bl, Bu, r]) *
                                                   ( Uy0_E[Bd, Bl, u] * Ux0_E[Br, Bd, l]))
                                              # Both U and ∂U has their final leg unscaled.
                                              # Here multiply the original S without exponent.
                                              @tensor dT2[d, r, u, l] := 
                                                  (dT2[D, R, U, L] *
                                                      Diagonal( Sx_2)[d, D] * Diagonal( Sx_2)[u, U] *
                                                      Diagonal( Sy_2)[r, R] * Diagonal( Sy_2)[l, L]) +
                                                  (T2[D, R, U, L] *
                                                      Diagonal(∂Sx_2)[d, D] * Diagonal( Sx_2)[u, U] *
                                                      Diagonal( Sy_2)[r, R] * Diagonal( Sy_2)[l, L]) +
                                                  (T2[D, R, U, L] *
                                                      Diagonal( Sx_2)[d, D] * Diagonal( Sx_2)[u, U] *
                                                      Diagonal(∂Sy_2)[r, R] * Diagonal( Sy_2)[l, L]) +
                                                  (T2[D, R, U, L] *
                                                      Diagonal( Sx_2)[d, D] * Diagonal(∂Sx_2)[u, U] *
                                                      Diagonal( Sy_2)[r, R] * Diagonal( Sy_2)[l, L]) +
                                                  (T2[D, R, U, L] *
                                                      Diagonal( Sx_2)[d, D] * Diagonal( Sx_2)[u, U] *
                                                      Diagonal( Sy_2)[r, R] * Diagonal(∂Sy_2)[l, L])

                                              iiter += 1
                                              iiter % 10 != 0 || (@info "ARPACK step $iiter.")
                                              vec(dT2)
                                          end,
                                          nothing,
                                          χc^4,
                                          χc^4)
        @tensor T_1[d, r, u, l] := T_STEP2[D, R, U, L] *
            Diagonal(Sy_in)[D, d] * Diagonal(Sx_in)[R, r] *
            Diagonal(Sy_in)[U, u] * Diagonal(Sx_in)[L, l]
        T_2 = Array(diffIsometry * vec(T_1))
        writedlm("T2_AD.$i.dat", reshape(T_2, size(T_STEP2)))
        # ScritiAD, = eigs(LinearMap{Float64}(v -> begin
        #                                         w = Array(diffIsometry * v)
        #                                         w .* sign.(T_2) .* sign.(vec(T_STEP2))
        #                                     end, nothing, χc^4, χc^4),
        #                  nev=nisoev, ritzvec=false, tol=1e-2, maxiter=30);
        # @show ScritiAD
        # @show abs.(ScritiAD)
    end

    # For lines that might violate invariant rule, put them into blocks
    #  s.t. the whole block itself does not violate.
    begin
        global Ux0, Ux1 = Ux0_2, Ux1_2
        global Uy0, Uy1 = Uy0_2, Uy1_2
        global Sx, Sy = Sx_2, Sy_2
    end
    kill_zeros!(Sx)
    kill_zeros!(Sy)
    #= End of Loop body =#

    Sx_real = [Sx; zeros(χc - length(Sx))]
    Sy_real = [Sy; zeros(χc - length(Sx))]
    global logSx = [logSx Sx_real];
    global logSy = [logSy Sy_real];
    global logUx = [logUx umark];
    global logFb = [logFb; sqrt(umark'umark)];

    # global logSo = [logSo [So; zeros(χc^2 - length(So))]];
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
        @tensor M1[bd, bl, bu, br] := Uy0[br, bd, u] * Diagonal(Sy*Zcur)[u, d] * Uy1[bl, bu, d]
        χbd, χbl, χbu, χbr = size(M1)
        _, _, χxc = size(Ux0)
        M1 = reshape(M1, (χbd*χbl, χbu*χbr))
        M1 = reshape(Ux1, (χbd*χbl, χxc))' * M1 * reshape(Ux0, (χbu*χbr, χxc))
        SSx = Diagonal(sqrt.(Sx*Zcur))
        lmul!(SSx, M1)
        rmul!(M1, SSx)

        SM1, _ = eigs(M1, nev=9, ritzvec=false)
        @show real.(SM1)
        @show imag.(SM1)

        if calc_critical_from_∂_trad
            # TODO: Use LinearMap.
            @tensor T1[u, l, d, r] := Uy0[br, bd, u] * Ux0[bu, br, l] * Uy1[bl, bu, d] * Ux1[bd, bl, r]
            @tensor M2[l, L, r, R] := T1[u, l, d, r] * Diagonal(Sy*Zcur)[u, U] * T1[D, L, U, R] * Diagonal(Sy*Zcur)[d, D]
            @tensor M2[l, L, r, R] := M2[l0, L0, r0, R0] * SSx[l0, l] * SSx[L0, L] * SSx[r0, r] * SSx[R0, R]

            SM2, _ = eigs(reshape(M2, (χxc^2, χxc^2)), nev=9, ritzvec=false)
            @show real.(SM2)
            @show imag.(SM2)
        end
    end

    # Print
    @show (i, Zcll, Zcur, exp(logZ))
    flush(stdout)
    flush(stderr)
end

