import TRG2
using LinearAlgebra
using TensorOperations
global χc = 40
global βc = log(1+sqrt(2))/2 - 0e-4;
global Zi = TRG2.zi_2Dising(1, βc);
global Ul, S0, Ur = svd(reshape(Zi, (4, 4)));
global kscal = 0.5

rmul!(Ur, Diagonal(S0.^((kscal+1)/2)));
rmul!(Ul, Diagonal(S0.^((kscal+1)/2)));
global S0b = S0.^(-kscal);
global Ul = reshape(Ul, (2, 2, 4));
global Ur = reshape(Ur, (2, 2, 4));

global (Zcur,
        Ux0, Ux1,
        Uy0, Uy1,
        Sx_in, Sy_in,
        Sx, Sy,
        So) = TRG2.bond_trg(Ul, Array(Ur),
                            Ul, Array(Ur),
                            ones(2),
                            ones(2),
                            S0b,
                            S0b,
                            χc,
                            kscal);

# global logSo = [So; zeros(χc^2 - length(So))];
global logSx = [Sx; zeros(χc - length(Sx))];
global logSy = [Sy; zeros(χc - length(Sx))];
global logUx = vec(Ux0[1:4, 1:4, 1:8]);
global logNrm = [vec(Ux0)'vec(Ux0)];
global logZ = log(Zcur) / 2;

for i = 1:40
    Ux0_prev = Ux0
    global (Zcur,
            Ux0, Ux1,
            Uy0, Uy1,
            Sx_in, Sy_in,
            Sx, Sy,
            So) = TRG2.bond_trg(Ux0, Array(Ux1),
                                Uy0, Array(Uy1),
                                Sx_in,
                                Sy_in,
                                Sx,
                                Sy,
                                χc,
                                kscal);
    Sx_real = [(x -> x^(kscal^-1) / (x^(kscal^-1*2) + 1e-5)).(Sx); zeros(χc - length(Sx))]
    Sy_real = [(x -> x^(kscal^-1) / (x^(kscal^-1*2) + 1e-5)).(Sy); zeros(χc - length(Sx))]
    # global logSo = [logSo [So; zeros(χc^2 - length(So))]];
    global logSx = [logSx Sx_real];
    global logSy = [logSy Sy_real];
    global logUx = [logUx vec(Ux0[1:4, 1:4, 1:8])];
    global logNrm = [logNrm; vec(Ux0)'vec(Ux0)];
    global logZ += log(Zcur) / 2^(i+1);

    #= Gauge fixing
    if size(Ux0) == size(Ux0_prev) # && i > 14
        Uytrans, Uxtrans = TRG2.gauge_u(Ux0, Ux0_prev)
        # NOTE: Must use := as overriding exists
        @tensor Ux0[i, j, k] := Uytrans[i, I] * Ux0[I, j, k]
        @tensor Ux0[i, j, k] := Uxtrans[j, J] * Ux0[i, J, k]
        @tensor Ux1[i, j, k] := Uytrans[i, I] * Ux1[I, j, k]
        @tensor Ux1[i, j, k] := Uxtrans[j, J] * Ux1[i, J, k]
        @tensor Uy0[i, j, k] := Uxtrans[i, I] * Uy0[I, j, k]
        @tensor Uy0[i, j, k] := Uytrans[j, J] * Uy0[i, J, k]
        @tensor Uy1[i, j, k] := Uxtrans[i, I] * Uy1[I, j, k]
        @tensor Uy1[i, j, k] := Uytrans[j, J] * Uy1[i, J, k]
        global Sx_in = Uxtrans * Diagonal(Sx_in) * Uxtrans'
        global Sy_in = Uytrans * Diagonal(Sy_in) * Uytrans'
    end =#

    #= Rotate-"back"
    if i % 2 == 0
        global Uy1, Ux1, Uy0, Ux0 = Ux1, Uy0, Ux0, Uy1
        global Sx, Sy = Sy, Sx
        global Sx_in, Sy_in = Sy_in, Sx_in
    end =#

    # Print
    @show (i, Zcur, exp(logZ))
end

