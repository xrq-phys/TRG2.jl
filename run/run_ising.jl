using TRG2
using LinearAlgebra
using TensorOperations

global χc = 40
# lo-temp phase -5.6165040e-8
# hi-temp phase -5.6165050e-8
# lo-temp phase -5.01530000e-8
# hi-temp phase -5.01530010e-8
global βc = log(1+sqrt(2))/2 - 5.01530000e-8;

# global χc = 80

global Zi = TRG2.zi_2Dising(1, βc);
global Ul, S0, Ur = svd(reshape(Zi, (4, 4)));
global kscal = 0.5

rmul!(Ur, Diagonal(S0.^((kscal+1)/2)));
rmul!(Ul, Diagonal(S0.^((kscal+1)/2)));
global S0b = S0.^(-kscal);
global Ul = reshape(Ul, (2, 2, 4));
global Ur = reshape(Ur, (2, 2, 4));

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

global logSx = [Sx; zeros(χc - length(Sx))];
global logSy = [Sy; zeros(χc - length(Sx))];
global logUx = zeros(1000); # vec(Ux0[1:4, 1:4, 1:8]);
global logFb = [sum(abs.(Ux0))];
TRG2.bond_scale!(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);

# Other loggings.
# global logSo = [So; zeros(χc^2 - length(So))];
global logZ = log(Zcur) / 2;
global logZll = [Zcll];

for i = 1:60
    Ux0_prev = Ux0
    TRG2.bond_merge!(Ux0, Ux1,
                     Uy0, Uy1,
                     Sx_in,
                     Sy_in);
    global (Zcll, Zcur,
            Ux0, Ux1,
            Uy0, Uy1,
            Sx_in, Sy_in,
            Sx, Sy,
            So) = TRG2.bond_trg(Ux0, Array(Ux1),
                                Uy0, Array(Uy1),
                                Sx,
                                Sy,
                                χc);
    Sx_real = [Sx; zeros(χc - length(Sx))]
    Sy_real = [Sy; zeros(χc - length(Sx))]
    global logSx = [logSx Sx_real];
    global logSy = [logSy Sy_real];
    global logUx = [logUx vec(Ux0[1:10, 1:10, 1:10])];
    global logFb = [logFb; sum(abs.(Ux0[1:10, 1:10, 1:10]))];
    TRG2.bond_scale!(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);

    # global logSo = [logSo [So; zeros(χc^2 - length(So))]];
    global logZ += log(Zcur) / 2^(i+1);
    global logZll = [logZll; Zcll];

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
    @show (i, Zcll, Zcur, exp(logZ))
end

