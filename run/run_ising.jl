using TRG2
using LinearAlgebra
using TensorOperations
using LinearMaps
using Arpack
include("run_disometry.jl")

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
global nisoev = 20

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
    TRG2.bond_merge!(Ux0, Ux1,
                     Uy0, Uy1,
                     Sx_in,
                     Sy_in);
end

global logSx = [Sx; zeros(χc - length(Sx))];
global logSy = [Sy; zeros(χc - length(Sx))];
global logUx = zeros(1000); # vec(Ux0[1:4, 1:4, 1:8]);
global logFb = [0.0];
TRG2.bond_scale!(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);

# Other loggings.
# global logSo = [So; zeros(χc^2 - length(So))];
global logZ = log(Zcur) / 2;
global logZll = [Zcll];

for i = 1:60
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
    calc_critical_from_∂ = false

    # Rescale weighted external bonds.
    if calc_critical_from_∂
        # Backup input for derivatives.
        global Ux0_STEP1 = copy(Ux0)
        global Ux1_STEP1 = copy(Ux1)
        global Uy0_STEP1 = copy(Uy0)
        global Uy1_STEP1 = copy(Uy1)
        global Sx_STEP1 = copy(Sx)
        global Sy_STEP1 = copy(Sy)
    end
    TRG2.bond_scale!(Ux0, Ux1, Uy0, Uy1, Sx, Sy, kscal);

    # RG forward.
    if calc_critical_from_∂
        # TRG2.bond_trg is not a !-method.
        #  not copy required.
        global Ux0_STEP2 = Ux0
        global Ux1_STEP2 = Ux1
        global Uy0_STEP2 = Uy0
        global Uy1_STEP2 = Uy1
        global Sx_STEP2 = Sx
        global Sy_STEP2 = Sy
    end
    global (Zcll, Zcur,
            Ux0_2, Ux1_2,
            Uy0_2, Uy1_2,
            Sx_in, Sy_in,
            Sx_2, Sy_2,
            So) = TRG2.bond_trg(Ux0, Array(Ux1),
                                Uy0, Array(Uy1),
                                Sx,
                                Sy,
                                χc);

    # Isometric marks.
    umark = copy(vec(Ux0_2[1:10, 1:10, 1:10]))

    # Weighted external bonds becomes inner now.
    # Merge.
    if calc_critical_from_∂
        global Ux0_2_STEP3 = Ux0_2
        global Ux1_2_STEP3 = Ux1_2
        global Uy0_2_STEP3 = Uy0_2
        global Uy1_2_STEP3 = Uy1_2
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

    if calc_critical_from_∂
        # Vectorize several objects.
        idEndUx0 = size(Ux0_STEP1)
        idEndUx1 = IdxUx0+ size(Ux1_STEP1)
        idEndUy0 = IdxUx1+ size(Uy0_STEP1)
        idEndUy1 = IdxUy0+ size(Uy1_STEP1)
        idEndSx  = IdxUy1+ size(Sx_STEP1)
        idEndSy  = IdxSx + size(Sy_STEP1)
        Scriti,  = eigs(LinearMap{Float64}(v -> begin
                                               ∂Ux0 = resize(v[         1:idEndUx0], size(Ux0_STEP1))
                                               ∂Ux1 = resize(v[idEndUx0+1:idEndUx1], size(Ux1_STEP1))
                                               ∂Uy0 = resize(v[idEndUx1+1:idEndUy0], size(Uy0_STEP1))
                                               ∂Uy1 = resize(v[idEndUy0+1:idEndUy1], size(Uy1_STEP1))
                                               ∂Sx  = resize(v[idEndUy1+1:idEndSx ], size(Sx_STEP1))
                                               ∂Sy  = resize(v[idEndSx +1:idEndSy ], size(Sy_STEP1))
                                               # Compute derivative.
                                               (∂Ux0_2, ∂Ux1_2,
                                                ∂Uy0_2, ∂Uy1_2,
                                                ∂Sx_2,
                                                ∂Sy_2) = run_bond_trg_diso(# Step 1.
                                                                           Ux0_STEP1, ∂Ux0,
                                                                           Ux1_STEP1, ∂Ux1,
                                                                           Uy0_STEP1, ∂Uy0,
                                                                           Uy1_STEP1, ∂Uy1,
                                                                           Sx_STEP1, ∂Sx,
                                                                           Sy_STEP1, ∂Sy,
                                                                           kscal,
                                                                           # Step 2.
                                                                           Ux0_STEP2,
                                                                           Ux1_STEP2,
                                                                           Uy0_STEP2,
                                                                           Uy1_STEP2,
                                                                           Sx_STEP2,
                                                                           Sy_STEP2,
                                                                           # Step 3.
                                                                           Ux0_2_STEP3,
                                                                           Ux1_2_STEP3,
                                                                           Uy0_2_STEP3,
                                                                           Uy1_2_STEP3,
                                                                           Sx_2,
                                                                           Sy_2)
                                               [vec(∂Ux0_2);
                                                vec(∂Ux1_2);
                                                vec(∂Uy0_2);
                                                vec(∂Uy1_2);
                                                vec(∂Sx);
                                                vec(∂Sy)]
                                           end,
                                           nothing,
                                           idEndSy,
                                           idEndSy),
                        nev=nisoev, ritzvec=false);
        @show log.(Scriti)./(0.5*log(2))
    end

    # For lines that might violate invariant rule, put them into blocks
    #  s.t. the whole block itself does not violate.
    begin
        global Ux0, Ux1 = Ux0_2, Ux1_2
        global Uy0, Uy1 = Uy0_2, Uy1_2
        global Sx, Sy = Sx_2, Sy_2
    end
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

    # Print
    @show (i, Zcll, Zcur, exp(logZ))
end

