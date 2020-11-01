using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Arpack

begin
    ElType = Float64
    χ2 = 20
    χc = 20

    A  = rand(χ2^2, χ2^2);
    ∂A = rand(χ2^2, χ2^2);

    Tx = LinearMap{ElType}(v -> A*v,
                           w -> A'w, χ2^2, χ2^2)
    # Adjoint maps.
    ∂Tx = LinearMap{ElType}(v -> ∂A*v,
                            w -> ∂A'w, χ2^2, χ2^2)

    # Value part.
    (Ux0_2isom, Sx_2, Ux1_2isom), = svds(Tx, nsv=χc)

    # SpGE→GE multiplication
    spmm(A, B) = Array(A * B)

    # Ux-part, diagonal part.
    begin
        local Ux0_2isom = reshape(Ux0_2isom, (χ2^2, χc))
        local Ux1_2isom = reshape(Ux1_2isom, (χ2^2, χc))
        local UdAVx = Ux0_2isom'spmm(∂Tx, Ux1_2isom)
        local Fij, UdAVSx, SUdAVx, US2x, VS2x, ∂LHS_ΓUx, ∂ΓUx, ∂LHS_ΠVx, ∂ΠVx
        ∂Sx = diag(UdAVx)
        Fij = [if i == j
                   0.0
               else
                   1.0 / (Sx_2[j]^2 - Sx_2[i]^2)
               end for i=1:χc, j=1:χc]
        UdAVSx = UdAVx * Diagonal(Sx_2)
        SUdAVx = Diagonal(Sx_2) * UdAVx
        ∂Ux0_2iso = Ux0_2isom * (Fij .*(UdAVSx + UdAVSx'))
        ∂Ux1_2iso = Ux1_2isom * (Fij .*(SUdAVx + SUdAVx'))
        US2x = Ux0_2isom * Diagonal(Sx_2.^2)
        VS2x = Ux1_2isom * Diagonal(Sx_2.^2)
        # U off-diagonal part.
        # LHS
        ∂LHS_ΓUx = spmm(∂Tx, Ux1_2isom) * Diagonal(Sx_2) + spmm(Tx, spmm(∂Tx', Ux0_2isom))
        ∂LHS_ΓUx -= Ux0_2isom * (UdAVSx + UdAVSx')
        # Trial: solution at U⊥'dAV⊥ = 0.
        ∂ΓUx = spmm(∂Tx, Ux1_2isom) * Diagonal(1.0 ./Sx_2) - Ux0_2isom * UdAVx * Diagonal(1.0 ./Sx_2)
        _, convUx = cg!(vec(∂ΓUx),
                        LinearMap{ElType}(∂v -> begin
                                              ∂Γ = reshape(∂v, (χ2^2, χc))
                                              vec(∂Γ*Diagonal(Sx_2.^2) - spmm(Tx, spmm(Tx', ∂Γ)) + US2x*Ux0_2isom'∂Γ)
                                          end, nothing, χ2^2*χc, χ2^2*χc),
                        vec(∂LHS_ΓUx), log=true)
        ∂Ux0_2iso += ∂ΓUx
        # V off-diagonal part.
        ∂LHS_ΠVx = spmm(∂Tx', Ux0_2isom) * Diagonal(Sx_2) + spmm(Tx', spmm(∂Tx, Ux1_2isom))
        ∂LHS_ΠVx -= Ux1_2isom * (SUdAVx + SUdAVx')
        ∂ΠVx = spmm(∂Tx', Ux0_2isom) * Diagonal(1.0 ./Sx_2) - Ux1_2isom * UdAVx'* Diagonal(1.0 ./Sx_2)
        _, convVx = cg!(vec(∂ΠVx),
                        LinearMap{ElType}(∂w -> begin
                                              ∂Π = reshape(∂w, (χ2^2, χc))
                                              vec(∂Π*Diagonal(Sx_2.^2) - spmm(Tx', spmm(Tx, ∂Π)) + VS2x*Ux1_2isom'∂Π)
                                          end, nothing, χ2^2*χc, χ2^2*χc),
                        vec(∂LHS_ΠVx), log=true)
        ∂Ux1_2iso += ∂ΠVx
    end

    ∂Ux0_2iso, ∂Sx, ∂Ux1_2iso
end

