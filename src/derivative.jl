# Derivative.
#

bond_trg_derivative(Ux0_s::Array{ElType, 3}, ∂Ux0_s::Array{ElType, 3}, # << This ∂ is imposed on S_innery Ux0. Not just Ux0.
                    Ux1_s::Array{ElType, 3}, ∂Ux1_s::Array{ElType, 3},
                    Uy0_s::Array{ElType, 3}, ∂Uy0_s::Array{ElType, 3},
                    Uy1_s::Array{ElType, 3}, ∂Uy1_s::Array{ElType, 3},
                    Sx_outer::Vector{ElType}, ∂Sx_outer::Vector{ElType}, # << Interface consistency. Not used in fact.
                    Sy_outer::Vector{ElType}, ∂Sy_outer::Vector{ElType},
                    Ux0_2iso::Array{ElType, 3},
                    Ux1_2iso::Array{ElType, 3},
                    Uy0_2iso::Array{ElType, 3},
                    Uy1_2iso::Array{ElType, 3},
                    Sx_2::Vector{ElType},
                    Sy_2::Vector{ElType}) where {ElType<:Real} = begin

    χ1, _, χk = size(Ux0_s)
    χ2, _, χc = size(Ux0_2iso)

    Tx = LinearMap{ElType}(v -> begin
                               M = reshape(v, (χ2, χ2))
                               @tensor M2[bu, bd] := Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * M[u, l])
                               @tensor M[d, r] = Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * M2[bu, bd])
                               vec(M)
                           end,
                           w -> begin
                               M = reshape(w, (χ2, χ2))
                               @tensor M2[bu, bd] := Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * M[d, r])
                               @tensor M[u, l] = Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * M2[bu, bd])
                               vec(M)
                           end, χ2^2, χ2^2)

    Ty = LinearMap{ElType}(v -> begin
                               M = reshape(v, (χ2, χ2))
                               @tensor M2[bl, br] := Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * M[r, u])
                               @tensor M[l, d] = Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * M2[bl, br])
                               vec(M)
                           end,
                           w -> begin
                               M = reshape(w, (χ2, χ2))
                               @tensor M2[bl, br] := Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * M[l, d])
                               @tensor M[r, u] = Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * M2[bl, br])
                               vec(M)
                           end, χ2^2, χ2^2)

    # Adjoint maps.
    ∂Tx = LinearMap{ElType}(v -> begin
                                M = reshape(v, (χ2, χ2))
                                @tensor M2[bu, bd] := Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * M[u, l])
                                @tensor ∂M2[bu, bd] := Uy0_s[br, bd, u] * (∂Ux0_s[bu, br, l] * M[u, l])
                                @tensor ∂M2[bu, bd] += ∂Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * M[u, l])
                                # M becomes ∂M_out now.
                                @tensor M[d, r] = Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * ∂M2[bu, bd])
                                @tensor M[d, r] += Uy1_s[bl, bu, d] * (∂Ux1_s[bd, bl, r] * M2[bu, bd])
                                @tensor M[d, r] += ∂Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * M2[bu, bd])
                                vec(M)
                            end,
                            w -> begin
                                M = reshape(w, (χ2, χ2))
                                @tensor M2[bu, bd] := Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * M[d, r])
                                @tensor ∂M2[bu, bd] := Uy1_s[bl, bu, d] * (∂Ux1_s[bd, bl, r] * M[d, r])
                                @tensor ∂M2[bu, bd] += ∂Uy1_s[bl, bu, d] * (Ux1_s[bd, bl, r] * M[d, r])
                                @tensor M[u, l] = Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * ∂M2[bu, bd])
                                @tensor M[u, l] += Uy0_s[br, bd, u] * (∂Ux0_s[bu, br, l] * M2[bu, bd])
                                @tensor M[u, l] += ∂Uy0_s[br, bd, u] * (Ux0_s[bu, br, l] * M2[bu, bd])
                                vec(M)
                            end, χ2^2, χ2^2)

    ∂Ty = LinearMap{ElType}(v -> begin
                                M = reshape(v, (χ2, χ2))
                                @tensor M2[bl, br] := Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * M[r, u])
                                @tensor ∂M2[bl, br] := Ux1_s[bd, bl, r] * (∂Uy0_s[br, bd, u] * M[r, u])
                                @tensor ∂M2[bl, br] += ∂Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * M[r, u])
                                @tensor M[l, d] = Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * ∂M2[bl, br])
                                @tensor M[l, d] += Uy1_s[bl, bu, d] * (∂Ux0_s[bu, br, l] * M2[bl, br])
                                @tensor M[l, d] += ∂Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * M2[bl, br])
                                vec(M)
                            end,
                            w -> begin
                                M = reshape(w, (χ2, χ2))
                                @tensor M2[bl, br] := Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * M[l, d])
                                @tensor ∂M2[bl, br] := Uy1_s[bl, bu, d] * (∂Ux0_s[bu, br, l] * M[l, d])
                                @tensor ∂M2[bl, br] += ∂Uy1_s[bl, bu, d] * (Ux0_s[bu, br, l] * M[l, d])
                                @tensor M[r, u] = Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * ∂M2[bl, br])
                                @tensor M[r, u] += Ux1_s[bd, bl, r] * (∂Uy0_s[br, bd, u] * M2[bl, br])
                                @tensor M[r, u] += ∂Ux1_s[bd, bl, r] * (Uy0_s[br, bd, u] * M2[bl, br])
                                vec(M)
                            end, χ2^2, χ2^2)

    # SpGE→GE multiplication
    spmm(A, B) = Array(A * B)

    # Ux-part, diagonal part.
    begin
        local Ux0_2isom = reshape(Ux0_2iso, (χ2^2, χc))
        local Ux1_2isom = reshape(Ux1_2iso, (χ2^2, χc))
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
                        vec(∂LHS_ΓUx), log=true, maxiter=800)
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
                        vec(∂LHS_ΠVx), log=true, maxiter=800)
        ∂Ux1_2iso += ∂ΠVx
        @show convUx
        @show convVx
        if !convUx.isconverged || !convVx.isconverged
            @show convUx.data[:resnorm][end]
            @show convVx.data[:resnorm][end]
        end
    end

    # Uy-part, diagonal part.
    begin
        local Uy0_2isom = reshape(Uy0_2iso, (χ2^2, χc))
        local Uy1_2isom = reshape(Uy1_2iso, (χ2^2, χc))
        local UdAVy = Uy0_2isom'spmm(∂Ty, Uy1_2isom)
        local Fij, UdAVSy, SUdAVy, US2y, VS2y, ∂LHS_ΓUy, ∂ΓUy, ∂LHS_ΠVy, ∂ΠVy
        ∂Sy = diag(UdAVy)
        Fij = [if i == j
                   0.0
               else
                   1.0 / (Sy_2[j]^2 - Sy_2[i]^2)
               end for i=1:χc, j=1:χc]
        UdAVSy = UdAVy * Diagonal(Sy_2)
        SUdAVy = Diagonal(Sy_2) * UdAVy
        ∂Uy0_2iso = Uy0_2isom * (Fij .*(UdAVSy + UdAVSy'))
        ∂Uy1_2iso = Uy1_2isom * (Fij .*(SUdAVy + SUdAVy'))
        US2y = Uy0_2isom * Diagonal(Sy_2.^2)
        VS2y = Uy1_2isom * Diagonal(Sy_2.^2)
        # U off-diagonal part.
        ∂LHS_ΓUy = spmm(∂Ty, Uy1_2isom) * Diagonal(Sy_2) + spmm(Ty, spmm(∂Ty', Uy0_2isom))
        ∂LHS_ΓUy -= Uy0_2isom * (UdAVSy + UdAVSy')
        ∂ΓUy = spmm(∂Ty, Uy1_2isom) * Diagonal(1.0 ./Sy_2) - Uy0_2isom * UdAVy * Diagonal(1.0 ./Sy_2)
        _, convUy = cg!(vec(∂ΓUy),
                        LinearMap{ElType}(∂v -> begin
                                              ∂Γ = reshape(∂v, (χ2^2, χc))
                                              vec(∂Γ*Diagonal(Sy_2.^2) - spmm(Ty, spmm(Ty', ∂Γ)) + US2y*Uy0_2isom'∂Γ)
                                          end, nothing, χ2^2*χc, χ2^2*χc),
                        vec(∂LHS_ΓUy), log=true, tol=1e-8, maxiter=800)
        ∂Uy0_2iso += ∂ΓUy
        # V off-diagonal part.
        ∂LHS_ΠVy = spmm(∂Ty', Uy0_2isom) * Diagonal(Sy_2) + spmm(Ty', spmm(∂Ty, Uy1_2isom))
        ∂LHS_ΠVy -= Uy1_2isom * (SUdAVy + SUdAVy')
        ∂ΠVy = spmm(∂Ty', Uy0_2isom) * Diagonal(1.0 ./Sy_2) - Uy1_2isom * UdAVy'* Diagonal(1.0 ./Sy_2)
        _, convVy = cg!(vec(∂ΠVy),
                        LinearMap{ElType}(∂w -> begin
                                              ∂Π = reshape(∂w, (χ2^2, χc))
                                              vec(∂Π*Diagonal(Sy_2.^2) - spmm(Ty', spmm(Ty, ∂Π)) + VS2y*Uy1_2isom'∂Π)
                                          end, nothing, χ2^2*χc, χ2^2*χc),
                        vec(∂LHS_ΠVy), log=true, tol=1e-8, maxiter=800)
        ∂Uy1_2iso += ∂ΠVy
        @show convUy
        @show convVy
        if !convUy.isconverged || !convVy.isconverged
            @show convUy.data[:resnorm][end]
            @show convVy.data[:resnorm][end]
        end
    end

    (reshape(∂Ux0_2iso, size(Ux0_2iso)),
     reshape(∂Ux1_2iso, size(Ux1_2iso)),
     reshape(∂Uy0_2iso, size(Uy0_2iso)),
     reshape(∂Uy1_2iso, size(Uy1_2iso)),
     ∂Sx,
     ∂Sy)
end

