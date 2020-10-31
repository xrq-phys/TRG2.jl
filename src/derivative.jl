
bond_trg_derivative(Ux0_s::Array{ElType, 3}, ∂Ux0_s::Array{ElType, 3}, # << This ∂ is imposed on S_innery Ux0. Not just Ux0.
                    Ux1_s::Array{ElType, 3}, ∂Ux1_s::Array{ElType, 3},
                    Uy0_s::Array{ElType, 3}, ∂Uy0_s::Array{ElType, 3},
                    Uy1_s::Array{ElType, 3}, ∂Uy1_s::Array{ElType, 3},
                    Sx_outer::Vector{ElType}, ∂Sx_outer::Vector{ElType},
                    Sy_outer::Vector{ElType}, ∂Sy_outer::Vector{ElType},
                    Ux0_2iso::Array{ElType, 3},
                    Ux1_2iso::Array{ElType, 3},
                    Uy0_2iso::Array{ElType, 3},
                    Uy1_2iso::Array{ElType, 3},
                    Sx_2::Vector{ElType},
                    Sy_2::Vector{ElType}) = begin

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
    Ux0_2isom = reshape(Ux0_2isom, (χ2^2, χc))
    Ux1_2isom = reshape(Ux1_2isom, (χ2^2, χc))
    UdAVx = Ux0_2isom'spmm(∂Tx, Ux1_2isom)
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

