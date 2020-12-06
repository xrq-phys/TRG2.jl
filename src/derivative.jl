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
                    Sy_2::Vector{ElType},
                    infox, infoy) where {ElType<:Real} = begin

    χ1, _, χk = size(Ux0_s)
    χ2, _, χs = size(Ux0_2iso)
    Ux0_2isom, Sx_2ext, Ux1_2isom = infox
    Uy0_2isom, Sy_2ext, Uy1_2isom = infoy
    _, χc = size(Ux0_2isom)

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
        local UdAVx = Ux0_2isom'spmm(∂Tx, Ux1_2isom)
        local Fij, UdAVSx, SUdAVx, US2x, VS2x, ∂LHS_ΓUx, ∂ΓUx, ∂LHS_ΠVx, ∂ΠVx
        ∂Sx = diag(UdAVx)[1:χs]
        Fij = [if i == j
                   0.0
               else
                   safercp(Sx_2ext[j]^2 - Sx_2ext[i]^2, 1.0, 1e-1, 1e-2)
               end for i=1:χc, j=1:χc]
        UdAVSx = UdAVx * Diagonal(Sx_2ext)
        SUdAVx = Diagonal(Sx_2ext) * UdAVx
        ∂Ux0_2iso = Ux0_2isom * (Fij .*(UdAVSx + UdAVSx'))[:, 1:χs]
        ∂Ux1_2iso = Ux1_2isom * (Fij .*(SUdAVx + SUdAVx'))[:, 1:χs]
    end

    # Uy-part, diagonal part.
    begin
        local UdAVy = Uy0_2isom'spmm(∂Ty, Uy1_2isom)
        local Fij, UdAVSy, SUdAVy, US2y, VS2y, ∂LHS_ΓUy, ∂ΓUy, ∂LHS_ΠVy, ∂ΠVy
        ∂Sy = diag(UdAVy)[1:χs]
        Fij = [if i == j
                   0.0
               else
                   safercp(Sy_2ext[j]^2 - Sy_2ext[i]^2, 1.0, 1e-1, 1e-2)
               end for i=1:χc, j=1:χc]
        UdAVSy = UdAVy * Diagonal(Sy_2ext)
        SUdAVy = Diagonal(Sy_2ext) * UdAVy
        ∂Uy0_2iso = Uy0_2isom * (Fij .*(UdAVSy + UdAVSy'))[:, 1:χs]
        ∂Uy1_2iso = Uy1_2isom * (Fij .*(SUdAVy + SUdAVy'))[:, 1:χs]
    end

    (reshape(∂Ux0_2iso, size(Ux0_2iso)),
     reshape(∂Ux1_2iso, size(Ux1_2iso)),
     reshape(∂Uy0_2iso, size(Uy0_2iso)),
     reshape(∂Uy1_2iso, size(Uy1_2iso)),
     ∂Sx,
     ∂Sy)
end

