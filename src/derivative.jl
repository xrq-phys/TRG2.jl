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

    # Ux-part.
    (∂Ux0_2iso, ∂Sx, ∂Ux1_2iso), = ∂lsvd(∂Tx, infox)

    # Uy-part.
    (∂Uy0_2iso, ∂Sy, ∂Uy1_2iso), = ∂lsvd(∂Ty, infoy)

    (reshape(∂Ux0_2iso, size(Ux0_2iso)),
     reshape(∂Ux1_2iso, size(Ux1_2iso)),
     reshape(∂Uy0_2iso, size(Uy0_2iso)),
     reshape(∂Uy1_2iso, size(Uy1_2iso)),
     ∂Sx,
     ∂Sy)
end

