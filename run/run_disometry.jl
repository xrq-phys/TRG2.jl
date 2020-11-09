# Dual of RG loop.
#

run_bond_trg_diso(#= Scale-bond input. =#
                  Ux0_s::AbstractArray{ElType, 3}, ∂Ux0_s::AbstractArray{ElType, 3},
                  Ux1_s::AbstractArray{ElType, 3}, ∂Ux1_s::AbstractArray{ElType, 3},
                  Uy0_s::AbstractArray{ElType, 3}, ∂Uy0_s::AbstractArray{ElType, 3},
                  Uy1_s::AbstractArray{ElType, 3}, ∂Uy1_s::AbstractArray{ElType, 3},
                  Sx_outer::Vector{ElType}, ∂Sx_outer::Vector{ElType},
                  Sy_outer::Vector{ElType}, ∂Sy_outer::Vector{ElType},
                  kscal::Real,
                  #= Scale-bond output.
                   & Bond-weighted-TRG input. =#
                  Ux0_s_scaled::AbstractArray{ElType, 3},
                  Ux1_s_scaled::AbstractArray{ElType, 3},
                  Uy0_s_scaled::AbstractArray{ElType, 3},
                  Uy1_s_scaled::AbstractArray{ElType, 3},
                  Sx_outer_scaled::Vector{ElType},
                  Sy_outer_scaled::Vector{ElType},
                  #= Bond-TRG output.
                   & Bond-merging input. =#
                  Ux0_2::AbstractArray{ElType, 3},
                  Ux1_2::AbstractArray{ElType, 3},
                  Uy0_2::AbstractArray{ElType, 3},
                  Uy1_2::AbstractArray{ElType, 3},
                  #= Bond-TRG output.
                   & loop output ("invariant"). =#
                  Sx_2::Vector{ElType},
                  Sy_2::Vector{ElType},
                  #= Bond-merging output
                   & loop output ("invariant").
                   - Not used in this routine. =#) where {ElType<:Real} = begin
    χo, _, χk = size(Ux0)

    # Scale bonds - Derivatives.
    ∂Sx_scatter = (kscal+1)/2 * ∂Sx_outer .* TRG2.inv_exp.(Sx_outer, (1-kscal)/2)
    ∂Sy_scatter = (kscal+1)/2 * ∂Sy_outer .* TRG2.inv_exp.(Sy_outer, (1-kscal)/2)
    ((U, S, ∂U, ∂S) -> begin
         rmul!(reshape(∂U, (χo^2, χk)), Diagonal(S.^((1+kscal)/2)))
         reshape(∂U, (χo^2, χk)) .+= reshape(U, (χo^2, χk)) * Diagonal(∂S)
     end).([Ux0_s,       Ux1_s,       Uy0_s,       Uy1_s   ],
           [Sx_outer,    Sx_outer,    Sy_outer,    Sy_outer],
           [∂Ux0_s,      ∂Ux1_s,      ∂Uy0_s,      ∂Uy1_s     ],
           [∂Sx_scatter, ∂Sx_scatter, ∂Sy_scatter, ∂Sy_scatter])
    ∂Sx_outer .*= -kscal * TRG2.inv_exp.(Sx_outer, kscal+1)
    ∂Sy_outer .*= -kscal * TRG2.inv_exp.(Sy_outer, kscal+1)
    # Values part are not executed as it will modify input.
    # Correct place to execute this block: before run_bond_trg_diso entry,
    #   and feed 
    # ((U, S) -> begin
    #      rmul!(reshape(U, (χo^2, χk)), Diagonal(S.^((1+kscal)/2)))
    #  end).([Ux0_s,    Ux1_s,    Uy0_s,    Uy1_s   ],
    #        [Sx_outer, Sx_outer, Sy_outer, Sy_outer])
    # Sx_outer .= inv_exp.(Sx_outer, kscal)
    # Sy_outer .= inv_exp.(Sy_outer, kscal)
    Ux0_s, Ux1_s, Uy0_s, Uy1_s, Sx_outer, Sy_outer = (Ux0_s_scaled,
                                                      Ux1_s_scaled,
                                                      Uy0_s_scaled,
                                                      Uy1_s_scaled,
                                                      Sx_outer_scaled,
                                                      Sy_outer_scaled)

    # This is the only derivative that requires value-part output as
    #  derivative-part input.
    (∂Ux0_2,
     ∂Ux1_2,
     ∂Uy0_2,
     ∂Uy1_2,
     ∂Sx,
     ∂Sy) = TRG2.bond_trg_derivative(Array(Ux0_s), ∂Ux0_s,
                                     Array(Ux1_s), ∂Ux1_s,
                                     Array(Uy0_s), ∂Uy0_s,
                                     Array(Uy1_s), ∂Uy1_s,
                                     Sx_outer, ∂Sx_outer,
                                     Sy_outer, ∂Sy_outer,
                                     Ux0_2, Array(Ux1_2),
                                     Uy0_2, Array(Uy1_2),
                                     Sx_2,
                                     Sy_2)
    # Update bond dimensions.
    χo, _, χc = size(Ux0_2)

    # Merge S.
    ((U, S, ∂U, ∂S) -> begin
         lmul!(Diagonal(S), reshape(∂U, (χo, χo*χc)))
         reshape(∂U, (χo, χo*χc)) .+= Diagonal(∂S) * reshape(U, (χo, χo*χc))
     end).([Ux0_2,    Uy0_2,    Ux1_2,    Uy1_2   ],
           [Sy_outer, Sx_outer, Sy_outer, Sx_outer],
           [∂Ux0_2,   ∂Uy0_2,   ∂Ux1_2,   ∂Uy1_2   ],
           [∂Sy_outer,∂Sx_outer,∂Sy_outer,∂Sx_outer])
    # ((U, S) -> begin
    #      lmul!(Diagonal(S), reshape(U, (χo, χo*χc)))
    #  end).([Ux0_2,    Uy0_2,    Ux1_2,    Uy1_2   ], 
    #        [Sy_outer, Sx_outer, Sy_outer, Sx_outer])

    # RG-forwarded derivatives.
    (∂Ux0_2, ∂Ux1_2,
     ∂Uy0_2, ∂Uy1_2,
     ∂Sx,
     ∂Sy)
end

