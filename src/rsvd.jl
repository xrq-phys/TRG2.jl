# Randomized SVD
#
using LinearAlgebra

lsvd(A; nsv=6, ncv=2*nsv) = begin
    m, n = size(A)
    ncv < min(m, n) || (ncv = min(m, n))
    nsv < ncv || throw(ArgumentError("Number of SV requested is too large."))

    Ql, = svd(Array(A*rand(n, ncv)))
    dAr = Array(Ql'A)
    Ur, sr, Vr = svd(dAr)

    # Return result as well as info tuple.
    (Ql*Ur[:, 1:nsv], sr[1:nsv], Vr[:, 1:nsv]), (Ql, Ur, sr, Vr)
end

rsvd(A; nsv=6, ncv=2*nsv) = begin
    m, n = size(A)
    ncv < min(m, n) || (ncv = min(m, n))
    nsv < ncv || throw(ArgumentError("Number of SV requested is too large."))

    Qr, = svd(Array(A'rand(m, ncv)))
    dAl = Array(A*Qr)
    Ul, sl, Vl = svd(dAl)

    (Ul[:, 1:nsv], sl[1:nsv], Qr*Vl[:, 1:nsv]), (Ul, sl, Vl, Qr)
end

∂lsvd(dA,
      lsvd_info::Tuple{AbstractMatrix{T},
                       AbstractMatrix{T},
                       AbstractVector{T},
                       AbstractMatrix{T}}; nsv=0) where {T} = begin
    Ql, Ur, sr, Vr = lsvd_info
    ncv = length(sr)
    nsv < ncv || throw(ArgumentError("Number of SV requested is too large."))
    nsv > 0 || (nsv = ncv)

    Ar #=Ql'A=# = Ur * Diagonal(sr) * Vr'
    dAr = Array(Ql'dA)
    Ur, dUr, sr, dsr, Vr, dVr = ∂svd(Ar, dAr, Ur, sr, Vr)
    (Ql*dUr[:, 1:nsv], dsr[1:nsv], dVr[:, 1:nsv]), nothing
end

∂rsvd(dA,
      rsvd_info::Tuple{AbstractMatrix{T},
                       AbstractVector{T},
                       AbstractMatrix{T},
                       AbstractMatrix{T}}; nsv=0) where {T} = begin
    Ul, sl, Vl, Qr = rsvd_info
    ncv = length(sl)
    nsv < ncv || throw(ArgumentError("Number of SV requested is too large."))
    nsv > 0 || (nsv = ncv)

    Al #=A*Qr=# = Ul * Diagonal(sl) * Vl'
    dAl = Array(dA*Qr)
    Ul, dUl, sl, dsl, Vl, dVl = ∂svd(Al, dAl, Ul, sl, Vl)
    (dUl[:, 1:nsv], dsl[1:nsv], Qr*dVl[:, 1:nsv]), nothing
end

