spn(i) = (i - 1.5) * 2

"""
    zi_2Dising(J2, β)

Generates partition tensor element for 2D Ising model.
```
     i
  j     l
     k
```
"""
zi_2Dising(J2, β) = begin
    Ip, Im = getint_ising()
    exp.((Ip.*J2 + Im).* β)
end


getint_ising() = begin
    Ip = reshape([(spn(i)*spn(j) + spn(k)*spn(l) for i=1:2, j=1:2, k=1:2, l=1:2)...], (2, 2, 2, 2))
    Im = reshape([(spn(j)*spn(k) + spn(l)*spn(i) for i=1:2, j=1:2, k=1:2, l=1:2)...], (2, 2, 2, 2))
    Ip, Im
end

