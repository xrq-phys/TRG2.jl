#!/usr/bin/env gnuplot

set format x "10^{%L}"
set format y "10^{%L}"
set xlabel "Matrix Size"
set ylabel "Elapsed Time (ms)"
set logscale xy
set xrange [100:10000]
set yrange [1:100000]
set key r b

plot \
	"svd_s_skx_xeon_8260_full.dat"     u ($1**2):($2/10**6):($3/10**6) w errorlines tit "Full SVD"             lt 1, \
	"svd_s_skx_xeon_8260_mpartial.dat" u ($1**2):($2/10**6):($3/10**6) w errorlines tit "Partial SVD (Matrix)" lt 2, \
	"svd_s_skx_xeon_8260_sparse.dat"   u ($1**2):($2/10**6):($3/10**6) w errorlines tit "Partial SVD (MMul)"   lt 4, \
	x**(5.0/2)/10**6 w l lt -1, \
	x**(6.0/2)/10**6 w l lt  0

pause -1

