#!/usr/bin/env gnuplot

set xrange [0:25]
set yrange [0:2.1]
set key t r

set arrow from 8.5,0 to 8.5,2.1 nohead

plot "trad40.dat" \
       u ($0):(-log($2)/2/pi+log($1)/2/pi) lt 1 tit "Should be 0.125", \
    "" u ($0):(-log($3)/2/pi+log($1)/2/pi) lt 2 tit "Should be 1.0", \
    "" u ($0):(-log($4)/2/pi+log($1)/2/pi) lt 3 tit "Should be 1.125", \
    "" u ($0):(-log($5)/2/pi+log($1)/2/pi) lt 3 notit, \
    "" u ($0):(-log(abs($6))/2/pi+log($1)/2/pi) lt 6 tit "Should be 2.0", \
    "" u ($0):(-log(abs($7))/2/pi+log($1)/2/pi) lt 6 notit, \
    "" u ($0):(-log(abs($8))/2/pi+log($1)/2/pi) lt 6 notit, \
    "" u ($0):(-log(abs($9))/2/pi+log($1)/2/pi) lt 6 notit, \
    "trad40_suppl.dat" \
       u ($0+0.5):(2*(-log($2)/2/pi+log($1)/2/pi)) lt 1 notit, \
    "" u ($0+0.5):(2*(-log($3)/2/pi+log($1)/2/pi)) lt 2 notit, \
    "" u ($0+0.5):(2*(-log($4)/2/pi+log($1)/2/pi)) lt 3 notit w lp, \
    "" u ($0+0.5):(2*(-log($5)/2/pi+log($1)/2/pi)) lt 3 notit, \
    "" u ($0+0.5):(2*(-log(abs($6))/2/pi+log($1)/2/pi)) lt 6 notit w lp, \
    "" u ($0+0.5):(2*(-log(abs($7))/2/pi+log($1)/2/pi)) lt 6 notit, \
    "" u ($0+0.5):(2*(-log(abs($8))/2/pi+log($1)/2/pi)) lt 6 notit, \
    "" u ($0+0.5):(2*(-log(abs($9))/2/pi+log($1)/2/pi)) lt 6 notit, \
    0.125 lt 0 notit, 1 lt 0 notit, 1.125 lt 0 notit, 2.0 lt 0 notit

pause -1;
