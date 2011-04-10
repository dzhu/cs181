set autoscale
unset log
unset label
set xtic auto
set ytic auto
set title "Number of hidden states vs Log Likelihood"
set xlabel "Number of hidden states"
set ylabel "log likelihood"
set terminal png size 800,600
set output '4civ_plot.png'
plot "4civ.dat" using 1:2 title '' with linespoints
