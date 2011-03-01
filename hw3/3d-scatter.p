# Example script for generating a 3D scatterplot using gnuplot.
# Be sure to revise the title, labels, etc to appropriately describe your plot.
# The data in the clustn.txt files should be listed in 3 columns, one for 
# each attribute, for example:
#     inst1-att1-value inst1-att2-value inst1-att3-value
#     inst2-att1-value inst2-att2-value inst2-att3-value
#     inst3-att1-value inst3-att2-value inst3-att3-value
#     ...
# This script can be run with: gnuplot 3d-scatter.p

set terminal pdf
set output "output.pdf"
set title "HAC, max, 3 attributes, k=4" 
set xlabel "Age" 
set xlabel  offset character -3, -2, 0 font "" textcolor lt -1 norotate
set xrange [ 0 : 1 ] noreverse nowriteback
set ylabel "Education" 
set ylabel  offset character 3, -2, 0 font "" textcolor lt -1 rotate by 90
set yrange [ 0 : 1 ] noreverse nowriteback
set zlabel "Income" offset 1, 0 
set zrange [ 0 : 1 ] noreverse nowriteback
splot "clust1.dat" with points title "Cluster 1",\
      "clust2.dat" with points title "Cluster 2"
