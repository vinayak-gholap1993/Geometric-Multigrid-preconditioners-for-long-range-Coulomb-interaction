# Gnuplot script file for plotting data in file "some_file.dat"
set autoscale
#set logscale y
set title "Error in charge denisities vs cutoff radius"
set xlabel "Cutoff radius for gaussian charges"
set ylabel "Absolute Error in total charge density"
set key outside
plot for [c=2:3] 'Total_charge_density_AbsErr_L2.dat' using 1:c with lines title columnheader

