# Gnuplot script file for plotting data in file "some_file.dat"
set autoscale
set logscale y
set xtic auto          # set xtics automatically
set ytic auto          # set ytics automatically
set title "Number of active cells Vs number of atoms"
set xlabel "Number of atoms"
set ylabel "Number of active cells"
#set style line 1 lt 1 linewidth 2 linecolor rgb "red"
#set style line 2 lt 1 linewidth 2 linecolor rgb "blue"
#set style line 3 lt 1 linewidth 2 linecolor rgb "green"
#set style line 4 lt 1 linewidth 2 linecolor rgb "black"
#set style line 5 lt 1 linewidth 2 linecolor rgb "cyan"
set key outside
plot for [c=2:7] 'ncells_per_atom.dat' using 1:c with lines title columnheader
