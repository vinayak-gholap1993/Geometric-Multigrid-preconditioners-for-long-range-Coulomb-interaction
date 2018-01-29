# Gnuplot script file for plotting data in file "some_file.dat"
set autoscale
#set logscale y
set title "Number of atoms Vs Wallclock time for RHS assembly in seconds"
set xlabel "Number of atoms"
set ylabel "RHS Assembly Wall clock time in seconds"
set key outside
plot for [c=2:4] 'RHS_assembly_atoms_Vs_walltime.dat' using 1:c with lines title columnheader
