# FIXME: add set terminal and plot to a file in png or eps formats
# Gnuplot script file for plotting data in file "some_file.dat"
set autoscale
#set logscale y
set title "Number of atoms Vs Wallclock time in seconds"
set xlabel "Number of atoms"
set ylabel "Wall clock time in seconds"

f(x) =  1.3804342597e+02 * x/8.

# set output "data.eps"
plot 'RELEASE_atoms_Vs_walltime.dat' using (column(1)):(column(2)) with lines title 'without', \
     'RELEASE_atoms_Vs_walltime.dat' using 1:3 with lines title 'with', \
      f(x) with lines title 'linear scaling'
