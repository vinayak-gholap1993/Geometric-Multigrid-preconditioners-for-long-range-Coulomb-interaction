#!/bin/bash
#   allocate 1 node with 40 CPU per node for 24 hours:
#PBS -l nodes=1:ppn=40,walltime=24:00:00
#   job name
#PBS -N Dealii-Project
#   stdout and stderr files:
#PBS -o $HOME/out.txt -e $HOME/err.txt
#   first non-empty non-comment line ends PBS options

# submit with: qsub <name>.sh
cd /home/woody/iwtm/iwtm003h/Dealii-Project/
module load openmpi/2.0.2-gcc
cd _build/src/
mpirun --npersocket 10 -np 20 main 8_atom_test.prm 2>&1 | tee _8_atom_test/toutput
mpirun --npersocket 10 -np 20 main 216_atom_test.prm 2>&1 | tee _216_atom_test/toutput

#mpirun --npersocket 10 -np 20 /_build/dft-mg C60_sd.prm 2>&1 | tee _C60_sd/toutput
