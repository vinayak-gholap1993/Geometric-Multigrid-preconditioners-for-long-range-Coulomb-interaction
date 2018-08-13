#!/bin/bash

#Go to build directory
cd ~/Desktop/Project/Dealii-Project/_build/
cmake ..
make distclean
make debug
cd src/

#Run the executable with MPI and passing the 
#prm file
for natoms in 8 216
do
echo Running $natoms atoms

for numprocs in 3
do
echo Running with $numprocs mpi processes
mpirun -np ${numprocs} main ${natoms}_atom_test.prm > ${natoms}_atom_test.mpirun=${numprocs}.output

done
done
exit 0
