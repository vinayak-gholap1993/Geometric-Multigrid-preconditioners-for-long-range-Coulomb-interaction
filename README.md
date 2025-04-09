Geometric multigrid preconditioners for the Poisson problem using the open-source C++ FEM library deal.II.

To apply geometric multigrid preconditioners to solve the Poisson problem using the atomistic-to-continuum approach. The Kelly-based error indicator
is used to perform h-adaptive refinement. The effect of different “smoothers” (Jacobi, Gauss-Seidel) is studied. A good preconditioner's (positive) influence on
the iterative solvers (e.g., Conjugate Gradient) is examined.

Tasks & Achievements:
▶️ Developed a C++ application code using the open-source FEM library deal.ii to evaluate the electrostatic potential in crystalline solids, employing an atomistic-to-continuum approach.
▶️ Computed electrostatic potential using an algorithmic structure that satisfies Poisson’s equation, incorporating charge smoothing via a Gaussian.
▶️ Determined source terms based on atomic charge distribution within the simulation domain.
▶️ Applied dipole expansion to provide the necessary boundary conditions for Poisson’s equation.
▶️ Highlighted the impact of selecting an effective preconditioner on iterative solvers for electrostatic problems.
▶️ Developed an optimized algorithm for numerical integration of charge density in atoms with local support, improving computational efficiency.
▶️ Achieved linear scaling of computation times by optimizing numerical integration for the Right-Hand-Side (RHS) vector of the problem.
▶️ Demonstrated computational efficiency and accuracy of the implemented algorithm through parallel computation using MPI, with convergence of total electrostatic energy.
▶️ Verified the method’s accuracy by comparing results against analytical solutions.
▶️ Conducted a study on the convergence of total electrostatic energy by varying vacuum size in the simulation lattice domain.
