#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <step_50.h>

using namespace dealii;

void check ()
{

  ParameterHandler prm;
  ParameterReader param(prm);
  param.declare_parameters();

  //Here create dynamically the list of all parameters required from the prm file for test purpose
  std::ostringstream oss;
  oss << "subsection Geometry" << std::endl
	 <<"    set Number of global refinement = 0 "<< std::endl
	<< "    set Domain limit left = 0" << std::endl
	<< "    set Domain limit right = 1" << std::endl
	<< "	set Mesh size = 0.25" << std::endl
	<< "	set Vacuum repetitions = 10" << std::endl
        <<"end" <<std::endl
       <<"subsection Misc"<<std::endl
	<< "    set Number of Adaptive Refinement = 4" << std::endl
        << "    set smoothing length = 0.5" << std::endl
	<< "    set Nonzero Density radius parameter around each charge = 3.5" << std::endl
	<< "	set Output and calculation of Analytical solution = false" << std::endl
	<< "	set Output of RHS field = false" << std::endl
	<< "	set Output of support of each atom = false" << std::endl
	<< "	set Flag for RHS evaluation optimization = true" << std::endl
	<< "	set Output time summary table = false" << std::endl
        <<"end"<<std::endl
        << "    set Polynomial degree = 1" << std::endl
        <<"subsection Solver input data"<<std::endl
        << "    set Preconditioner = GMG" << std::endl
        <<"end"<<std::endl
       <<"subsection Problem Selection"<<std::endl
        << "    set Problem = GaussianCharges" << std::endl
        << "    set Dimension = 3" << std::endl
	<< "	set Homogeneous Boundary Conditions = false" << std::endl
        <<"end"<<std::endl
       <<"subsection Lammps data"<<std::endl
      << "  set Lammps input file = " << SOURCE_DIR << "/atom_n1_2.data" << std::endl
      <<"end"<<std::endl;

  prm.parse_input_from_string(oss.str().c_str());

  prm.enter_subsection ("Geometry");
  unsigned int number_of_global_refinement =prm.get_integer("Number of global refinement");
  double domain_size_left     = prm.get_double ("Domain limit left");
  double domain_size_right     = prm.get_double ("Domain limit right");
  double mesh_size_h = prm.get_double ("Mesh size");
  unsigned int repetitions_for_vacuum = prm.get_integer ("Vacuum repetitions");
  prm.leave_subsection ();

  prm.enter_subsection ("Misc");
  unsigned int number_of_adaptive_refinement_cycles      = prm.get_integer ("Number of Adaptive Refinement");
  double r_c = prm.get_double ("smoothing length");
  double nonzero_density_radius_parameter = prm.get_double("Nonzero Density radius parameter around each charge");
  bool flag_analytical_solution = prm.get_bool ("Output and calculation of Analytical solution");
  bool flag_rhs_field = prm.get_bool ("Output of RHS field");
  bool flag_atoms_support = prm.get_bool ("Output of support of each atom");
  bool flag_rhs_assembly = prm.get_bool ("Flag for RHS evaluation optimization");
  const bool & flag_output_time = prm.get_bool ("Output time summary table");
  prm.leave_subsection ();

  const unsigned int Degree = prm.get_integer("Polynomial degree");

  prm.enter_subsection("Solver input data");
  std::string PreconditionerType = (prm.get("Preconditioner"));
  prm.leave_subsection();

  prm.enter_subsection("Problem Selection");
  std::string Problemtype= (prm.get("Problem"));
  const unsigned int d = prm.get_integer("Dimension");
  const bool & flag_boundary_conditions = prm.get_bool ("Homogeneous Boundary Conditions");
  prm.leave_subsection();

  prm.enter_subsection("Lammps data");
  std::string LammpsInputFile = (prm.get("Lammps input file"));
  prm.leave_subsection();

	  if (d == 2)
	  {
	      Step50::LaplaceProblem<2> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
						mesh_size_h, repetitions_for_vacuum, number_of_global_refinement, number_of_adaptive_refinement_cycles,
						r_c, nonzero_density_radius_parameter, flag_rhs_assembly, flag_analytical_solution, flag_rhs_field,
						flag_atoms_support, flag_boundary_conditions, flag_output_time);
	      laplace_problem.run ();
	  }
	  else if (d == 3)
	  {
	      Step50::LaplaceProblem<3> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
						mesh_size_h, repetitions_for_vacuum, number_of_global_refinement, number_of_adaptive_refinement_cycles,
						r_c, nonzero_density_radius_parameter, flag_rhs_assembly, flag_analytical_solution, flag_rhs_field,
						flag_atoms_support, flag_boundary_conditions, flag_output_time);
	      laplace_problem.run ();
	  }
          else if (d != 2 && d != 3)
          {
              AssertThrow(false, ExcMessage("Only 2d and 3d dimensions are supported."));
          }


}


int main (int argc, char *argv[])
{

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  check ();

  return 0;
}
