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

  deallog << "DEBUG:" << SOURCE_DIR << std::endl;

  //Here create dynamically the list of all parameters required from the prm file for test purpose
  std::ostringstream oss;
  oss << "subsection Geometry" << std::endl
         <<"set Number of global refinement = 4 "<< std::endl
        << "set Domain limit left = 0.0" << std::endl
        << "set Domain limit right = 1.0" << std::endl
        <<"end" <<std::endl
       <<"subsection Misc"<<std::endl
        << "set Number of Adaptive Refinement = 5" << std::endl
        << "set smoothing length = 0.5" << std::endl
        << "set Nonzero Density radius parameter around each charge = 3" << std::endl
        <<"end"<<std::endl
        << "set Polynomial degree = 1" << std::endl
        <<"subsection Solver input data"<<std::endl
        << "set Preconditioner = GMG" << std::endl
        <<"end"<<std::endl
       <<"subsection Problem Selection"<<std::endl
        << "set Problem = Step16" << std::endl
        << "set Dimension = 3" << std::endl
        <<"end"<<std::endl
       <<"subsection Lammps data"<<std::endl
      << "set Lammps input file = " << SOURCE_DIR << "/atom_2.data" << std::endl
      <<"end"<<std::endl;

  prm.parse_input_from_string(oss.str().c_str());

}


int main ()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  check ();

  return 0;
}

