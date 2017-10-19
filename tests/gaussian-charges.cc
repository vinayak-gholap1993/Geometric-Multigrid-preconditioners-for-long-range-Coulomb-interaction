#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <step_50.h>




void check ()
{
    /*
  ParameterHandler prm;
  ParameterReader param(prm);
  std::ifstream in(p);

  param.read_parameters(in);
  param.parse_input(in);
  */

  deallog << "DEBUG:" << SOURCE_DIR << std::endl;

  /*
  std::ostringstream oss;
  oss << "set Lammps input file = " << SOURCE_DIR << "/atom_2.data" << std::endl
      << "blablabla" << std::endl
      << "...." << std::endl;
  param.parse_input_from_string(oss.str().c_str());
  */
}


int main ()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  check ();

  return 0;
}
