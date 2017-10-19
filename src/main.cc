
#include "step_50.h"
using namespace dealii;


int main (int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    try
    {

        //deallog.depth_console(3);

        AssertThrow(argc > 1, ExcMessage ("Invalid inputs"));

        std::string parameter_name (argv[1]);

        ParameterHandler prm;
        ParameterReader param(prm);
        param.read_parameters(parameter_name);

        prm.enter_subsection("Problem Selection");
        std::string Problemtype= (prm.get("Problem"));
        const int d = prm.get_integer("Dimension");    // set default to two in parameter class
        prm.leave_subsection();

        prm.enter_subsection("Solver input data");
        std::string PreconditionerType = (prm.get("Preconditioner"));
        prm.leave_subsection();

        prm.enter_subsection("Lammps data");
        std::string LammpsInputFile = (prm.get("Lammps input file"));
        prm.leave_subsection();

        const unsigned int Degree = prm.get_integer("Polynomial degree");

//        std::ostringstream oss;
//        oss << "set Lammps input file = " << SOURCE_DIR << "/atom_2.data" << std::endl;
//        prm.parse_input_from_string(oss.str().c_str());


        if (d == 2)
        {
           Step50::LaplaceProblem<2> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile);
           laplace_problem.run ();
        }
        else if (d == 3)
        {
            Step50::LaplaceProblem<3> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile);
            laplace_problem.run ();
        }
        else if (d != 2 && d != 3)
        {
            AssertThrow(false, ExcMessage("Only 2d and 3d dimensions are supported."));
        }


    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return 0;
}
