#include <step_50.h>
using namespace dealii;



int main (int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    try
    {        
        using namespace Step50;

        //deallog.depth_console(3);

        AssertThrow(argc > 1, ExcMessage ("Invalid inputs"));

        std::string parameter_name (argv[1]);

        ParameterHandler prm;
        ParameterReader param(prm);
        param.declare_parameters();
        param.read_parameters(parameter_name);

        prm.enter_subsection ("Geometry");
         unsigned int number_of_global_refinement =prm.get_integer("Number of global refinement");
        double domain_size_left     = prm.get_double ("Domain limit left");
        double domain_size_right     = prm.get_double ("Domain limit right");
        prm.leave_subsection ();

        prm.enter_subsection ("Misc");
        unsigned int number_of_adaptive_refinement_cycles      = prm.get_integer ("Number of Adaptive Refinement");
        double r_c = prm.get_double ("smoothing length");
        double nonzero_density_radius_parameter ;//= prm.get_double("Nonzero Density radius parameter around each charge");
        prm.leave_subsection ();

        const unsigned int Degree = prm.get_integer("Polynomial degree");

        prm.enter_subsection("Solver input data");
        std::string PreconditionerType = (prm.get("Preconditioner"));
        prm.leave_subsection();

        prm.enter_subsection("Problem Selection");
        std::string Problemtype= (prm.get("Problem"));
        const unsigned int d = prm.get_integer("Dimension");
        prm.leave_subsection();

        prm.enter_subsection("Lammps data");
        std::string LammpsInputFile = (prm.get("Lammps input file"));
        prm.leave_subsection();

        std::vector<double> r_c_variation {2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0};
        for(const auto & i : r_c_variation)
            {
                nonzero_density_radius_parameter = i;
                std::cout<<"cutoff radius: "<<nonzero_density_radius_parameter<<std::endl;

        if (d == 2)
        {
            LaplaceProblem<2> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                              number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
            laplace_problem.run ();
        }
        else if (d == 3)
        {
            LaplaceProblem<3> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                              number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
            laplace_problem.run ();
        }
        else if (d != 2 && d != 3)
        {
            AssertThrow(false, ExcMessage("Only 2d and 3d dimensions are supported."));
        }

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

