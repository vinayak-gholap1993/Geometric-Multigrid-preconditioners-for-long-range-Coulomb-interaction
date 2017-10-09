
#include "step-50.h"
using namespace dealii;


class ParameterReader: public Subscriptor
{
public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);

private:
    void declare_parameters();
    ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    :
    prm(paramhandler)
{}

void ParameterReader::declare_parameters()
{
    prm.enter_subsection("Geometry");
    {
        prm.declare_entry("Number of global refinement","2",Patterns::Integer(),
                          "The uniform global mesh refinement on the Domain in the power of 4");

        prm.declare_entry("Domain limit left","-1",Patterns::Double(),
                          "Left limit of domain");

        prm.declare_entry("Domain limit right","1",Patterns::Double(),
                          "Right limit of domain");
    }
    prm.leave_subsection();


    prm.enter_subsection("Problem Selection");
    {
        prm.declare_entry ("Problem","Step16",Patterns::Selection("Step16 | GaussianCharges"),
                           "Problem definition for RHS Function");

        prm.declare_entry ("Dimension", "2", Patterns::Integer(), "Problem space dimension");
    }
    prm.leave_subsection();

    prm.enter_subsection("Misc");
    {
        prm.declare_entry ("Number of Adaptive Refinement","2",Patterns::Integer(),
                           "Number of Adaptive refinement cycles to be done");

        prm.declare_entry ("smoothing length", "0.5", Patterns::Double(),
                           "The smoothing length parameter for each Gaussian atom");

        prm.declare_entry ("Nonzero Density radius parameter around each charge","3",Patterns::Double(),
                           "Set the parameter to localize the density around each charge where it is nonzero");
    }
    prm.leave_subsection();

    prm.declare_entry("Polynomial degree", "1", Patterns::Integer(),
                      "Polynomial degree of finite elements");

    prm.enter_subsection("Solver input data");
    {
        prm.declare_entry ("Preconditioner","GMG",Patterns::Selection("GMG | Jacobi"),
                           "Preconditioner type to be applied to the system matrix");
    }
    prm.leave_subsection();

    prm.enter_subsection("Lammps data");
    {
        prm.declare_entry ("Lammps input file","atom_8.data",Patterns::Anything(),
                           "Lammps input file with atoms, charges and positions");
    }
    prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string &parameter_file)
{
    declare_parameters();
    prm.parse_input(parameter_file);
}



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

        std::ostringstream oss;
        oss << "set Lammps input file = " << SOURCE_DIR << "/_build/atom_2.data" << std::endl;
        prm.parse_input_from_string(oss.str().c_str());


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
