//This is a test for 3 atoms problem with cell data transfer for rhs assembly optimization
//The test will check for the correctness of cell data transfer upon grid refinement

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <step_50.h>
#include <assert.h>

using namespace dealii;
using namespace Step50;

template <int dim>
class Test_LaplaceProblem : public Step50::LaplaceProblem<dim>
{
protected:
    using Step50::LaplaceProblem<dim>::read_lammps_input_file;
    using Step50::LaplaceProblem<dim>::refine_grid;
    using Step50::LaplaceProblem<dim>::setup_system;
    using Step50::LaplaceProblem<dim>::rhs_assembly_optimization;
    using Step50::LaplaceProblem<dim>::pack_function;
    using Step50::LaplaceProblem<dim>::unpack_function;
    using Step50::LaplaceProblem<dim>::prepare_for_coarsening_and_refinement;
    using Step50::LaplaceProblem<dim>::project_cell_data;
    using Step50::LaplaceProblem<dim>::grid_output_debug;
    using Step50::LaplaceProblem<dim>::assemble_system;
    using Step50::LaplaceProblem<dim>::assemble_multigrid;
    using Step50::LaplaceProblem<dim>::solve;

    const bool flag_rhs_assembly;
    const unsigned int degree;
    ParameterHandler &prm;
    unsigned int number_of_global_refinement , number_of_adaptive_refinement_cycles;
    double domain_size_left , domain_size_right;
    std::string Problemtype, PreconditionerType, LammpsInputFilename;
    double r_c, nonzero_density_radius_parameter;

public:
    Test_LaplaceProblem (const unsigned int Degree , ParameterHandler &prm,
                         const std::string &Problemtype, const std::string &PreconditionerType,
                         const std::string &LammpsInputFile, const double &domain_size_left,
                         const double &domain_size_right, const unsigned int &number_of_global_refinement,
                         const unsigned int &number_of_adaptive_refinement_cycles, const double &r_c,
                         const double &nonzero_density_radius_parameter, const bool & flag_rhs_assembly_)
        : Step50::LaplaceProblem<dim> (Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile,
                                       domain_size_left, domain_size_right, number_of_global_refinement,
                                       number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter,flag_rhs_assembly_),

          flag_rhs_assembly(flag_rhs_assembly_),
          degree(Degree),
          prm(prm),
          number_of_global_refinement(number_of_global_refinement),
          number_of_adaptive_refinement_cycles(number_of_adaptive_refinement_cycles),
          domain_size_left(domain_size_left),
          domain_size_right(domain_size_right),
          Problemtype(Problemtype),
          PreconditionerType(PreconditionerType),
          LammpsInputFilename(LammpsInputFile),
          r_c(r_c),
          nonzero_density_radius_parameter(nonzero_density_radius_parameter)
    { }

    void run ( );

};

template class Test_LaplaceProblem<2>;
template class Test_LaplaceProblem<3>;

template <int dim>
void Test_LaplaceProblem<dim>::run( )
{
    Step50::LaplaceProblem<dim>::pcout << "Dimension:	" << dim << std::endl;
    Timer timer;
    timer.start();
    Step50::LaplaceProblem<dim>::read_lammps_input_file(LammpsInputFilename);

    for (unsigned int cycle=0; cycle<number_of_adaptive_refinement_cycles; ++cycle)
    {
        Step50::LaplaceProblem<dim>::pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube (Step50::LaplaceProblem<dim>::triangulation,domain_size_left,domain_size_right);

            Step50::LaplaceProblem<dim>::triangulation.refine_global (number_of_global_refinement);
        }
        else
            Step50::LaplaceProblem<dim>::refine_grid ();

        Step50::LaplaceProblem<dim>::pcout << "   Number of active cells:       "<< Step50::LaplaceProblem<dim>::triangulation.n_global_active_cells() << std::endl;

        Step50::LaplaceProblem<dim>::setup_system ();

        Step50::LaplaceProblem<dim>::pcout << "   Number of degrees of freedom: " << Step50::LaplaceProblem<dim>::mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level=0; level<Step50::LaplaceProblem<dim>::triangulation.n_global_levels(); ++level)
            Step50::LaplaceProblem<dim>::pcout << Step50::LaplaceProblem<dim>::mg_dof_handler.n_dofs(level) << (level == Step50::LaplaceProblem<dim>::triangulation.n_global_levels()-1 ? ")" : ", ");
        Step50::LaplaceProblem<dim>::pcout << std::endl;

        if((cycle == 0) && (flag_rhs_assembly))
            Step50::LaplaceProblem<dim>::rhs_assembly_optimization( );

        if(dim == 2)
            grid_output_debug(cycle);

        Step50::LaplaceProblem<dim>::assemble_system();

        if(PreconditionerType == "GMG")
            Step50::LaplaceProblem<dim>::assemble_multigrid ();

        Step50::LaplaceProblem<dim>::solve ();

    }
    timer.stop();
//    Step50::LaplaceProblem<dim>::pcout << "\nElapsed wall time: " << timer.wall_time() << " seconds.\n"<<std::endl;
    timer.reset();
}

void check ()
{

    ParameterHandler prm;
    ParameterReader param(prm);
    param.declare_parameters();

    std::ostringstream oss;
    oss << "subsection Geometry" << std::endl
        <<"    set Number of global refinement = 1 "<< std::endl
        << "    set Domain limit left = 0" << std::endl
        << "    set Domain limit right = 1" << std::endl
        <<"end" <<std::endl
        <<"subsection Misc"<<std::endl
        << "    set Number of Adaptive Refinement = 3" << std::endl
        << "    set smoothing length = 0.1" << std::endl
        << "    set Nonzero Density radius parameter around each charge = 3.5" << std::endl
        <<"end"<<std::endl
        << "    set Polynomial degree = 1" << std::endl
        <<"subsection Solver input data"<<std::endl
        << "    set Preconditioner = GMG" << std::endl
        <<"end"<<std::endl
        <<"subsection Problem Selection"<<std::endl
        << "    set Problem = GaussianCharges" << std::endl
        << "    set Dimension = 2" << std::endl
        <<"end"<<std::endl
        <<"subsection Lammps data"<<std::endl
        << "  set Lammps input file = " << SOURCE_DIR << "/atom_3.data" << std::endl
        <<"end"<<std::endl;

    prm.parse_input_from_string(oss.str().c_str());

    prm.enter_subsection ("Geometry");
    unsigned int number_of_global_refinement =prm.get_integer("Number of global refinement");
    double domain_size_left     = prm.get_double ("Domain limit left");
    double domain_size_right     = prm.get_double ("Domain limit right");
    prm.leave_subsection ();

    prm.enter_subsection ("Misc");
    unsigned int number_of_adaptive_refinement_cycles      = prm.get_integer ("Number of Adaptive Refinement");
    double r_c = prm.get_double ("smoothing length");
    double nonzero_density_radius_parameter = prm.get_double("Nonzero Density radius parameter around each charge");
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

    //If rhs optimization needed turn flag to true
    bool flag_rhs_assembly = true;


    if (d == 2)
    {
        Test_LaplaceProblem<2> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter, flag_rhs_assembly);
        test_laplace_problem.run();
    }
    else if (d == 3)
    {
        Test_LaplaceProblem<3> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter, flag_rhs_assembly);
        test_laplace_problem.run();
    }
    else if (d != 2 && d != 3)
    {
        AssertThrow(false, ExcMessage("Only 2d and 3d dimensions are supported."));
    }

}

int main (int argc, char *argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 2);

    check ();

    return 0;
}


