#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <step_50.h>

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
    using Step50::LaplaceProblem<dim>::assemble_system;

    ConditionalOStream                        pcout;
    parallel::distributed::Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>    mg_dof_handler;

    typedef LA::MPI::SparseMatrix matrix_t;
    typedef LA::MPI::Vector vector_t;

    matrix_t system_matrix;

    IndexSet locally_relevant_set;

    ConstraintMatrix     hanging_node_constraints;
    ConstraintMatrix     constraints;

    vector_t       solution;
    vector_t       system_rhs;

    const unsigned int degree;

    ParameterHandler &prm;

    unsigned int number_of_global_refinement , number_of_adaptive_refinement_cycles;
    double domain_size_left , domain_size_right;
    std::string Problemtype, PreconditionerType, LammpsInputFilename;
    std::shared_ptr<Function<dim>> rhs_func;
    std::shared_ptr<Function<dim>> coeff_func;
    bool lammpsinput;
    unsigned int number_of_atoms;
    std::vector<Point<dim> > atom_positions;
    unsigned int * atom_types;
    double * charges;
    double r_c, nonzero_density_radius_parameter;

    typedef typename parallel::distributed::Triangulation<dim>::cell_iterator cell_it;
    std::map<cell_it, std::set<unsigned int> > charges_list_for_each_cell;

  public:
    Test_LaplaceProblem (const unsigned int Degree , ParameterHandler &prm,
                         std::string &Problemtype, std::string &PreconditionerType, std::string &LammpsInputFile,
                         double &domain_size_left, double &domain_size_right, unsigned int &number_of_global_refinement, unsigned int &number_of_adaptive_refinement_cycles,
                         double &r_c, double &nonzero_density_radius_parameter) : Step50::LaplaceProblem<dim> (Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile,
                                                                                                               domain_size_left, domain_size_right, number_of_global_refinement,
                                                                                                               number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter),
        pcout (std::cout,(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)),
        triangulation (MPI_COMM_WORLD,Triangulation<dim>::
                      limit_level_difference_at_vertices,
                      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
        fe (Degree),
        mg_dof_handler (triangulation),
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
    {
        pcout<<"Problem type is:   " << Problemtype<<std::endl;

        if (Problemtype == "Step16")
        {
            rhs_func   = std::make_shared<Step16::RightHandSide<dim>>();
            coeff_func = std::make_shared<Step16::Coefficient<dim>>();
        }
        if(Problemtype == "GaussianCharges")
        {
            rhs_func   = std::make_shared<GaussianCharges::RightHandSide<dim>>();
            coeff_func = std::make_shared<GaussianCharges::Coefficient<dim>>();
        }


    }

    void run ();

};

template class Test_LaplaceProblem<2>;
template class Test_LaplaceProblem<3>;

template <int dim>
void Test_LaplaceProblem<dim>::run()
{
    Timer timer;
    Step50::LaplaceProblem<dim>::read_lammps_input_file(LammpsInputFilename);
    for (unsigned int cycle=0; cycle<number_of_adaptive_refinement_cycles; ++cycle)
        // first mesh size 4^2 = 16*16*16 and then 2 refinements
    {
        timer.start();

        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation,domain_size_left,domain_size_right);

            triangulation.refine_global (number_of_global_refinement);  //eg. first mesh size 4^2 = 16*16*16
        }
        else
            Step50::LaplaceProblem<dim>::refine_grid ();

        pcout << "   Number of active cells:       "<< triangulation.n_global_active_cells() << std::endl;

        Step50::LaplaceProblem<dim>::setup_system ();

        pcout << "   Number of degrees of freedom: " << mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
            pcout << mg_dof_handler.n_dofs(level) << (level == triangulation.n_global_levels()-1 ? ")" : ", ");
        pcout << std::endl;

        Step50::LaplaceProblem<dim>::rhs_assembly_optimization(atom_positions);
        Step50::LaplaceProblem<dim>::assemble_system (atom_positions, charges, charges_list_for_each_cell);

        timer.stop();
        //std::cout << "   Elapsed CPU time: " << timer() << " seconds."<<std::endl;
        //std::cout << "   Elapsed wall time: " << timer.wall_time() << " seconds."<<std::endl;
        timer.reset();

        // Print the charges densities i.e. system rhs norms to compare with rhs optimization
        pcout << "   L2 rhs norm " << std::setprecision(10) << std::scientific << system_rhs.l2_norm() << std::endl;
        pcout << "   LInfinity rhs norm " << std::setprecision(10) << std::scientific << system_rhs.linfty_norm() << std::endl;
    }
}

void check ()
{
  ParameterHandler prm;
  ParameterReader param(prm);
  param.declare_parameters();

  //Here create dynamically the list of all parameters required from the prm file for test purpose
  std::ostringstream oss;
  oss << "subsection Geometry" << std::endl
         <<"    set Number of global refinement = 4 "<< std::endl
        << "    set Domain limit left = -2.5" << std::endl
        << "    set Domain limit right = 2.5" << std::endl
        <<"end" <<std::endl
       <<"subsection Misc"<<std::endl
        << "    set Number of Adaptive Refinement = 1" << std::endl
        << "    set smoothing length = 0.5" << std::endl
//        << "    set Nonzero Density radius parameter around each charge = 3" << std::endl
        <<"end"<<std::endl
        << "    set Polynomial degree = 1" << std::endl
        <<"subsection Solver input data"<<std::endl
        << "    set Preconditioner = GMG" << std::endl
        <<"end"<<std::endl
       <<"subsection Problem Selection"<<std::endl
        << "    set Problem = GaussianCharges" << std::endl
        << "    set Dimension = 3" << std::endl
        <<"end"<<std::endl
       <<"subsection Lammps data"<<std::endl
      << "  set Lammps input file = " << SOURCE_DIR << "/atom_2.data" << std::endl
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
  double nonzero_density_radius_parameter ;//= prm.get_double("Nonzero Density radius parameter around each charge");
  prm.leave_subsection ();

  const unsigned int Degree = prm.get_integer("Polynomial degree");

  prm.enter_subsection("Solver input data");
  std::string PreconditionerType = (prm.get("Preconditioner"));
  prm.leave_subsection();

  prm.enter_subsection("Problem Selection");
  std::string Problemtype= (prm.get("Problem"));
  const unsigned int d = prm.get_integer("Dimension");    // set default to two in parameter class
  prm.leave_subsection();

  prm.enter_subsection("Lammps data");
  std::string LammpsInputFile = (prm.get("Lammps input file"));
  prm.leave_subsection();

  std::vector<double> r_c_variation {2.0,2.5,3.0,3.5,4.0};
  for(const auto & i : r_c_variation)
      {
          nonzero_density_radius_parameter = i;//prm.get_double("Nonzero Density radius parameter around each charge");
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              std::cout<<"cutoff radius: "<<nonzero_density_radius_parameter<<std::endl;

          if (d == 2)
          {
                Test_LaplaceProblem<2> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                            number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
                test_laplace_problem.run();

//              Step50::LaplaceProblem<2> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
//                                                number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
//              laplace_problem.run ();
          }
          else if (d == 3)
          {
                  Test_LaplaceProblem<3> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                              number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
                  test_laplace_problem.run();

//              Step50::LaplaceProblem<3> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
//                                                number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
//              laplace_problem.run ();
          }
          else if (d != 2 && d != 3)
          {
              AssertThrow(false, ExcMessage("Only 2d and 3d dimensions are supported."));
          }
      }


}


int main (int argc, char *argv[])
{

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  check ();

  return 0;
}
