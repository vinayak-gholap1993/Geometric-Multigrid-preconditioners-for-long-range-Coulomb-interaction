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
    void charge_density_test(const std::vector<Point<dim> > & , double * ,
                             const std::map<typename parallel::distributed::Triangulation<dim>::cell_iterator, std::set<unsigned int> > &, bool & );

    const unsigned int degree;
    ParameterHandler &prm;
    unsigned int number_of_global_refinement , number_of_adaptive_refinement_cycles;
    double domain_size_left , domain_size_right;
    std::string Problemtype, PreconditionerType, LammpsInputFilename;
    double r_c, nonzero_density_radius_parameter;

  public:
    Test_LaplaceProblem (const unsigned int Degree , ParameterHandler &prm,
                         std::string &Problemtype, std::string &PreconditionerType, std::string &LammpsInputFile,
                         double &domain_size_left, double &domain_size_right, unsigned int &number_of_global_refinement, unsigned int &number_of_adaptive_refinement_cycles,
                         double &r_c, double &nonzero_density_radius_parameter) : Step50::LaplaceProblem<dim> (Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile,
                                                                                                               domain_size_left, domain_size_right, number_of_global_refinement,
                                                                                                               number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter),
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

    void run (bool &);
    ~Test_LaplaceProblem(){}

};

template class Test_LaplaceProblem<2>;
template class Test_LaplaceProblem<3>;

template <int dim>
void Test_LaplaceProblem<dim>::run(bool &flag_rhs_assembly)
{
    Timer timer;
    Step50::LaplaceProblem<dim>::flag_rhs_assembly = flag_rhs_assembly;
    Step50::LaplaceProblem<dim>::read_lammps_input_file(LammpsInputFilename);
    for (unsigned int cycle=0; cycle<number_of_adaptive_refinement_cycles; ++cycle)
        // first mesh size 4^2 = 16*16*16 and then 2 refinements
    {
        timer.start();

        Step50::LaplaceProblem<dim>::pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube (Step50::LaplaceProblem<dim>::triangulation,domain_size_left,domain_size_right);

            Step50::LaplaceProblem<dim>::triangulation.refine_global (number_of_global_refinement);  //eg. first mesh size 4^2 = 16*16*16
        }
        else
            Step50::LaplaceProblem<dim>::refine_grid ();

        Step50::LaplaceProblem<dim>::pcout << "   Number of active cells:       "<< Step50::LaplaceProblem<dim>::triangulation.n_global_active_cells() << std::endl;

        Step50::LaplaceProblem<dim>::setup_system ();

        Step50::LaplaceProblem<dim>::pcout << "   Number of degrees of freedom: " << Step50::LaplaceProblem<dim>::mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level=0; level<Step50::LaplaceProblem<dim>::triangulation.n_global_levels(); ++level)
            Step50::LaplaceProblem<dim>::pcout << Step50::LaplaceProblem<dim>::mg_dof_handler.n_dofs(level) << (level == Step50::LaplaceProblem<dim>::triangulation.n_global_levels()-1 ? ")" : ", ");
        Step50::LaplaceProblem<dim>::pcout << std::endl;

        if(flag_rhs_assembly != 0)
            Step50::LaplaceProblem<dim>::rhs_assembly_optimization(Step50::LaplaceProblem<dim>::atom_positions);

        Step50::LaplaceProblem<dim>::assemble_system (Step50::LaplaceProblem<dim>::atom_positions, Step50::LaplaceProblem<dim>::charges, Step50::LaplaceProblem<dim>::charges_list_for_each_cell
                                                      ,flag_rhs_assembly);
        charge_density_test(Step50::LaplaceProblem<dim>::atom_positions, Step50::LaplaceProblem<dim>::charges, Step50::LaplaceProblem<dim>::charges_list_for_each_cell
        ,flag_rhs_assembly);

        timer.stop();
        //std::cout << "   Elapsed CPU time: " << timer() << " seconds."<<std::endl;
        //std::cout << "   Elapsed wall time: " << timer.wall_time() << " seconds."<<std::endl;
        timer.reset();

        // Print the charges densities i.e. system rhs norms to compare with rhs optimization
        Step50::LaplaceProblem<dim>::pcout << "   L2 rhs norm " << std::setprecision(10) << std::scientific << Step50::LaplaceProblem<dim>::system_rhs.l2_norm() << std::endl;
        Step50::LaplaceProblem<dim>::pcout << "   LInfinity rhs norm " << std::setprecision(10) << std::scientific << Step50::LaplaceProblem<dim>::system_rhs.linfty_norm() << std::endl;
    }
}

//Integrate for each cell the charge density for associated atom list
//Add all cell contribution charge sensities to check if some error due to rhs assembly optimization
//Ideally we consider Charge neutral system
template <int dim>
void Test_LaplaceProblem<dim>::charge_density_test(const std::vector<Point<dim> > & atom_positions, double * charges,
                                                   const std::map<typename parallel::distributed::Triangulation<dim>::cell_iterator, std::set<unsigned int> > &charges_list_for_each_cell
                                                   , bool & flag_rhs_assembly)
{
    const QGauss<dim>  quadrature_formula(degree+1);

    FEValues<dim> fe_values (Step50::LaplaceProblem<dim>::fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);

    const unsigned int   dofs_per_cell = Step50::LaplaceProblem<dim>::fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double>    coefficient_values (n_q_points);
    std::vector<double>    density_values (n_q_points);

    double r = 0.0, r_squared = 0.0;
    const double r_c_squared_inverse = 1.0 / (r_c * r_c);

    const double constant_value = 4.0 * (numbers::PI)  / (std::pow(r_c, 3) * std::pow(numbers::PI, 1.5));

    std::set<unsigned int> set_atom_indices;
    typedef LA::MPI::Vector vector_t;
    vector_t total_charge_densities;
    total_charge_densities.reinit(Step50::LaplaceProblem<dim>::mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

    typename DoFHandler<dim>::active_cell_iterator
    cell = Step50::LaplaceProblem<dim>::mg_dof_handler.begin_active(),
    endc = Step50::LaplaceProblem<dim>::mg_dof_handler.end();
    for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit (cell);

            Step50::LaplaceProblem<dim>::coeff_func->value_list (fe_values.get_quadrature_points(),
                                    coefficient_values);

                    const std::vector<Point<dim> > & quadrature_points = fe_values.get_quadrature_points();
                    set_atom_indices =  charges_list_for_each_cell.at(cell);

                    for(unsigned int q_points = 0; q_points < n_q_points; ++q_points)
                        {
                            density_values[q_points] = 0.0;
                                {
                                //With rhs assembly optimization
                                //If flag != 0 iterate only over the neighouring atoms and apply rhs optimization
                                if(flag_rhs_assembly != 0)
                                    {
                                            for(const auto & a : set_atom_indices)
                                            {
                                                //std::cout<< *iter << " ";
                                                r = 0.0;
                                                r_squared = 0.0;

                                                const Point<dim> Xi = atom_positions[a];
                                                r = Xi.distance(quadrature_points[q_points]);
                                                r_squared = r * r;

                                                density_values[q_points] +=  constant_value *
                                                                             exp(-r_squared * r_c_squared_inverse) *
                                                                             charges[a];
                                            }
                                    }
                                //Without optimization
                                //If flag == 0 iterate over all the atoms in the domain, i.e. do not optimize the assembly
                                if(flag_rhs_assembly == 0)
                                    {
                                        for(unsigned int k = 0; k < atom_positions.size(); ++k)
                                            {
                                                r = 0.0;
                                                r_squared = 0.0;

                                                const Point<dim> Xi = atom_positions[k];
                                                r = Xi.distance(quadrature_points[q_points]);
                                                r_squared = r * r;

                                                density_values[q_points] +=  constant_value *
                                                        exp(-r_squared * r_c_squared_inverse) *
                                                        charges[k];
                                            }
                                    }
                                }
                        }
                    set_atom_indices.clear();

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                        cell_rhs(i) += (density_values[q_point]* fe_values.JxW(q_point));
                }

            cell->get_dof_indices (local_dof_indices);
            Step50::LaplaceProblem<dim>::constraints.distribute_local_to_global (cell_rhs, local_dof_indices, total_charge_densities);
        }

    total_charge_densities.compress(VectorOperation::add);
    Step50::LaplaceProblem<dim>::pcout << "Total charge density over the domain after rhs assembly optimization " << std::setprecision(10) << std::scientific
                                        << total_charge_densities.l2_norm() << std::endl;
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
  const unsigned int d = prm.get_integer("Dimension");
  prm.leave_subsection();

  prm.enter_subsection("Lammps data");
  std::string LammpsInputFile = (prm.get("Lammps input file"));
  prm.leave_subsection();

  bool flag_rhs_assembly;
  std::vector<double> r_c_variation {2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0};
  for(const auto & i : r_c_variation)
      {
          nonzero_density_radius_parameter = i;
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              std::cout<<"cutoff radius: "<<std::fixed<<std::setprecision(2)<<nonzero_density_radius_parameter<<std::endl;

          if (d == 2)
          {
                Test_LaplaceProblem<2> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                            number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
//                test_laplace_problem.run();
                flag_rhs_assembly = 0;
                std::cout << "Without rhs assembly optimization" <<std::endl;
                test_laplace_problem.run(flag_rhs_assembly);
//                test_laplace_problem.~Test_LaplaceProblem();

                Test_LaplaceProblem<2> test_laplace_problem_with_rhs_optimiation(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                            number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
                flag_rhs_assembly = 1;
                std::cout << "Rhs assembly optimization ENABLED" <<std::endl;
                test_laplace_problem_with_rhs_optimiation.run(flag_rhs_assembly);

          }
          else if (d == 3)
          {
                  Test_LaplaceProblem<3> test_laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                              number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
//                  test_laplace_problem.run();
                  flag_rhs_assembly = 0;
                  std::cout << "Without rhs assembly optimization" <<std::endl;
                  test_laplace_problem.run(flag_rhs_assembly);
//                  test_laplace_problem.~Test_LaplaceProblem();

                  Test_LaplaceProblem<3> test_laplace_problem_with_rhs_optimiation(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile, domain_size_left, domain_size_right,
                                                              number_of_global_refinement, number_of_adaptive_refinement_cycles, r_c, nonzero_density_radius_parameter);
                  flag_rhs_assembly = 1;
                  std::cout << "Rhs assembly optimization ENABLED" <<std::endl;
                  test_laplace_problem_with_rhs_optimiation.run(flag_rhs_assembly);

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
