/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2003 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Guido Kanschat and Timo Heister
 */





#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include<deal.II/base/parameter_handler.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/cell_id.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/timer.h>

namespace LA
{
#ifdef USE_PETSC_LA
using namespace dealii::LinearAlgebraPETSc;
#else
using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <memory>
#include <string>
#include <set>



namespace Step16
{
using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide():Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  /*component = 0*/) const;
};

template <int dim>
class Coefficient : public Function<dim>
{
public:
    Coefficient () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &/*p*/,
                                  const unsigned int /*component = 0*/) const
{
    return 10.0;
}

template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
    if (p.square() < 0.5*0.5)
        return 5;
    else
        return 1;
}
}

// Test for two charges at origin with neutral charge system i.e. Homogeneous
// Dirichlet B.C.
namespace GaussianCharges
{
using namespace dealii;

const double r_c = 0.5;

const double r_c_squared_inverse = 1.0 / (r_c * r_c);


template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide():Function<dim>() {}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

template <int dim>
class Coefficient : public Function<dim>
{
public:
    Coefficient () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,const unsigned int /*component = 0*/) const
{
    double radial_distance_squared = 0.0, return_value = 0.0, constant_value = 0.0;


    radial_distance_squared = p.square();  // r^2 = r_x^2 + r_y^2+ r_z^2

    constant_value = radial_distance_squared * r_c_squared_inverse;

    return_value = ((8.0 * exp(-4.0 * constant_value ) - exp(-constant_value))/(std::pow(r_c,3) * std::pow(numbers::PI, 1.5))) ;
    return return_value;
}

template <int dim>
double Coefficient<dim>::value (const Point<dim> &,
                                const unsigned int) const
{
    return 1;
}
}

/*
namespace YetAnotherProblem
        {
            template <int dim>
            class RightHandSide : public Function<dim>
            {

            }
        }
*/


namespace Step50
{
using namespace dealii;



template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem (const unsigned int , ParameterHandler &, std::string &, std::string &, std::string &);
    void run ();

private:
    void setup_system (unsigned int &, std::vector<Point<dim> > &, double *);
    void assemble_system (unsigned int &, std::vector<Point<dim> > &, double *);
    void assemble_multigrid ();
    void solve ();
    void refine_grid ();
    //void solution_gradient();
    void read_lammps_input_file(const std::string& filename);
    void output_results (const unsigned int cycle) const;

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

    MGLevelObject<matrix_t> mg_matrices;
    MGLevelObject<matrix_t> mg_interface_matrices;
    MGConstrainedDoFs                    mg_constrained_dofs;

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
    //std::set<unsigned int> nonzero_density_cells;

};


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


template <int dim>
LaplaceProblem<dim>::LaplaceProblem (const unsigned int degree , ParameterHandler &param,
                                     std::string &Problemtype, std::string &PreconditionerType, std::string &LammpsInputFile)
    :
    pcout (std::cout,
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           == 0)),
    triangulation (MPI_COMM_WORLD,Triangulation<dim>::
                  limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe (degree),
    mg_dof_handler (triangulation),
    degree(degree),
    prm(param),
    Problemtype(Problemtype),
    PreconditionerType(PreconditionerType),
    LammpsInputFilename(LammpsInputFile)

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


template <int dim>
void LaplaceProblem<dim>::read_lammps_input_file(const std::string& filename)
{

    std::ifstream file(filename);
    unsigned int count = 0;
    std::string input;

    double a = 0.0, b = 0.0;

    Point<dim> p;



if(dim == 3)
    {

    if(file.is_open())
        {
            lammpsinput = 1;
            while(!file.eof())
                {
                    if(count == 2)
                        {
                            file >> number_of_atoms;
                            std::cout<< "Number of atoms: " << number_of_atoms<< std::endl;
                            atom_types = new unsigned int [number_of_atoms]();
                            charges = new double [number_of_atoms]();
                            atom_positions.resize(number_of_atoms);
                        }
                    else if(count == 35)
                        {
                            for(unsigned int i = 0; i < number_of_atoms; ++i)
                                {
                                    file >> a ;
                                    file >> b;
                                    file >> atom_types[i];
                                    file >> charges[i];
                                    file >> p(0);
                                    file >> p(1);
                                    file >> p(2);

                                    atom_positions[i] = p;

                                    /*
                                    const Point<dim> test1 = atom_positions[i];
                                    std::cout << test1 <<std::endl;

                                    std::cout<< "atom types: "<< atom_types[i]<< "  "<<
                                                "charges: "<<charges[i]<< "  "<<
                                                "atom pos: "<<p<<std::endl;
                                    */

                                }
                        }
                    else
                        {
                            file >> input;
                            //std::cout<< input << "  "<< count<<std::endl;
                        }
                    count++;
                }
        }
    else
        {
            lammpsinput = 0;
            pcout<<"Unable to open the file."<< std::endl;
        }
    file.close();
    }
else
    {
        lammpsinput = 0;
        pcout<< "\nReading of Lammps input file implemented for 3D only\n" <<std::endl;
    }
}


template <int dim>
void LaplaceProblem<dim>::setup_system (unsigned int &number_of_atoms, std::vector<Point<dim> > &atom_positions, double * charges)
{
    mg_dof_handler.distribute_dofs (fe);
    mg_dof_handler.distribute_mg_dofs (fe);

    DoFTools::extract_locally_relevant_dofs (mg_dof_handler,
            locally_relevant_set);

    solution.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    system_rhs.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

    constraints.reinit (locally_relevant_set);
    hanging_node_constraints.reinit (locally_relevant_set);
    DoFTools::make_hanging_node_constraints (mg_dof_handler, hanging_node_constraints);
    DoFTools::make_hanging_node_constraints (mg_dof_handler, constraints);

    //typename FunctionMap<dim>::type      dirichlet_boundary;
    std::set<types::boundary_id>         dirichlet_boundary;
    typename FunctionMap<dim>::type      dirichlet_boundary_functions;
    ZeroFunction<dim>                    homogeneous_dirichlet_bc ;
    dirichlet_boundary.insert(0);
    dirichlet_boundary_functions[0] = &homogeneous_dirichlet_bc;
    VectorTools::interpolate_boundary_values (mg_dof_handler,
            dirichlet_boundary_functions,
            constraints);
    constraints.close ();
    hanging_node_constraints.close ();

    DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(), mg_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (mg_dof_handler, dsp, constraints);
    system_matrix.reinit (mg_dof_handler.locally_owned_dofs(), dsp, MPI_COMM_WORLD, true);


    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(mg_dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(mg_dof_handler, dirichlet_boundary);


    const unsigned int n_levels = triangulation.n_global_levels();

    mg_interface_matrices.resize(0, n_levels-1);
    mg_interface_matrices.clear_elements ();
    mg_matrices.resize(0, n_levels-1);
    mg_matrices.clear_elements ();

    for (unsigned int level=0; level<n_levels; ++level)
    {
        DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(level),
                                   mg_dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(mg_dof_handler, dsp, level);

        mg_matrices[level].reinit(mg_dof_handler.locally_owned_mg_dofs(level),
                                  mg_dof_handler.locally_owned_mg_dofs(level),
                                  dsp,
                                  MPI_COMM_WORLD, true);

        mg_interface_matrices[level].reinit(mg_dof_handler.locally_owned_mg_dofs(level),
                                            mg_dof_handler.locally_owned_mg_dofs(level),
                                            dsp,
                                            MPI_COMM_WORLD, true);
    }

    this->number_of_atoms = number_of_atoms;
    this->atom_positions = atom_positions;
    this->charges = charges;

/*
    typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();

    double r = 0.0;

    for(; cell!= endc; ++cell)
        if (cell->is_locally_owned())
            {
                for(unsigned int vertex_number = 0; vertex_number < GeometryInfo<dim>::vertices_per_cell; ++vertex_number)
                    {
                        for(unsigned int i = 0; i < number_of_atoms; ++i)
                            {
                                r = 0.0;
                                const Point<dim> Xi = atom_positions[i];
                                r = Xi.distance(cell->vertex(vertex_number));
                                if( r < nonzero_density_radius_parameter * r_c)
                                   nonzero_density_cells.insert(cell->id());

                            }
                    }
            }
            */
}



template <int dim>
void LaplaceProblem<dim>::assemble_system (unsigned int &number_of_atoms, std::vector<Point<dim> > &atom_positions, double * charges)
{
    const QGauss<dim>  quadrature_formula(degree+1);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);


    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double>    coefficient_values (n_q_points);
    std::vector<double>    density_values (n_q_points);

    this->number_of_atoms = number_of_atoms;
    this->atom_positions = atom_positions;
    this->charges = charges;

    double r = 0.0, r_squared = 0.0;
    const double r_c_squared_inverse = 1.0 / (r_c * r_c);

    /*
    const Point<dim> test = atom_positions[3];
    std::cout<< "This data: \n" << this->number_of_atoms << " " << this->charges[3] << " " << test << std::endl;
    */

    const double constant_value = 4.0 * (numbers::PI)  / (std::pow(r_c, 3) * std::pow(numbers::PI, 1.5));

    typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();
    for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit (cell);

            coeff_func->value_list (fe_values.get_quadrature_points(),
                                    coefficient_values);


            // evaluate RHS function at quadrature points.
            if(lammpsinput == 0)
                {
                    rhs_func->value_list (fe_values.get_quadrature_points(),
                                         density_values);
                }
            else if(lammpsinput != 0)
                {
                    const std::vector<Point<dim> > & quadrature_points = fe_values.get_quadrature_points();
                    for(unsigned int q_points = 0; q_points < n_q_points; ++q_points)
                        {
                            density_values[q_points] = 0.;


                            // FIXME: figure out which cells have non-zero contribution from density for which atoms
                            // maybe keep std::set<unsigned int> attached to a cell and in loop below only
                            // go over atoms which have non-zero contribution.
                            // for starters, do this association during setup_system(), i.e.
                            // after each adaptive refinement.
                            // TODO: add 1 unit test with 8 atoms and several refinement steps
                            // TODO: add 1 unit test with 2 atom of oposite charge NOT at the same pont,
                            // make sure the solution agrees with analytical solution
                            for(unsigned int k = 0; k < number_of_atoms; ++k)
                            {
                                r = 0.0;
                                r_squared = 0.0;

                                const Point<dim> Xi = atom_positions[k];
                                r = Xi.distance(quadrature_points[q_points]);
                                r_squared = r * r;

                                density_values[q_points] +=  constant_value *
                                                             exp(-r_squared * r_c_squared_inverse) *
                                                             this->charges[k];
                            }
                        }
                }


            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += (coefficient_values[q_point] *
                                             fe_values.shape_grad(i,q_point) *
                                             fe_values.shape_grad(j,q_point) *
                                             fe_values.JxW(q_point));

                            cell_rhs(i) += (fe_values.shape_value(i,q_point) *
                                           density_values[q_point] *
                                           fe_values.JxW(q_point));

                }

            cell->get_dof_indices (local_dof_indices);
            constraints.distribute_local_to_global (cell_matrix, cell_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}



template <int dim>
void LaplaceProblem<dim>::assemble_multigrid ()
{
    QGauss<dim>  quadrature_formula(1+degree);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double>    coefficient_values (n_q_points);



    std::vector<ConstraintMatrix> boundary_constraints (triangulation.n_global_levels());
    ConstraintMatrix empty_constraints;
    for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
    {
        IndexSet dofset;
        DoFTools::extract_locally_relevant_level_dofs (mg_dof_handler, level, dofset);
        boundary_constraints[level].reinit(dofset);
        boundary_constraints[level].add_lines (mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines (mg_constrained_dofs.get_boundary_indices(level));

        boundary_constraints[level].close ();
    }

    typename DoFHandler<dim>::cell_iterator cell = mg_dof_handler.begin(),
                                            endc = mg_dof_handler.end();

    for (; cell!=endc; ++cell)
        if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
        {
            cell_matrix = 0;
            fe_values.reinit (cell);

            coeff_func->value_list (fe_values.get_quadrature_points(),
                                    coefficient_values);

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += (coefficient_values[q_point] *
                                             fe_values.shape_grad(i,q_point) *
                                             fe_values.shape_grad(j,q_point) *
                                             fe_values.JxW(q_point));

            cell->get_mg_dof_indices (local_dof_indices);

            boundary_constraints[cell->level()].distribute_local_to_global (cell_matrix,local_dof_indices,
                    mg_matrices[cell->level()]);


            const IndexSet &interface_dofs_on_level
                = mg_constrained_dofs.get_refinement_edge_indices(cell->level());
            const unsigned int lvl = cell->level();

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    if (interface_dofs_on_level.is_element(local_dof_indices[i])   // at_refinement_edge(i)
                            &&
                            !interface_dofs_on_level.is_element(local_dof_indices[j])   // !at_refinement_edge(j)
                            &&
                            (
                                (!mg_constrained_dofs.is_boundary_index(lvl, local_dof_indices[i])
                                 &&
                                 !mg_constrained_dofs.is_boundary_index(lvl, local_dof_indices[j])
                                ) // ( !boundary(i) && !boundary(j) )
                                ||
                                (
                                    mg_constrained_dofs.is_boundary_index(lvl, local_dof_indices[i])
                                    &&
                                    local_dof_indices[i]==local_dof_indices[j]
                                ) // ( boundary(i) && boundary(j) && i==j )
                            )
                       )
                    {
                    }
                    else
                    {
                        cell_matrix(i,j) = 0;
                    }


            empty_constraints.distribute_local_to_global (cell_matrix,
                    local_dof_indices,
                    mg_interface_matrices[cell->level()]);
        }

    for (unsigned int i=0; i<triangulation.n_global_levels(); ++i)
    {
        mg_matrices[i].compress(VectorOperation::add);
        mg_interface_matrices[i].compress(VectorOperation::add);
    }
}




template <int dim>
void LaplaceProblem<dim>::solve ()
{
    SolverControl solver_control (500, 1e-8*system_rhs.l2_norm(), false);
    SolverCG<vector_t> solver (solver_control);

    if(PreconditionerType == "GMG")
    {

       // MGTransferPrebuilt<vector_t> mg_transfer(hanging_node_constraints, mg_constrained_dofs);
        MGTransferPrebuilt<vector_t> mg_transfer( mg_constrained_dofs);     // Marked Deprecated due to unused constraints Matrix
        mg_transfer.build_matrices(mg_dof_handler);

        matrix_t &coarse_matrix = mg_matrices[0];

        SolverControl coarse_solver_control (1000, 1e-10, false, false);
        SolverCG<vector_t> coarse_solver(coarse_solver_control);
        PreconditionIdentity id;
        MGCoarseGridIterativeSolver<vector_t, SolverCG<vector_t>, matrix_t, PreconditionIdentity > coarse_grid_solver(coarse_solver,
            coarse_matrix,
            id);

        typedef LA::MPI::PreconditionJacobi Smoother;
        MGSmootherPrecondition<matrix_t, Smoother, vector_t> mg_smoother;
        mg_smoother.initialize(mg_matrices, Smoother::AdditionalData(0.5));
        mg_smoother.set_steps(2);

        mg::Matrix<vector_t> mg_matrix(mg_matrices);
        mg::Matrix<vector_t> mg_interface_up(mg_interface_matrices);
        mg::Matrix<vector_t> mg_interface_down(mg_interface_matrices);


        Multigrid<vector_t > mg(mg_matrix,
                            coarse_grid_solver,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother);

        mg.set_edge_matrices(mg_interface_down, mg_interface_up);

        PreconditionMG<dim, vector_t, MGTransferPrebuilt<vector_t> >
        preconditioner(mg_dof_handler, mg, mg_transfer);

        solver.solve (system_matrix, solution, system_rhs,
                      preconditioner);

    }

    else if (PreconditionerType == "Jacobi")
    {
        typedef LA::MPI::PreconditionJacobi JacobiPreconditioner;
        JacobiPreconditioner preconditionJacobi;
        preconditionJacobi.initialize (system_matrix, JacobiPreconditioner::AdditionalData(0.6));

        solver.solve (system_matrix, solution, system_rhs,
                      preconditionJacobi);

    }


    pcout << "   Starting value " << solver_control.initial_value() << std::endl;
    pcout << "   CG converged in " << solver_control.last_step() << " iterations." << std::endl;
    pcout << "   Convergence value " << solver_control.last_value() << std::endl;
    pcout << "   L1 solution norm " << solution.l1_norm() << std::endl;
    pcout << "   L2 solution norm " << solution.l2_norm() << std::endl;
    pcout << "   LInfinity solution norm " << solution.linfty_norm() << std::endl;

    constraints.distribute (solution);
}




template <int dim>
void LaplaceProblem<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    LA::MPI::Vector temp_solution;
    temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
    temp_solution = solution;

    KellyErrorEstimator<dim>::estimate (static_cast<DoFHandler<dim>&>(mg_dof_handler),
                                        QGauss<dim-1>(degree+1),
                                        typename FunctionMap<dim>::type(),
                                        temp_solution,
                                        estimated_error_per_cell);

    const double threshold = 0.6 * Utilities::MPI::max(estimated_error_per_cell.linfty_norm(), MPI_COMM_WORLD);
    GridRefinement::refine (triangulation, estimated_error_per_cell, threshold);

    // parallel::distributed::GridRefinement::
    //refine_and_coarsen_fixed_fraction (triangulation,
    //                                estimated_error_per_cell,
    //                             0.3, 0.0);

    triangulation.execute_coarsening_and_refinement ();
}
/*
template <int dim>
class Function_Map : public Function<dim>
{
public:
    Function_Map(Function<dim> & scalar_function,int selected_component,int n_components);
    double value (const Point<dim> &p, int component) const
    {
        if (component == selected_component)
            return scalar_function.gradient(p);
        else
            return 0.0;
    }
};


template <int dim>
void LaplaceProblem<dim>::solution_gradient()
{
    DoFHandler<dim> dof_vector(triangulation);
    Vector<double> potential = solution;
    FEFieldFunction<dim> fe_field(dof_vector,potential);

    QGauss<dim>  quadrature(1+degree);
    Vector<double> grad_solution;

    VectorFunctionFromScalarFunctionObject func_map(std_cxx1x::bind(&FEFieldFunction::gradient,
                                                                    fe_field,
                                                                    std_cxx1x::_1), 0, 3);

    VectorTools::project(dof_vector, constraints,quadrature,fe_field,grad_solution );

}

*/



template <int dim>
void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
{
    DataOut<dim> data_out;

    LA::MPI::Vector temp_solution;
    temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
    temp_solution = solution;


    LA::MPI::Vector temp = solution;
    system_matrix.residual(temp,solution,system_rhs);
    LA::MPI::Vector res_ghosted = temp_solution;
    res_ghosted = temp;

    data_out.attach_dof_handler (mg_dof_handler);
    data_out.add_data_vector (temp_solution, "solution");
    data_out.add_data_vector (res_ghosted, "res");
    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches (0);

    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 5) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
            filenames.push_back (std::string("solution-") +
                                 Utilities::int_to_string (cycle, 5) +
                                 "." +
                                 Utilities::int_to_string(i, 4) +
                                 ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string (cycle, 5) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        const std::string
        visit_master_filename = ("solution-" +
                                 Utilities::int_to_string (cycle, 5) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        //data_out.write_visit_record (visit_master, filenames);        // Marked Deprectaed
        DataOutBase::write_visit_record (visit_master, filenames);

        //std::cout << "   wrote " << pvtu_master_filename << std::endl;

    }
}



template <int dim>
void LaplaceProblem<dim>::run ()
{
    prm.enter_subsection ("Geometry");
    domain_size_left     = prm.get_double ("Domain limit left");
    domain_size_right     = prm.get_double ("Domain limit right");
    number_of_global_refinement =prm.get_integer("Number of global refinement");
    prm.leave_subsection ();

    prm.enter_subsection ("Misc");
    number_of_adaptive_refinement_cycles      = prm.get_integer ("Number of Adaptive Refinement");
    r_c = prm.get_double ("smoothing length");
    nonzero_density_radius_parameter = prm.get_double("Nonzero Density radius parameter around each charge");
    prm.leave_subsection ();

    Timer timer;

    read_lammps_input_file(LammpsInputFilename);


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
            refine_grid ();

        pcout << "   Number of active cells:       "<< triangulation.n_global_active_cells() << std::endl;

        setup_system (number_of_atoms, atom_positions, charges);

        pcout << "   Number of degrees of freedom: " << mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
            pcout << mg_dof_handler.n_dofs(level) << (level == triangulation.n_global_levels()-1 ? ")" : ", ");
        pcout << std::endl;

        assemble_system (number_of_atoms, atom_positions, charges);
        assemble_multigrid ();

        solve ();

        timer.stop();
        //std::cout << "   Elapsed CPU time: " << timer() << " seconds."<<std::endl;
        //std::cout << "   Elapsed wall time: " << timer.wall_time() << " seconds."<<std::endl;
        timer.reset();

        //solution_gradient();
        output_results (cycle);
    }
}
}


int main (int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    try
    {
        using namespace dealii;
        using namespace Step50;

        //deallog.depth_console(3);

        AssertThrow(argc > 1, ExcMessage ("Invalid inputs"));

        std::string parameter_name (argv[1]);

        ParameterHandler prm;
        ParameterReader param(prm);
        param.read_parameters(parameter_name);

        prm.enter_subsection("Problem Selection");
        std::string Problemtype= (prm.get("Problem"));
        const unsigned int d = prm.get_integer("Dimension");    // set default to two in parameter class
        prm.leave_subsection();

        prm.enter_subsection("Solver input data");
        std::string PreconditionerType = (prm.get("Preconditioner"));
        prm.leave_subsection();

        prm.enter_subsection("Lammps data");
        std::string LammpsInputFile = (prm.get("Lammps input file"));
        prm.leave_subsection();

        const unsigned int Degree = prm.get_integer("Polynomial degree");

        if (d == 2)
        {
            LaplaceProblem<2> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile);
            laplace_problem.run ();
        }
        else if (d == 3)
        {
            LaplaceProblem<3> laplace_problem(Degree , prm ,Problemtype, PreconditionerType, LammpsInputFile);
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
