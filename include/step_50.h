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


#ifndef STEP_50_H
#define STEP_50_H



#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

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
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

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
#include <map>
#include <iomanip>
#include <limits>

using namespace dealii;

namespace Step50
{
using namespace dealii;

template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem (const unsigned int , ParameterHandler &, const std::string &, const std::string &, const std::string &,
		    const std::string &, const double &, const double &, const double &, const unsigned int &,
		    const unsigned int &, const unsigned int &, const double &, const double &,
		    const bool &, const bool &, const bool &, const bool &, const bool &, const unsigned int &);
    ~LaplaceProblem();
    void run ();

protected:
    void setup_system (const unsigned int &);
    void assemble_system ();
    void assemble_multigrid ();
    void solve ();
    void refine_grid (const unsigned int &);
    //void solution_gradient();
    void read_lammps_input_file(const std::string& filename);
    void output_results (const unsigned int cycle) const;
    void rhs_assembly_optimization();
    void grid_output_debug(const unsigned int );
    void pack_function(const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &,
                       const typename parallel::distributed::Triangulation<dim,dim>::CellStatus , void *);
    void unpack_function(const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &,
                         const typename parallel::distributed::Triangulation<dim,dim>::CellStatus , const void *);
    void prepare_for_coarsening_and_refinement ();
    void project_cell_data();
    void postprocess_electrostatic_energy();
    double long_ranged_potential(const Point<dim> & , const Point<dim> &, const double &) const;
    const double short_ranged_potential(const Point<dim> & , const Point<dim> &, const double &);
    void compute_charge_densities();
    void compute_moments();

    ConditionalOStream                        pcout;
    TimerOutput computing_timer;

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

    const unsigned int number_of_global_refinement , number_of_adaptive_refinement_cycles;
    const double domain_size_left , domain_size_right, mesh_size_h;
    const unsigned int repetitions_for_vacuum;
    const std::string Problemtype, PreconditionerType, LammpsInputFilename, Boundary_conditions;
    std::shared_ptr<Function<dim>> rhs_func;
    std::shared_ptr<Function<dim>> coeff_func;
    std::unique_ptr<Function<dim>> exact_solution;
    bool lammpsinput;
    const bool flag_analytical_solution, flag_rhs_field, flag_atoms_support,
		flag_rhs_assembly, flag_output_time;
    unsigned int number_of_atoms;
    std::vector<Point<dim> > atom_positions;
    std::vector<unsigned int> atom_types;
    std::vector<double> charges;
    const double r_c, nonzero_density_radius_parameter;

    typedef typename parallel::distributed::Triangulation<dim>::cell_iterator cell_it;
    std::map<cell_it, std::set<unsigned int> > charges_list_for_each_cell;
    std::size_t data_size_in_bytes;
    unsigned int offset;
    Tensor<1, dim, double> dipole_moment;
    Tensor<2, dim, double> quadrupole_moment;
    std::map<cell_it, std::vector<double> > density_values_for_each_cell;
    const unsigned int quadrature_degree_rhs;

};
}


class ParameterReader: public Subscriptor
{
public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);
    void declare_parameters();
private:
    ParameterHandler &prm;
};

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

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    const double r_c;
    RightHandSide(const double &_r_c): Function<dim>(),r_c(_r_c) {}
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

//Added analytical solution output to the mesh
//To visualise the required domain size to ensure correct DBC assumption
template <int dim>
class Analytical_Solution : public Function<dim>
{
public:
    const double &r_c;
    const std::vector<Point<dim> > &atom_positions;
    const std::vector<double> &charges;
    Analytical_Solution(const double &_r_c, const std::vector<Point<dim> > &_pos, const std::vector<double> &_charges)
	: Function<dim>(),r_c(_r_c),atom_positions(_pos),charges(_charges) {}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

template <int dim>
class Analytical_Solution_without_lammps : public Function<dim>
{
public:
    const double r_c;
    Analytical_Solution_without_lammps(const double &_r_c): Function<dim>(), r_c(_r_c) {}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

// Non-zero DBC for potential by employment of quadrupole expansion
template <int dim>
class NonZeroDBC : public Function<dim>
{
public:
    const Point<dim> x0;
    const Tensor<1, dim, double> p0;
    const Tensor<2, dim, double> Q0;

    NonZeroDBC(const Point<dim> &x0_,
	       const Tensor<1, dim, double> &p0_,
	       const Tensor<2, dim, double> &Q0_): Function<dim>(),x0(x0_), p0(p0_), Q0(Q0_){}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,const unsigned int /*component = 0*/) const
{
    const double r_c_squared_inverse = 1.0 / (r_c * r_c);
    const double radial_distance_squared = p.square();  // r^2 = r_x^2 + r_y^2+ r_z^2
    const double constant_value = radial_distance_squared * r_c_squared_inverse;

    return (8.0 * exp(-4.0 * constant_value ) - exp(-constant_value))/(std::pow(r_c,3) * std::pow(numbers::PI, 1.5)) ;
}

template <int dim>
double Coefficient<dim>::value (const Point<dim> &,
                                const unsigned int) const
{
    return 1;
}

template <int dim>
double Analytical_Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
    Assert(atom_positions.size() == charges.size(), ExcInternalError());
    double return_value = 0.0;
    for (unsigned int i = 0; i < charges.size(); ++i)
	{
	    const double radial_distance = atom_positions[i].distance(p);
	    if(radial_distance < 1e-10)
		return_value += charges[i] * 2.0 / (std::sqrt(numbers::PI) * r_c);
	    else
		return_value += charges[i] * (erf(radial_distance / r_c)/ radial_distance);
	}
    return return_value;
}

template <int dim>
double Analytical_Solution_without_lammps<dim>::value(const Point<dim> &p, const unsigned int) const
{
    const double radial_distance = std::sqrt(p.square());
    return (erf(2.0 * radial_distance / r_c) - erf(radial_distance / r_c)) / (4.0 * numbers::PI *radial_distance) ;
}

template <int dim>
double NonZeroDBC<dim>::value(const Point<dim> &p, const unsigned int) const
{
    const Tensor<1, dim, double> x_diff = p - x0;
    const double x_diff_norm = x_diff.norm();
    const auto x_Q_x = contract3(x_diff, Q0, x_diff);
    return (p0 * x_diff) / (std::pow(x_diff_norm,3)) + (0.5 * x_Q_x) / (std::pow(x_diff_norm,5));
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






#endif // STEP_50_H

