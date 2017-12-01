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

using namespace dealii;

namespace Step50
{
using namespace dealii;

template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem (const unsigned int , ParameterHandler &, std::string &, std::string &, std::string &,
                    double &, double &, unsigned int &, unsigned int &,
                    double &, double &);
    void run (bool &);
    bool flag_rhs_assembly;

protected:
    void setup_system ();
    void assemble_system (const std::vector<Point<dim> > &, double *,
                          const std::map<typename parallel::distributed::Triangulation<dim>::cell_iterator, std::set<unsigned int> > & , bool &);
    void assemble_multigrid ();
    void solve ();
    void refine_grid ();
    //void solution_gradient();
    void read_lammps_input_file(const std::string& filename);
    void output_results (const unsigned int cycle) const;
    void rhs_assembly_optimization(const std::vector<Point<dim> > &);
    void grid_output_debug(const std::map<typename parallel::distributed::Triangulation<dim>::cell_iterator, std::set<unsigned int> > &);
    void pack_function(const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &,
		       const typename parallel::distributed::Triangulation<dim,dim>::CellStatus , void *);
    void unpack_function(const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &,
			 const typename parallel::distributed::Triangulation<dim,dim>::CellStatus , const void *);

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

    typedef typename parallel::distributed::Triangulation<dim>::cell_iterator cell_it;
    std::map<cell_it, std::set<unsigned int> > charges_list_for_each_cell;

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
    double r_c;
    RightHandSide(double _r_c):Function<dim>() { r_c = _r_c;}
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
    double r_c;
    Analytical_Solution(double _r_c):Function<dim>() {r_c = _r_c;}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

template <int dim>
class Analytical_Solution_without_lammps : public Function<dim>
{
public:
    double r_c;
    Analytical_Solution_without_lammps(double _r_c):Function<dim>() {r_c = _r_c;}
    virtual double value (const Point<dim>   &p,  const unsigned int  /*component = 0*/) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,const unsigned int /*component = 0*/) const
{
    const double r_c_squared_inverse = 1.0 / (r_c * r_c);
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

template <int dim>
double Analytical_Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
    double radial_distance = std::sqrt(p.square());
    double return_value = 0;

    return_value = (erf(radial_distance / r_c) / radial_distance) ;
    return return_value;
}

template <int dim>
double Analytical_Solution_without_lammps<dim>::value(const Point<dim> &p, const unsigned int) const
{
    double radial_distance = std::sqrt(p.square());
    double return_value = 0;

    return_value = ((erf(2.0 * radial_distance / r_c) - erf(radial_distance / r_c)) / (4.0 * numbers::PI *radial_distance)) ;
    return return_value;
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

