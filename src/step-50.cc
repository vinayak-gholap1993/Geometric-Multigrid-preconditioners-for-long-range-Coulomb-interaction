#include <step_50.h>


using namespace dealii;
using namespace Step50;

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
    prm.parse_input(parameter_file);
}


template <int dim>
LaplaceProblem<dim>::LaplaceProblem (const unsigned int degree , ParameterHandler &param,
                                     const std::string &Problemtype, const std::string &PreconditionerType, const std::string &LammpsInputFile,
                                     const double &domain_size_left, const double &domain_size_right, const unsigned int &number_of_global_refinement,
                                     const unsigned int &number_of_adaptive_refinement_cycles,
                                     const double &r_c, const double &nonzero_density_radius_parameter, const bool &flag_rhs_assembly)
    :
    flag_rhs_assembly(flag_rhs_assembly),
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
    pcout<<"Preconditioner :    " << PreconditionerType<<std::endl;
    if(flag_rhs_assembly)
	pcout<<"Rhs assembly optimization ENABLED"<<std::endl;
    else
	pcout<<"Without rhs assembly optimization"<<std::endl;

    if (Problemtype == "Step16")
    {
        rhs_func   = std::make_shared<Step16::RightHandSide<dim>>();
        coeff_func = std::make_shared<Step16::Coefficient<dim>>();
    }
    if(Problemtype == "GaussianCharges")
    {
        rhs_func   = std::make_shared<GaussianCharges::RightHandSide<dim>>(r_c);
        coeff_func = std::make_shared<GaussianCharges::Coefficient<dim>>();
    }


}

template <int dim>
LaplaceProblem<dim>::~LaplaceProblem ()
{
    triangulation.clear();
    mg_dof_handler.clear();
    if(flag_rhs_assembly)
	charges_list_for_each_cell.clear();
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
                    pcout<< "Number of atoms: " << number_of_atoms<< std::endl;
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
			file >> p(2); //For 2d test case comment
//			file>>input;

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
void LaplaceProblem<dim>::rhs_assembly_optimization()
{
    typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();

    double distance_from_vertex_to_atom = 0.0;

    for(; cell!= endc; ++cell)
        if (cell->is_locally_owned())
        {
            std::set<unsigned int> atom_indices;

            for(unsigned int i = 0; i < this->atom_positions.size(); ++i)
            {
                for(unsigned int vertex_number = 0; vertex_number < GeometryInfo<dim>::vertices_per_cell; ++vertex_number)
                {
                    distance_from_vertex_to_atom = 0.0;
                    const Point<dim> Xi = this->atom_positions[i];
                    distance_from_vertex_to_atom = Xi.distance(cell->vertex(vertex_number));
                    if( distance_from_vertex_to_atom < nonzero_density_radius_parameter * r_c)
                    {
                        atom_indices.insert(i);
                    }
                }
            }

            this->charges_list_for_each_cell.insert(std::make_pair(cell, atom_indices));

            //clear std::set content for next cell
            atom_indices.clear();
        }

    std::set<unsigned int>::iterator iter;
    typename std::map<cell_it, std::set<unsigned int> >::iterator it;

//        //Print the contents of std::map as cell level, index : atom_list
//        for(it = charges_list_for_each_cell.begin(); it != charges_list_for_each_cell.end(); ++it)
//            {
//                std::cout<< it->first->level()<<" "<<it->first->index() << ":" ;
//                for(iter = it->second.begin(); iter != it->second.end(); ++iter)
//                    std::cout<< *iter << " ";
//                std::cout<< std::endl;
//            }
}

// Output the grid with atoms list for each cell
// Preferable to run only for one refinement
template <int dim>
void LaplaceProblem<dim>::grid_output_debug(const unsigned int cycle)
{
    std::map<types::global_dof_index, Point<dim> > support_points;
    MappingQ1<dim> mapping;
    DoFTools::map_dofs_to_support_points(mapping, mg_dof_handler, support_points);

    const std::string base_filename =
        "grid" + dealii::Utilities::int_to_string(dim) + "_p" + "_cycle"+ dealii::Utilities::int_to_string(cycle) + dealii::Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    const std::string filename =  base_filename + ".gp";
    std::ofstream f(filename.c_str());

    f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
      << "set output \"" << base_filename << ".png\"" << std::endl
      << "set size square" << std::endl
      << "set view equal xy" << std::endl
      << "unset xtics" << std::endl
      << "unset ytics" << std::endl
      << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
    GridOut().write_gnuplot(triangulation, f);
    f << "e" << std::endl;

    for (auto it : this->charges_list_for_each_cell)
    {
        f << it.first->center() << " \"";
        for (auto el : it.second)
            f << el << ", ";
        f << "\"\n";
    }

    f << std::flush;

    f << "e" << std::endl;

//        Output another grid with flag output for atom presence on each cell
//        if atom assigned to the cell flag 1 else flag 0
    const std::string base_filename_2 =
        "grid_atom_presence" + dealii::Utilities::int_to_string(dim) + "_p" + "_cycle"+ dealii::Utilities::int_to_string(cycle) + dealii::Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    const std::string filename_2 =  base_filename_2 + ".gp";
    std::ofstream g(filename_2.c_str());

    g << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
      << "set output \"" << base_filename_2 << ".png\"" << std::endl
      << "set size square" << std::endl
      << "set view equal xy" << std::endl
      << "unset xtics" << std::endl
      << "unset ytics" << std::endl
      << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
    GridOut().write_gnuplot(triangulation, g);
    g << "e" << std::endl;

    for (auto it : this->charges_list_for_each_cell)
    {
        g << it.first->center() << " \"";
        if(it.second.empty())
            g << 0;
        else
            g << 1;
        g << "\"\n";
    }

    g << std::flush;

    g << "e" << std::endl;

}

template <int dim>
void LaplaceProblem<dim>::pack_function(const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &cell,
                                        const typename parallel::distributed::Triangulation<dim,dim>::CellStatus status, void *data)
{
    if (status==parallel::distributed::Triangulation<dim,dim>::CELL_COARSEN)
    {
        Assert(cell->has_children(), ExcInternalError());
    }
    else
    {
        Assert(!cell->has_children(), ExcInternalError());
    }

    unsigned int * data_store = reinterpret_cast<unsigned int *>(data);

    std::set<unsigned int> set_atom_indices;
    std::vector<unsigned int> vec_atom_indices;
    set_atom_indices = this->charges_list_for_each_cell.at(cell);
    std::copy(set_atom_indices.begin(), set_atom_indices.end(), std::back_inserter(vec_atom_indices));
    const unsigned int n_indices = vec_atom_indices.size();
    Assert (sizeof(unsigned int) * (n_indices+1) <= this->data_size_in_bytes,
            ExcInternalError());
    std::memcpy(data_store, &n_indices, sizeof(unsigned int));
    data_store++;
    std::memcpy(data_store, &vec_atom_indices[0], sizeof(unsigned int)*n_indices);
    set_atom_indices.clear();
    vec_atom_indices.clear();
}

template <int dim>
void LaplaceProblem<dim>::unpack_function (const typename parallel::distributed::Triangulation<dim,dim>::cell_iterator &cell,
        const typename parallel::distributed::Triangulation<dim,dim>::CellStatus status, const void *data)
{
    Assert ((status!=parallel::distributed::Triangulation<dim,dim>::CELL_COARSEN),
            ExcNotImplemented());
    if (status==parallel::distributed::Triangulation<dim,dim>::CELL_REFINE)
    {
        Assert(cell->has_children(), ExcInternalError());
    }
    else
    {
        Assert(!cell->has_children(), ExcInternalError());
    }

    (void) status;
    const unsigned int * data_store = reinterpret_cast<const unsigned int *>(data);
    unsigned int n_indices = 0;
    std::memcpy(&n_indices, data_store, sizeof(unsigned int));
    data_store++;
    std::vector<unsigned int> vec_atom_indices(n_indices);
    std::memcpy(&vec_atom_indices[0], data_store, sizeof(unsigned int) * n_indices);

    // print debug
//    if(n_indices != 0)
//    {
//	    std::cout << "cell with center " << cell->center() << " has " << n_indices << " values:" << std::endl;
//	    for (auto &ind : vec_atom_indices)
//		std::cout <<" " << ind;
//	    std::cout << std::endl;
//    }

    std::set<unsigned int> set_atom_indices;	//(vec_atom_indices.begin(), vec_atom_indices.end());
    std::copy(vec_atom_indices.begin(), vec_atom_indices.end(), std::inserter(set_atom_indices, set_atom_indices.begin()));

    if(cell->has_children())
    {
        for (unsigned int child=0; child<cell->n_children(); ++child)
            if (cell->child(child)->is_locally_owned())
            {
//                Assert(this->charges_list_for_each_cell.find(cell->child(child)) == this->charges_list_for_each_cell.end(),
//                       ExcInternalError());
                this->charges_list_for_each_cell[cell->child(child)] = set_atom_indices;
            }
    }
    else
    {
//	Assert(this->charges_list_for_each_cell.find(cell) == this->charges_list_for_each_cell.end(),
//	       ExcInternalError());
        this->charges_list_for_each_cell[cell] = set_atom_indices;
    }
    vec_atom_indices.clear();
    set_atom_indices.clear();
}

template <int dim>
void LaplaceProblem<dim>::prepare_for_coarsening_and_refinement ( )
{
    unsigned int number_of_values = 0;
    for (typename parallel::distributed::Triangulation<dim>::active_cell_iterator it = triangulation.begin_active();
            it != triangulation.end(); it++)
        if (it->is_locally_owned())
            number_of_values = std::max(static_cast<unsigned int>(this->charges_list_for_each_cell.at(it).size()),
                                        number_of_values);

    number_of_values = Utilities::MPI::max(number_of_values, triangulation.get_communicator ()) + 1;
    Assert (number_of_values > 0, ExcInternalError());
    this->data_size_in_bytes = sizeof(unsigned int) * number_of_values;
    this->offset = triangulation.register_data_attach(data_size_in_bytes, std::bind(&Step50::LaplaceProblem<dim>::pack_function,
                   this,
                   std::placeholders::_1,
                   std::placeholders::_2,
                   std::placeholders::_3));


}

template <int dim>
void LaplaceProblem<dim>::project_cell_data()
{
    triangulation.notify_ready_to_unpack(this->offset, std::bind(&Step50::LaplaceProblem<dim>::unpack_function,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2,
                                         std::placeholders::_3));
}

template <int dim>
void LaplaceProblem<dim>::setup_system ()
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
}



template <int dim>
void LaplaceProblem<dim>::assemble_system ()
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

    double r = 0.0, r_squared = 0.0;
    const double r_c_squared_inverse = 1.0 / (r_c * r_c);

    /*
    const Point<dim> test = atom_positions[3];
    std::cout<< "This data: \n" << this->number_of_atoms << " " << this->charges[3] << " " << test << std::endl;
    */

    const double constant_value = 4.0 * (numbers::PI)  / (std::pow(r_c, 3) * std::pow(numbers::PI, 1.5));

    std::set<unsigned int> set_atom_indices;

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
		if(flag_rhs_assembly)
		    set_atom_indices = this->charges_list_for_each_cell.at(cell);
//                    pcout<<"Printing the cell->atom_indices: "<<std::endl;
//                    for(const auto & b : set_atom_indices)
//                        {
//                            pcout<< b << ", ";
//                        }
//                    std::cout<<std::endl;
                for(unsigned int q_points = 0; q_points < n_q_points; ++q_points)
                {
                    density_values[q_points] = 0.0;

                    // FIXME: figure out which cells have non-zero contribution from density for which atoms
                    // maybe keep std::set<unsigned int> attached to a cell and in loop below only
                    // go over atoms which have non-zero contribution.
                    // for starters, do this association during setup_system(), i.e.
                    // after each adaptive refinement.
                    // TODO: add 1 unit test with 8 atoms and several refinement steps
                    // TODO: add 1 unit test with 2 atom of oposite charge NOT at the same pont,
                    // make sure the solution agrees with analytical solution

                    //If flag = false iterate over all the atoms in the domain, i.e. do not optimize the assembly
                    if(!flag_rhs_assembly)

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
                                                         this->charges[k];
                        }

                    }


                    //If flag = true iterate only over the neighouring atoms and apply rhs optimization
                    if(flag_rhs_assembly)

                    {
                        //To be checked
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
                                                         this->charges[a];
                        }
                    }


                }
		if(flag_rhs_assembly)
		    set_atom_indices.clear();
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
        MGTransferPrebuilt<vector_t> mg_transfer( mg_constrained_dofs);
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


    pcout << "   Starting value " << std::fixed << solver_control.initial_value() << std::endl;
    pcout << "   CG converged in " << solver_control.last_step() << " iterations." << std::endl;
    pcout << "   Convergence value " << std::scientific << solver_control.last_value() << std::endl;
    pcout << "   L1 solution norm " << std::setprecision(10) << std::scientific << solution.l1_norm() << std::endl;
    pcout << "   L2 solution norm " << std::setprecision(10) << std::scientific << solution.l2_norm() << std::endl;
    pcout << "   LInfinity solution norm " << std::setprecision(10) << std::scientific << solution.linfty_norm() << std::endl;
    // Print the charges densities i.e. system rhs norms to compare with rhs optimization
//    pcout << "   L1 rhs norm " << std::setprecision(10) << std::scientific << system_rhs.l1_norm() << std::endl;
//    pcout << "   L2 rhs norm " << system_rhs.l2_norm() << std::endl;
//    pcout << "   LInfinity rhs norm " << system_rhs.linfty_norm() << std::endl;

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

    if((lammpsinput != 0) && (flag_rhs_assembly))
        prepare_for_coarsening_and_refinement();

    parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> soltrans(mg_dof_handler);
    triangulation.prepare_coarsening_and_refinement();

    LA::MPI::Vector previous_solution;
    previous_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
    previous_solution = solution;
    soltrans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement ();

    if((lammpsinput != 0) && (flag_rhs_assembly))
	project_cell_data();

    setup_system();

    soltrans.interpolate(solution);
    constraints.set_zero (solution);

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

    LA::MPI::Vector analytical_sol_ghost;

//Output the analytical solution on mesh only for Gaussian charges problem with or without LAMMPS input
    if(Problemtype == "GaussianCharges")
    {
        if(lammpsinput == 0)
        {
            //Need to implement the analytical sol for the problem on paper
            LA::MPI::Vector analytical_sol;
            analytical_sol.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
            VectorTools::interpolate (mg_dof_handler, GaussianCharges::Analytical_Solution_without_lammps<dim> (r_c), analytical_sol);
            analytical_sol_ghost.reinit(mg_dof_handler.locally_owned_dofs(),locally_relevant_set,MPI_COMM_WORLD);
            analytical_sol_ghost = analytical_sol;
            data_out.add_data_vector (analytical_sol_ghost, "Analytical_Solution_without_lammps");
        }
        else if (lammpsinput != 0)
        {
            LA::MPI::Vector analytical_sol;
            analytical_sol.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
            VectorTools::interpolate (mg_dof_handler, GaussianCharges::Analytical_Solution<dim> (r_c), analytical_sol);
            analytical_sol_ghost.reinit(mg_dof_handler.locally_owned_dofs(),locally_relevant_set,MPI_COMM_WORLD);
            analytical_sol_ghost = analytical_sol;
            data_out.add_data_vector (analytical_sol_ghost, "Analytical_sol_atoms");
        }
    }

    LA::MPI::Vector interpolated_rhs;
    interpolated_rhs.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    LA::MPI::Vector interpolated_rhs_ghost;
    interpolated_rhs_ghost.reinit(mg_dof_handler.locally_owned_dofs(), locally_relevant_set, MPI_COMM_WORLD);

//Output the rhs to mesh for visualisation

    if((lammpsinput != 0) && (number_of_atoms < 10))
    {
        if (Problemtype == "Step16")
            VectorTools::interpolate (mg_dof_handler, Step16::RightHandSide<dim> (), interpolated_rhs);
        if (Problemtype == "GaussianCharges")
            VectorTools::interpolate (mg_dof_handler, GaussianCharges::RightHandSide<dim> (r_c), interpolated_rhs);

        interpolated_rhs_ghost = interpolated_rhs;
        data_out.add_data_vector (interpolated_rhs_ghost, "interpolated_rhs");
    }
    LA::MPI::Vector system_rhs_ghost;
    system_rhs_ghost.reinit(mg_dof_handler.locally_owned_dofs(), locally_relevant_set, MPI_COMM_WORLD);
    system_rhs_ghost = system_rhs;
    data_out.add_data_vector (system_rhs_ghost, "system_rhs");

//Output support for rhs of each atom with 1 being atom present in the cell

    std::vector<Vector<float>> support(number_of_atoms,
				       Vector<float>(this->triangulation.n_active_cells()));

    if((lammpsinput != 0) && (flag_rhs_assembly))
    {
	    unsigned int cell_index = 0;
	    std::set<unsigned int> set_atom_indices;
	    for (auto cell: this->mg_dof_handler.active_cell_iterators())
	    {
		if (cell->is_locally_owned())
		{
		    set_atom_indices = this->charges_list_for_each_cell.at(cell);
		    if(!set_atom_indices.empty())
		    {
			for(auto i: set_atom_indices)
			    support[i](cell_index) = 1.0;
		    }
		}
		cell_index++;
		set_atom_indices.clear();
	    }
	    Assert (cell_index == this->triangulation.n_active_cells(),
		    ExcInternalError());
	    for (unsigned int i = 0; i < number_of_atoms; i++)
	    {
		data_out.add_data_vector (support[i],
					  std::string("support_") +
					  dealii::Utilities::int_to_string(i));
	    }
    }

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
        DataOutBase::write_visit_record (visit_master, filenames);

        //std::cout << "   wrote " << pvtu_master_filename << std::endl;

    }
}



template <int dim>
void LaplaceProblem<dim>::run ()
{
    Timer timer_test (triangulation.get_communicator(), true);

    pcout << "Dimension:	" << dim << std::endl;
    Timer timer;
    read_lammps_input_file(LammpsInputFilename);

    for (unsigned int cycle=0; cycle<number_of_adaptive_refinement_cycles; ++cycle)
    {
        timer.start();

        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
//	    GridGenerator::hyper_cube (triangulation,domain_size_left,domain_size_right); // 1 cell i.e. 2^(0*dim)
//	    triangulation.refine_global (number_of_global_refinement);  // formula: 2^(num * dim)

	    // Here domain_left and right need to be according to the LAMMPS atom file xlo and xhi
	    // Need to set #Global_ref = 0
	    const static double a = 0.5;
	    const double N = 2 *(domain_size_right - domain_size_left);	// N = | left - right |/ a
	    const double M = 10; // Setting the vaccum around the lattice as 10 * a
//	    const double total_domain_length_1D = (N + 2 * M) * a;
	    const double repetitions_in_each_direction = 2 * (N + 2 * M);   // in terms of 'h' grid size, h = a/2
	    std::vector< unsigned int > repetitions;
	    repetitions.push_back (repetitions_in_each_direction);
	    if(dim >= 2)
		repetitions.push_back (repetitions_in_each_direction);
	    if(dim >= 3)
		repetitions.push_back (repetitions_in_each_direction);

	    const Point<dim> lower_left = (dim == 2
					   ?
					   Point<dim> (domain_size_left - (M *a), domain_size_left - (M *a))
					   :
					   Point<dim> (domain_size_left - (M *a), domain_size_left - (M *a), domain_size_left - (M *a)));
	    const Point<dim> upper_right = (dim == 2
					    ?
					    Point<dim> (domain_size_right + (M *a), domain_size_right + (M *a))
					    :
					    Point<dim> (domain_size_right + (M *a), domain_size_right + (M *a), domain_size_right + (M *a)));

	    GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions, lower_left, upper_right, false);
        }
	else
	    refine_grid ();

        pcout << "   Number of active cells:       "<< triangulation.n_global_active_cells() << std::endl;

	if(cycle == 0)
	    setup_system ();

        pcout << "   Number of degrees of freedom: " << mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
            pcout << mg_dof_handler.n_dofs(level) << (level == triangulation.n_global_levels()-1 ? ")" : ", ");
        pcout << std::endl;

        if((cycle == 0) && (flag_rhs_assembly))
            rhs_assembly_optimization();

        if(dim == 2)
            grid_output_debug(cycle);

        assemble_system ();

        if(PreconditionerType == "GMG")
            assemble_multigrid ();

        solve ();

        timer.stop();
//	pcout << "   Elapsed wall time: " << timer.wall_time() << " seconds."<<std::endl;
        timer.reset();

        //solution_gradient();
        output_results (cycle);
    }
    timer_test.stop();
    pcout << "   \nElapsed wall time: " << timer_test.wall_time() << " seconds.\n"<<std::endl;
    timer_test.reset();
}


//explicit instantiation for template class
template class Step50::LaplaceProblem<2>;
template class Step50::LaplaceProblem<3>;
