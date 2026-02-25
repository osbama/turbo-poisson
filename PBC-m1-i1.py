__author__ = 'obm'


# Solves the Gauss's law.
#PBC is applied. The average potential in the cell is zero
#Mixed formulation of Poisson equation. Jellium is added via Lagrange multipliers
#Energy output is in Hartree units (mind the 4\pi)
#Distance unit is Bohr
#Charge unit is electrons
#Dielectric constant is relative (unitless like in Gaussian unit formalism)


class PeriodicBC(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS, lx = 1., ly=1.,lz=1., length_scaling = 1.):
        SubDomain.__init__(self)
        self.tol = tolerance
        self.lx = lx/length_scaling
        self.ly = ly/length_scaling
        self.lz = lz/length_scaling
        self.length_scaling = length_scaling

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
       return bool((near(x[0], 0.) or near(x[1], 0.) or near(x[2],0.)) and
                  (not ((near(x[0], 0.) and near(x[1], self.ly) and between(x[2], (0.,self.lz)) ) or
                    (near(x[0], self.lx) and near(x[1], 0.) and between(x[2], (0.,self.lz)) ) or
                    (near(x[0], self.lx) and between(x[1], (0.,self.ly)) and near(x[2], 0.) ) or
                    (near(x[0], 0.) and between(x[1], (0.,self.ly)) and near(x[2], self.lz) ) or
                    (between(x[0], (0.,self.lx)) and near(x[1], 0.) and near(x[2], self.lz) ) or
                    (between(x[0], (0.,self.lx)) and near(x[1], self.ly) and near(x[2], 0.) )
                  )) and on_boundary)

    def map(self, x, y):
       #surfaces to be folded. Inside statement exclusively handles overlap edges
       #L,L lines fold to 0,0
       if near(x[0], self.lx) and near(x[1], self.ly) and near(x[2], self.lz): #line
           y[0] = x[0] - self.lx
           y[1] = x[1] - self.ly
           y[2] = x[2] - self.lz
       elif near(x[0], self.lx) and  near(x[2], self.lz):
           y[0] = x[0] - self.lx
           y[1] = x[1]
           y[2] = x[2] - self.lz
       elif near(x[0], self.lx) and  near(x[1], self.ly):
           y[0] = x[0] - self.lx
           y[1] = x[1] - self.ly
           y[2] = x[2]
       elif near(x[1], self.ly) and  near(x[2], self.lz):
           y[0] = x[0]
           y[1] = x[1] - self.ly
           y[2] = x[2] - self.lz
       elif near(x[0], self.lx):
           y[0] = x[0] - self.lx
           y[1] = x[1]
           y[2] = x[2]
       elif near(x[1], self.ly):
           y[0] = x[0]
           y[1] = x[1] - self.ly
           y[2] = x[2]
       elif near(x[2], self.lz):
           y[0] = x[0]
           y[1] = x[1]
           y[2] = x[2] -self.lz
       else :
           y[0] = -1000
           y[1] = -1000
           y[2] = -1000


class extended_normalize:
    """Normalize part or whole of vector.

    V    = Functionspace we normalize in

    u    = Function where part is normalized

    part = The index of the part of the mixed function space
           that we want to normalize.

    For example. When solving for velocity and pressure coupled in the
    Navier-Stokes equations we sometimes (when there is only Neuman BCs
    on pressure) need to normalize the pressure.

    Example of use:
    mesh = UnitSquare(1, 1)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    VQ = V * Q
    up = Function(VQ)
    normalize_func = extended_normalize(VQ, 2)
    up.vector()[:] = 2.
    print 'before ', up.vector().array().astype('I')
    normalize_func(up.vector())
    print 'after ', up.vector().array().astype('I')

    results in:
        before [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
        after  [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0]
    """
    def __init__(self, V, part='entire vector'):
        self.part = part
        if isinstance(part, int):
            self.u = Function(V)
            v = TestFunction(V)
            self.c = assemble(Constant(1., cell=V.cell())*dx, mesh=V.mesh())
            self.pp = ['0']*self.u.value_size()
            self.pp[part] = '1'
            self.u0 = interpolate(Expression(self.pp, element=V.ufl_element()), V)
            self.x0 = self.u0.vector()
            self.C1 = assemble(v[self.part]*dx)
        else:
            self.u = Function(V)
            self.vv = self.u.vector()

    def __call__(self, v):
        if isinstance(self.part, int):
            # assemble into c1 the part of the vector that we want to normalize
            c1 = self.C1.inner(v)
            if abs(c1) > 1.e-8:
                # Perform normalization
                self.x0[:] = self.x0[:]*(c1/self.c)
                v.axpy(-1., self.x0)
                self.x0[:] = self.x0[:]*(self.c/c1)
        else:
            # normalize entire vector
            #dummy = normalize(v) # does not work in parallel
            #self.vv = Vector(v)
            self.vv[:] = 1./v.size()
            c = v.inner(self.vv)
            self.vv[:] = c
            v.axpy(-1., self.vv)
def PBC_m1_i1(argv):
 MPI.barrier(comm)


 #Location of the charge
 Chg_locx=Chg_locxC*alat*celldm1
 Chg_locy=Chg_locyC*alat*celldm2
 Chg_locz=Chg_loczC*alat*celldm2


 source_str = "{chg_tot}*exp(-((pow(x[0]-{chg_locx}, 2.0)+pow(x[1]-{chg_locy}, 2.0)+pow(x[2]-{chg_locz}, 2.0)))/(2.0*pow({chg_spread},2.0)))/pow((pow(2.0*pi,0.5)*{chg_spread}),3.0)"\
             .format(chg_tot=Chg_tot,chg_locx=Chg_locx,chg_locy=Chg_locy,chg_locz=Chg_locz,chg_spread=Chg_spread)
 if mpiRank==0 and Silence<1  :
    print "rho:",source_str
 eps_str="0.5*({eps2}-{eps1})*erf((x[2]-{z0_B})/{beta})*erf((x[2]-{z0_T})/{beta}) + 0.5*({eps1}+{eps2})"\
         .format(eps2=Eps2,eps1=Eps1,z0_B=Z0_B,z0_T=Z0_T,beta=Beta)
 if mpiRank==0 and Silence<1  :
    print "Eps:",eps_str

 ####Automated solution from Documented example using lagrange multipliers method (not the krylov method)
 # Create supercell and mesh and function space

 mesh = BoxMesh(0.0,0.0,0.0,celldm1*alat,celldm2*alat,celldm3*alat,Mesh1,Mesh2,Mesh3)
 pb = PeriodicBC(lx=celldm1,ly=celldm2,lz=celldm3)

 BDM = FunctionSpace(mesh, "BDM", 1, constrained_domain=pb)
 DG = FunctionSpace(mesh, "DG", 0, constrained_domain=pb)
 R = FunctionSpace(mesh, 'R', 0, constrained_domain=pb)
 W = MixedFunctionSpace([BDM, DG, R])

 (sigma, u, mu) = TrialFunctions(W)
 (tau, v, nu) = TestFunctions(W)

 # Define source function
 rho = Expression(source_str)
 epsilon=Expression(eps_str)

 # Define variational form
 a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx - inner(u, nu)*dx - inner(mu, v)*dx
 L = - 4*np.pi*rho*v*dx

 uh = Function(W)


 if auto_adaptive1 :
    if mpiRank==0 and Silence<1  :
       print "Starting Auto adaptive algorithm 1"
    #Adaptive solver
    problem = LinearVariationalProblem(a, L, uh)
    #E
    #print "uh",uh[4]
    #exit()
    #E = div(as_vector((uh[0],uh[1])))*rho*dx #does not converge
    #E = div(uh.sub(0))*rho*dx #error

    #sigma, phi, mu = uh.split() #error
    #E=phi*rho*dx

    a, b, c = split(uh)
    E=0.5*b*rho*dx

    solver = AdaptiveLinearVariationalSolver(problem, E)
    solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "mumps"
    solver.parameters["linear_variational_solver"]["linear_solver"] = "mumps"
    solver.parameters["max_iterations"] = mesh_maxiter


    if args.showconfig :
       for ranks in range(0,MPI.size(comm)):
          if mpiRank==ranks :
             print "Solver configuration for process ",ranks," :"
             #print solver.methods()
             #print solver.preconditioners()
             print solver.parameters
             for parameter in solver.parameters.iteritems():
                print parameter
             print "Error control parameters:"
             for parameter in solver.parameters.get('error_control'):
               print parameter
               print solver.parameters["error_control"][parameter]
             print "solver parameters:"
             for parameter in solver.parameters.get('linear_variational_solver'):
               print parameter
               print solver.parameters["linear_variational_solver"][parameter]
          MPI.barrier(comm)
       exit()
    #solve
    solver.solve(MeshTol)
    sigma, phi, mu = uh.split()
    # Report results
    mesh_new=mesh.leaf_node()
    V = FunctionSpace(mesh_new, "DG", 0)
    chg_ongrid_new=Function(V)
    chg_ongrid_new.interpolate(rho)
    chg_new = chg_ongrid_new*dx
    chg_new_int = assemble(chg_new)
    V = FunctionSpace(mesh.root_node(), "DG", 0)
    chg_ongrid_old=Function(V)
    chg_ongrid_old.interpolate(rho)
    chg_old = chg_ongrid_old*dx
    chg_old_int = assemble(chg_old)
    #E integral
    E_new = phi.leaf_node()*chg_ongrid_new*dx
    E_int_new = assemble(E_new)
    E_int_new = E_int_new*0.5
    E_old=phi.root_node()*chg_ongrid_old*dx
    E_int_old = assemble(E_old)
    E_int_old = E_int_old*0.5
 elif single_shot :
    if mpiRank==0 and Silence<1  :
       print "Starting single shot algorithm"
    solve(a == L, uh,solver_parameters={"linear_solver": "mumps"})

    # Report results
    sigma, phi, mu = uh.split()
    V = FunctionSpace(mesh, "DG", 0)
    chg_ongrid=Function(V)
    chg_ongrid.interpolate(rho)
    chg = chg_ongrid*dx
    chg_int = assemble(chg)

    #E integral
    E = phi*rho*dx
    E_int = assemble(E)
    E_int = E_int*0.5

 else :
    if mpiRank==0 :
       print "Please select a method for solution"
    exit()


 ############REPORTING
 #print "depth",mesh.depth()
 #print "depth",mesh.leaf_node()
 if mpiRank == 0 :
   if Silence < 1 :
      if auto_adaptive1 :
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
         print "                 RESULTS                                          "
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
         print "alat:",alat," celldm1:",celldm1," celldm2:",celldm2," celldm3:",celldm3
         print "Number of cells before adaptive optimization:",mesh.num_cells()
         print "Number of cells after adaptive optimization:",mesh.leaf_node().num_cells()
         print "Total energy calculated using initial mesh:",E_int_old
         print "Total energy calculated using final mesh:",E_int_new
         print "Integrated charge in the inital mesh:",chg_old_int
         print "Integrated charge in the final mesh:",chg_new_int
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
      elif single_shot :
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
         print "                 RESULTS                                          "
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
         print "alat:",alat," celldm1:",celldm1," celldm2:",celldm2," celldm3:",celldm3
         print "Number of cells:",mesh.num_cells()
         print "Total energy:",E_int
         print "Integrated charge:",chg_int
         print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
   elif Silence == 1  :
      if auto_adaptive1 :
         print "#Old number of cells, New number of cells, alat, E_old, E_new, chg_int_old, chg_int_new:%12.8e %12.8e %12.8e %12.8e %12.8e %12.8e %12.8e "%(mesh.num_cells(),mesh.leaf_node().num_cells(),alat, E_int_old,E_int_new,chg_old_int,chg_new_int)
      elif single_shot :
         print "#number of cells, alat, E, chg_int:%12.8e %12.8e %12.8e %12.8e "%(mesh.num_cells(),alat, E_int,chg_int)
   elif Silence > 1 :
      if auto_adaptive1 :
         print E_int_new
      if single_shot :
         print E_int

 #Save file in vtk format
 if  fout_level != 0 and Silence<1:
    if fout_level == 1 :
       filepot = File(fout_prefix+"-pot.pvd")
       filepot << u.root_node()
       filepot << u.leaf_node()
    elif fout_level== 2 :
       filepot = File(fout_prefix+"-pot.pvd")
       filepot << u.root_node()
       filepot << u.leaf_node()
       filechg = File(fout_prefix+"-chg.pvd")
       filechg << chg_ongrid_old
       filechg << chg_ongrid_new
       print "Do not forget epsilon: "+fout_prefix+"-eps.pvd"
    elif fout_level== 3 :
       filepot = File(fout_prefix+"-pot.pvd")
       filepot << u.root_node()
       filepot << u.leaf_node()
       filechg = File(fout_prefix+"-chg.pvd")
       filechg << chg_ongrid_old
       filechg << chg_ongrid_new
       print "Do not forget epsilon: "+fout_prefix+"-eps.pvd"
       filemesh = File(fout_prefix+"-mesh.pvd")
       filemesh << mesh.root_node()
       filemesh << mesh.leaf_node()


 #######FINAL
 if Silence < 1 :
   if auto_adaptive1:
      solver.summary()
   list_timings()