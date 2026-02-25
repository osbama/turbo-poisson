from dolfin.cpp.common import MPI, info, has_mpi, mpi_comm_world, set_log_level, list_timings
from dolfin.cpp.function import near, between
from dolfin.cpp.io import File, interactive
from dolfin.cpp.la import list_linear_solver_methods, has_linear_algebra_backend
from dolfin.cpp.mesh import SubDomain, BoxMesh

__author__ = 'obm'

###OBM 2014####
# This is intended for Appendix A of Komsa, Hannu-Pekka and Pasquarello, Alfredo,doi:10.1103/PhysRevLett.110.095505
# Calculates the Energy of a gaussian charge sitting in a box filled with air.
#PBC is applied. The average potential in the cell is zero
#Energy output is in Hartree units (mind the 4\pi)
#Distance unit is Bohr
#Charge unit is electrons
#Dielectric constant is relative (unitless like in Gaussian unit formalism)

from dolfin import *
#from dolfin.fem.formmanipulations import increase_order
#from ufl import VectorElement
import numpy as np
import argparse
import datetime
#from math import pow
#from numpy import array, sqrt


run_id = datetime.datetime.now().strftime('%d%m%Y-%H%M')
run_id = run_id.strip(' \t\n\r')
#Settings etc.
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with TPETSc. Exiting.")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

#Parallelization stuff
if not has_mpi():
    info("This requires the parallel version. Exiting")
    exit()

comm = mpi_comm_world()
mpiRank = MPI.rank(comm)

#
parser = argparse.ArgumentParser(prog='poisson-3D-pbc-avz-v.2',
                                 description='A Fenics based program for Poisson equation with periodic boundary conditions')
parser.add_argument('--alat', dest='alat', help="lattice constant", default=1.0, type=float, action="store")
parser.add_argument('--celldm', nargs=3, dest='celldm', help="celldm 1 2 and 3", type=float, default=[1.0, 1.0, 1.0])
parser.add_argument('--mesh', dest='meshdm', nargs=3, default=[64, 64, 64], type=int,
                    help="Number of mesh points in three dimensions")
parser.add_argument('--charge-model', dest='chgmodel', choices=['gaussian'], default='gaussian',
                    help="Charge distrubution  model.")
parser.add_argument('--chg-parameters', dest='chgparam', nargs='+', default=[1.0, 0.5, 0.5, 0.5, 0.1], type=float,
                    help="Charge distrubution model parameters")
parser.add_argument('--charge-position-units', dest='chgposunit', choices=['alat', 'bohr', 'crystal'], default='alat',
                    help="Charge position units")
parser.add_argument('--charge-spread-units', dest='chgsprunit', choices=['alat', 'bohr'], default='bohr',
                    help="Charge spread units")
parser.add_argument('--epsilon-model', dest='epsmodel', choices=['constant', 'slab', 'left-right', 'top-bottom'],
                    default='constant', help="Dielectric function model.")
parser.add_argument('--epsilon-parameters', dest='epsparam', nargs='+', type=float, default=[1.0],
                    help="Dielectric function model parameters.")
parser.add_argument('--epsilon-position-units', dest='epsposunit', choices=['alat', 'bohr', 'crystal'], default='alat',
                    help="Charge position units")
parser.add_argument('--epsilon-spread-units', dest='epssprunit', choices=['alat', 'bohr'], default='bohr',
                    help="Charge spread units")
parser.add_argument('--show_config', dest='showconfig', action='store_true', default=False)
parser.add_argument('--adaptive_tol', dest='MeshTol', help="Adaptive mesh algorithm tolerance", type=float, default=0.1)
parser.add_argument('--adaptive_maxiter', dest='mesh_maxiter', help="Adaptive mesh algorithm max iterations", type=int,
                    default=100)
parser.add_argument('--adaptive_refine', dest='mesh_refrat', help="Adaptive mesh algorithm refine ratio", type=float,
                    default=0.5)
parser.add_argument('--silent', dest='silent', action='store_true', default=False,
                    help="Produces minimal screen output")
parser.add_argument('--adaptive', dest='aalgo', default='none', choices=['none', 'fenics'], help="Adaptive algorithms.")
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help="Produces maximum screen output")
parser.add_argument('--output-prefix', dest='fout_prefix', default=run_id, help="prefix of output files")
parser.add_argument('--linear_solver', dest='lin_solv', default='superlu_dist', help="Linear solver that will be used")
parser.add_argument('--file-output-level', dest='fout_level', type=int, default=0,
                    help="Number of files to be produced: 1: Only the potential; 2: files in 1 and charge, epsilon;  3: files in 2 and mesh")
parser.add_argument('--interactive', dest='interactive_graph', action='store_true', default=False, help="Produces interactive graphs")
MPI.barrier(comm)
args = parser.parse_args()
#args = MPI.broadcast(comm,args,0)

lin_solv = args.lin_solv.strip(' \t\n\r')
fout_prefix = args.fout_prefix.strip(' \t\n\r')
fout_level = args.fout_level
mesh_maxiter = args.mesh_maxiter

Silence = 0
if args.silent:
    Silence = 1
if args.debug:
    Silence = -9
if Silence > 0:
    set_log_level(30)
elif Silence == 0:
    set_log_level(PROGRESS)
elif Silence == -9:
    set_log_level(DEBUG)

if Silence < 1:
    print "Process", mpiRank, " ready"

MPI.barrier(comm)

if mpiRank == 0 and Silence < 1:
    print "OBM's yet another poisson solver."
    print "Intended for charge state corrections"
    print "Parallel version .2 alpha"
    print "Run id:", run_id
    print "Total number of processors:", MPI.size(comm)
    print "Units:"
    print "Coordinates: alat (Bohr)"
    print "Energy: Hartree"
    print "Epsilon: Relative (Gauss)"
    print "Charge: Electrons"

if fout_level != 0 and Silence < 1:
    if fout_level == 1:
        print "Potential will be written in: " + fout_prefix + "-pot.pvd"
    elif fout_level == 2:
        print "Potential will be written in: " + fout_prefix + "-pot.pvd"
        print "Charge will be written in: " + fout_prefix + "-chg.pvd"
        print "Epsilon will be written in: " + fout_prefix + "-eps.pvd"
    elif fout_level == 3:
        print "Potential will be written in: " + fout_prefix + "-pot.pvd"
        print "Charge will be written in: " + fout_prefix + "-chg.pvd"
        print "Epsilon will be written in: " + fout_prefix + "-eps.pvd"
        print "Mesh will be written in: " + fout_prefix + "-mesh.pvd"
    else:
        print "Level ", fout_level, " is not supported"
        exit()

#Do not forget to converge mesh sizes!
#In this program, the mesh sizes will be varied. Set the for loop for the range
Mesh1 = args.meshdm[0]
Mesh2 = args.meshdm[1]
Mesh3 = args.meshdm[2]
MeshTol = args.MeshTol
if mpiRank == 0 and Silence < 1:
    print "Mesh:", Mesh1, "x", Mesh2, "x", Mesh3
    if args.aalgo == 'fenics':
        print "Adaptive mesh: enabled"
        print "Tolerance:", MeshTol
        print "Maximum iterations:", mesh_maxiter
        #print "Refine ratio:",mesh_refrat
    print "------------------------------------------------"
#cell size parameters
#(in this program, alat will be varied, the size of the box is alat*celldm)
alat = args.alat
celldm1 = args.celldm[0]
celldm2 = args.celldm[1]
celldm3 = args.celldm[2]
if mpiRank == 0 and Silence < 1:
    print "Supercell:"
    print "alat=", alat
    print "celldm1=", celldm1, " celldm2=", celldm2, " celldm3=", celldm3
    print "------------------------------------------------"


#charge parameters
if args.chgmodel == 'gaussian':
    if len(args.chgparam) < 5:
        print "Not enough parameters for charge model"
        exit()
    if len(args.chgparam) > 5:
        print "Warning: Ignoring some parameters for charge"
    if args.chgposunit == 'alat':
        Chg_locx = args.chgparam[1] * alat
        Chg_locy = args.chgparam[2] * alat
        Chg_locz = args.chgparam[3] * alat
    elif args.chgposunit == 'crystal':
        Chg_locx = args.chgparam[1] * alat * celldm1
        Chg_locy = args.chgparam[2] * alat * celldm2
        Chg_locz = args.chgparam[3] * alat * celldm3
    elif args.chgposunit == 'bohr':
        Chg_locx = args.chgparam[1]
        Chg_locy = args.chgparam[2]
        Chg_locz = args.chgparam[3]
    if args.chgsprunit == 'alat':
        Chg_spread = args.chgparam[4] * alat
    elif args.chgsprunit == 'bohr':
        Chg_spread = args.chgparam[4]
    Chg_tot = args.chgparam[0]
    source_str = "{chg_tot}*exp(-((pow(x[0]-{chg_locx}, 2.0)+pow(x[1]-{chg_locy}, 2.0)+pow(x[2]-{chg_locz}, 2.0)))/(2.0*pow({chg_spread},2.0)))/pow((pow(2.0*pi,0.5)*{chg_spread}),3.0)" \
        .format(chg_tot=Chg_tot, chg_locx=Chg_locx, chg_locy=Chg_locy, chg_locz=Chg_locz, chg_spread=Chg_spread)
    if mpiRank == 0 and Silence < 1:
        print "Charge:"
        print "Total=", Chg_tot
        print "x=", Chg_locx / alat, " (", Chg_locx, " Bohr) y=", Chg_locy / alat, " (", Chg_locy, " Bohr) z=", Chg_locz / alat, " (", Chg_locz, "  Bohr)"
        print "spread=", Chg_spread / alat, " (", Chg_spread, " Bohr)"
        print "Expression(", source_str, ")"
        print "------------------------------------------------"


#dielectric function parameters
if args.epsmodel == 'slab':
    if len(args.epsparam) < 5:
        print "Not enough parameters for slab model"
        exit()
    if len(args.epsparam) > 5:
        print "Warning: Ignoring some parameters for epsilon"
    Eps2 = args.epsparam[0]
    Eps1 = args.epsparam[1]
    if args.epsposunit == 'alat':
        Z0_B = args.epsparam[2] * alat
        Z0_T = args.epsparam[3] * alat
    elif args.epsposunit == 'crystal':
        Z0_B = args.epsparam[2] * alat * celldm3
        Z0_T = args.epsparam[3] * alat * celldm3
    elif args.epsposunit == 'bohr':
        Z0_B = args.epsparam[2]
        Z0_T = args.epsparam[3]
    if args.epssprunit == 'alat':
        Beta = args.epsparam[4] * alat
    elif args.epssprunit == 'bohr':
        Beta = args.epsparam[4]
    eps_str = "0.5*({eps2}-{eps1})*erf((x[2]-{z0_B})/{beta})*erf((x[2]-{z0_T})/{beta}) + 0.5*({eps1}+{eps2})" \
        .format(eps2=Eps2, eps1=Eps1, z0_B=Z0_B, z0_T=Z0_T, beta=Beta)
    eps_gradx_str = "0"
    eps_grady_str = "0"
    eps_gradz_str = "0.56418958355*(({eps2}-{eps1})/{beta})*(exp(-pow((x[2]-{z0_B})/{beta},2))*erf((x[2]-{z0_T})/{beta})+erf((x[2]-{z0_B})/{beta})*exp(-pow((x[2]-{z0_T})/{beta},2)))" \
        .format(eps2=Eps2, eps1=Eps1, z0_B=Z0_B, z0_T=Z0_T, beta=Beta)
    if mpiRank == 0 and Silence < 1:
        print "Epsilon:"
        print "Outside=", Eps2
        print "Inside=", Eps1
        print "Start=", Z0_B / alat, " (", Z0_B, " Bohr) End=", Z0_T / alat, " (", Z0_T, " Bohr)"
        print "spread=", Beta / alat, " (", Beta, " Bohr)"
        print "epsilon=Expression(", eps_str, ")"
        print "epsilon_grad=Expression(", eps_gradx_str, ",", eps_grady_str, ",", eps_gradz_str, ")"
        print "------------------------------------------------"
elif args.epsmodel == 'left-right' or args.epsmodel == 'top-bottom':
    if len(args.epsparam) < 4:
        print "Not enough parameters for ", args.epsmodel, " model"
        exit()
    if len(args.epsparam) > 4:
        print "Warning: Ignoring some parameters for epsilon"
    Eps2 = args.epsparam[0]
    Eps1 = args.epsparam[1]
    if args.epsposunit == 'alat':
        Start = args.epsparam[2] * alat
    elif args.epsposunit == 'crystal':
        if args.epsmodel == 'left-right':
            Start = args.epsparam[2] * alat * celldm1
        elif args.epsmodel == 'top-bottom':
            Start = args.epsparam[2] * alat * celldm3
    elif args.epsposunit == 'bohr':
        Start = args.epsparam[2]
    if args.epssprunit == 'alat':
        Beta = args.epsparam[3] * alat
    elif args.epssprunit == 'bohr':
        Beta = args.epsparam[3]
    if args.epsmodel == 'left-right':
        eps_str = "0.5*({eps2}-{eps1})*erf((x[0]-{start})/{beta}) + 0.5*({eps1}+{eps2})" \
            .format(eps2=Eps2, eps1=Eps1, start=Start, beta=Beta)
    elif args.epsmodel == 'top-bottom':
        eps_str = "0.5*({eps2}-{eps1})*erf((x[2]-{start})/{beta}) + 0.5*({eps1}+{eps2})" \
            .format(eps2=Eps2, eps1=Eps1, start=Start, beta=Beta)
    #eps_gradx_str="0.56418958355*(({eps2}-{eps1})/{beta})*exp(-pow((x[0]-{start})/{beta},2))"\
    #               .format(eps2=Eps2,eps1=Eps1,start=Start,beta=Beta)
    #eps_grady_str="0"
    #eps_gradz_str="0"
    if mpiRank == 0 and Silence < 1:
        print "Epsilon:"
        if args.epsmodel == 'left-right':
            print "Left=", Eps2
            print "Right=", Eps1
        elif args.epsmodel == 'top-bottom':
            print "Top=", Eps2
            print "Bottom=", Eps1
        print "Start=", Start / alat, " (", Start, " Bohr)"
        print "spread=", Beta / alat, " (", Beta, " Bohr)"
        print "epsilon=Expression(", eps_str, ")"
        #print "epsilon_grad=Expression(",eps_gradx_str,",",eps_grady_str,",",eps_gradz_str,")"
        print "------------------------------------------------"
elif args.epsmodel == 'constant':
    if len(args.epsparam) < 1:
        print "Not enough parameters for constant epsilon"
        exit()
    if len(args.epsparam) > 1:
        print "Warning: Ignoring some parameters for epsilon"
    Eps = args.epsparam[0]
    eps_str = "{eps}" \
        .format(eps=Eps)
    if mpiRank == 0 and Silence < 1:
        print "Epsilon=", Eps
        print "Constant(", eps_str, ")"
        print "------------------------------------------------"


class PeriodicBC(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS, lx=1., ly=1., lz=1., length_scaling=1.):
        SubDomain.__init__(self)
        self.tol = tolerance
        self.lx = lx / length_scaling
        self.ly = ly / length_scaling
        self.lz = lz / length_scaling
        self.length_scaling = length_scaling

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0.) or near(x[1], 0.) or near(x[2], 0.)) and
                    (not ((near(x[0], 0.) and near(x[1], self.ly) and between(x[2], (0., self.lz)) ) or
                          (near(x[0], self.lx) and near(x[1], 0.) and between(x[2], (0., self.lz)) ) or
                          (near(x[0], self.lx) and between(x[1], (0., self.ly)) and near(x[2], 0.) ) or
                          (near(x[0], 0.) and between(x[1], (0., self.ly)) and near(x[2], self.lz) ) or
                          (between(x[0], (0., self.lx)) and near(x[1], 0.) and near(x[2], self.lz) ) or
                          (between(x[0], (0., self.lx)) and near(x[1], self.ly) and near(x[2], 0.) )
                    )) and on_boundary)

    def map(self, x, y):
        #surfaces to be folded. Inside statement exclusively handles overlap edges
        #L,L lines fold to 0,0
        if near(x[0], self.lx) and near(x[1], self.ly) and near(x[2], self.lz):  #line
            y[0] = x[0] - self.lx
            y[1] = x[1] - self.ly
            y[2] = x[2] - self.lz
        elif near(x[0], self.lx) and near(x[2], self.lz):
            y[0] = x[0] - self.lx
            y[1] = x[1]
            y[2] = x[2] - self.lz
        elif near(x[0], self.lx) and near(x[1], self.ly):
            y[0] = x[0] - self.lx
            y[1] = x[1] - self.ly
            y[2] = x[2]
        elif near(x[1], self.ly) and near(x[2], self.lz):
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
            y[2] = x[2] - self.lz
        else:
            y[0] = -10000
            y[1] = -10000
            y[2] = -10000


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
            self.c = assemble(Constant(1., cell=V.cell()) * dx, mesh=V.mesh())
            self.pp = ['0'] * self.u.value_size()
            self.pp[part] = '1'
            self.u0 = interpolate(Expression(self.pp, element=V.ufl_element()), V)
            self.x0 = self.u0.vector()
            self.C1 = assemble(v[self.part] * dx)
        else:
            self.u = Function(V)
            self.vv = self.u.vector()

    def __call__(self, v):
        if isinstance(self.part, int):
            # assemble into c1 the part of the vector that we want to normalize
            c1 = self.C1.inner(v)
            if abs(c1) > 1.e-8:
                # Perform normalization
                self.x0[:] = self.x0[:] * (c1 / self.c)
                v.axpy(-1., self.x0)
                self.x0[:] = self.x0[:] * (self.c / c1)
        else:
            # normalize entire vector
            #dummy = normalize(v) # does not work in parallel
            #self.vv = Vector(v)
            self.vv[:] = 1. / v.size()
            c = v.inner(self.vv)
            self.vv[:] = c
            v.axpy(-1., self.vv)


MPI.barrier(comm)


class epsilon_func(Expression):
    def eval(self, values, x):
        val = sin(x[0])
        values[0] = val
        values[1] = val
        values[2] = val

    def value_shape(self):
        return (3,)


#This is where the main begins

parameters["form_compiler"]["optimize"] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-Ofast -flto'
#Show the available settings. Note that this method always show the default settings, not the ones set by the user
if args.showconfig:
    for ranks in range(0, MPI.size(comm)):
        if mpiRank == 0:
            print "Available solvers as reported by master process:"
            list_linear_solver_methods()
            if mpiRank == ranks:
                print parameters["form_compiler"]["quadrature_degree"]


####Automated solution from Documented example using lagrange multipliers method (not the krylov method)
# Create supercell and mesh and function space

mesh = BoxMesh(0.0, 0.0, 0.0, celldm1 * alat, celldm2 * alat, celldm3 * alat, Mesh1, Mesh2, Mesh3)
pb = PeriodicBC(lx=celldm1 * alat, ly=celldm2 * alat, lz=celldm3 * alat)

BDM = FunctionSpace(mesh, "BDM", 2, constrained_domain=pb)
DG = FunctionSpace(mesh, "DG", 0, constrained_domain=pb)
R = FunctionSpace(mesh, 'R', 0, constrained_domain=pb)
W = MixedFunctionSpace([BDM, DG, R])

(sigma, u, mu) = TrialFunctions(W)
(tau, v, nu) = TestFunctions(W)

# Define source function
rho = Expression(source_str, element=DG.ufl_element())
if args.epsmodel == 'constant':
    epsilon = Constant(eps_str)
    a = (dot(sigma, tau) + div(tau) * u + epsilon * div(sigma) * v) * dx - inner(u, nu) * dx - inner(mu, v) * dx
else:
    #epsilon=interpolate(Expression(eps_str),DG)
    epsilon = Expression(eps_str, element=DG.ufl_element())
    a = (dot(sigma, tau) + div(tau) * u + div(epsilon * sigma) * v) * dx - inner(u, nu) * dx - inner(mu, v) * dx
    #epsilon_grad=Expression((eps_gradx_str,eps_grady_str,eps_gradz_str),element=BDM.ufl_element()) #now it is a vector
    #a = (dot(sigma, tau) + div(tau)*u + epsilon*v*div(sigma)+dot(epsilon_grad,sigma)*v)*dx - inner(u, nu)*dx - inner(mu, v)*dx
# Define variational form
#a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx - inner(u, nu)*dx - inner(mu, v)*dx

L = - 4 * np.pi * ((rho * v) * dx)

if args.showconfig:
    for ranks in range(0, MPI.size(comm)):
        if mpiRank == 0:
            print "Available solvers as reported by master process:"
            list_linear_solver_methods()
            if mpiRank == ranks:
                print "Quadrature degree:", parameters["form_compiler"]["quadrature_degree"]


##
#Auto adaptive
##
if args.aalgo == 'fenics':
    if mpiRank == 0 and Silence < 1:
        print "Starting Auto adaptive algorithm 1"
    #Error
    uw = Function(W)
    (sigma, u, mu) = split(uw)
    #print "rho element", rho.ufl_element().degree()
    M = u * rho * dx

    #Adaptive solver
    problem = LinearVariationalProblem(a, L, uw)
    solver = AdaptiveLinearVariationalSolver(problem, M)
    solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = lin_solv
    solver.parameters["linear_variational_solver"]["linear_solver"] = lin_solv
    solver.parameters["max_iterations"] = mesh_maxiter

    if args.showconfig:
        for ranks in range(0, MPI.size(comm)):
            if mpiRank == ranks:
                print "**--Solver configuration for process ", ranks, " :--**"
                info(solver.default_parameters(), True)
                print "Quadrature degree:", parameters["form_compiler"]["quadrature_degree"]
            MPI.barrier(comm)
    #solve
    solver.solve(MeshTol)
    # Report results
    V_new = FunctionSpace(mesh.leaf_node(), "DG", 0)
    chg_ongrid_new = Function(V_new)
    chg_ongrid_new.interpolate(rho)
    chg_new = chg_ongrid_new * dx
    chg_new_int = assemble(chg_new)
    eps_ongrid_new = Function(V_new)
    eps_ongrid_new.interpolate(epsilon)
    V_old = FunctionSpace(mesh.root_node(), "DG", 0)
    chg_ongrid_old = Function(V_old)
    chg_ongrid_old.interpolate(rho)
    chg_old = chg_ongrid_old * dx
    chg_old_int = assemble(chg_old)
    eps_ongrid_old = Function(V_old)
    eps_ongrid_old.interpolate(epsilon)
    #E integral
    sigma_new, phi_new, mu_new = split(uw.leaf_node())
    E_new = phi_new * chg_ongrid_new * dx
    E_int_new = assemble(E_new)
    E_int_new = E_int_new * 0.5
    sigma_old, phi_old, mu_old = split(uw.root_node())
    E_old = phi_old * chg_ongrid_old * dx
    E_int_old = assemble(E_old)
    E_int_old = E_int_old * 0.5




##
#  Without adaptive
##

elif args.aalgo == 'none':
    if mpiRank == 0 and Silence < 1:
        print "Starting single shot algorithm"
    uw = Function(W)
    solve(a == L, uw, solver_parameters={"linear_solver": lin_solv})

    # Report results
    sigma, phi, mu = uw.split()
    V = FunctionSpace(mesh, "DG", 0)
    chg_ongrid = Function(V)
    chg_ongrid.interpolate(rho)
    chg = chg_ongrid * dx
    chg_int = assemble(chg)
    eps_ongrid = Function(V)
    eps_ongrid.interpolate(epsilon)
    #E integral
    E = phi * rho * dx
    E_int = assemble(E)
    E_int = E_int * 0.5

else:
    if mpiRank == 0:
        print "Please select a method for solution"
    exit()


############REPORTS
#print "depth",mesh.depth()
#print "depth",mesh.leaf_node()
if mpiRank == 0:
    if Silence < 1:
        if args.aalgo == 'fenics':
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
            print "                 RESULTS                                          "
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
            print "alat:", alat, " celldm1:", celldm1, " celldm2:", celldm2, " celldm3:", celldm3
            print "Number of cells before adaptive optimization:", mesh.num_cells()
            print "Number of cells after adaptive optimization:", mesh.leaf_node().num_cells()
            print "Total energy calculated using initial mesh:", E_int_old
            print "Total energy calculated using final mesh:", E_int_new
            print "Integrated charge in the inital mesh:", chg_old_int
            print "Integrated charge in the final mesh:", chg_new_int
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        elif args.aalgo == 'none':
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
            print "                 RESULTS                                          "
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
            print "alat:", alat, " celldm1:", celldm1, " celldm2:", celldm2, " celldm3:", celldm3
            print "Number of cells:", mesh.num_cells()
            print "Total energy:", E_int
            print "Integrated charge:", chg_int
            print "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    elif Silence == 1:
        if args.aalgo == 'fenics':
            print "#Old number of cells, New number of cells, celldm1,celldm2,celldm3,alat, E_old, E_new, chg_int_old, chg_int_new:%12.8e %12.8e %8f %8f %8f %12.8e %12.8e %12.8e %12.8e %12.8e " % (
                mesh.num_cells(), mesh.leaf_node().num_cells(), celldm1, celldm2, celldm3, alat, E_int_old, E_int_new,
                chg_old_int, chg_new_int)
        elif args.aalgo == 'none':
            print "#number of cells, celldm1,celldm2,celldm3,alat, E, chg_int:%12.8e %8f %8f %8f %12.8e %12.8e %12.8e " % (
                mesh.num_cells(), celldm1, celldm2, celldm3, alat, E_int, chg_int)
    elif Silence > 1:
        if args.aalgo == 'fenics':
            print E_int_new
        if args.aalgo == 'none':
            print E_int
if interactive_graph :
    plot (mesh.leaf_node())
    interactive()
    
#Save file in vtk format
if fout_level != 0:
    if args.aalgo == 'none':
        if fout_level == 1:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi
        elif fout_level == 2:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi
            filechg = File(fout_prefix + "-chg.pvd")
            filechg << chg_ongrid
            fileeps = File(fout_prefix + "-eps.pvd")
            fileeps << eps_ongrid
        elif fout_level == 3:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi
            filechg = File(fout_prefix + "-chg.pvd")
            filechg << chg_ongrid
            fileeps = File(fout_prefix + "-eps.pvd")
            fileeps << eps_ongrid
            filemesh = File(fout_prefix + "-mesh.pvd")
            filemesh << mesh
    else:
        sigma_old, phi_old, mu_old = uw.root_node().split()
        sigma_new, phi_new, mu_new = uw.leaf_node().split()
        if fout_level == 1:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi_old
            filepot << phi_new
        elif fout_level == 2:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi_old
            filepot << phi_new
            filechg = File(fout_prefix + "-chg.pvd")
            filechg << chg_ongrid_old
            filechg << chg_ongrid_new
            fileeps = File(fout_prefix + "-eps.pvd")
            fileeps << eps_ongrid_old
            fileeps << eps_ongrid_new
        elif fout_level == 3:
            filepot = File(fout_prefix + "-pot.pvd")
            filepot << phi_old
            filepot << phi_new
            filechg = File(fout_prefix + "-chg.pvd")
            filechg << chg_ongrid_old
            filechg << chg_ongrid_new
            fileeps = File(fout_prefix + "-eps.pvd")
            fileeps << eps_ongrid_old
            fileeps << eps_ongrid_new
            filemesh = File(fout_prefix + "-mesh.pvd")
            filemesh << mesh.root_node()
            filemesh << mesh.leaf_node()


#######FINAL
if Silence < 1:
    if args.aalgo == 'fenics':
        solver.summary()
    list_timings()
