__author__ = 'obm'
from dolfin.cpp.common import MPI, info, has_mpi, mpi_comm_world, set_log_level, list_timings
from dolfin.cpp.function import near, between
from dolfin.cpp.io import File, interactive
from dolfin.cpp.la import list_linear_solver_methods, has_linear_algebra_backend
from dolfin.cpp.mesh import SubDomain, BoxMesh, Mesh, MeshFunction


__author__ = 'obm'

###OBM 2015####
#Periodic file reader test
#


from dolfin import *
#from dolfin.fem.formmanipulations import increase_order
#from ufl import VectorElement
import numpy as np
import argparse
import datetime
import os.path
#from math import pow
#from numpy import array, sqrt


run_id = datetime.datetime.now().strftime('%d%m%Y-%H%M')
run_id = run_id.strip(' \t\n\r')
#Settings etc.

parameters["linear_algebra_backend"] = "PETSc"

#Parallelization stuff
if not has_mpi():
    info("This requires the parallel version. Exiting")
    exit()

comm = mpi_comm_world()
mpiRank = MPI.rank(comm)

#
parser = argparse.ArgumentParser(prog='file reader test',
                                 description='test for memory conservative reading of files')
parser.add_argument('--alat', dest='alat', help="lattice constant", default=1.0, type=float, action="store")
parser.add_argument('--celldm', nargs=3, dest='celldm', help="celldm 1 2 and 3", type=float, default=[1.0, 1.0, 1.0])
parser.add_argument('--mesh', dest='meshdm', nargs=3, default=[64, 64, 64], type=int,
                    help="Number of mesh points in three dimensions")
parser.add_argument('--epsilon-model', dest='epsmodel', choices=['constant', 'slab', 'left-right', 'top-bottom','read'],
                    default='constant', help="Dielectric function model.")
parser.add_argument('--epsilon-parameters', dest='epsparam', nargs='+', type=str, default=[1.0],
                    help="Dielectric function model parameters.")
parser.add_argument('--epsilon-position-units', dest='epsposunit', choices=['alat', 'bohr', 'crystal'], default='alat',
                    help="Charge position units")
parser.add_argument('--epsilon-spread-units', dest='epssprunit', choices=['alat', 'bohr'], default='bohr',
                    help="Charge spread units")
parser.add_argument('--silent', dest='silent', action='store_true', default=False,
                    help="Produces minimal screen output")
parser.add_argument('--output-prefix', dest='fout_prefix',type=str, default=run_id, help="prefix of output files")
parser.add_argument('--dump_eps', dest='dump_eps', action='store_true',default=False,
                    help="Write the epsilon to a file")


MPI.barrier(comm)
args = parser.parse_args()
#args = MPI.broadcast(comm,args,0)

####################################
#Reflexes for the input
#####################################
fout_prefix = args.fout_prefix.strip(' \t\n\r')


Silence = 0
if args.silent:
    Silence = 1
if Silence > 0:
    set_log_level(30)
elif Silence == 0:
    set_log_level(PROGRESS)
elif Silence == -9:
    set_log_level(DEBUG)

#Mesh
#Do not forget to converge mesh sizes!
#In this program, the mesh sizes will be varied. Set the for loop for the range
Mesh1 = args.meshdm[0]
Mesh2 = args.meshdm[1]
Mesh3 = args.meshdm[2]
#cell size parameters
#(in this program, alat will be varied, the size of the box is alat*celldm)
alat = args.alat
celldm1 = args.celldm[0]
celldm2 = args.celldm[1]
celldm3 = args.celldm[2]
start_point=Point(0.0,0.0,0.0)
end_point=Point(celldm1 * alat, celldm2 * alat, celldm3 * alat)
mesh = BoxMesh(start_point,end_point, Mesh1, Mesh2, Mesh3)


Q  = FunctionSpace(mesh,"DG",0)

#dielectric function parameters
if args.epsmodel == 'slab':
    if len(args.epsparam) < 5:
        print "Not enough parameters for slab model"
        exit()
    if len(args.epsparam) > 5:
        print "Warning: Ignoring some parameters for epsilon"
    Eps2 = float(args.epsparam[0])
    Eps1 = float(args.epsparam[1])
    if args.epsposunit == 'alat':
        Z0_B = float(args.epsparam[2]) * alat
        Z0_T = float(args.epsparam[3]) * alat
    elif args.epsposunit == 'crystal':
        Z0_B = float(args.epsparam[2]) * alat * celldm3
        Z0_T = float(args.epsparam[3]) * alat * celldm3
    elif args.epsposunit == 'bohr':
        Z0_B = float(args.epsparam[2])
        Z0_T = float(args.epsparam[3])
    if args.epssprunit == 'alat':
        Beta = float(args.epsparam[4]) * alat
    elif args.epssprunit == 'bohr':
        Beta = float(args.epsparam[4])
    eps_str = "0.5*({eps2}-{eps1})*erf((x[2]-{z0_B})/{beta})*erf((x[2]-{z0_T})/{beta}) + 0.5*({eps1}+{eps2})" \
        .format(eps2=Eps2, eps1=Eps1, z0_B=Z0_B, z0_T=Z0_T, beta=Beta)
    eps_gradx_str = "0"
    eps_grady_str = "0"
    eps_gradz_str = "0.56418958355*(({eps2}-{eps1})/{beta})*(exp(-pow((x[2]-{z0_B})/{beta},2))*erf((x[2]-{z0_T})/{beta})+erf((x[2]-{z0_B})/{beta})*exp(-pow((x[2]-{z0_T})/{beta},2)))" \
        .format(eps2=Eps2, eps1=Eps1, z0_B=Z0_B, z0_T=Z0_T, beta=Beta)
    epsilon = Expression(eps_str, element=Q.ufl_element())
elif args.epsmodel == 'left-right' or args.epsmodel == 'top-bottom':
    if len(args.epsparam) < 4:
        print "Not enough parameters for ", args.epsmodel, " model"
        exit()
    if len(args.epsparam) > 4:
        print "Warning: Ignoring some parameters for epsilon"
    Eps2 = float(args.epsparam[0])
    Eps1 = float(args.epsparam[1])
    if args.epsposunit == 'alat':
        Start = float(args.epsparam[2]) * alat
    elif args.epsposunit == 'crystal':
        if args.epsmodel == 'left-right':
            Start = float(args.epsparam[2]) * alat * celldm1
        elif args.epsmodel == 'top-bottom':
            Start = float(args.epsparam[2]) * alat * celldm3
    elif args.epsposunit == 'bohr':
        Start = float(args.epsparam[2])
    if args.epssprunit == 'alat':
        Beta = float(args.epsparam[3]) * alat
    elif args.epssprunit == 'bohr':
        Beta = float(args.epsparam[3])
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
    epsilon = Expression(eps_str, element=Q.ufl_element())
elif args.epsmodel == 'constant':
    if len(args.epsparam) < 1:
        print "Not enough parameters for constant epsilon"
        exit()
    if len(args.epsparam) > 1:
        print "Warning: Ignoring some parameters for epsilon"
    Eps = float(args.epsparam[0])
    eps_str = "{eps}" \
        .format(eps=Eps)
    epsilon = Constant(eps_str, element=Q.ufl_element())
elif args.epsmodel == 'read' :
    if len(args.epsparam) < 6:
        print "Not enough parameters for constant epsilon"
        exit()
    if len(args.epsparam) > 6:
        print "Warning: Ignoring some parameters for epsilon"
    epsilon_code = """

class Epsilon_code : public Expression
{
public:

  // Create expression with 6 components
  Epsilon() : Expression(6) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const double * const * x_a = cell.midpoint;
    double x_c[3];
    uint folded_index;
    uint folded_index_temp,r_temp;
    x_c=cell.midpoint.x;

    while (x_c > xpmax) {
        x_c = x_c -xpmax;
    }
    while (y_c > ypmax) {
        y_c = y_c -ypmax;
    }
    while (z_c > zpmax) {
        z_c = z_c -zpmax;
    }

    values[0] = (*c00)[folded_index];
    values[1] = (*c01)[folded_index];
    values[2] = (*c02)[folded_index];
    values[3] = (*c11)[folded_index];
    values[4] = (*c12)[folded_index];
    values[5] = (*c22)[folded_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<double> > c00;
  std::shared_ptr<MeshFunction<double> > c01;
  std::shared_ptr<MeshFunction<double> > c02;
  std::shared_ptr<MeshFunction<double> > c11;
  std::shared_ptr<MeshFunction<double> > c12;
  std::shared_ptr<MeshFunction<double> > c22;
  double xpmax;
  double ypmax;
  double zpmax;
};
"""
    c00 = MeshFunction("double", submesh, args.epsparam[0])
    c01 = MeshFunction("double", submesh, args.epsparam[1])
    c02 = MeshFunction("double", submesh, args.epsparam[2])
    c11 = MeshFunction("double", submesh, args.epsparam[3])
    c12 = MeshFunction("double", submesh, args.epsparam[4])
    c22 = MeshFunction("double", submesh, args.epsparam[5])
    epsilonc = Expression(cppcode=epsilon_code,domain=mesh)
    epsilonc.c00 = c00
    epsilonc.c01 = c01
    epsilonc.c02 = c01
    epsilonc.c11 = c11
    epsilonc.c12 = c12
    epsilonc.c22 = c11
    epsilon = as_matrix(((epsilonc[0], epsilonc[1],epsilonc[2]), (epsilonc[1], epsilonc[3],epsilonc[4]), (epsilonc[2], epsilonc[4],epsilonc[5])))

if args.dump_eps:
      eps_ongrid = Function(Q)
      eps_ongrid.interpolate(epsilon)
      fileeps = File(fout_prefix + "-eps.xml.gz")
      fileeps << eps_ongrid
