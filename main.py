__author__ = 'obm'

###OBM 2014####
## "TURBO" EMT solver. Intended for charge state corrections. Contains PBC and Drichlet solvers.

import argparse
from dolfin import *
import datetime


def main(argv):
   run_id=datetime.datetime.now().strftime('%d%m%Y-%H%M')
   run_id=run_id.strip(' \t\n\r')
   #Settings etc.
   if not has_linear_algebra_backend("PETSc"):
       info("DOLFIN has not been configured with TPETSc. Exiting.")
       exit()
   if not has_cgal():
       print "DOLFIN must be compiled with CGAL to run this demo."
       exit(0)
   parameters["linear_algebra_backend"] = "PETSc"
   
   #Parallelization stuff
   if not has_mpi():
      info("This requires the parallel version. Exiting")
      exit()
   comm = mpi_comm_world()
   mpiRank = MPI.rank(comm)
   #
   parser = argparse.ArgumentParser(prog='poisson-3D-pbc-avz-saconv',description='A Fenics based program for Poisson equation with periodic boundary conditions')
   parser.add_argument('--alat',dest='alat',help="lattice constant",default=1.0,type=float,action="store")
   parser.add_argument('--celldm',nargs=3,dest='celldm',help="celldm 1 2 and 3",default=[1.0,1.0,1.0])
   parser.add_argument('--mesh',dest='meshdm',nargs=3,default=[64,64,64],type=int,help="Number of mesh points in three dimensions")
   parser.add_argument('--chg_loc',dest='chgloc',nargs=3,default=[0.5,0.5,0.5],type=float,help="Number of mesh points in three dimensions")
   parser.add_argument('--chg_spread',dest='chgsigma',help="Sigma of charge distrubution",default=0.1,type=float,action="store")
   parser.add_argument('--chg_tot',dest='chgtot',help="Total charge",default=1.0,type=float,action="store")
   parser.add_argument('--show_config', dest='showconfig', action='store_true',default=False)
   parser.add_argument('--adaptive_tol', dest='MeshTol',help="Adaptive mesh algorithm tolerance",type=float,default=0.1)
   parser.add_argument('--adaptive_maxiter',dest='mesh_maxiter',help="Adaptive mesh algorithm max iterations",type=int,default=100)
   parser.add_argument('--adaptive_refine',dest='mesh_refrat',help="Adaptive mesh algorithm refine ratio",type=float,default=0.5)
   parser.add_argument('--silent', dest='silent', action='store_true',default=False,help="Produces minimal screen output")
   parser.add_argument('--adaptive', dest='aalgo',default=1,type=int,help="Adaptive algorithms. 0: No adaptive mesh 1: AdaptiveLinearSolver ")
   parser.add_argument('--debug', dest='debug', action='store_true',default=False,help="Produces maximum screen output")
   parser.add_argument('--output_prefix', dest='fout_prefix',default=run_id,help="prefix of output files")
   parser.add_argument('--file_output_level', dest='fout_level',type=int,default=0,help="Number of files to be produced: 1: Only the potential; 2: files in 1 and charge, epsilon;  3: files in 2 and mesh")
   MPI.barrier(comm)
   args = parser.parse_args()
   #args = MPI.broadcast(comm,args,0)
   if args.aalgo==0 :
      single_shot=True
      auto_adaptive1=False
   elif args.aalgo==1 :
      single_shot=False
      auto_adaptive1=True
   else:
      print "Unsuported adaptive algorithm:",args.aalgo
      exit()
   fout_prefix=args.fout_prefix.strip(' \t\n\r')
   fout_level=args.fout_level
   mesh_maxiter=args.mesh_maxiter
   
   Silence=0
   if args.silent :
      Silence=1
   if args.debug :
      Silence=-9
   if Silence > 0 :
      set_log_level(30)
   elif Silence == 0 :
      set_log_level(PROGRESS)
   elif Silence == -9 :
      set_log_level(DEBUG)
   
   if Silence<1:
      print "Process",mpiRank," ready"
   
   MPI.barrier(comm)
   
   if mpiRank==0 and Silence<1:
      print "OBM's yet another poisson solver."
      print "Intended for charge state corrections"
      print "Parallel version .1 alpha"
      print "Run id:",run_id
      print "Total number of processors:",MPI.size(comm)
      print "Units:"
      print "Coordinates: alat (Bohr)"
      print "Energy: Hartree"
      print "Epsilon: Relative (Gauss)"
      print "Charge: Electrons"
   
   if  fout_level != 0 and Silence<1:
      if fout_level == 1 :
         print "Potential will be written in: "+fout_prefix+"-pot.pvd"
      elif fout_level== 2 :
         print "Potential will be written in: "+fout_prefix+"-pot.pvd"
         print "Charge will be written in: "+fout_prefix+"-chg.pvd"
         print "Epsilon will be written in: "+fout_prefix+"-eps.pvd"
      elif fout_level== 3 :
         print "Potential will be written in: "+fout_prefix+"-pot.pvd"
         print "Charge will be written in: "+fout_prefix+"-chg.pvd"
         print "Epsilon will be written in: "+fout_prefix+"-eps.pvd"
         print "Mesh will be written in: "+fout_prefix+"-mesh.pvd"
      else:
         print "Level ",fout_level," is not supported"
         exit()
   #Do not forget to converge mesh sizes!
   #In this program, the mesh sizes will be varied. Set the for loop for the range
   Mesh1=args.meshdm[0]
   Mesh2=args.meshdm[1]
   Mesh3=args.meshdm[2]
   MeshTol=args.MeshTol
   if mpiRank==0 and Silence<1 :
      print "Mesh:",Mesh1,"x",Mesh2,"x",Mesh3
      if auto_adaptive1 :
         print "Adaptive mesh: enabled"
         print "Tolerance:",MeshTol
         print "Maximum iterations:",mesh_maxiter
         #print "Refine ratio:",mesh_refrat
      print "------------------------------------------------"
   #cell size parameters
   #(in this program, alat will be varied, the size of the box is alat*celldm)
   alat=args.alat
   celldm1=args.celldm[0]
   celldm2=args.celldm[1]
   celldm3=args.celldm[2]
   if mpiRank==0  and Silence<1 :
      print "Supercell:"
      print "alat=",alat
      print "celldm1=",celldm1," celldm2=",celldm2," celldm3=",celldm3
      print "------------------------------------------------"
   
   
   #charge parameters
   #(in this program, the charge position is in crystal coordinates)
   Chg_locxC=args.chgloc[0]
   Chg_locyC=args.chgloc[1]
   Chg_loczC=args.chgloc[2]
   Chg_spread=args.chgsigma
   Chg_tot=args.chgtot
   if mpiRank==0 and Silence<1  :
     print "Charge:"
     print "Total=",Chg_tot
     print "x=",Chg_locxC," (",Chg_locxC*celldm1*alat," Bohr) y=",Chg_locyC," (",Chg_locyC*celldm2*alat," Bohr) z=",Chg_loczC," (",Chg_locxC*celldm1*alat,"  Bohr)"
     print "spread=",Chg_spread, " Bohr (",Chg_spread/alat," in alat coordinates)"
     print "------------------------------------------------"
   
   
   #dielectric function parameters
   #(disabled in this program)
   Eps2=1.0
   Eps1=2.45
   Z0_B=1.78
   Z0_T=6.22
   Beta=0.208645

if __name__ == "__main__":
   main(sys.argv[1:])
