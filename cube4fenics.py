from dolfin.cpp.mesh import MeshFunction, BoxMesh, Point,cells

__author__ = 'obm'
import numpy as np
from math import ceil, floor, sqrt
from dolfin import *
#from fenicstools import *
import argparse
import datetime

run_id = datetime.datetime.now().strftime('%d%m%Y-%H%M')
run_id = run_id.strip(' \t\n\r')

parser = argparse.ArgumentParser(prog='Polarizability to dielectric',
                                 description='A Fenics based program for convering from polarizability to dielectric const')
parser.add_argument('--output-prefix', dest='fout_prefix',type=str, default=run_id, help="prefix of output files")
parser.add_argument('--celldm', nargs=3, dest='celldm', help="celldm 1 2 and 3", type=float, default=[1.0, 1.0, 1.0])
parser.add_argument('--mesh', dest='meshdm', nargs=3, default=[64, 64, 64], type=int,
                    help="Number of mesh points in three dimensions")
parser.add_argument('--cube_file', dest='read_cube', default='none', help="The cube file that contains polarizability")
parser.add_argument('--use_mesh', dest='external_mesh', default='none', help="Use this mesh XML to calculate polarizability")
args = parser.parse_args()
fout_prefix = args.fout_prefix.strip(' \t\n\r')
read_cube=args.read_cube.strip(' \t\n\r')
class MyExpr3D(Expression):
    def __init__(self, cell_fun):
        assert(cell_fun.dim()==3)
        self.cell_fun = cell_fun
    def eval_cell(self, values, x, cell):
        values[0] = self.cell_fun[cell.index]
class cube_file:
  def __init__(self, fname):
    f = open(fname, 'r')
    for i in range(2): f.readline() # echo comment
    tkns = f.readline().split() # number of atoms included in the file followed by the position of the origin of the volumetric data
    self.natoms = int(tkns[0])
    self.origin = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
# The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector.
    tkns = f.readline().split() #
    self.NX = int(tkns[0])
    self.X = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
    tkns = f.readline().split() #
    self.NY = int(tkns[0])
    self.Y = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
    tkns = f.readline().split() #
    self.NZ = int(tkns[0])
    self.Z = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
# The last section in the header is one line for each atom consisting of 5 numbers, the first is the atom number, second (?), the last three are the x,y,z coordinates of the atom center.
    self.atoms = []
    for i in range(self.natoms):
      tkns = f.readline().split()
      self.atoms.append([tkns[0], tkns[2], tkns[3], tkns[4]])
# Volumetric data
    self.data = np.zeros((self.NX,self.NY,self.NZ))
    i=0
    for s in f:
      for v in s.split():
        self.data[i/(self.NY*self.NZ), (i/self.NZ)%self.NY, i%self.NZ] = float(v)
        i+=1
    if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"

  def dump(self, f):
# output Gaussian cube into file descriptor "f".
# Usage pattern: f=open('filename.cube'); cube.dump(f); f.close()
    print >>f, "CUBE file\nOBM tests"
    print >>f, "%4d %.6f %.6f %.6f" % (self.natoms, self.origin[0], self.origin[1], self.origin[2])
    print >>f, "%4d %.6f %.6f %.6f"% (self.NX, self.X[0], self.X[1], self.X[2])
    print >>f, "%4d %.6f %.6f %.6f"% (self.NY, self.Y[0], self.Y[1], self.Y[2])
    print >>f, "%4d %.6f %.6f %.6f"% (self.NZ, self.Z[0], self.Z[1], self.Z[2])
    for atom in self.atoms:
      print >>f, "%s %d %s %s %s" % (atom[0], 0, atom[1], atom[2], atom[3])
    for ix in xrange(self.NX):
      for iy in xrange(self.NY):
         for iz in xrange(self.NZ):
            print >>f, "%.5e " % self.data[ix,iy,iz],
            if (iz % 6 == 5): print >>f, ''
         print >>f,  ""

  def mask_sphere(self, R, Cx,Cy,Cz):
# produce spheric volume mask with radius R and center @ [Cx,Cy,Cz]
# can be used for integration over spherical part of the volume
    m=0*self.data
    for ix in xrange( int(ceil((Cx-R)/self.X[0])), int(floor((Cx+R)/self.X[0])) ):
      ryz=sqrt(R**2-(ix*self.X[0]-Cx)**2)
      for iy in xrange( int(ceil((Cy-ryz)/self.Y[1])), int(floor((Cy+ryz)/self.Y[1])) ):
          rz=sqrt(ryz**2 - (iy*self.Y[1]-Cy)**2)
          for iz in xrange( int(ceil((Cz-rz)/self.Z[2])), int(floor((Cz+rz)/self.Z[2])) ):
              m[ix,iy,iz]=1
    return m
  def calculate_Pv(self,num_boxes_x,num_boxes_y,num_boxes_z,dump_prefix):
#Calculates dipole moment per volume, num_boxes_(x,y,z) determine the number of boxes in each cardinal direction
     start_point=Point(0.0,0.0,0.0)
     end_point=Point((self.NX * self.X[0]+self.NY * self.Y[0]+self.NZ * self.Z[0]),(self.NX * self.X[1]+self.NY * self.Y[1]+self.NZ * self.Z[1]),(self.NX * self.X[2]+self.NY * self.Y[2]+self.NZ * self.Z[2]) )
     print start_point
     print end_point
     mesh_cube = BoxMesh(start_point,end_point, int(self.NX), int(self.NY), int(self.NZ))
     mesh_diel = BoxMesh(start_point,end_point, int(self.NX/num_boxes_x), int(self.NY/num_boxes_y), int(self.NZ/num_boxes_z))
     cube_inmesh=MeshFunction("double",mesh_cube,3)
     dump_data_rx=MeshFunction("double",mesh_diel,3)
     dump_data_ry=MeshFunction("double",mesh_diel,3)
     dump_data_rz=MeshFunction("double",mesh_diel,3)
     filled_cell_markers = dolfin.CellFunction("bool", mesh_cube)

     increment=self.NX*self.NY*self.NZ/100
     all_elements=self.NX*self.NY*self.NZ
     counter=0
     per_counter=0
     max_x=self.NX * self.X[0]+self.NY*self.Y[0]+self.NZ*self.Z[0]
     max_y=self.NX * self.X[1]+self.NY*self.Y[1]+self.NZ*self.Z[1]
     max_z=self.NX * self.X[2]+self.NY*self.Y[2]+self.NZ*self.Z[2]
     counter=0
     for cell in cells(mesh_cube) :
        #counter+=1
        cur_x=cell.midpoint().x()
        cur_y=cell.midpoint().y()
        cur_z=cell.midpoint().z()
        ix=floor(self.NX*cur_x/max_x)
        iy=floor(self.NY*cur_y/max_y)
        iz=floor(self.NZ*cur_z/max_z)
        cube_inmesh[cell]=cube_inmesh[cell]+self.data[ix,iy,iz]
     #   print counter,"/",all_elements," ",ix,",",iy,",",iz," : ",self.data[ix,iy,iz]
     #This one is faster (but less safe)
     #for iz in xrange(self.NX):
     # for iy in xrange(self.NY):
     #    for ix in xrange(self.NZ):
     #      cube_inmesh[counter]=self.data[ix,iy,iz]
     #      counter=counter+1
     print "Writing ",dump_prefix,"-cube.pvd"
     #cube_dump_file = File(dump_prefix + "-cube.pvd")
     #cube_dump_file << cube_inmesh
     print "Writing ",dump_prefix,"-cube.xml.gz"
     #cube_dump_file = File(dump_prefix + "-cube.xml.gz")
     #cube_dump_file << cube_inmesh
     #Too slow
     #for cell_cube in cells(mesh_cube) :
     #    for cell_diel in cells(mesh_diel) :
     #        if cell_diel.contains(cell_cube.midpoint()) :
     #            dump_data_rx[cell_diel]=dump_data_rx[cell_diel]+(cell_cube.midpoint().x()-cell_cube.midpoint().x())*cube_inmesh[cell_cube]
     #            dump_data_ry[cell_diel]=dump_data_ry[cell_diel]+(cell_cube.midpoint().y()-cell_cube.midpoint().y())*cube_inmesh[cell_cube]
     #            dump_data_rz[cell_diel]=dump_data_rz[cell_diel]+(cell_cube.midpoint().z()-cell_cube.midpoint().z())*cube_inmesh[cell_cube]
     #            break
     p=MyExpr3D(cube_inmesh)
     V=FunctionSpace(mesh_diel,"Lagrange",1)
     r=Function(V)
     r2=Function(V)
     #dipole_vector_values = np.zeros(mesh.num_vertices()*3)
     #dipole_vector_values[::2] = mesh_diel.coordinates().sum(1)
     parameters['allow_extrapolation'] = True
     r=project(p,V)

     c00_file = File(dump_prefix + "-x.pvd")
     c00_file<<r



cube=cube_file(read_cube)
print "Nx: %d"%cube.NX
print "Ny: %d"%cube.NY
print "Nz: %d"%cube.NZ
cube.calculate_Pv(5,5,5,fout_prefix)