"""
Heat conduction problem using NURBS

Equation to solve
      a(v,u) = <v,f> + <v,\overline{q}>|_{Gamma_b}

      where
      a(u,v) = int_Omega (\nabla v^T * kappa * \nabla u ) dOmega
      <v,f> = int_Omega (v * f) dOmega
      <v,\overline{q}>|_{Gamma_b} = int_Gamma_b (v * \overline{q}) dGamma_b

Geometry --> a rectangle of 1 per 1 where the bottom right corner is rounded
      inversely by a quarter circle of radius 0.5

Boundary conditions --> Left wall: Dirichlet u=0
                        Right wall: Dirichlet u=1
                        Top & Bottom & Cylinder: Neumann (fixed \overline{q}, may not be the same)

kappa --> thermal conductivity
f --> source term
"""
from __future__ import print_function
from fenics import *
from mshr import *
import math
import matplotlib.pyplot as plt
from tIGAr import *
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
from numpy import array

# Constant parameters
kappa = 1.0
Length = 1.0
Height = 1.0
f = Constant(0.0)

# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 0.0))

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], Length))

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], 0.0))

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], Height))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]>=Length/2 and x[0]<=Length and x[1]>=0.0 and x[1]<=Height/2)

# Create mesh
"""
channel = Rectangle(Point(0, 0), Point(Length, Length))
cylinder = Circle(Point(Length, 0), Length/2)
domain = channel - cylinder
mesh = generate_mesh(domain, 50)
"""
# Geometry creation
nx = 5 # number of horizontal points
ny = nx # numer of vertical points
cpArray = [[[0 for k in range(0,2)] for j in range(0,ny)] for i in range(0,nx)]
Ax = Length/(nx-1)
Ay = Height/(ny-1)
cpArray[0][0][0] = -1.0;
cpArray[0][0][1] = -1.0
for i in range(0,nx):
    for j in range(0,ny):
        if i == 0:
            cpArray[i][j][0] = -1.0
        else:
            cpArray[i][j][0] = cpArray[i-1][j][0] + Ax;
        if j == 0:
            cpArray[i][j][1] = -1.0
        else:
            cpArray[i][j][1] = cpArray[i][j-1][1] + Ay;

# Open knot vectors for a one-Bezier-element bi-unit square.
uKnots = [-1.0,-1.0,-1.0,1.0,1.0,1.0]
vKnots = [-1.0,-1.0,-1.0,1.0,1.0,1.0]
uKnots = []
vKnots = []
for i in range(0,nx):
    if i ==0:
        for n_rep in range(0,nx):
            uKnots.append(-1)
    elif i == nx-2:
        for n_rep in range(0,nx):
            uKnots.append(1)
    else:
        uKnots.append(-1 + 2/(nx-1) * i)

for i in range(0,ny):
    if i ==0:
        for n_rep in range(0,ny):
            vKnots.append(-1)
    elif i == nx-2:
        for n_rep in range(0,ny):
            vKnots.append(1)
    else:
        vKnots.append(-1 + 2/(ny-1) * i)
print(uKnots)
# Refinement of the mesh
N_LEVELS = 3

####### Geometry creation #######

# Parameter determining level of refinement
REF_LEVEL = N_LEVELS+3

# Create initial mesh
ikNURBS = NURBS_ik([uKnots,vKnots],cpArray)

# Refinement
numNewKnots = 1
for i in range(0,REF_LEVEL):
    numNewKnots *= 2
h = 2.0/float(numNewKnots)
numNewKnots -= 1
knotList = []
for i in range(0,numNewKnots):
    knotList += [float(i+1)*h-1.0,]
newKnots = array(knotList)
print(knotList)
ikNURBS.refine(0,newKnots)
ikNURBS.refine(1,newKnots)

# Output in PetIGA format
if(mpirank==0):
    PetIGA().write("out.dat",ikNURBS)
MPI.barrier(worldcomm)

# Creating a control mesh from the NURBS
splineMesh = NURBSControlMesh(ikNURBS,useRect=True)

# Create a spline generator for a spline with a single scalar field on the
# given control mesh, where the scalar field is the same as the one used
# to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
splineGenerator = EqualOrderSpline(1,splineMesh)
