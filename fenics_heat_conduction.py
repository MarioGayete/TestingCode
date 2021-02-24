"""
Heat conduction using FEniCS and tIGAr solver
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
import matplotlib.pyplot as plt
from tIGAr import *
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
from numpy import array
import math

# Constant parameters
kappa = 1.0
Length = 1.0
q_top = 0.0;
q_bottom = 0.0;
q_cyl = 0.0;
u_left = 0;
u_right = 1;
f = Constant(0.0)

# Number of levels of refinement with which to run the Poisson problem.
# (Note: Paraview output files will correspond to the last/highest level
# of refinement.)
N_LEVELS = 3

# Array to store error at different refinement levels:
L2_errors = zeros(N_LEVELS)

# Create classes for defining parts of the boundaries
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
        return (on_boundary and near(x[1], Length))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]>=Length/2 and x[0]<=Length and x[1]>=0.0 and x[1]<=Length/2)




# Create mesh
channel = Rectangle(Point(0, 0), Point(Length, Length))
cylinder = Circle(Point(Length, 0), Length/2)
domain = channel - cylinder
mesh = generate_mesh(domain, 50)

# Define function spaces
V = FunctionSpace(mesh, 'P', 1)

# Define boundaries
left  =  Left() #'on_boundary && near(x[0], 0)'
right  = Right()#' on_boundary && near(x[0], Length)'
top    = Top()#'on_boundary && near(x[1], Length)'
bottom = Bottom()#'on_boundary && near(x[1], 0)'
cylinder = Cylinder()#'on_boundary && x[0]>Length/2 && x[0]<Length && x[1]>0.0 && x[1]<Length/2'

# Mark boundaries
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
right.mark(boundary_markers, 2)
bottom.mark(boundary_markers, 3)
top.mark(boundary_markers, 4)
cylinder.mark(boundary_markers, 5)

# Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define each tipe of BC
boundary_conditions = {1: {'Dirichlet': u_left},
                       2: {'Dirichlet': u_right},
                       3: {'Neumann':   q_bottom},
                       4: {'Neumann':   q_top},
                       5: {'Neumann':   q_cyl}}

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Collect Dirichlet conditions
bcu = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
                         boundary_markers, i)
        bcu.append(bc)

# Collect Neumann Bounary conditions
integrals_N = []
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        if boundary_conditions[i]['Neumann'] != 0:
            g = boundary_conditions[i]['Neumann']
            integrals_N.append(g*v*ds(i))

# Define the variational problem
F = kappa*dot(grad(v), grad(u))*dx + \
   - dot(f,v)*dx \
   + sum(integrals_N)
a, L = lhs(F), rhs(F)


# Solving the problem
u_h = Function(V)
solve(a == L, u_h, bcu)


# Plot solution
plot(u_h, title='Temperature')
plt.savefig('heat_fenics/fenics_HC.png');
plot(mesh,title='Mesh')
plt.savefig('heat_fenics/fenics_HC_mesh.png')

# Save the subdomains
file = File("heat_fenics/subdomains.pvd")
file << boundary_markers

# Save to file
# Create VTK file for saving solution
vtkfile = File('heat_fenics/solution.pvd')
vtkfile << (u_h)
