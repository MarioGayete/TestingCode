"""
Author: Mario Gayete
Last edited in: March 8th, 2021


Solving the Oseen equations using NURBS

Equation to solve
      a(v,u) - b(p,v) + S(u,v) + b(q,u) = L(v)

      where
      a(u,v) := (sigma u + (beta·nabla)u,v)_Omega + mu(nabla u, nabla v)_Omega
      b(p,v) := (p,nabla · v)_Omega
      S(u,v) := delta_0 {(tau curl(Lapl(u), Lapl(v))_h + <h^2[(beta · nabla)u] x n,[(beta · nabla)v] x n>_{F_i}}
      L(v) := (f,v)_Omega + delta_0 (tau curl(f),curl(Lapl(v)))_h

      Lapl(u) :=  sigma u + (beta·nabla)u - mu Delta u

Constants:
    sigma
    delta_0
    tau
    beta

Geometry --> a rectangle of 1 per 1

Problem to solve --> Potential Flow proposed on the paper
    u = grad(g)
    p = -0.5 |grad(h)|^2 + 14/5

    h = x^3 - 3xy^2
    f = 0
    beta = u
"""
from __future__ import print_function
from fenics import *
from mshr import *
import math
import matplotlib.pyplot as plt_fig
from tIGAr import *
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
from igakit.cad import *
from igakit.plot import plt
from numpy import array
from numpy import linspace
from numpy import pi
from numpy import zeros
from numpy import ogrid

# Constant parameters
Length = 1.0
Height = 1.0
f = Constant(0.0)
sigma = 0.0
Re = 1e3
mu = 1./Re
delta0 = 1e-9


####### Geometry creation #######
print("Generating the geometry...")
points = np.zeros((2,2,2), dtype='d')
s = slice(0, +1.0, 2j)
x, y = np.ogrid[0:Length:2j, 0:Height:2j]
points = np.zeros((2,2,2), dtype='d')
points[...,0] = x
points[...,1] = y
srf = bilinear(points)

plt.plot(srf)
plt_fig.savefig('savings/surface.png')

# Refinement of the mesh
print("Refining mesh...")
N_LEVELS = 3

# Finding the minimum and maximum value of the knots in both directions
max_uKnots = max(srf.knots[0])
min_uKnots = min(srf.knots[0])
max_vKnots = max(srf.knots[1])
min_vKnots = min(srf.knots[1])

# Parameter determining level of refinement
REF_LEVEL = N_LEVELS+3

# Refinement
to_insert_u = linspace(min_uKnots,max_uKnots,2**REF_LEVEL)[1:-1]
to_insert_v = linspace(min_vKnots,max_vKnots,2**REF_LEVEL)[1:-1]
srf.refine(0,to_insert_u)
srf.refine(1,to_insert_v)

plt.plot(srf)
plt_fig.savefig('savings/refined_surface.png')

# Creating a control mesh from the NURBS
splineMesh = NURBSControlMesh(srf,useRect=False)

# Create a spline generator for a spline with a single scalar field on the
# given control mesh, where the scalar field is the same as the one used
# to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
splineGenerator = EqualOrderSpline(1,splineMesh)

#### Setting Boundary Conditions #######
# Set Dirichlet Boundary conditions (homogeneous BC's)
field = 0
scalarSpline = splineGenerator.getScalarSpline(field)
for parametricDirection in [0,1]:
    for side in [0,1]:
        sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
        splineGenerator.addZeroDofs(field,sideDofs)

#### ANALYSIS ####
# Choose the quadrature degree to be used throughout the analysis.
QUAD_DEG = 4

# Create the extracted spline directly from the generator.
# As of version 2019.1, this is required for using quad/hex elements in
# parallel.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

# Trial and test Functions
print("Generating the system...")
Q = FunctionSpace(srf, 'P', 1)
V = VectorFunctionSpace(srf, 'P', 2)

u = spline.rationalize(TrialFunction(V))
v = spline.rationalize(TestFunction(V))
p = spline.rationalize(TrialFunction(Q))
q = spline.rationalize(TestFunction(Q))

# Generate the manufactured solution
x = spline.spatialCoordinates()
h = x[0]**3 - 3*x[0]*x[1]**2
u_sol = spline.grad(h)
p_sol = -1./2. * (u_sol[0]**2+u_sol[1]**2) + 14./5.

# Determination of the coefficients
beta = u_sol

""" Reminder
a(v,u) - b(p,v) + S(u,v) + b(q,u) = L(v)
where
a(u,v) := (sigma u + (beta·nabla)u,v)_Omega + mu(nabla u, nabla v)_Omega
b(p,v) := (p,nabla · v)_Omega
S(u,v) := delta_0 {(tau curl(Lapl(u), Lapl(v))_h + <h^2[(beta · nabla)u] x n,[(beta · nabla)v] x n>_{F_i}}
L(v) := (f,v)_Omega + delta_0 (tau curl(f),curl(Lapl(v)))_h
"""
# Set up and solve the Oseen problem
div_v = spline.div(u_sol)
print(div_v)
print("---------------")
a1 = dot(sigma,u) + inner(beta,spline.grad(u))
a = inner(a1,v)*spline.dx + mu* inner(spline.grad(u),spline.grad(v)) * spline.dx
b1 = inner(p_sol,spline.div(v))*spline.dx
#b2 = inner(q,spline.grad(u))*spline.dx
L = inner(f,v)*spline.dx

# Solve the system
# FEniCS Function objects are always in the homogeneous representation; it
# is a good idea to name variables in such a way as to recall this.
print("Solving the system...")
u_hom = Function(spline.V)
p_hom = Function(spline.V)
spline.solveLinearVariationalProblem(a-b1==L,u_hom)

####### Postprocessing #######
print("Saving data...")
# The solution, u, is in the homogeneous representation.
u_hom.rename("u","u")
File("results/u.pvd") << u_hom

# To visualize correctly in Paraview, we need the geometry information
# as well.
nsd = 3
for i in range(0,nsd+1):
    name = "F"+str(i)
    spline.cpFuncs[i].rename(name,name)
    File("results/"+name+"-file.pvd") << spline.cpFuncs[i]
