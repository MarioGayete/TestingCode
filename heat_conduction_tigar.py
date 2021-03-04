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

Boundary conditions --> Walls --> fixed temperature u = 0

kappa --> thermal conductivity
f --> source term
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

# Constant parameters
kappa = 1.0
Length = 1.0
Height = 1.0

####### Geometry creation #######
print("Generating the geometry...")
c1 = circle(radius = Length/2., center = (Length,0), angle = (pi/2.,pi))
left = line(p0 =(0, Height) , p1 =(0, 0))
# right = line(p0 =(Length, Length/2.), p1 =(Length, Height))
top = line(p0 =(Length, Height), p1 =(0, Height))
# bottom = line(p0 =(0, 0), p1 =(Length/2., 0))

perimeter_top = join(top,left,axis=0)

srf = ruled(c1,perimeter_top)

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

# Output in PetIGA format
if(mpirank==0):
    PetIGA().write("out.dat",srf)
MPI.barrier(worldcomm)

# Creating a control mesh from the NURBS
splineMesh = NURBSControlMesh(srf,useRect=True)

# Create a spline generator for a spline with a single scalar field on the
# given control mesh, where the scalar field is the same as the one used
# to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
splineGenerator = EqualOrderSpline(1,splineMesh)

# Set Dirichlet Boundary conditions (left temperature at 0 and right temperature at 1)
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
u = spline.rationalize(TrialFunction(spline.V))
v = spline.rationalize(TestFunction(spline.V))

# Generate the internal heat sources vector
x = spline.spatialCoordinates()
soln = sin(pi*x[0])*sin(pi*x[1])
f = -spline.div(spline.grad(soln))

# Set up and solve the Heat conduction problem
a = kappa*inner(spline.grad(u),spline.grad(v))*spline.dx
L = inner(f,v)*spline.dx

# Solve the system
# FEniCS Function objects are always in the homogeneous representation; it
# is a good idea to name variables in such a way as to recall this.
print("Solving the system...")
u_hom = Function(spline.V)
spline.solveLinearVariationalProblem(a==L,u_hom)

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

# Useful notes for plotting:
#
#  In Paraview, the data in these different files can be combined with the
#  Append Attributes filter, then an appropriate vector field for the mesh
#  warping and the weighted solution can be created using the Calculator
#  filter.  E.g., in this case, the vector field to warp by would be
#
#   (F0/F3-coordsX)*iHat + (F1/F3-coordsY)*jHat + (F2/F3-coordsZ)*kHat
#
#  in Paraview Calculator syntax, and the solution would be u/F3.
