# DeviceForge 1D Poisson Reference Solver

## Overview

DeviceForge includes a verified one-dimensional electrostatic Poisson
reference solver for semiconductor and dielectric simulations.

The solver is intended to provide a numerically well-defined reference
implementation against which higher-dimensional DeviceForge solvers can
be developed and verified.

The current implementation supports:

-   one-dimensional uniform Cartesian grids
-   Laplace and Poisson problems
-   prescribed volumetric charge density
-   spatially varying relative permittivity
-   multiple dielectric or semiconductor material regions
-   conservative material-interface treatment
-   Dirichlet boundary conditions
-   mixed Dirichlet-Neumann boundary conditions
-   sparse linear-system assembly
-   configurable linear solvers
-   direct and iterative solution methods
-   node-centred and face-centred electrostatic postprocessing
-   analytical and manufactured-solution verification

The solver is implemented by:

``` text
src/deviceforge/solvers/poisson_solver.py
```

------------------------------------------------------------------------

# Governing Equation

The solver evaluates the one-dimensional electrostatic Poisson equation
in conservative form:

\[
-`\frac{d}{dx}`{=tex}`\left`{=tex}(`\varepsilon`{=tex}\_r(x)`\frac{d\phi}{dx}`{=tex}`\right`{=tex})
= `\frac{\rho(x)}{\varepsilon_0}`{=tex} \]

where:

-   (`\phi`{=tex}) is the electrostatic potential in volts
-   (`\rho`{=tex}) is volumetric charge density in C/m³
-   (`\varepsilon`{=tex}\_0) is the vacuum permittivity
-   (`\varepsilon`{=tex}\_r(x)) is the local relative permittivity

The vacuum permittivity used by DeviceForge is:

\[ `\varepsilon`{=tex}\_0 =
8.8541878128`\times10`{=tex}\^{-12}`\text{ F/m}`{=tex} \]

Equivalently:

\[
`\frac{d}{dx}`{=tex}`\left`{=tex}(`\varepsilon`{=tex}\_0`\varepsilon`{=tex}\_r(x)`\frac{d\phi}{dx}`{=tex}`\right`{=tex})=-`\rho`{=tex}(x)
\]

When no charge-density field is supplied, DeviceForge treats the charge
density as zero and the equation reduces to Laplace's equation.

------------------------------------------------------------------------

# Computational Domain

The reference solver currently supports a uniform one-dimensional
Cartesian grid.

For grid points (x_0,x_1,`\ldots`{=tex},x\_{N-1}), the uniform grid
spacing is:

\[ `\Delta `{=tex}x=x\_{i+1}-x_i. \]

Electrostatic potential, relative permittivity, and charge density are
stored at grid nodes. Several conservative electrostatic quantities are
also evaluated at the faces between adjacent nodes.

------------------------------------------------------------------------

# Material Permittivity

Relative permittivity is obtained from the materials assigned to the
DeviceForge device. Therefore, (`\varepsilon`{=tex}\_r) may vary
spatially across the computational domain.

This allows the reference solver to represent structures containing
multiple materials, including semiconductor-dielectric and
dielectric-dielectric interfaces.

All grid points must have (`\varepsilon`{=tex}\_r\>0), and the device
must provide complete material coverage of the computational domain.

------------------------------------------------------------------------

# Harmonic Face Permittivity

For adjacent node-centred values (`\varepsilon`{=tex}*{r,i}) and
(`\varepsilon`{=tex}*{r,i+1}), DeviceForge calculates face-centred
relative permittivity using:

\[ `\varepsilon`{=tex}*{r,i+1/2} =
`\frac{2\varepsilon_{r,i}\varepsilon_{r,i+1}}`{=tex}
{`\varepsilon`{=tex}*{r,i}+`\varepsilon`{=tex}\_{r,i+1}}. \]

Harmonic averaging is used because electrostatic flux must remain
conservative across discontinuous material interfaces.

------------------------------------------------------------------------

# Spatial Discretisation

For an interior grid point (i), the conservative centred discretisation
is:

\[ -`\varepsilon`{=tex}*{r,i-1/2}`\phi`{=tex}*{i-1} +
`\left`{=tex}(`\varepsilon`{=tex}*{r,i-1/2}+`\varepsilon`{=tex}*{r,i+1/2}`\right`{=tex})`\phi`{=tex}*i -
`\varepsilon`{=tex}*{r,i+1/2}`\phi`{=tex}\_{i+1} =
`\frac{\rho_i\Delta x^2}{\varepsilon_0}`{=tex}. \]

This produces a tridiagonal sparse linear system. For constant relative
permittivity, the formulation reduces to the familiar second-order
centred Poisson discretisation.

------------------------------------------------------------------------

# Linear System

The discretised electrostatic problem is written as:

\[ A`\phi`{=tex}=b. \]

The coefficient matrix is initially assembled using SciPy LIL sparse
storage. After assembly and boundary-condition application, it is
converted to CSR format before being supplied to the configured linear
solver.

------------------------------------------------------------------------

# Boundary Conditions

## Dirichlet Boundary Conditions

Dirichlet conditions prescribe:

\[ `\phi`{=tex}=V\_{`\mathrm{specified}`{=tex}}. \]

DeviceForge requires Dirichlet boundary values to use units of `V`.

## Neumann Boundary Conditions

Neumann conditions prescribe the outward-normal potential derivative:

\[ `\frac{\partial\phi}{\partial n}`{=tex}=g. \]

DeviceForge requires Neumann values to use units of `V/m`.

At the left boundary:

\[ -`\frac{d\phi}{dx}`{=tex}=g_L. \]

At the right boundary:

\[ `\frac{d\phi}{dx}`{=tex}=g_R. \]

For a linear potential with slope (s):

\[ g_L=-s,`\qquad `{=tex}g_R=+s. \]

------------------------------------------------------------------------

# Supported Boundary Combinations

The current solver supports:

``` text
Dirichlet + Dirichlet
Dirichlet + Neumann
Neumann   + Dirichlet
```

A boundary condition must be specified at both endpoints. An endpoint
may not simultaneously contain both Dirichlet and Neumann conditions.

Pure-Neumann problems are currently rejected because they do not
uniquely define the absolute electrostatic potential without an
additional gauge condition.

------------------------------------------------------------------------

# Charged Neumann Boundary Treatment

For Poisson problems containing volumetric charge, the Neumann endpoint
equations include a half-cell source contribution.

Right boundary:

\[ `\varepsilon`{=tex}*{r,N-1/2}(`\phi`{=tex}*N-`\phi`{=tex}*{N-1}) =
`\varepsilon`{=tex}*{r,N}g_R`\Delta `{=tex}x +
`\frac{\rho_N\Delta x^2}{2\varepsilon_0}`{=tex}. \]

Left boundary:

\[ `\varepsilon`{=tex}\_{r,1/2}(`\phi`{=tex}\_0-`\phi`{=tex}*1) =
`\varepsilon`{=tex}*{r,0}g_L`\Delta `{=tex}x +
`\frac{\rho_0\Delta x^2}{2\varepsilon_0}`{=tex}. \]

The source contribution is positive on both endpoint right-hand sides.
The derivative term uses boundary-node relative permittivity, while
matrix coupling uses adjacent harmonic face permittivity.

------------------------------------------------------------------------

# Symmetric Dirichlet Elimination

Dirichlet conditions are imposed using symmetric elimination:

1.  transfer the fixed column contribution to the right-hand side
2.  zero the corresponding column
3.  zero the corresponding row
4.  replace the diagonal entry with one
5.  set the right-hand-side entry to the prescribed potential

This preserves matrix symmetry. For supported one-dimensional problems,
the resulting operator is symmetric positive definite, enabling
conjugate-gradient solution.

------------------------------------------------------------------------

# Linear Solvers

The Poisson solver delegates the assembled system to an object
satisfying `LinearSolverProtocol`.

The default solver is `SparseDirectSolver`.

Supported linear-algebra components include:

``` text
DenseDirectSolver
SparseDirectSolver
ConjugateGradientSolver
IdentityPreconditioner
JacobiPreconditioner
```

This separates the physical Poisson formulation from the numerical
linear-algebra backend.

------------------------------------------------------------------------

# Solver Result

The solver returns a `SimulationResult` whose primary field is
`electrostatic_potential` in `V`.

Diagnostics include:

-   convergence status
-   iteration count
-   residual history
-   runtime
-   solver name
-   backend name

Metadata includes equation type, spatial dimension, discretisation
method, interface averaging, linear solver information, matrix storage
and dimensions, matrix non-zero count and density, permittivity range,
charge-density range, grid-point count, and grid spacing.

------------------------------------------------------------------------

# Electrostatic Postprocessing

Available derived quantities include:

-   node-centred electric field
-   electric displacement
-   electrostatic energy density
-   face-centred electric field
-   face-centred relative permittivity
-   face-centred electric displacement

## Electric Field

\[ E=-`\frac{d\phi}{dx}`{=tex}. \]

Interior nodes use a centred second-order derivative. Endpoints use
second-order one-sided derivatives when at least three grid points are
available. A two-point grid necessarily uses the available first-order
difference.

## Face-Centred Electric Field

\[ E\_{i+1/2} = -`\frac{\phi_{i+1}-\phi_i}{\Delta x}`{=tex}. \]

## Electric Displacement

\[ D=`\varepsilon`{=tex}\_0`\varepsilon`{=tex}\_rE. \]

For conservative interface analysis:

\[ D\_{i+1/2} =
`\varepsilon`{=tex}*0`\varepsilon`{=tex}*{r,i+1/2}E\_{i+1/2}. \]

In an uncharged dielectric stack, normal electric displacement should
remain continuous across material interfaces.

## Electrostatic Energy Density

\[
u=`\frac{1}{2}`{=tex}`\varepsilon`{=tex}\_0`\varepsilon`{=tex}\_rE\^2.
\]

------------------------------------------------------------------------

# Verification Strategy

The 1D reference solver is verified using independent analytical and
numerical tests that cover physical behaviour and numerical properties.

## Laplace Verification

For zero charge and constant permittivity:

\[ `\frac{d^2\phi}{dx^2}`{=tex}=0. \]

With fixed endpoint potentials:

\[ `\phi`{=tex}(x)=`\phi`{=tex}\_L+`\frac{\phi_R-\phi_L}{L}`{=tex}x. \]

The numerical solution reproduces this linear profile and its constant
electric field.

## Uniform-Charge Verification

For constant charge and permittivity with zero potential at both ends:

\[ `\phi`{=tex}(x) =
`\frac{\rho}{2\varepsilon_0\varepsilon_r}`{=tex}x(L-x). \]

This verifies source sign, source scaling, grid-spacing scaling,
potential curvature, and electric-field calculation.

## Heterogeneous Dielectric Verification

At a charge-free dielectric interface:

\[ D_n=`\varepsilon`{=tex}\_0`\varepsilon`{=tex}\_rE_n \]

must remain continuous. Tests confirm face-centred displacement
continuity to approximately floating-point accuracy.

## Mixed Dirichlet-Neumann Verification

Both `Dirichlet + Neumann` and `Neumann + Dirichlet` orientations are
tested against linear analytical solutions, including outward-normal
sign conventions, matrix assembly, direct solution, and
conjugate-gradient solution.

## Charged Mixed-Boundary Verification

For left Dirichlet and right Neumann conditions with constant charge and
permittivity:

\[ `\phi`{=tex}(x) = `\phi`{=tex}\_L +
`\left`{=tex}(g_R+`\frac{\rho L}{\varepsilon_0\varepsilon_r}`{=tex}`\right`{=tex})x -
`\frac{\rho x^2}{2\varepsilon_0\varepsilon_r}`{=tex}. \]

The numerical implementation reproduces this quadratic solution to
approximately machine precision. Both mixed-boundary orientations are
tested with direct and conjugate-gradient solvers.

## Matrix Verification

Mixed-boundary systems are explicitly tested for matrix symmetry and
positive definiteness. This verifies Neumann row scaling, symmetric
Dirichlet elimination, and compatibility with conjugate-gradient
solution.

------------------------------------------------------------------------

# Manufactured-Solution Verification

The smooth manufactured solution is:

\[
`\phi`{=tex}(x)=`\sin`{=tex}`\left`{=tex}(`\frac{\pi x}{L}`{=tex}`\right`{=tex}).
\]

The corresponding charge density for constant relative permittivity is:

\[ `\rho`{=tex}(x) = `\varepsilon`{=tex}\_0`\varepsilon`{=tex}\_r
`\left`{=tex}(`\frac{\pi}{L}`{=tex}`\right`{=tex})\^2
`\sin`{=tex}`\left`{=tex}(`\frac{\pi x}{L}`{=tex}`\right`{=tex}). \]

with:

\[ `\phi`{=tex}(0)=0,`\qquad`{=tex}`\phi`{=tex}(L)=0. \]

------------------------------------------------------------------------

# Grid-Convergence Results

The RMS error is:

\[ e =
`\sqrt{\frac{1}{N}\sum_i(\phi_{h,i}-\phi_{\mathrm{exact},i})^2}`{=tex}.
\]

The observed convergence order is:

\[ p=`\frac{\log(e_h/e_{h/2})}{\log 2}`{=tex}. \]

    Grid points            RMS error   Error ratio   Observed order
  ------------- -------------------- ------------- ----------------
             21   1.420642634989e-03            \-               \-
             41   3.591331920374e-04      3.955754         1.983953
             81   9.031491950553e-05      3.976455         1.991483
            161   2.264743183942e-05      3.987866         1.995617

The measured convergence rate approaches (p=2), confirming second-order
spatial convergence for a smooth one-dimensional Poisson problem.

------------------------------------------------------------------------

# Verification Example

The standalone manufactured-solution study is located at:

``` text
examples/verification/poisson_1d_grid_convergence.py
```

It produces:

``` text
convergence_results.csv
error_convergence.png
runtime_scaling.png
numerical_vs_analytical.png
```

The runtime figure is an execution diagnostic and should not be
interpreted as a formal computational-complexity benchmark because
startup and interpreter overhead can dominate small 1D problems.

------------------------------------------------------------------------

# Mixed-Boundary Example

A standalone charged mixed-boundary example is provided at:

``` text
examples/electrostatics/mixed_boundary_1d.py
```

It demonstrates the outward-normal Neumann convention, charged endpoint
half-cell treatment, and comparison against the analytical quadratic
solution.

------------------------------------------------------------------------

# Visualisation

DeviceForge provides 1D electrostatic plotting functions for:

-   potential
-   electric field
-   electric displacement
-   electrostatic energy density
-   relative permittivity
-   solver residual history
-   face-centred electric displacement

Face-centred electric-displacement visualisation includes stable
handling of effectively constant conservative flux so that
floating-point roundoff is not visually exaggerated.

------------------------------------------------------------------------

# Current Scope

The current 1D reference solver supports:

-   one-dimensional Cartesian grids
-   uniform spacing
-   linear electrostatic Poisson equation
-   prescribed fixed charge density
-   spatially varying relative permittivity
-   heterogeneous materials
-   harmonic material-interface averaging
-   Dirichlet boundary conditions
-   mixed Dirichlet-Neumann boundary conditions
-   dense direct solution
-   sparse direct solution
-   conjugate-gradient solution
-   supported linear preconditioning
-   sparse CSR matrix storage
-   node-centred electrostatic postprocessing
-   conservative face-centred postprocessing

------------------------------------------------------------------------

# Current Limitations

The current reference solver does not yet provide:

-   pure-Neumann gauge handling
-   nonuniform grids
-   nonlinear semiconductor Poisson coupling
-   self-consistent mobile carrier concentrations
-   drift-diffusion transport
-   quantum corrections
-   transient electrostatics

These capabilities belong to later DeviceForge physics layers rather
than the current linear 1D Poisson reference problem.

------------------------------------------------------------------------

# Relationship to the 2D Solver

The verified one-dimensional solver acts as the numerical reference for
the DeviceForge two-dimensional electrostatic architecture.

``` text
Verified 1D reference solver
        ↓
Apply equivalent conservative formulation in 2D
        ↓
Verify 2D analytical behaviour and convergence
        ↓
Extend to semiconductor electrostatics
        ↓
Introduce nonlinear carrier physics
        ↓
Drift-diffusion transport
        ↓
3D device simulation
```

The 1D implementation provides a controlled environment in which
boundary treatment, interface averaging, sparse matrix assembly, solver
interfaces, and verification methodology can be established before
extension to higher dimensions.

------------------------------------------------------------------------

# Reference-Solver Status

The DeviceForge 1D Poisson solver is considered a verified reference
implementation for the currently supported linear electrostatic problem.

Verification currently includes:

-   Laplace analytical solutions
-   uniform-charge analytical solutions
-   heterogeneous dielectric-interface tests
-   conservative displacement continuity
-   mixed Dirichlet-Neumann solutions
-   charged mixed-boundary analytical solutions
-   outward-normal sign tests
-   matrix symmetry tests
-   matrix positive-definiteness tests
-   sparse direct solution
-   conjugate-gradient solution
-   manufactured-solution grid convergence
-   measured second-order spatial accuracy
-   electrostatic postprocessing regression tests
-   visualisation regression tests

This reference implementation provides the baseline numerical behaviour
against which subsequent DeviceForge electrostatic solvers should be
tested.
