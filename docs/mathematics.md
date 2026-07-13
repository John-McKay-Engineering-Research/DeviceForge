# Mathematical Foundations

## 1. Overview

The initial DeviceForge numerical model solves two-dimensional electrostatic boundary-value problems on structured Cartesian grids.

The first implemented benchmark solves the Laplace equation:

[
\nabla^2 \phi = 0
]

where (\phi) is the electrostatic potential.

The Laplace equation represents a charge-free electrostatic domain. It provides an important first validation case because its behaviour is well understood and analytical solutions are available for simple geometries.

Later versions of DeviceForge will extend this formulation to the Poisson equation:

[
\nabla \cdot \left(\varepsilon \nabla \phi\right) = -\rho
]

where:

* (\varepsilon) is the material permittivity
* (\rho) is the charge density

---

## 2. Two-Dimensional Laplace Equation

For a two-dimensional Cartesian domain, the Laplace equation is:

[
\frac{\partial^2 \phi}{\partial x^2}
+
\frac{\partial^2 \phi}{\partial y^2}
====================================

0
]

The current reference solver assumes uniform grid spacing:

[
\Delta x = \Delta y = h
]

Using central finite differences:

[
\frac{\partial^2 \phi}{\partial x^2}
\approx
\frac{
\phi_{i+1,j}
------------

2\phi_{i,j}
+
\phi_{i-1,j}
}{h^2}
]

and:

[
\frac{\partial^2 \phi}{\partial y^2}
\approx
\frac{
\phi_{i,j+1}
------------

2\phi_{i,j}
+
\phi_{i,j-1}
}{h^2}
]

Substituting these approximations into the Laplace equation gives:

[
\phi_{i+1,j}
+
\phi_{i-1,j}
+
\phi_{i,j+1}
+
\phi_{i,j-1}
------------

# 4\phi_{i,j}

0
]

Rearranging:

[
\phi_{i,j}
==========

\frac{1}{4}
\left(
\phi_{i+1,j}
+
\phi_{i-1,j}
+
\phi_{i,j+1}
+
\phi_{i,j-1}
\right)
]

This expression forms the basis of the current Jacobi iteration.

---

## 3. Jacobi Iteration

The Jacobi method calculates every updated grid value using values from the previous iteration:

[
\phi_{i,j}^{(k+1)}
==================

\frac{1}{4}
\left(
\phi_{i+1,j}^{(k)}
+
\phi_{i-1,j}^{(k)}
+
\phi_{i,j+1}^{(k)}
+
\phi_{i,j-1}^{(k)}
\right)
]

where:

* (k) is the current iteration
* (k+1) is the next iteration

Because all new values depend only on the previous field, Jacobi updates can be calculated independently. This property makes the method well suited to parallel implementation on multicore CPUs and GPUs.

Jacobi iteration is easy to understand and validate, although it generally converges more slowly than Gauss-Seidel, Successive Over-Relaxation and multigrid methods.

---

## 4. Boundary Conditions

DeviceForge initially supports two boundary-condition categories.

### 4.1 Dirichlet Boundary Conditions

Dirichlet conditions prescribe the field value:

[
\phi = \phi_0
]

For the rectangular validation problem:

[
\phi(0,y) = 0
]

and:

[
\phi(L,y) = 1
]

These conditions represent fixed electrical potentials at the left and right boundaries.

### 4.2 Neumann Boundary Conditions

Neumann conditions prescribe the normal derivative:

[
\frac{\partial \phi}{\partial n} = g
]

The current Jacobi implementation supports homogeneous Neumann conditions:

[
\frac{\partial \phi}{\partial n} = 0
]

This represents zero normal potential gradient.

In the discrete implementation, the boundary value is copied from its nearest interior neighbour.

---

## 5. Analytical Validation Case

The first benchmark uses:

* (0\ \text{V}) at the left boundary
* (1\ \text{V}) at the right boundary
* zero normal gradient at the top and bottom boundaries

The analytical solution is:

[
\phi(x) = \frac{x}{L}
]

The potential therefore increases linearly from left to right and remains constant in the vertical direction.

This provides several validation criteria:

* Dirichlet values remain fixed.
* Top and bottom normal gradients approach zero.
* The centre-line potential matches a linear analytical profile.
* The numerical residual decreases with iteration.
* Grid refinement should improve agreement with the analytical solution.

---

## 6. Convergence Criterion

The current solver records the maximum absolute change between consecutive iterations:

[
r^{(k)}
=======

\max_{i,j}
\left|
\phi_{i,j}^{(k+1)}
------------------

\phi_{i,j}^{(k)}
\right|
]

The solver is considered converged when:

[
r^{(k)} \leq \varepsilon
]

where (\varepsilon) is the requested numerical tolerance.

This is currently a solution-change criterion rather than a direct discrete PDE residual. A future solver revision may report both:

* solution-change norm
* discrete equation residual

---

## 7. Electric Field

Once electrostatic potential has been solved, the electric field can be calculated from:

[
\mathbf{E} = -\nabla \phi
]

In two dimensions:

[
E_x = -\frac{\partial \phi}{\partial x}
]

[
E_y = -\frac{\partial \phi}{\partial y}
]

The field magnitude is:

[
|\mathbf{E}|
============

\sqrt{
E_x^2 + E_y^2
}
]

Electric-field post-processing will be introduced after the initial potential visualisation workflow.

---

## 8. Planned Extensions

The numerical formulation will later be extended to include:

* unequal grid spacing
* spatially varying permittivity
* Poisson source terms
* semiconductor charge density
* sparse matrix assembly
* Gauss-Seidel iteration
* Successive Over-Relaxation
* Conjugate Gradient methods
* multigrid acceleration
* nonlinear semiconductor equations
* drift-diffusion transport
* three-dimensional discretisation
