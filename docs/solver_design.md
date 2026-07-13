# Solver Design

## 1. Purpose

The DeviceForge solver layer provides a common interface for numerical algorithms while keeping device definition, physical modelling, execution backend and result presentation separate.

All numerical solvers inherit from `BaseSolver` and return a `SimulationResult`.

The standard execution pattern is:

```python
result = solver.solve(simulation)
```

---

## 2. Solver Interface

Every solver must provide:

* A public name
* Simulation validation
* A `solve()` method
* Convergence diagnostics
* Runtime measurement
* A standard `SimulationResult`

The base interface allows different numerical algorithms to be used without changing the rest of DeviceForge.

Planned implementations include:

* Jacobi
* Gauss-Seidel
* Successive Over-Relaxation
* Conjugate Gradient
* Multigrid
* C++ solver backends
* OpenMP implementations
* CUDA implementations

---

## 3. Separation of Responsibilities

### Simulation

The `Simulation` object defines:

* Device
* Grid
* Boundary conditions
* Initial potential
* Requested tolerance
* Maximum iterations

### SolverConfiguration

The solver configuration defines:

* Solver tolerance limit
* Solver iteration limit
* Backend identity

### Solver

The solver:

1. Validates the simulation.
2. Creates the initial potential field.
3. Applies the numerical update.
4. Reapplies boundary conditions.
5. Records convergence history.
6. Stops when converged or when the iteration limit is reached.
7. Returns a `SimulationResult`.

### SimulationResult

The result stores:

* Solved fields
* Convergence state
* Iteration count
* Residual history
* Runtime
* Solver identity
* Backend identity
* Numerical metadata

---

## 4. Effective Solver Settings

Both `Simulation` and `SolverConfiguration` contain tolerance and maximum-iteration settings.

The current design uses the stricter constraint.

Effective tolerance:

[
\varepsilon_{\text{effective}}
==============================

\min
\left(
\varepsilon_{\text{simulation}},
\varepsilon_{\text{solver}}
\right)
]

Effective iteration limit:

[
N_{\text{effective}}
====================

\min
\left(
N_{\text{simulation}},
N_{\text{solver}}
\right)
]

This prevents either configuration from silently weakening another component's requested limits.

---

## 5. Jacobi Solver Workflow

The initial Jacobi solver follows this process:

```text
Validate simulation
        ↓
Create initial potential field
        ↓
Apply fixed Dirichlet values
        ↓
Copy the previous field
        ↓
Update interior points
        ↓
Apply homogeneous Neumann conditions
        ↓
Reapply Dirichlet conditions
        ↓
Calculate maximum potential change
        ↓
Record convergence history
        ↓
Check tolerance
        ↓
Return SimulationResult
```

---

## 6. Current Limitations

The initial Jacobi solver supports:

* Two-dimensional grids
* Uniform spacing
* Laplace equation
* NumPy execution
* Dirichlet boundaries
* Homogeneous Neumann boundaries

It does not yet support:

* Poisson charge terms
* Unequal grid spacing
* Spatially varying permittivity
* Non-zero Neumann values
* Internal interfaces
* Three-dimensional fields
* Sparse matrix formulations
* GPU execution
* Distributed domain decomposition

---

## 7. Performance Considerations

Jacobi requires a complete copy of the previous potential field for every iteration.

For a grid with (N) points:

* Computational complexity per iteration is approximately (O(N)).
* Memory requirement includes at least two potential arrays.
* Convergence may require many iterations for fine grids.

Despite its slow convergence, Jacobi is valuable as:

* A readable reference implementation
* An analytical validation baseline
* A parallel-computing example
* A CPU/GPU benchmarking workload
* A comparison point for faster solvers

---

## 8. Future Solver Comparison

DeviceForge will compare solvers using:

* Convergence state
* Iteration count
* Final residual
* Runtime
* Memory usage
* Accuracy against analytical solutions
* Grid-size scaling
* CPU-thread scaling
* GPU speed-up
* Parallel efficiency

A common `SimulationResult` interface ensures these metrics can be compared consistently.
