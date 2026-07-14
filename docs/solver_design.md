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

# 5. Implemented Solvers

## Jacobi

Characteristics

- Explicit iteration
- Fully vectorisable
- Excellent reference implementation
- Simple mathematics
- Slower convergence

Applications

- Validation
- GPU implementations
- Teaching
- Baseline benchmarking

---

## Gauss-Seidel

Characteristics

- In-place updates
- Uses newly computed values immediately
- Lower memory requirements
- Faster convergence than Jacobi in theory
- More difficult to parallelise

Applications

- Classical PDE solvers
- Reference implementation for SOR

---

## Successive Over-Relaxation (SOR)

Characteristics

- Extension of Gauss-Seidel
- Relaxation factor ω
- Can dramatically reduce iteration count
- Sensitive to ω

Applications

- Faster stationary iterative solver
- Baseline before multigrid

## 6. Solver Workflows

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

# 7. Iterative Solver Benchmarking

## Purpose

DeviceForge includes a reproducible benchmarking framework for evaluating iterative numerical solvers against a common reference problem.

The objective of these benchmarks is to compare both the numerical behaviour and computational performance of different solution algorithms while using an identical physical model and convergence criteria.

The current benchmark problem solves the two-dimensional Laplace equation on a rectangular silicon domain with fixed-potential (Dirichlet) boundaries on the left and right edges and homogeneous Neumann boundaries on the top and bottom edges.

---

## Implemented Solvers

The current benchmark compares:

- Jacobi Iteration
- Gauss-Seidel Iteration
- Successive Over-Relaxation (SOR)

Multiple SOR relaxation factors can be evaluated within the same benchmark.

---

## Benchmark Metrics

Each benchmark records:

- Convergence status
- Total iteration count
- Solver runtime
- Final recorded solution-change residual
- Maximum absolute error relative to the analytical solution

These metrics provide a balanced comparison between numerical accuracy and computational efficiency.

---

## Analytical Validation

The benchmark uses a problem with a known analytical solution:

\[
\phi(x)=\frac{x}{L}
\]

where:

- Left boundary = 0 V
- Right boundary = 1 V
- Top boundary = zero normal gradient
- Bottom boundary = zero normal gradient

The analytical solution increases linearly across the computational domain.

Each solver is compared against this analytical solution by computing the maximum absolute error:

\[
E_{\max}
=
\max
\left|
\phi_{\mathrm{numerical}}
-
\phi_{\mathrm{analytical}}
\right|
\]

This provides a quantitative measure of numerical accuracy independent of convergence rate.

---

## Benchmark Outputs

The benchmark automatically generates:

- Residual history comparison
- Runtime comparison
- Iteration-count comparison
- Analytical-error comparison

These figures are written to:

```text
figures/benchmarks/
```

Example output:

```text
solver_convergence_comparison.png
solver_runtime_comparison.png
solver_iteration_comparison.png
solver_error_comparison.png
```

---

## Interpretation of Results

Benchmark results should be interpreted carefully.

A solver requiring fewer iterations does not necessarily execute faster.

Likewise, a faster runtime does not necessarily imply superior numerical performance.

Several factors contribute to overall solver performance:

- Numerical convergence rate
- Cost per iteration
- Memory-access patterns
- Vectorisation
- Cache efficiency
- Parallelisation capability
- Hardware implementation

For example, the current Jacobi implementation performs vectorised NumPy operations, while the initial Gauss-Seidel and SOR implementations use explicit Python loops.

Consequently, Jacobi may execute faster despite requiring more numerical iterations.

This distinction highlights the important difference between:

- algorithmic efficiency
- implementation efficiency
- hardware efficiency

---

## Future Benchmark Extensions

The benchmarking framework is designed to grow alongside DeviceForge.

Planned comparisons include:

- Different computational grid sizes
- Non-uniform meshes
- Three-dimensional simulations
- Sparse matrix implementations
- GPU acceleration
- OpenMP implementations
- Distributed-memory execution (MPI)
- C++ numerical backends
- Surrogate-assisted numerical solvers

The benchmarking framework will provide quantitative evidence of the performance gains achieved by each successive numerical implementation.

---

## 8. Current Limitations

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

## 9. Performance Considerations

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

## 10. Future Solver Comparison

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
