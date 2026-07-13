# Examples

## Two-Dimensional Laplace Rectangle

The first DeviceForge demonstration solves the two-dimensional Laplace equation on a rectangular silicon domain.

### Boundary Conditions

* Left boundary: (0\ \text{V})
* Right boundary: (1\ \text{V})
* Top boundary: zero normal gradient
* Bottom boundary: zero normal gradient

### Expected Solution

The analytical potential is:

[
\phi(x) = \frac{x}{L}
]

The potential increases linearly from the left contact to the right contact.

### Running the Example

From the DeviceForge repository root:

```powershell
python examples/laplace_rectangle.py
```

The example:

1. Creates a structured two-dimensional grid.
2. Creates a silicon device region.
3. Defines Dirichlet and Neumann boundary conditions.
4. Runs the NumPy Jacobi solver.
5. Prints numerical convergence information.
6. Displays the electrostatic-potential field.
7. Displays the convergence history.
8. Saves both figures.

### Generated Figures

The example generates:

```text
figures/examples/laplace_potential.png
figures/examples/jacobi_convergence.png
```

### Example Terminal Output

```text
DeviceForge Jacobi Demonstration
====================================
Converged:       True
Iterations:      <solver dependent>
Final residual:  <solver dependent>
Runtime:         <hardware dependent> s
Solver:          jacobi
Backend:         numpy
Grid shape:      (101, 51)
```

Exact iteration counts and runtimes depend on the selected tolerance, grid size and host hardware.

### Validation

The computed centre-line profile is validated through automated tests against the expected linear analytical solution.

The test suite also verifies:

* preservation of fixed boundary values
* zero-gradient behaviour
* convergence-history consistency
* solver iteration limits
* rejection of unsupported grid spacing
* rejection of unsupported Neumann values
