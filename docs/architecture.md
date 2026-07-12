# DeviceForge Software Architecture

## 1. Purpose

DeviceForge is an open-source research framework for semiconductor device simulation, high-performance computing, machine-learning surrogate modelling and multi-objective design optimisation.

The initial implementation will focus on two-dimensional semiconductor electrostatics. The architecture is intended to support later development of:

* Three-dimensional device simulation
* Coupled semiconductor transport equations
* C++ numerical backends
* OpenMP CPU parallelisation
* CUDA GPU acceleration
* MPI-distributed simulation campaigns
* PyTorch and TensorFlow surrogate models
* Sensitivity analysis
* Multi-objective optimisation
* Interactive visualisation
* Graphical comparison of initial and optimised device configurations

DeviceForge will not be designed as a collection of independent scripts. It will be structured as a modular scientific software framework with clearly separated responsibilities.

---

## 2. Primary Design Goals

The architecture is guided by the following goals.

### 2.1 Modularity

Physics models, geometry definitions, numerical solvers, computing backends, optimisation methods and user interfaces should remain separate.

A change to the user interface should not require changing the numerical solver.

A change from a NumPy solver to a CUDA solver should not require redefining the semiconductor device.

### 2.2 Extensibility

The first version will support two-dimensional structured grids, but the core data model should allow three-dimensional grids and fields to be introduced later.

The project should also allow additional:

* Materials
* Device geometries
* PDE models
* Boundary conditions
* Solvers
* Compute backends
* Surrogate models
* Optimisation methods
* Visualisation modes

### 2.3 Reproducibility

A simulation should be fully defined by a serialisable configuration containing:

* Geometry
* Grid resolution
* Material definitions
* Doping profiles
* Applied voltages
* Boundary conditions
* Physics model
* Solver settings
* Compute backend
* Optimisation settings

The same configuration should produce repeatable results when run with the same software version and numerical tolerances.

### 2.4 Validation

Numerical correctness must be prioritised before performance optimisation.

Each solver should be validated using:

* Analytical solutions
* Manufactured solutions
* Grid-convergence studies
* Cross-comparison between independent solver implementations
* Regression tests

### 2.5 Performance Portability

The same physical problem should eventually be executable using different backends:

* NumPy reference implementation
* Serial C++
* OpenMP C++
* CUDA
* Future distributed backends

### 2.6 Interactive Engineering Output

DeviceForge should not operate only as a command-line or backend library.

Users should be able to:

* Define a semiconductor device
* View its geometry and material regions
* Change device parameters
* Run simulations
* Inspect potential and electric-field distributions
* Compare solver performance
* Explore sensitivity results
* Run optimisation studies
* Interactively inspect Pareto-optimal designs
* Compare initial and optimised device configurations
* View two-dimensional and, later, three-dimensional device results

---

## 3. High-Level Architecture

DeviceForge is divided into several architectural layers.

```text
┌─────────────────────────────────────────────────────────┐
│                 User Interaction Layer                  │
│   Dashboard · Command Line · Examples · Configuration   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Application / Workflow Layer              │
│ Simulation · Benchmarking · Dataset Generation          │
│ Sensitivity Analysis · Optimisation · Model Training    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 Domain and Physics Layer                │
│ Grid · Field · Device · Region · Material               │
│ Boundary Conditions · Electrostatics · Transport        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Numerical Methods Layer                  │
│ Discretisation · Matrix Assembly · Iterative Solvers    │
│ Convergence Control · Nonlinear Coupling                │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Compute Backend Layer                 │
│ NumPy · C++ · OpenMP · CUDA · Future MPI Backend        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Results and Presentation Layer             │
│ Fields · Metrics · Plots · 2D/3D Visualisation          │
│ Reports · Interactive Comparisons · Export              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Repository Modules

The initial Python package is divided into the following modules.

```text
src/deviceforge/
├── core/
├── geometry/
├── physics/
├── solvers/
├── postprocessing/
└── visualisation/
```

Future modules will be added only when the relevant functionality is implemented.

```text
src/deviceforge/
├── backends/
├── machine_learning/
├── optimisation/
├── sensitivity/
├── distributed/
├── configuration/
└── workflows/
```

---

## 5. Core Module

The `deviceforge.core` package contains the fundamental objects used throughout the framework.

These classes should not depend on graphical-interface libraries, machine-learning frameworks or specific hardware backends.

### 5.1 Grid

The `Grid` class defines the computational domain.

Responsibilities:

* Store the number of grid points
* Store spatial spacing
* Store domain origin
* Determine dimensionality
* Calculate physical dimensions
* Generate coordinate arrays
* Validate grid definitions

Illustrative interface:

```python
grid = Grid(
    shape=(200, 100),
    spacing=(1.0e-9, 1.0e-9),
    origin=(0.0, 0.0),
)
```

The same class should later support:

```python
grid = Grid(
    shape=(200, 100, 80),
    spacing=(1.0e-9, 1.0e-9, 1.0e-9),
    origin=(0.0, 0.0, 0.0),
)
```

The `Grid` class should be dimension-aware rather than explicitly limited to two dimensions.

### 5.2 Field

The `Field` class represents a physical quantity defined on a grid.

Examples include:

* Electrostatic potential
* Charge density
* Donor concentration
* Acceptor concentration
* Relative permittivity
* Electron concentration
* Hole concentration
* Electric-field magnitude
* Electric-field vector components

Illustrative interface:

```python
potential = Field(
    name="electrostatic_potential",
    units="V",
    grid=grid,
    values=potential_array,
)
```

Responsibilities:

* Associate numerical data with a grid
* Store physical units
* Validate array dimensions
* Support scalar and vector fields
* Provide metadata for visualisation and export

### 5.3 Material

The `Material` class stores physical properties.

Initial properties may include:

* Relative permittivity
* Electron affinity
* Band gap
* Intrinsic carrier concentration
* Electron mobility
* Hole mobility
* Thermal conductivity

Initial materials may include:

* Silicon
* Silicon dioxide
* Generic conductor
* Generic dielectric

Later materials may include:

* Silicon carbide
* Gallium nitride
* Germanium
* User-defined materials

### 5.4 Region

A `Region` associates a geometric area or volume with:

* A material
* Donor concentration
* Acceptor concentration
* Region name
* Region type
* Optional model parameters

Example regions include:

* Source
* Drain
* Channel
* Substrate
* Oxide
* Gate
* Contact

Regions should be represented using grid-aligned masks during the initial structured-grid implementation.

### 5.5 BoundaryCondition

Boundary conditions should be represented explicitly rather than embedded directly inside solver code.

Initial boundary-condition types:

* Dirichlet
* Neumann

Future types:

* Robin
* Periodic
* Interface conditions
* Floating contacts

Illustrative interface:

```python
left_contact = BoundaryCondition(
    name="left_contact",
    condition_type="dirichlet",
    value=0.0,
)
```

### 5.6 Device

The `Device` class describes a complete semiconductor device.

Responsibilities:

* Hold the computational grid
* Store device regions
* Store contacts
* Store material assignments
* Store doping distributions
* Store boundary conditions
* Validate the physical configuration
* Provide geometry information to visualisation tools

The `Device` class should describe the problem but should not solve it.

### 5.7 Simulation

The `Simulation` class coordinates a simulation run.

Responsibilities:

1. Validate the device.
2. Initialise physical fields.
3. Apply the selected physics model.
4. Apply the selected discretisation.
5. Pass the problem to the selected solver.
6. Execute the selected compute backend.
7. Calculate derived quantities.
8. Return a `SimulationResult`.

Illustrative interface:

```python
simulation = Simulation(
    device=device,
    physics_model=ElectrostaticModel(),
    solver=SORSolver(),
    backend="numpy",
)

result = simulation.run()
```

### 5.8 SimulationResult

The `SimulationResult` class contains all relevant outputs from a simulation.

Expected contents:

* Potential field
* Electric-field components
* Electric-field magnitude
* Charge-density field
* Convergence status
* Number of iterations
* Residual history
* Runtime
* Solver name
* Backend name
* Device parameters
* Derived device metrics
* Software version
* Execution metadata

The result object should be usable by:

* Plotting tools
* Benchmarking workflows
* Optimisation workflows
* Dataset generation
* Regression tests
* Interactive dashboards
* Report-generation tools

---

## 6. Geometry Module

The `deviceforge.geometry` package defines semiconductor-device geometry independently of the numerical solver.

### 6.1 Initial Two-Dimensional Geometry

The initial implementation will use structured Cartesian grids and Boolean masks.

Possible initial devices include:

* Rectangular Laplace benchmark
* Abrupt PN junction
* PIN junction
* Simplified MOS capacitor
* Simplified MOSFET-like structure

Geometry functions should generate named regions rather than directly changing physical fields.

Example:

```python
device = create_pn_junction(
    grid=grid,
    junction_position=50.0e-9,
    donor_density=1.0e22,
    acceptor_density=1.0e22,
)
```

### 6.2 Future Three-Dimensional Geometry

The three-dimensional implementation should use the same public `Device`, `Region`, `Material` and `Grid` interfaces.

Possible future geometries include:

* Three-dimensional PN junction
* Fin-shaped semiconductor structure
* Nanowire-like structure
* Three-dimensional MOSFET-inspired geometry

Three-dimensional geometry will require:

* Volumetric masks
* Greater memory control
* 3D boundary surfaces
* 3D visualisation
* More advanced meshing or domain decomposition

---

## 7. Physics Module

The `deviceforge.physics` package defines physical models and source terms.

### 7.1 Initial Electrostatic Model

The first implemented equation will be:

[
\nabla \cdot \left(\varepsilon \nabla \phi\right) = -\rho
]

where:

* (\phi) is electrostatic potential
* (\varepsilon) is material permittivity
* (\rho) is charge density

The initial fixed-charge model may use:

[
\rho = q(N_D - N_A)
]

The electric field is calculated using:

[
\mathbf{E} = -\nabla \phi
]

### 7.2 Future Physical Models

Future development may include:

* Mobile carrier concentrations
* Poisson-Boltzmann coupling
* Electron continuity equation
* Hole continuity equation
* Drift-diffusion transport
* Recombination-generation models
* Temperature-dependent material properties
* Electrothermal coupling
* Quantum correction models

Each physics model should expose a common interface where practical.

---

## 8. Numerical Solver Architecture

The `deviceforge.solvers` package defines numerical algorithms.

### 8.1 Solver Interface

All solvers should expose a common interface.

```python
result = solver.solve(problem)
```

A solver should accept:

* Discretised problem
* Initial solution
* Boundary conditions
* Numerical tolerances
* Maximum iteration count
* Compute backend

A solver should return:

* Solution
* Convergence status
* Residual history
* Iteration count
* Runtime
* Diagnostic information

### 8.2 Initial Solvers

The planned implementation sequence is:

1. Jacobi
2. Gauss-Seidel
3. Successive Over-Relaxation
4. Conjugate Gradient
5. Sparse direct or iterative methods
6. Newton-based nonlinear methods
7. Gummel-style coupled iteration

### 8.3 Solver Configuration

Solver settings should be represented using configuration objects.

```python
solver_config = SolverConfig(
    tolerance=1.0e-8,
    max_iterations=10_000,
    relaxation_factor=1.8,
)
```

This avoids passing many unrelated arguments between functions.

---

## 9. Compute Backend Architecture

The compute backend determines how numerical work is executed.

The device, physics and solver interfaces should remain as independent as possible from the backend.

### 9.1 NumPy Backend

Purpose:

* Reference implementation
* Numerical validation
* Readability
* Rapid algorithm development

### 9.2 C++ Backend

Purpose:

* Performance-critical numerical kernels
* Professional C++ software design
* Sparse matrix operations
* Improved memory control

### 9.3 OpenMP Backend

Purpose:

* Shared-memory CPU parallelisation
* Thread-scaling studies
* Multicore performance benchmarking

### 9.4 CUDA Backend

Purpose:

* GPU acceleration
* Structured-grid kernels
* Parallel iterative updates
* GPU memory and kernel-performance studies

### 9.5 Distributed Backend

The initial distributed-computing implementation will distribute independent simulation cases using MPI.

Examples:

* Dataset generation
* Sensitivity-analysis samples
* Parameter sweeps
* Optimisation evaluations

Future development may include distributed domain decomposition for individual large simulations.

---

## 10. Machine-Learning Architecture

Machine learning will be added after a validated simulator and dataset-generation workflow are available.

The initial surrogate architecture should use a common interface.

```python
model.fit(features, targets)
predictions = model.predict(features)
model.save(path)
```

Planned implementations:

* PyTorch neural-network surrogate
* TensorFlow/Keras neural-network surrogate
* Gaussian-process surrogate
* Scikit-learn baseline models

The optimisation and visualisation layers should interact with a generic surrogate interface rather than depending directly on one ML framework.

### 10.1 Distributed Machine Learning

Distributed ML should be treated separately from distributed simulation.

Planned approaches include:

* PyTorch DistributedDataParallel
* TensorFlow distribution strategies
* Multi-GPU training
* Future multi-node training

Benchmarks may compare:

* Single CPU
* Single GPU
* Multiple GPUs
* PyTorch and TensorFlow implementations

---

## 11. Optimisation Architecture

The optimisation layer will search the semiconductor design space.

### 11.1 Design Variables

Possible design variables include:

* Device dimensions
* Junction location
* Doping concentration
* Doping-gradient parameters
* Applied voltage
* Temperature
* Oxide thickness
* Region dimensions
* Material selection

### 11.2 Objectives

Possible objectives include:

* Minimise peak electric field
* Reduce electric-field concentration
* Minimise leakage proxy
* Improve electrostatic control
* Maximise capacitance proxy
* Minimise simulation cost
* Balance performance and manufacturability constraints

### 11.3 Methods

Planned methods include:

* Latin hypercube sampling
* Sobol sensitivity analysis
* Morris screening
* Bayesian optimisation
* Genetic algorithms
* NSGA-II multi-objective optimisation

### 11.4 Optimisation Result

An optimisation result should contain:

* Candidate designs
* Objective values
* Constraint values
* Pareto-optimal set
* Selected recommended designs
* Original design
* Optimised design
* Simulation results for comparison
* Surrogate uncertainty where available

---

## 12. User Interface Architecture

DeviceForge should include an interactive engineering interface in addition to its Python API.

The initial user interface will likely be implemented as a browser-based application using Streamlit and Plotly.

The user interface must remain separate from the simulation core.

The dashboard should call public DeviceForge APIs rather than directly accessing internal arrays or solver implementation details.

### 12.1 Main Interface Sections

The dashboard should contain the following sections.

#### Device Builder

The user can define:

* Device type
* Device dimensions
* Grid resolution
* Materials
* Doping levels
* Region dimensions
* Applied voltages
* Temperature

The interface should display a live representation of the device geometry.

#### Simulation Controls

The user can select:

* Physics model
* Numerical solver
* Compute backend
* Convergence tolerance
* Maximum iteration count
* Grid resolution

The user can then start a simulation.

#### Results Viewer

The results viewer should display:

* Device geometry
* Material map
* Doping map
* Potential field
* Electric-field magnitude
* Electric-field vectors
* Charge-density map
* Convergence history
* Solver metrics
* Runtime information

#### Performance Viewer

The performance section should display:

* Python versus C++ runtime
* Serial versus OpenMP runtime
* CPU versus GPU runtime
* Speed-up
* Parallel efficiency
* Memory consumption
* Mesh-scaling results

#### Sensitivity Viewer

The sensitivity-analysis section should display:

* Parameter importance
* First-order Sobol indices
* Total-order Sobol indices
* Interaction effects
* Input-output response plots

#### Optimisation Viewer

The optimisation interface should allow users to:

* Select design variables
* Select optimisation objectives
* Define variable bounds
* Define constraints
* Select optimisation algorithm
* Start an optimisation run
* Monitor optimisation progress

### 12.2 Pareto-Front Interaction

For multi-objective optimisation, the user should be able to:

* View an interactive Pareto front
* Hover over candidate designs
* Click a candidate design
* Inspect its device parameters
* Load the selected design into the simulation viewer
* Compare it with the baseline design

### 12.3 Initial and Optimised Device Comparison

The interface should provide a split-screen comparison.

```text
┌──────────────────────────┬──────────────────────────┐
│     Initial Device       │     Optimised Device     │
├──────────────────────────┼──────────────────────────┤
│ Geometry                 │ Geometry                 │
│ Doping distribution      │ Doping distribution      │
│ Potential field          │ Potential field          │
│ Electric field           │ Electric field           │
│ Device metrics           │ Device metrics           │
│ Runtime                  │ Runtime                  │
└──────────────────────────┴──────────────────────────┘
```

This comparison is intended to make optimisation results visually understandable.

### 12.4 Two-Dimensional Visualisation

Initial 2D outputs should include:

* Filled contour plots
* Heatmaps
* Region outlines
* Contact labels
* Vector fields
* Line profiles
* Cross-sections
* Interactive zoom
* Hover values

### 12.5 Three-Dimensional Visualisation

The future 3D interface should support:

* Rotatable device geometry
* Volume rendering
* Surface contours
* Slice planes
* Clipping planes
* Iso-surfaces
* Cross-sectional views
* Region transparency
* Optimised geometry comparison

Possible visualisation libraries include:

* Plotly
* PyVista
* VTK

The 3D viewer should consume the same `SimulationResult` data model used by the 2D viewer.

---

## 13. Presentation and Reporting

DeviceForge should support several presentation formats.

### 13.1 Interactive Dashboard

Used for:

* Parameter exploration
* Simulation inspection
* Optimisation interaction
* Portfolio demonstrations

### 13.2 Static Scientific Figures

Used for:

* GitHub README
* Technical reports
* LinkedIn posts
* Benchmark documentation

### 13.3 Exported Results

Planned export formats include:

* CSV
* JSON
* NumPy arrays
* PNG
* SVG
* VTK-compatible formats for 3D
* YAML simulation configuration

### 13.4 Automated Report Generation

A future reporting workflow may generate a summary containing:

* Device configuration
* Solver configuration
* Simulation metrics
* Potential and electric-field plots
* Benchmark results
* Sensitivity results
* Optimisation results
* Recommended device configuration

---

## 14. Application Workflows

### 14.1 Single Simulation Workflow

```text
User defines device
        ↓
Device configuration is validated
        ↓
Grid and regions are created
        ↓
Physics model is initialised
        ↓
Numerical system is assembled
        ↓
Solver executes on selected backend
        ↓
Derived fields and metrics are calculated
        ↓
SimulationResult is returned
        ↓
Results are visualised or exported
```

### 14.2 Dataset-Generation Workflow

```text
Parameter space is defined
        ↓
Samples are generated
        ↓
Simulation cases are distributed
        ↓
Each case produces device metrics
        ↓
Results are validated and stored
        ↓
Dataset is prepared for ML training
```

### 14.3 Surrogate-Modelling Workflow

```text
Simulation dataset
        ↓
Preprocessing and normalisation
        ↓
Training and validation split
        ↓
Surrogate model training
        ↓
Model validation
        ↓
Prediction-speed benchmarking
        ↓
Model export
```

### 14.4 Optimisation Workflow

```text
Design variables and objectives
        ↓
Initial design sampling
        ↓
Simulator or surrogate evaluation
        ↓
Optimisation algorithm
        ↓
Candidate device configurations
        ↓
Pareto-optimal designs
        ↓
High-fidelity simulation validation
        ↓
Interactive device comparison
```

---

## 15. Configuration Architecture

Simulation configurations should eventually be serialisable.

Example YAML configuration:

```yaml
device:
  type: pn_junction
  dimension: 2

grid:
  shape: [200, 100]
  spacing: [1.0e-9, 1.0e-9]

materials:
  semiconductor: silicon

doping:
  donor_density: 1.0e22
  acceptor_density: 1.0e22

boundary_conditions:
  left:
    type: dirichlet
    value: 0.0
  right:
    type: dirichlet
    value: 0.7

physics:
  model: electrostatic

solver:
  method: sor
  tolerance: 1.0e-8
  maximum_iterations: 10000
  relaxation_factor: 1.8

backend:
  type: numpy
```

The configuration layer should validate:

* Required fields
* Units
* Value ranges
* Supported model names
* Compatible solver and backend combinations

---

## 16. Error Handling and Logging

DeviceForge should provide clear exceptions for:

* Invalid grid shapes
* Invalid spacing
* Overlapping or missing regions
* Unknown materials
* Invalid boundary conditions
* Solver non-convergence
* Unsupported dimensions
* Backend unavailability
* Invalid optimisation constraints
* Model-loading failures

Library code should use structured logging rather than relying on uncontrolled print statements.

Log messages may include:

* Simulation start
* Grid size
* Selected solver
* Selected backend
* Convergence progress
* Runtime
* Warnings
* Validation failures
* Optimisation progress

---

## 17. Testing Architecture

### 17.1 Unit Tests

Unit tests will cover:

* Grid construction
* Field validation
* Material properties
* Region masks
* Boundary-condition assignment
* Differential operators
* Residual calculations
* Post-processing functions

### 17.2 Integration Tests

Integration tests will cover:

* Complete Laplace simulations
* Complete Poisson simulations
* Device construction
* Solver-to-result workflows
* Dashboard-to-simulation workflows
* Dataset generation
* Optimisation evaluation

### 17.3 Regression Tests

Regression tests will ensure that:

* Solver refactoring does not change validated results
* C++ and CUDA implementations remain consistent with the reference solver
* Performance changes do not alter numerical outputs
* ML model changes do not silently change data-processing behaviour

### 17.4 Validation Cases

Initial validation cases include:

* Linear potential between parallel boundaries
* Analytical Laplace solutions
* Manufactured Poisson solutions
* Grid-convergence studies
* Simplified abrupt PN-junction behaviour

---

## 18. Two-Dimensional Development Plan

The initial 2D development sequence is:

1. Implement the `Grid` class.
2. Implement the `Field` class.
3. Implement boundary-condition objects.
4. Implement basic rectangular geometry.
5. Implement the Jacobi Laplace solver.
6. Validate the solver.
7. Add electric-field post-processing.
8. Add 2D visualisation.
9. Add Gauss-Seidel and SOR.
10. Add semiconductor materials and doping.
11. Implement a 2D Poisson model.
12. Create a PN-junction demonstration.
13. Add benchmarking.
14. Add an interactive 2D dashboard.
15. Add C++ and accelerated backends.
16. Add surrogate modelling and optimisation.
17. Display optimised 2D device configurations.

---

## 19. Three-Dimensional Extension Plan

Three-dimensional development will begin only after the 2D framework is validated and stable.

The first 3D milestone will be a structured-grid Laplace or Poisson benchmark using the same public interfaces as the 2D implementation.

The planned 3D sequence is:

1. Extend grid coordinate generation to three dimensions.
2. Extend scalar and vector fields to 3D arrays.
3. Implement 3D finite-difference operators.
4. Implement 3D boundary surfaces.
5. Add 3D solver kernels.
6. Validate against a known cuboid-domain problem.
7. Add 3D field slicing and visualisation.
8. Add GPU memory and performance benchmarks.
9. Create a basic 3D semiconductor device.
10. Extend optimisation outputs to 3D geometry.
11. Display initial and optimised 3D configurations interactively.

---

## 20. Dependency Rules

To prevent circular dependencies and tightly coupled code, the following dependency direction should be maintained.

```text
core
  ↑
geometry and physics
  ↑
solvers and backends
  ↑
postprocessing
  ↑
workflows, optimisation and machine learning
  ↑
visualisation and user interface
```

The core package must not depend on:

* Streamlit
* Plotly
* PyTorch
* TensorFlow
* CUDA-specific Python libraries
* Optimisation libraries

The user interface may depend on the public APIs of all lower layers.

---

## 21. Initial Public API

The intended future user-facing API may resemble:

```python
from deviceforge import Grid, Simulation
from deviceforge.geometry import create_pn_junction
from deviceforge.physics import ElectrostaticModel
from deviceforge.solvers import SORSolver
from deviceforge.visualisation import plot_result

grid = Grid(
    shape=(200, 100),
    spacing=(1.0e-9, 1.0e-9),
)

device = create_pn_junction(
    grid=grid,
    donor_density=1.0e22,
    acceptor_density=1.0e22,
    applied_voltage=0.7,
)

simulation = Simulation(
    device=device,
    physics_model=ElectrostaticModel(),
    solver=SORSolver(
        tolerance=1.0e-8,
        max_iterations=10_000,
    ),
    backend="numpy",
)

result = simulation.run()

plot_result(result)
```

The dashboard should use the same public API.

---

## 22. Initial User-Interface Concept

A possible dashboard layout is:

```text
┌───────────────────────────────────────────────────────────────┐
│                        DeviceForge                            │
├───────────────────┬───────────────────────────────────────────┤
│ Device Parameters │ Device Geometry                           │
│                   │                                           │
│ Device type       │ 2D or 3D interactive view                │
│ Dimensions        │                                           │
│ Doping            ├───────────────────────────────────────────┤
│ Voltage           │ Simulation Results                        │
│ Temperature       │                                           │
│ Grid resolution   │ Potential · Electric Field · Charge       │
│                   │                                           │
│ Solver Settings   ├───────────────────────────────────────────┤
│                   │ Metrics and Convergence                    │
│ Solver            │                                           │
│ Backend           │ Runtime · Iterations · Residual            │
│ Tolerance         │                                           │
│                   ├───────────────────────────────────────────┤
│ [Run Simulation]  │ Optimisation and Pareto Front             │
│ [Run Optimiser]   │                                           │
└───────────────────┴───────────────────────────────────────────┘
```

---

## 23. Current Scope

The current development scope is limited to:

* Two-dimensional structured grids
* Electrostatic Laplace and Poisson problems
* NumPy-based numerical solvers
* Boundary-condition handling
* Electric-field calculation
* Solver convergence analysis
* Scientific 2D visualisation
* Architecture suitable for later UI integration

The following are planned but not yet implemented:

* C++
* OpenMP
* CUDA
* MPI
* PyTorch
* TensorFlow
* Surrogate modelling
* Optimisation
* Interactive dashboard
* Three-dimensional simulation

---

## 24. Architectural Principle

The central architectural principle of DeviceForge is:

> A semiconductor device should be defined once, simulated using interchangeable numerical and computational backends, and presented through both programmatic and interactive interfaces.

This will allow DeviceForge to evolve from an initial 2D electrostatic solver into a wider platform for semiconductor simulation, HPC experimentation, machine-learning acceleration and optimised device-design visualisation.
