# KSP-QP: Krylov Semismooth-Newton Proximal Method of Multipliers

A C++17 solver for convex **Quadratic Programs (QPs)** and **Linear Programs (LPs)**, with Python bindings via pybind11. The solver implements a Primal-Dual Proximal Method of Multipliers (PMM) outer loop with a Semi-Smooth Newton (SSN) inner solver, accelerated by iterative Krylov methods.

---

## Problem form

KSP-QP solves problems of the form:

```
min   c^T x  +  (1/2) x^T Q x
s.t.  A x  = b
      lw  <=  B x  <=  uw
      lx  <=  x    <=  ux
```

where `Q` is a symmetric positive semi-definite matrix (set `Q = 0` for LPs).

---

## Requirements

### C++ build
- CMake >= 3.16
- A C++17-compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- Internet access for the first build (CMake auto-downloads Eigen and GoogleTest)

### Python bindings (optional)
- Python 3.9+
- `pip install numpy scipy matplotlib pandas`
- `pip install qpalm osqp` (only needed for the comparison benchmark)

---

## Getting the code

```bash
git clone https://github.com/dayeonryoo/KSP-QP
cd KSP-QP
```

---

## Building

### Option A — C++ executables

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

> **Portability note:** `CMakeLists.txt` compiles with `-march=native`, which tunes for the CPU
> doing the build and is not portable to other machines — a binary built this way can crash with
> `Illegal instruction` on a different/older CPU. This is safe as long as everyone builds from
> source on their own machine (the normal workflow here); it becomes a problem if you copy a
> compiled `build/` to another machine, bake one into a Docker image that runs elsewhere, or
> ship a prebuilt release binary. In those cases, switch `-march=native` to a portable baseline
> (e.g. `-march=x86-64-v2`) first.

This produces two executables inside `build/`:

| Executable | Source | Description |
|---|---|---|
| `ksp_qp_netlib` | `src/netlib.cpp` | Runs the solver on Netlib LPs (`.mps`) |
| `ksp_qp_maros_meszaros` | `src/maros_meszaros.cpp` | Runs the solver on Maros-Meszaros QPs (`.SIF`) |

PDE-constrained QPs are generated in Python (`python/pde_generator.py`) rather than by a C++
driver, so that `include/` holds solver code only — see
[the PDE benchmark](#pde-constrained-qp-benchmarks-smooth-l2-regularized) below.

`ksp_qp_netlib` and `ksp_qp_maros_meszaros` each take `--name`, `--tol`,
`--max-iter`, and `--time-limit` flags. `--name` picks a single problem to solve, or `all` to
sweep every problem the driver knows about (both also
take `--root` to point at a different data directory, and write a CSV when `--name all` is
used). Run either of them with `--help` for the full flag list. Run them from the repo root so
their default (relative) data paths resolve, or pass `--root`. Each driver still hardcodes its
`PrintWhen`/`PrintWhat` settings and (beyond the flags above) some legacy alternates only
reachable by editing `main()` — see
["Running the C++ drivers"](#running-the-c-drivers) below.

### Option B — Python bindings

All commands are run from the `python/` subdirectory.

```bash
cd python
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python3)
cmake --build . --config Release
cd ..
```

This places `ksp_qp_bind.cpython-<tag>-<platform>.so` directly in `python/`, so
`import ksp_qp_bind` works without any installation step.

---

## Quick start

### C++ — solve a problem from matrices

```cpp
#include "ksp_qp.hpp"
#include "problem.hpp"

using T   = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;

int main() {
    int n = 2;  // variables
    int m = 1;  // equality constraints
    int l = 0;  // general inequality rows (none here)

    // Q = [[2, 0], [0, 2]]  (positive definite)
    SpMat Q(n, n);
    Q.insert(0, 0) = 2.0;
    Q.insert(1, 1) = 2.0;
    Q.makeCompressed();

    // c = [-2, -5]
    Vec c(n);
    c << -2.0, -5.0;

    // A x = b  →  x[0] + x[1] = 1
    SpMat A(m, n);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 1.0;
    A.makeCompressed();

    Vec b(m);
    b << 1.0;

    // B (empty — no general inequality constraints)
    SpMat B(l, n);

    // Bounds on x: 0 <= x <= inf
    Vec lx = Vec::Zero(n);
    Vec ux = Vec::Constant(n, 1e20);

    // Bounds on Bx (empty)
    Vec lw(l), uw(l);

    T   tol        = 1e-6;
    int max_iter   = 10000;
    double time_limit = 60.0;   // seconds

    Problem<T> prob(Q, A, B, c, b, /*obj_const=*/0.0,
                    lx, ux, lw, uw,
                    tol, max_iter, time_limit,
                    PrintWhen::EVERY10, PrintWhat::TUNING);

    KSP_QP<T> solver(prob);
    Solution<T> sol = solver.solve();

    sol.print_summary();
    // sol.opt == 0  →  optimal
    // sol.obj_val   →  optimal objective value
    // sol.x         →  primal solution vector
    return 0;
}
```

### C++ — solve from an MPS/SIF file

```cpp
#include "ksp_qp.hpp"
#include "problem.hpp"
#include "mps_format_parser.hpp"

using T = double;

int main() {
    MpsFormatParser<T> parser;
    ParsedModel<T> model = parser.parse("path/to/problem.mps");
    KSPQPdata<T>   pd    = parser.to_kspqp(model);

    T   tol        = 1e-6;
    int max_iter   = 1000000;
    double time_limit = 600.0;   // seconds

    Problem<T>  prob(pd, tol, max_iter, time_limit,
                     PrintWhen::EVERY10, PrintWhat::TUNING);
    KSP_QP<T>  solver(prob);
    Solution<T> sol = solver.solve();

    sol.print_summary();
    return 0;
}
```

### Python — solve from a SIF/MPS file

```python
import sys
sys.path.insert(0, "path/to/KSP-QP/python")
import ksp_qp_bind

result = ksp_qp_bind.solve_from_sif(
    "path/to/KSP-QP/data/maros_meszaros/QAFIRO.SIF",
    tol        = 1e-6,
    max_iter   = 1_000_000_000,
    time_limit = 600.0,
)

print("Status      :", result["status"])       # 0 = optimal
print("Objective   :", result["obj_val"])
print("Solve time  :", result["solving_time"], "s")
print("PMM iters   :", result["pmm_iter"])
print("SSN iters   :", result["ssn_iter"])
print("Krylov iters:", result["krylov_iter"])
```

### Python — parse a SIF file and feed to another solver

```python
import sys, scipy.sparse as sp, numpy as np
sys.path.insert(0, "path/to/KSP-QP/python")
import ksp_qp_bind

pd = ksp_qp_bind.parse_sif("path/to/problem.SIF")

# Reconstruct scipy sparse matrices
Q = sp.csc_matrix((pd["Q_data"], pd["Q_indices"], pd["Q_indptr"]), shape=pd["Q_shape"])
A = sp.csc_matrix((pd["A_data"], pd["A_indices"], pd["A_indptr"]), shape=pd["A_shape"])
B = sp.csc_matrix((pd["B_data"], pd["B_indices"], pd["B_indptr"]), shape=pd["B_shape"])

c, b   = pd["c"],  pd["b"]
lx, ux = pd["lx"], pd["ux"]
lw, uw = pd["lw"], pd["uw"]
n, m, l = pd["n"], pd["m"], pd["l"]
```

---

## API reference

### C++ `Problem<T>`

| Field | Type | Description |
|---|---|---|
| `Q` | `SpMat` | `n×n` symmetric PSD quadratic cost matrix |
| `A` | `SpMat` | `m×n` equality constraint matrix |
| `B` | `SpMat` | `l×n` general inequality constraint matrix |
| `c` | `Vec` | `n`-dim linear cost vector |
| `b` | `Vec` | `m`-dim RHS of equality constraints |
| `lx`, `ux` | `Vec` | `n`-dim variable bounds |
| `lw`, `uw` | `Vec` | `l`-dim bounds on `Bx` |
| `tol` | `T` | Primal-dual termination tolerance (default `1e-6`) |
| `max_iter` | `int` | Maximum PMM outer iterations |
| `time_limit` | `double` | Wall-clock limit in seconds (default `600`) |
| `when` | `PrintWhen` | `NEVER` / `EVERY10` / `ALWAYS` |
| `what` | `PrintWhat` | `NONE` / `MINIMAL` / `SSN` / `TUNING` / `FULL` (see ["Tuning: printing and timers"](#tuning-printing-and-timers)) |

### C++ `Solution<T>`

| Field | Type | Description |
|---|---|---|
| `opt` | `int` | Termination status (see table below) |
| `x` | `Vec` | Primal solution |
| `y1` | `Vec` | Dual variables for `Ax = b` |
| `y2` | `Vec` | Dual variables for `lw ≤ Bx ≤ uw` |
| `z` | `Vec` | Dual variables for variable bounds |
| `obj_val` | `T` | Primal objective value |
| `PMM_iter` | `int` | PMM outer iterations performed |
| `SSN_iter` | `int` | Total SSN inner iterations |
| `Krylov_iter` | `int` | Total Krylov iterations |
| `solving_time` | `double` | Wall-clock solve time in seconds |

### Termination status codes

| `opt` | Meaning |
|---|---|
| `0` | Optimal solution found |
| `-2` | Primal infeasibility detected |
| `-3` | Dual infeasibility detected |
| `-1` | Numerical error |
| `1` | Maximum PMM iterations reached |
| `2` | Maximum SSN iterations reached |
| `3` | Line search failed |
| `4` | Time limit exceeded |

### Python `ksp_qp_bind`

**`solve_from_sif(filename, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0)`**
Parse and solve a SIF/MPS file. Returns a dict with keys:
`status`, `obj_val`, `solving_time`, `pmm_iter`, `ssn_iter`, `krylov_iter`.

**`parse_sif(filename)`**
Parse a SIF/MPS file and return problem data as numpy arrays.
Returns a dict with keys: `n`, `m`, `l`, `Q_*`, `A_*`, `B_*`, `c`, `b`, `lx`, `ux`, `lw`, `uw`, `obj_const`.
Sparse matrices are in CSC format (`_data`, `_indices`, `_indptr`, `_shape`).

---

## Running the C++ drivers

Each driver takes a small set of `--flag value` command-line options (see `--help`), and
defaults to paths relative to the repo root — **run them from the repo root**, or pass `--root`
to point at your clone from elsewhere. No editing or rebuilding is needed just to change `tol`,
`max_iter`, `time_limit`, or (where applicable) which data file to load.

### `ksp_qp_netlib` — Netlib LPs (`src/netlib.cpp`)

```bash
./build/ksp_qp_netlib [--root DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE]
```

Solves `<root>/<PROBLEM>.mps` (default: `data/netlib/AFIRO.mps`), printing the solution summary.
Pass `--name all` to sweep every Netlib LP with a known reference objective value (the same
`name -> obj_val` map used historically), checking each result against it and appending a row
to `<root>/results/netlib_all.csv` (override with `--out`). For a QPALM/OSQP comparison with
performance profiles, use `python/benchmark_netlib.py` instead (see below).

Two infeasibility-detection alternates are also included, commented out at the bottom of the
file (sweep the Netlib-infeasible set, or solve one infeasible LP by name) — these predate
`--name` and still require editing `main()` (uncomment, remove the `/* ... */`, rebuild) since
they check for *detected infeasibility* rather than an objective value.

### `ksp_qp_maros_meszaros` — Maros-Meszaros QPs (`src/maros_meszaros.cpp`)

```bash
./build/ksp_qp_maros_meszaros [--root DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE] [--cooldown S]
```

Solves `<root>/<PROBLEM>.SIF` (default: `data/maros_meszaros/AUG2DCQP.SIF`), printing the
solution summary. Pass `--name all` to sweep the full Maros-Meszaros set against its built-in
reference objectives, appending a row to `<root>/results/maros_meszaros_all.csv` (override with
`--out`) for each — `--cooldown` (default 0s) sleeps between problems in this mode.

For comparing against QPALM/OSQP rather than just checking against reference objectives, use
the Python benchmark instead (see below) — that's what produces performance profiles.

### PDE-constrained QPs — generated in Python

There is no C++ driver for these: the generators live in `python/pde_generator.py` (Q1 FEM and
FD assembly, built on the element kernels in `python/fem_q1.py`), so `include/` stays
solver-only. They build the L2-regularized PDE-constrained control problems of
(Pearson & Gondzio, 2017):

| Choice | Generator | Problem |
|---|---|---|
| `poisson` | `make_poisson_l2_control` | 2D Poisson control, control-constrained |
| `poisson_state` | `make_poisson_l2_state_control` | 2D Poisson control, state-constrained |
| `convdiff` | `make_convdiff_l2_control` | 2D convection-diffusion control |

Each returns a `KSPQPdata`; `.to_dict()` gives the CSC/numpy dict that
`ksp_qp_bind.solve_from_data()` and `benchmark_common.kspqp_to_qpalm()` consume.
`generate_pde_l2_qp(choice, nc, beta, ...)` wraps all three and returns that dict directly:

```python
import pde_generator, ksp_qp_bind          # from the python/ directory
pd = pde_generator.generate_pde_l2_qp("convdiff", nc=6, beta=1e-2,
                                      y_lower=0.0, y_upper=0.2,
                                      u_lower=-0.75, u_upper=0.75, eps=0.01)
res = ksp_qp_bind.solve_from_data(pd, 1e-6, 10**9, 600.0)
```

`nc` is the grid exponent (grid size `2^nc + 1` per direction). `lumped_mass` swaps the
consistent Q1 mass matrix for the lumped one, and `discretization` selects `"fem"` (default)
or `"fd"` (5-point stencil with first-order upwind convection; always lumped).

---

## Running the benchmarks

The Python benchmark scripts compare **KSP-QP vs QPALM vs OSQP** and live in `python/`.
Build the Python binding first (see "Building" above), then `pip install qpalm osqp numpy scipy
matplotlib pandas`.

### Maros-Meszaros QP benchmark

```bash
cd python
python3 benchmark_mm.py
```

Runs the full Maros-Meszaros test set. Writes `results/comparison_mm.csv` plus Dolan-Moré
performance profiles (`results/performance_profile_mm*.pdf/.png`, by time and by iteration count).

```
--root DIR             override project root (default: parent of script)
--tol 1e-6             primal-dual tolerance
--time-limit 600       per-problem time limit in seconds
--solver {ksp-qp,qpalm,osqp} [...]   which solvers to run (default: all three)
--out PREFIX           output file prefix (default: comparison_mm)
--cooldown 0           seconds to sleep between problems (avoids CPU throttling)
```

### Netlib LP benchmark

```bash
cd python
python3 benchmark_netlib.py
```

Runs the full Netlib LP test set. Writes `results/comparison_netlib.csv` plus Dolan-Moré
performance profiles (`results/performance_profile_netlib*.pdf/.png`). Same flags as
`benchmark_mm.py`.

### PDE-constrained QP benchmarks (smooth, L2-regularized)

```bash
cd python
python3 benchmark_l2pde.py
```

Produces three tables — `poisson_control`, `poisson_state`, `convdiff_both` — written to
`results/l2_<table>.csv`. The problems come from `pde_generator.py`
(Pearson & Gondzio, 2017); the sweep parameters follow their tables.

```
--root DIR              override project root (default: parent of script)
--tol 1e-6              solver tolerance
--time-limit 600        per-problem time limit in seconds (10 min default)
--table {poisson_control,poisson_state,convdiff_both} [...]   default: all three
--nc N [N ...]          grid exponents to sweep (default: 7 8 9 10)
--solver {ksp-qp,qpalm,osqp} [...]   default: all three
--cooldown 0            seconds to sleep between problems
--lumped-mass {0,1}     0 = consistent mass matrix (default), 1 = lumped
--discretization {fem,fd}   spatial discretization (default: fem)
--out PREFIX            output file prefix
```

---

## Tuning: printing and timers

The solver has two independent knobs for diagnosing/tuning performance: **runtime printing**
(what gets printed to stdout while solving, controlled by `Problem<T>`'s `when`/`what` fields)
and a **compile-time step timer** (per-phase wall-clock breakdown of the SSN inner loop,
printed to stderr).

### Runtime printing — `PrintWhen` / `PrintWhat` (`include/printing.hpp`)

`PrintWhen` controls *how often* a line is printed per PMM iteration:

| `PrintWhen` | Behavior |
|---|---|
| `NEVER` | No output |
| `EVERY10` | Print every 10th PMM iteration |
| `ALWAYS` | Print every PMM iteration |

`PrintWhat` controls *how much* is printed on each line:

| `PrintWhat` | Columns shown |
|---|---|
| `NONE` | Nothing (overrides `PrintWhen`) |
| `MINIMAL` | Iteration counts, residuals |
| `SSN` | + Krylov/factorization counts, PMM params (`mu`, `rho`, `eps`), line-search/Krylov failures — printed at every SSN iteration, not just per PMM iteration |
| `TUNING` | Same columns as `SSN`, at the normal per-PMM-iteration cadence |
| `FULL` | + objective value, but without the Krylov/factorization/failure columns |

Set these on the `Problem<T>` constructor, e.g. `Problem<T> prob(pd, tol, max_iter, time_limit,
PrintWhen::EVERY10, PrintWhat::TUNING);`. `TUNING` is the most useful combination for tuning
PMM/SSN hyperparameters — it shows residuals, `mu`/`rho`/`eps`, and failure counts together.
Turning printing off (`PrintWhen::NEVER` or `PrintWhat::NONE`) removes the stdout overhead
entirely, which matters when timing large sweeps.

### Compile-time step timers — `SSN_ENABLE_TIMERS` (`include/ssn.hpp`)

A separate, more granular timer instruments the phases inside each SSN iteration (system prep,
linear solve, preconditioner assembly/analyze/factorize, Krylov solve, LDLT fallback,
line search, state update). It is gated by a macro so it compiles to zero overhead when off:

```cpp
// include/ssn.hpp
#ifndef SSN_ENABLE_TIMERS
#define SSN_ENABLE_TIMERS 0   // set to 1 to enable
#endif
```

Enable it either by editing that line directly, or by passing the define at configure time
without touching the source:

```bash
cmake -B build -DCMAKE_CXX_FLAGS="-DSSN_ENABLE_TIMERS=1"
cmake --build build --config Release
```

When enabled, every SSN iteration prints a line like this to **stderr** (independent of the
`PrintWhen`/`PrintWhat` settings above):

```
[Timer] ssn_iter=3 total=0.1234s | prep=0.0012 linear_solve=0.1180 (prec_setup=0.0500 [assembly=0.0100 analyze=0.0150 factorize=0.0250] krylov_solve=0.0680) linesearch=0.0030 state_update=0.0012
```

If the Krylov solve falls back to a dense LDLT factorization for that iteration, a second line
reports the LDLT analyze/factorize/solve breakdown. This is the tool to use when profiling
*where* time goes inside the solver (e.g. preconditioner factorization vs. CG iterations);
`PrintWhat::TUNING` is the tool for watching *convergence behavior* (residuals, PMM parameters)
across iterations.

---

## Project structure

```
KSP-QP/
├── include/
│   ├── problem.hpp          # Problem data structure
│   ├── solution.hpp         # Solution data structure
│   ├── ksp_qp.hpp/.tpp     # Main solver
│   ├── ssn.hpp/.tpp         # SSN inner solver
│   ├── mps_format_parser.hpp/.tpp # MPS/SIF/QPS file parser
│   ├── ksp_qp_types.hpp       # ParsedModel/KSPQPdata data structures
│   ├── schur_operator.hpp    # Schur complement linear operator
│   ├── schur_preconditioner.hpp
│   ├── printing.hpp         # PrintWhen/PrintWhat runtime printing
│   └── record_result.hpp
├── src/
│   ├── netlib.cpp           # Netlib LP benchmark runner
│   └── maros_meszaros.cpp   # Maros-Meszaros QP benchmark runner
├── python/
│   ├── ksp_qp_bind.cpp          # pybind11 bindings
│   ├── fem_q1.py                 # Q1 finite-element reference-element kernels
│   ├── pde_generator.py          # Builds PDE-constrained QPs (Q1 FEM / FD)
│   ├── benchmark_common.py       # shared QPALM/OSQP conversion + runner helpers
│   ├── benchmark_mm.py           # Maros-Meszaros benchmark vs QPALM/OSQP
│   ├── benchmark_netlib.py       # Netlib LP benchmark vs QPALM/OSQP
│   ├── benchmark_infeas.py       # infeasible-instance benchmark vs QPALM/OSQP
│   ├── benchmark_l2pde.py        # L2 PDE-constrained benchmark vs QPALM/OSQP
│   ├── tests/                    # unittest suite for the Python generators
│   └── CMakeLists.txt            # Python binding build config
├── data/
│   ├── netlib/              # Netlib LP instances (.mps)
│   ├── netlib_infeas/       # Infeasible Netlib instances (.mps)
│   ├── maros_meszaros/      # Maros-Meszaros QP instances (.SIF)
│   └── kennington/          # Kennington LP instances (.mps)
├── results/                 # Output CSVs and plots
└── CMakeLists.txt           # Main build configuration
```
