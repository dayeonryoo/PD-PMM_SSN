# SSN-PMM: Semi-Smooth Newton Proximal Method of Multipliers

A C++17 solver for convex **Quadratic Programs (QPs)** and **Linear Programs (LPs)**, with Python bindings via pybind11. The solver implements a Primal-Dual Proximal Method of Multipliers (PMM) outer loop with a Semi-Smooth Newton (SSN) inner solver, accelerated by iterative Krylov methods.

---

## Problem form

SSN-PMM solves problems of the form:

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
git clone https://github.com/<your-org>/PD-PMM_SSN.git
cd PD-PMM_SSN
```

---

## Building

### Option A — C++ executables

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

This produces four executables inside `build/`:

| Executable | Description |
|---|---|
| `ssn_pmm_netlib` | Runs the solver on the Netlib LP test set |
| `ssn_pmm_maros_meszaros` | Runs the solver on the Maros-Meszaros QP test set |
| `ssn_pmm_pde` | Runs the solver on a PDE-constrained problem |
| `mps_parser` | Standalone MPS/SIF file parser demo |

### Option B — Python bindings

All commands are run from the `python/` subdirectory.

```bash
cd python
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python3)
cmake --build . --config Release
cd ..
```

This places `ssn_pmm_bind.cpython-<tag>-<platform>.so` directly in `python/`, so
`import ssn_pmm_bind` works without any installation step.

---

## Quick start

### C++ — solve a problem from matrices

```cpp
#include "SSN_PMM.hpp"
#include "Problem.hpp"

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

    SSN_PMM<T> solver(prob);
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
#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "MpsParser.hpp"

using T = double;

int main() {
    MpsParser<T>   parser;
    ParsedModel<T> model = parser.parse("path/to/problem.mps");
    PDPMMdata<T>   pd    = parser.to_pdpmm(model);

    T   tol        = 1e-6;
    int max_iter   = 1000000;
    double time_limit = 600.0;   // seconds

    Problem<T>  prob(pd, tol, max_iter, time_limit,
                     PrintWhen::EVERY10, PrintWhat::TUNING);
    SSN_PMM<T>  solver(prob);
    Solution<T> sol = solver.solve();

    sol.print_summary();
    return 0;
}
```

### Python — solve from a SIF/MPS file

```python
import sys
sys.path.insert(0, "path/to/PD-PMM_SSN/python")
import ssn_pmm_bind

result = ssn_pmm_bind.solve_from_sif(
    "path/to/PD-PMM_SSN/data/maros_meszaros/QAFIRO.SIF",
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
sys.path.insert(0, "path/to/PD-PMM_SSN/python")
import ssn_pmm_bind

pd = ssn_pmm_bind.parse_sif("path/to/problem.SIF")

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
| `what` | `PrintWhat` | `NONE` / `TUNING` |

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

### Python `ssn_pmm_bind`

**`solve_from_sif(filename, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0)`**
Parse and solve a SIF/MPS file. Returns a dict with keys:
`status`, `obj_val`, `solving_time`, `pmm_iter`, `ssn_iter`, `krylov_iter`.

**`parse_sif(filename)`**
Parse a SIF/MPS file and return problem data as numpy arrays.
Returns a dict with keys: `n`, `m`, `l`, `Q_*`, `A_*`, `B_*`, `c`, `b`, `lx`, `ux`, `lw`, `uw`, `obj_const`.
Sparse matrices are in CSC format (`_data`, `_indices`, `_indptr`, `_shape`).

---

## Running the benchmarks

### Maros-Meszaros QP benchmark (Python, compares SSN-PMM vs QPALM vs OSQP)

```bash
pip install qpalm osqp numpy scipy matplotlib pandas

# Build the Python binding first (see "Building" above)

cd python
python3 benchmark_mm.py
```

Results are written to `results/comparison_mm.csv` and a Dolan-Moré performance profile is saved as `results/performance_profile_mm.pdf/.png`.

Optional arguments:
```
--root /path/to/PD-PMM_SSN   # override project root (default: parent of script)
--tol 1e-6                   # primal-dual tolerance
--time-limit 600             # per-problem time limit in seconds
```

### Netlib LP benchmark (C++)

```bash
./build/ssn_pmm_netlib
```

Results are appended to `results/*.csv`.

---

## Project structure

```
PD-PMM_SSN/
├── include/
│   ├── Problem.hpp          # Problem data structure
│   ├── Solution.hpp         # Solution data structure
│   ├── SSN_PMM.hpp/.tpp     # Main solver
│   ├── SSN.hpp/.tpp         # SSN inner solver
│   ├── MpsParser.hpp/.tpp   # MPS/SIF file parser
│   ├── SchurOperator.hpp    # Schur complement linear operator
│   ├── SchurPreconditioner.hpp
│   ├── Printing.hpp
│   └── RecordResult.hpp
├── src/
│   ├── netlib.cpp           # Netlib LP benchmark runner
│   ├── maros_meszaros.cpp   # Maros-Meszaros QP benchmark runner
│   ├── PDE.cpp              # PDE-constrained problem runner
│   └── readingMps.cpp       # MPS parser demo
├── python/
│   ├── ssn_pmm_bind.cpp     # pybind11 bindings
│   ├── benchmark_mm.py      # Python benchmark script
│   └── CMakeLists.txt       # Python binding build config
├── data/
│   ├── netlib/              # Netlib LP instances (.mps)
│   ├── netlib_infeas/       # Infeasible Netlib instances (.mps)
│   ├── maros_meszaros/      # Maros-Meszaros QP instances (.SIF)
│   └── kennington/          # Kennington LP instances (.mps)
├── results/                 # Output CSVs and plots
└── CMakeLists.txt           # Main build configuration
```
