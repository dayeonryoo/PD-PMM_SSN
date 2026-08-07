"""
Benchmark SSN-PMM vs QPALM vs OSQP on L1/L2-regularized PDE-constrained QP problems (see include/PDEgenerator.hpp).

Four named tables are produced:

  poisson_vary_n    Poisson control: varying grid size n and L1-reg alpha1; alpha2 fixed

  poisson_vary_a2   Poisson control: varying L2-reg alpha2; n, alpha1 fixed

  convdiff_vary_n   Conv-diff control: varying grid size n and L1-reg alpha1; alpha2 fixed

  convdiff_vary_a2  Conv-diff control: varying L2-reg alpha2; n, alpha1 fixed

Outputs
-------
  results/poisson_vary_n.csv
  results/poisson_vary_a2.csv
  results/convdiff_vary_n.csv
  results/convdiff_vary_a2.csv

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Install dependencies
----------------------------
  pip install numpy scipy qpalm osqp

Step 2 - Build the SSN-PMM Python binding (if not already built)
-----------------------------------------------------------------
  cd python && mkdir -p build && cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

Step 3 - Run the benchmark
---------------------------
python3 benchmark_pde.py

Settings: tol = 1e-6, time limit = 600 s (10 min), max iterations = infinity by default.
        --root:       to change the output directory (default: results/).
        --out:        to change the output file prefix (default: comparison_mm).
        --solver:     to select which solvers to run among ssn-pmm, qpalm, osqp (default: all three).
        --table:      to select which tables to run among poisson_vary_n, poisson_vary_a2, convdiff_vary_n, convdiff_vary_a2 (default: all four).
        --nc:         to select which grid exponents to run for vary-n tables (default: 6 7 8 9 10; see sweep parameters).
        --tol:        to change the solver tolerance (default: 1e-6).
        --time-limit: to change the solver time limit in seconds (default: 600).
        --cooldown:   to change the cooldown time in seconds between solver runs (default: 0).
        --lumped-mass: 0 to use the consistent mass matrix (default), 1 to use the lumped mass matrix.
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    import ssn_pmm_bind
except ModuleNotFoundError:
    sys.exit(
        "Cannot find ssn_pmm_bind. "
        "Build it first:\n"
        "  cd python && mkdir build && cd build\n"
        "  cmake .. && cmake --build . --config Release"
    )

from benchmark_common import (
    pdpmm_to_qpalm,
    run_qpalm,
    run_osqp,
    run_solvers,
    _write_csv,
    _load_existing_rows,
)

# ---------------------------------------------------------------------------
# Sweep parameters: See (Gondzio, Pougkakiotis & Pearson 2022) for the original tables and parameter choices.
# ---------------------------------------------------------------------------

POISSON_VARY_N_NC     = [6, 7, 8, 9, 10]
POISSON_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6, 0.0]
POISSON_VARY_N_ALPHA2 = 1e-2

POISSON_VARY_A2_NC     = 8
POISSON_VARY_A2_ALPHA1 = 1e-2
POISSON_VARY_A2_ALPHA2 = [1e-2, 1e-4, 1e-6, 0.0]

CONVDIFF_VARY_N_NC     = [6, 7, 8, 9, 10]
CONVDIFF_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6, 0.0]
CONVDIFF_VARY_N_ALPHA2 = 1e-2

CONVDIFF_VARY_A2_NC     = 9
CONVDIFF_VARY_A2_ALPHA1 = 1e-2
CONVDIFF_VARY_A2_ALPHA2 = [1e-2, 1e-4, 1e-6, 0.0]

ALL_TABLES = ["poisson_vary_n", "poisson_vary_a2", "convdiff_vary_n", "convdiff_vary_a2"]

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _n_display(nc: int) -> int:
    """n = 3 * np."""
    return 3*(2**nc+1)**2


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _worker_ssn(problem, nc, alpha1, alpha2, lumped_mass, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_l1l2_qp(problem, nc, alpha1, alpha2,
                                                     lumped_mass=lumped_mass)
        result["n_vars"] = pd_data["n"]
        result["res"] = ssn_pmm_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(problem, nc, alpha1, alpha2, lumped_mass, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_l1l2_qp(problem, nc, alpha1, alpha2,
                                                     lumped_mass=lumped_mass)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(problem, nc, alpha1, alpha2, lumped_mass, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_l1l2_qp(problem, nc, alpha1, alpha2,
                                                     lumped_mass=lumped_mass)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# Core per-problem runner
# ---------------------------------------------------------------------------

def run_one(result: dict, problem: str, nc: int, alpha1: float, alpha2: float,
            lumped_mass: bool,
            tol: float, time_limit: float, max_iter: int, solvers: set,
            cooldown: float = 0.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses."""
    n_disp = _n_display(nc)
    a2_str = f"{alpha2:.0e}" if alpha2 > 0 else "0"
    print(f"  {problem} nc={nc} (n={n_disp:.2e})  "
          f"alpha1={alpha1:.0e}  alpha2={a2_str}  "
          f"mass={'lumped' if lumped_mass else 'consistent'}", flush=True)

    worker_args = (problem, nc, alpha1, alpha2, lumped_mass)
    return run_solvers(result, worker_args, _worker_ssn, _worker_qpalm, _worker_osqp,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "problem", "nc", "n_display", "alpha1", "alpha2", "lumped_mass",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


# ---------------------------------------------------------------------------
# Table runners
# ---------------------------------------------------------------------------

def _run_vary_n(problem: str, nc_list: list[int], alpha1_list: list[float],
                alpha2: float, lumped_mass: bool, tol: float, time_limit: float,
                max_iter: int, result_dir: Path, solvers: set,
                cooldown: float = 0.0, name_prefix: str = "") -> list[dict]:
    label = f"{problem}_vary_n"
    csv_path = result_dir / f"{name_prefix}{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(alpha1_list)
    done = 0

    def _flush() -> None:
        _write_csv(csv_path, rows, CSV_FIELDS)

    for nc in nc_list:
        for alpha1 in alpha1_list:
            done += 1
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"problem": problem, "nc": nc, "n_display": _n_display(nc),
                   "alpha1": alpha1, "alpha2": alpha2, "lumped_mass": int(lumped_mass)}
            rows.append(row)
            run_one(row, problem, nc, alpha1, alpha2, lumped_mass, tol, time_limit, max_iter,
                    solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_vary_a2(problem: str, nc: int, alpha1: float, alpha2_list: list[float],
                 lumped_mass: bool, tol: float, time_limit: float, max_iter: int,
                 result_dir: Path, solvers: set, cooldown: float = 0.0,
                 name_prefix: str = "") -> list[dict]:
    label = f"{problem}_vary_a2"
    csv_path = result_dir / f"{name_prefix}{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(alpha2_list)

    def _flush() -> None:
        _write_csv(csv_path, rows, CSV_FIELDS)

    for done, alpha2 in enumerate(alpha2_list, 1):
        print(f"\n[{label}  {done}/{n_total}]")
        row = {"problem": problem, "nc": nc, "n_display": _n_display(nc),
               "alpha1": alpha1, "alpha2": alpha2, "lumped_mass": int(lumped_mass)}
        rows.append(row)
        run_one(row, problem, nc, alpha1, alpha2, lumped_mass, tol, time_limit, max_iter,
                solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",       default=str(HERE.parent))
    parser.add_argument("--tol",        type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--nc",         type=int,   nargs="+", default=None,
                        help="Grid exponents for vary-n tables (overrides defaults)")
    parser.add_argument("--table",      nargs="+",  default=ALL_TABLES,
                        choices=ALL_TABLES, metavar="TABLE",
                        help=f"Which tables to run (default: all). "
                             f"Choices: {' '.join(ALL_TABLES)}")
    parser.add_argument("--solver",     nargs="+",  default=["ssn-pmm", "qpalm", "osqp"],
                        choices=["ssn-pmm", "qpalm", "osqp"], metavar="SOLVER",
                        help="Solvers to run (default: all three). Choices: ssn-pmm qpalm osqp")
    parser.add_argument("--cooldown",   type=float, default=0.0,
                        help="Seconds to sleep between solver runs to prevent CPU throttling (default: 0)")
    parser.add_argument("--lumped-mass", type=int, default=0, choices=[0, 1], metavar="{0,1}",
                        help="0 to use the consistent mass matrix (default), "
                             "1 to use the lumped mass matrix.")
    parser.add_argument("--out",       default="",
                        help="Prefix for output filenames (e.g. '0508' -> '0508_poisson_vary_n.csv')")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000
    tables     = set(args.table)
    solvers    = set(args.solver)
    lumped_mass = bool(args.lumped_mass)
    name_prefix = f"{args.out}_" if args.out else ""

    # ---- poisson_vary_n ----------------------------------------------------
    if "poisson_vary_n" in tables:
        nc_list = args.nc or POISSON_VARY_N_NC
        _run_vary_n("poisson", nc_list, POISSON_VARY_N_ALPHA1,
                   POISSON_VARY_N_ALPHA2, lumped_mass, tol, time_limit, max_iter, result_dir,
                   solvers, cooldown, name_prefix)

    # ---- poisson_vary_a2 ---------------------------------------------------
    if "poisson_vary_a2" in tables:
        _run_vary_a2("poisson", POISSON_VARY_A2_NC, POISSON_VARY_A2_ALPHA1,
                    POISSON_VARY_A2_ALPHA2, lumped_mass, tol, time_limit, max_iter, result_dir,
                    solvers, cooldown, name_prefix)

    # ---- convdiff_vary_n ---------------------------------------------------
    if "convdiff_vary_n" in tables:
        nc_list = args.nc or CONVDIFF_VARY_N_NC
        _run_vary_n("convdiff", nc_list, CONVDIFF_VARY_N_ALPHA1,
                   CONVDIFF_VARY_N_ALPHA2, lumped_mass, tol, time_limit, max_iter, result_dir,
                   solvers, cooldown, name_prefix)

    # ---- convdiff_vary_a2 --------------------------------------------------
    if "convdiff_vary_a2" in tables:
        _run_vary_a2("convdiff", CONVDIFF_VARY_A2_NC, CONVDIFF_VARY_A2_ALPHA1,
                    CONVDIFF_VARY_A2_ALPHA2, lumped_mass, tol, time_limit, max_iter, result_dir,
                    solvers, cooldown, name_prefix)


if __name__ == "__main__":
    main()
