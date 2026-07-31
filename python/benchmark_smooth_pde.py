"""
Benchmark SSN-PMM vs QPALM vs OSQP on L2-regularized PDE-constrained QP problems.

Five named tables are produced:

  poisson_control     2D Poisson control, control-constrained
                      (u >= 0, u <= per-beta upper bound)

  poisson_state       2D Poisson control, state-constrained
                      (-0.1 <= y <= per-beta upper bound)

  convdiff_both       2D convection-diffusion control,
                      both state and control bounds

  poisson3d_control   3D Poisson control, control-constrained

  helmholtz           2D Helmholtz control (L = -Delta - k^2), state-constrained,
                      same target/boundary condition as poisson_state,
                      swept over k in {20, 50} and per-(k,beta) state bounds

Outputs
-------
  results/smooth_poisson_control.csv
  results/smooth_poisson_state.csv
  results/smooth_convdiff_both.csv
  results/smooth_poisson3d_control.csv
  results/smooth_helmholtz.csv

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Build the SSN-PMM Python binding (if not already built)
-----------------------------------------------------------------
  cd python && mkdir -p build && cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

Step 2 - Run the benchmark
---------------------------
  # Full run (all four tables)
  python3 benchmark_smooth_pde.py

  # Quick smoke-test (small nc, short time limit)
  python3 benchmark_smooth_pde.py --table poisson_control --nc 2 3 --time-limit 30

  # Single table
  python3 benchmark_smooth_pde.py --table poisson_state --nc 2 3 4 --time-limit 120

  # Prefix output filenames (e.g. '0727' -> '0727_smooth_poisson_control.csv')
  python3 benchmark_smooth_pde.py --name 0727

Available table names: poisson_control  poisson_state  convdiff_both  poisson3d_control  helmholtz

Settings: tol = 1e-9, time limit = 10 min by default.
"""

import sys
import os
import csv
import math
import time
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

try:
    import qpalm
except ModuleNotFoundError:
    sys.exit("Cannot find qpalm. Install it with: pip install qpalm")

try:
    import osqp
except ModuleNotFoundError:
    sys.exit("Cannot find osqp. Install it with: pip install osqp")

# Reuse the solver-wrapper / conversion / crash-tolerant-subprocess helpers
# already written for the repo's other PDE benchmark, rather than
# duplicating them a third time (they're already duplicated once between
# benchmark_mm.py and benchmark_pde.py).
from benchmark_pde import (
    pdpmm_to_qpalm,
    run_qpalm,
    run_osqp,
    _run_isolated,
    QPALM_SOLVED,
    OSQP_SOLVED,
)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DEFAULT_EPS = 0.01
DEFAULT_WX  = -1.0 / math.sqrt(2.0)
DEFAULT_WY  =  1.0 / math.sqrt(2.0)

INF = math.inf

# 2D Poisson control, control-constrained (u >= 0, u <= u_upper(beta))
TABLE1_NC    = [7, 8, 9, 10]
TABLE1_BETAS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
TABLE1_U_UPPER = {
    1.0: 0.01, 1e-1: 0.1, 1e-2: 1.0, 1e-3: 3.0, 1e-4: 20.0, 1e-5: 100.0, 1e-6: 300.0,
}

# 2D Poisson control, state-constrained (-0.1 <= y <= y_upper(beta))
TABLE2_NC    = [7, 8, 9, 10]
TABLE2_BETAS = [1.0, 1e-2, 1e-4, 1e-6]
TABLE2_Y_UPPER = {1.0: 0.002, 1e-2: 0.175, 1e-4: 0.9, 1e-6: 1.0}

# 2D convection-diffusion control, both state and control bounds
TABLE3_NC    = [7, 8, 9, 10]
TABLE3_BETAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
TABLE3_Y_UPPER = {1e-1: 0.2, 1e-2: 0.5, 1e-3: 0.5, 1e-4: 0.75, 1e-5: 0.75}
TABLE3_U_BOUND = {1e-1: 0.75, 1e-2: 2.0, 1e-3: 3.0, 1e-4: 5.0, 1e-5: 6.0}

# 3D Poisson control, control-constrained
TABLE4_NC      = [4, 5]
TABLE4_BETAS   = TABLE1_BETAS
TABLE4_U_UPPER = TABLE1_U_UPPER

# 2D Helmholtz control (L = -Delta - k^2), state-constrained, same
# target/boundary as poisson_state (y = sin(pi x1) sin(pi x2) on boundary).
# (k, beta, y_upper); y_lower = -y_upper (symmetric state bounds), u unconstrained.
TABLE5_NC = [7, 8, 9, 10]
TABLE5_CONFIGS = [
    (20.0, 1e-2, 0.0005),
    (20.0, 1e-4, 0.05),
    (20.0, 1e-6, 0.6),
    (50.0, 1e-2, 1e-5),
    (50.0, 1e-4, 0.001),
]

ALL_TABLES = ["poisson_control", "poisson_state", "convdiff_both", "poisson3d_control", "helmholtz"]


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _n_display(nc: int, dim: int) -> int:
    """n = 2 * np (x = [y; u]), np = (2^nc+1)^dim."""
    n1d = 2 ** nc + 1
    return 2 * (n1d ** dim)


def _fmt_bound(v: float) -> str:
    if v == INF:
        return "+inf"
    if v == -INF:
        return "-inf"
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param):
    return ssn_pmm_bind.generate_pde_control_qp(
        choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param
    )


def _worker_ssn(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param,
                 tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param)
        result["n_vars"] = pd_data["n"]
        result["res"] = ssn_pmm_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param,
                   tol, time_limit, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param,
                  tol, time_limit, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# Core per-problem runner
# ---------------------------------------------------------------------------

def run_one(result: dict, choice: str, dim: int, nc: int, beta: float,
            y_lower: float, y_upper: float, u_lower: float, u_upper: float,
            eps: float, wx: float, wy: float,
            tol: float, time_limit: float, max_iter: int, solvers: set,
            cooldown: float = 0.0, k_param: float = 20.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses.

    Mutates `result` in place and calls `flush_cb()` (if given) right after
    each solver finishes, so partial results are persisted immediately.
    """
    n_disp = _n_display(nc, dim)
    k_str = f"  k={k_param:g}" if choice == "helmholtz" else ""
    print(f"  {choice} nc={nc} (n={n_disp:.2e})  alpha2={beta:.0e}{k_str}  "
          f"y=[{_fmt_bound(y_lower)},{_fmt_bound(y_upper)}]  "
          f"u=[{_fmt_bound(u_lower)},{_fmt_bound(u_upper)}]", flush=True)

    common_args = (choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, wx, wy, k_param)

    # --- SSN-PMM ---
    if "ssn-pmm" in solvers:
        ssn_out = _run_isolated(_worker_ssn, (*common_args, tol, time_limit, max_iter))
        if "error" in ssn_out:
            print(f"    SSN-PMM  ERROR — {ssn_out['error']}")
            result.update(ssn_status=-99, ssn_solved=0, ssn_time=float("inf"),
                          pmm_iter=-1, ssn_iter=-1, pmm_tol_achieved=float("nan"),
                          ssn_obj=float("nan"))
        else:
            r = ssn_out["res"]
            result.update(
                ssn_status=r["status"], ssn_solved=int(r["status"] == 0),
                ssn_time=r["run_time"],
                pmm_iter=r["pmm_iter"], ssn_iter=r["ssn_iter"],
                pmm_tol_achieved=r["pmm_tol_achieved"],
                ssn_obj=r["obj_val"],
            )
            ok = "OK" if r["status"] == 0 else f"status={r['status']}"
            print(f"    SSN-PMM  {ok:8s}  t={r['run_time']:.2f}s  "
                  f"{r['pmm_iter']}({r['ssn_iter']})[tol={r['pmm_tol_achieved']:.2e}]")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    # --- QPALM ---
    if "qpalm" in solvers:
        qpalm_out = _run_isolated(_worker_qpalm, (*common_args, tol, time_limit))
        if "error" in qpalm_out:
            print(f"    QPALM    ERROR — {qpalm_out['error']}")
            result.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=float("inf"),
                          qpalm_iter=-1, qpalm_inner_iter=-1, qpalm_obj=float("nan"),
                          qpalm_tol_achieved=float("nan"))
        else:
            r = qpalm_out["res"]
            result.update(
                qpalm_status=r["status"], qpalm_solved=int(r["status"] == QPALM_SOLVED),
                qpalm_time=r["run_time"], qpalm_iter=r["outer_iter"],
                qpalm_inner_iter=r["inner_iter"], qpalm_obj=r["obj_val"],
                qpalm_tol_achieved=r["tol_achieved"],
            )
            ok = "OK" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
            print(f"    QPALM    {ok:8s}  t={r['run_time']:.2f}s  "
                  f"{r['outer_iter']}({r['inner_iter']})[tol={r['tol_achieved']:.2e}]")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    # --- OSQP ---
    if "osqp" in solvers:
        osqp_out = _run_isolated(_worker_osqp, (*common_args, tol, time_limit))
        if "error" in osqp_out:
            print(f"    OSQP     ERROR — {osqp_out['error']}")
            result.update(osqp_status=-99, osqp_solved=0, osqp_time=float("inf"),
                          osqp_iter=-1, osqp_obj=float("nan"),
                          osqp_tol_achieved=float("nan"))
        else:
            r = osqp_out["res"]
            result.update(
                osqp_status=r["status"], osqp_solved=int(r["status"] == OSQP_SOLVED),
                osqp_time=r["run_time"], osqp_iter=r["outer_iter"], osqp_obj=r["obj_val"],
                osqp_tol_achieved=r["tol_achieved"],
            )
            ok = "OK" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
            print(f"    OSQP     {ok:8s}  t={r['run_time']:.2f}s  iter={r['outer_iter']}  "
                  f"tol={r['tol_achieved']:.2e}")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    return result


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "table", "choice", "nc", "n_display", "alpha2", "k_param",
    "y_lower", "y_upper", "u_lower", "u_upper",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole CSV from `rows`. Called after every solver finishes."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_rows(path: Path) -> list[dict]:
    """Load previously written rows so a rerun appends instead of overwriting."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Table runners
# ---------------------------------------------------------------------------

def _run_poisson_control(nc_list, betas, tol, time_limit, max_iter, result_dir,
                         solvers, cooldown, name_prefix):
    label = "poisson_control"
    csv_path = result_dir / f"{name_prefix}smooth_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for beta in betas:
            done += 1
            u_upper = TABLE1_U_UPPER[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "poisson", "nc": nc,
                   "n_display": _n_display(nc, 2), "alpha2": beta,
                   "y_lower": -INF, "y_upper": INF, "u_lower": 0.0, "u_upper": u_upper}
            rows.append(row)
            run_one(row, "poisson", 2, nc, beta, -INF, INF, 0.0, u_upper,
                    DEFAULT_EPS, DEFAULT_WX, DEFAULT_WY,
                    tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_poisson_state(nc_list, betas, tol, time_limit, max_iter, result_dir,
                       solvers, cooldown, name_prefix):
    label = "poisson_state"
    csv_path = result_dir / f"{name_prefix}smooth_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for beta in betas:
            done += 1
            y_upper = TABLE2_Y_UPPER[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "poisson_state", "nc": nc,
                   "n_display": _n_display(nc, 2), "alpha2": beta,
                   "y_lower": -0.1, "y_upper": y_upper, "u_lower": -INF, "u_upper": INF}
            rows.append(row)
            run_one(row, "poisson_state", 2, nc, beta, -0.1, y_upper, -INF, INF,
                    DEFAULT_EPS, DEFAULT_WX, DEFAULT_WY,
                    tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_convdiff_both(nc_list, betas, tol, time_limit, max_iter, result_dir,
                       solvers, cooldown, name_prefix):
    label = "convdiff_both"
    csv_path = result_dir / f"{name_prefix}smooth_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for beta in betas:
            done += 1
            y_upper = TABLE3_Y_UPPER[beta]
            u_bound = TABLE3_U_BOUND[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "convdiff", "nc": nc,
                   "n_display": _n_display(nc, 2), "alpha2": beta,
                   "y_lower": 0.0, "y_upper": y_upper,
                   "u_lower": -u_bound, "u_upper": u_bound}
            rows.append(row)
            run_one(row, "convdiff", 2, nc, beta, 0.0, y_upper, -u_bound, u_bound,
                    DEFAULT_EPS, DEFAULT_WX, DEFAULT_WY,
                    tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_poisson3d_control(nc_list, betas, tol, time_limit, max_iter, result_dir,
                           solvers, cooldown, name_prefix):
    label = "poisson3d_control"
    csv_path = result_dir / f"{name_prefix}smooth_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for beta in betas:
            done += 1
            u_upper = TABLE4_U_UPPER[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "poisson3d", "nc": nc,
                   "n_display": _n_display(nc, 3), "alpha2": beta,
                   "y_lower": -INF, "y_upper": INF, "u_lower": 0.0, "u_upper": u_upper}
            rows.append(row)
            run_one(row, "poisson3d", 3, nc, beta, -INF, INF, 0.0, u_upper,
                    DEFAULT_EPS, DEFAULT_WX, DEFAULT_WY,
                    tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_helmholtz(nc_list, configs, tol, time_limit, max_iter, result_dir,
                   solvers, cooldown, name_prefix):
    label = "helmholtz"
    csv_path = result_dir / f"{name_prefix}smooth_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(configs)
    done = 0

    def _flush():
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for k_param, beta, y_upper in configs:
            done += 1
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "helmholtz", "nc": nc,
                   "n_display": _n_display(nc, 2), "alpha2": beta, "k_param": k_param,
                   "y_lower": -y_upper, "y_upper": y_upper, "u_lower": -INF, "u_upper": INF}
            rows.append(row)
            run_one(row, "helmholtz", 2, nc, beta, -y_upper, y_upper, -INF, INF,
                    DEFAULT_EPS, DEFAULT_WX, DEFAULT_WY,
                    tol, time_limit, max_iter, solvers, cooldown,
                    k_param=k_param, flush_cb=_flush)
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
    parser.add_argument("--tol",        type=float, default=1e-9)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--nc",         type=int,   nargs="+", default=None,
                        help="Grid exponents (overrides each table's default nc list)")
    parser.add_argument("--table",      nargs="+",  default=ALL_TABLES,
                        choices=ALL_TABLES, metavar="TABLE",
                        help=f"Which tables to run (default: all). "
                             f"Choices: {' '.join(ALL_TABLES)}")
    parser.add_argument("--solver",     nargs="+",  default=["ssn-pmm", "qpalm", "osqp"],
                        choices=["ssn-pmm", "qpalm", "osqp"], metavar="SOLVER",
                        help="Solvers to run (default: all three). Choices: ssn-pmm qpalm osqp")
    parser.add_argument("--cooldown",   type=float, default=3.0,
                        help="Seconds to sleep between solver runs to prevent CPU throttling (default: 3)")
    parser.add_argument("--name",       default="",
                        help="Prefix for output filenames (e.g. '0727' -> '0727_smooth_poisson_control.csv')")
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
    name_prefix = f"{args.name}_" if args.name else ""

    if "poisson_control" in tables:
        nc_list = args.nc or TABLE1_NC
        _run_poisson_control(nc_list, TABLE1_BETAS, tol, time_limit, max_iter,
                             result_dir, solvers, cooldown, name_prefix)

    if "poisson_state" in tables:
        nc_list = args.nc or TABLE2_NC
        _run_poisson_state(nc_list, TABLE2_BETAS, tol, time_limit, max_iter,
                           result_dir, solvers, cooldown, name_prefix)

    if "convdiff_both" in tables:
        nc_list = args.nc or TABLE3_NC
        _run_convdiff_both(nc_list, TABLE3_BETAS, tol, time_limit, max_iter,
                           result_dir, solvers, cooldown, name_prefix)

    if "poisson3d_control" in tables:
        nc_list = args.nc or TABLE4_NC
        _run_poisson3d_control(nc_list, TABLE4_BETAS, tol, time_limit, max_iter,
                              result_dir, solvers, cooldown, name_prefix)

    if "helmholtz" in tables:
        nc_list = args.nc or TABLE5_NC
        _run_helmholtz(nc_list, TABLE5_CONFIGS, tol, time_limit, max_iter,
                       result_dir, solvers, cooldown, name_prefix)


if __name__ == "__main__":
    main()
