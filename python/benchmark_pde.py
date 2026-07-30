"""
Benchmark SSN-PMM vs QPALM vs OSQP on L1/L2-regularized PDE-constrained QP problems.

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

Step 1 - Build the SSN-PMM Python binding (if not already built)
-----------------------------------------------------------------
  cd python && mkdir -p build && cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

Step 2 - Run the benchmark
---------------------------
  # Full run (all four tables, all nc levels; may take hours for nc >= 9)
  python3 benchmark_pde.py

  # Quick smoke-test (nc = 6 only, 60-second time limit, two tables)
  python3 benchmark_pde.py --table convdiff_vary_n convdiff_vary_a2 --nc 6 --time-limit 60

  # Single table
  python3 benchmark_pde.py --table poisson_vary_n --nc 6 7 8 --time-limit 120

  # Prefix output filenames (e.g. '0508' -> '0508_poisson_vary_n.csv')
  python3 benchmark_pde.py --name 0508

Available table names: poisson_vary_n  poisson_vary_a2  convdiff_vary_n  convdiff_vary_a2

Settings: tol = 1e-6, time limit = 60 s by default.
"""

import sys
import os
import csv
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

POISSON_VARY_N_NC     = [6, 7, 8, 9, 10]
POISSON_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6, 0.0]
POISSON_VARY_N_ALPHA2 = 1e-2

POISSON_VARY_A2_NC     = 8
POISSON_VARY_A2_ALPHA1 = 1e-6
POISSON_VARY_A2_ALPHA2 = [1e-2, 1e-4, 1e-6, 0.0]

CONVDIFF_VARY_N_NC     = [6, 7, 8, 9, 10]
CONVDIFF_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6, 0.0]
CONVDIFF_VARY_N_ALPHA2 = 1e-2

CONVDIFF_VARY_A2_NC     = 9
CONVDIFF_VARY_A2_ALPHA1 = 1e-6
CONVDIFF_VARY_A2_ALPHA2 = [1e-2, 1e-4, 1e-6, 0.0]

ALL_TABLES = ["poisson_vary_n", "poisson_vary_a2", "convdiff_vary_n", "convdiff_vary_a2"]

QPALM_SOLVED = qpalm.Info.SOLVED
OSQP_SOLVED  = 1


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _n_display(nc: int) -> int:
    """n = 3 * np."""
    return 3*(2**nc+1)**2


def _fmt_n(n: int) -> str:
    exp = int(np.floor(np.log10(n)))
    coef = n / 10**exp
    return f"{coef:.2f}·10^{exp}"


def _fmt_alpha(a: float) -> str:
    if a == 0.0:
        return "0"
    exp = int(round(np.log10(a)))
    return f"10^{exp}"


def _fmt_ssn(r: dict) -> str:
    if r.get("ssn_status") != 0:
        return "FAIL"
    return f"{r['pmm_iter']}({r['ssn_iter']})[{r['pmm_tol_achieved']:.2e}]"


def _fmt_qpalm(r: dict) -> str:
    if r.get("qpalm_status") != QPALM_SOLVED:
        return "FAIL"
    return f"{r['qpalm_iter']}({r['qpalm_inner_iter']})"


def _fmt_osqp(r: dict) -> str:
    if r.get("osqp_status") != OSQP_SOLVED:
        return "FAIL"
    return str(r["osqp_iter"])


def _fmt_time(r: dict, key: str, solved_key: str, solved_val) -> str:
    if r.get(solved_key) != solved_val:
        return "—"
    return f"{r[key]:.2f}"


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _make_upper_triangular(Q):
    Q_coo = Q.tocoo()
    entries: dict[tuple[int, int], float] = {}
    for r, c, v in zip(Q_coo.row.tolist(), Q_coo.col.tolist(), Q_coo.data.tolist()):
        key = (min(r, c), max(r, c))
        entries.setdefault(key, v)
    if not entries:
        return sp.csc_matrix(Q.shape)
    rows, cols, vals = zip(*((k[0], k[1], v) for k, v in entries.items()))
    return sp.coo_matrix((vals, (rows, cols)), shape=Q.shape).tocsc()


def pdpmm_to_qpalm(pd: dict):
    n, ell = pd["n"], pd["l"]
    INF = 1e30

    Q = sp.csc_matrix(
        (pd["Q_data"], pd["Q_indices"], pd["Q_indptr"]), shape=pd["Q_shape"]
    )
    A = sp.csc_matrix(
        (pd["A_data"], pd["A_indices"], pd["A_indptr"]), shape=pd["A_shape"]
    )
    B = sp.csc_matrix(
        (pd["B_data"], pd["B_indices"], pd["B_indptr"]), shape=pd["B_shape"]
    )

    Q_upper = _make_upper_triangular(Q)
    q = np.asarray(pd["c"], dtype=np.float64)

    row_blocks  = [A]
    bmin_blocks = [np.asarray(pd["b"],  dtype=np.float64)]
    bmax_blocks = [np.asarray(pd["b"],  dtype=np.float64)]

    if ell > 0:
        row_blocks.append(B)
        bmin_blocks.append(np.asarray(pd["lw"], dtype=np.float64))
        bmax_blocks.append(np.asarray(pd["uw"], dtype=np.float64))

    row_blocks.append(sp.eye(n, format="csc"))
    bmin_blocks.append(np.asarray(pd["lx"], dtype=np.float64))
    bmax_blocks.append(np.asarray(pd["ux"], dtype=np.float64))

    C    = sp.vstack(row_blocks, format="csc")
    bmin = np.clip(np.concatenate(bmin_blocks), -INF, INF)
    bmax = np.clip(np.concatenate(bmax_blocks), -INF, INF)

    return Q_upper, q, C, bmin, bmax, n, int(C.shape[0])


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def run_qpalm(qpalm_data: tuple, tol: float, time_limit: float) -> dict:
    Q_upper, q, C, bmin, bmax, n, m_total = qpalm_data

    data      = qpalm.Data(n, m_total)
    data.Q    = Q_upper
    data.q    = q
    data.A    = C
    data.bmin = bmin
    data.bmax = bmax

    settings            = qpalm.Settings()
    settings.eps_abs    = tol
    settings.eps_rel    = tol
    settings.max_iter   = 2_000_000_000
    settings.time_limit = time_limit
    settings.verbose    = 0
    settings.scaling    = 10

    solver = qpalm.Solver(data, settings)
    solver.solve()

    info = solver.info
    return {
        "status":       int(info.status_val),
        "obj_val":      float(info.objective),
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter_out),
        "inner_iter":   int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }


def run_osqp(qpalm_data: tuple, tol: float, time_limit: float) -> dict:
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data

    prob = osqp.OSQP()
    prob.setup(
        Q_upper, q, C, bmin, bmax,
        eps_abs    = tol,
        eps_rel    = tol,
        max_iter   = 2_000_000_000,
        time_limit = time_limit,
        verbose    = False,
        scaling    = 10,
    )
    res = prob.solve()

    return {
        "status":       int(res.info.status_val),
        "obj_val":      float(res.info.obj_val),
        "run_time":     float(res.info.run_time),
        "outer_iter":   int(res.info.iter),
        "inner_iter":   0,
        "tol_achieved": max(float(res.info.prim_res), float(res.info.dual_res)),
    }


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _worker_ssn(problem, nc, alpha1, alpha2, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
        result["n_vars"] = pd_data["n"]
        result["res"] = ssn_pmm_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(problem, nc, alpha1, alpha2, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(problem, nc, alpha1, alpha2, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _run_isolated(target_func, args: tuple) -> dict:
    """Spawn a fresh process, run target_func(*args, conn), return the sent dict.

    If the worker dies without sending a result (segfault, OOM-kill by the
    OS, etc.), returns an {"error": ...} dict instead of raising, so a single
    crashed problem size doesn't abort the whole benchmark sweep.
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=target_func, args=(*args, child_conn))
    p.start()
    child_conn.close()
    try:
        out = parent_conn.recv()
    except EOFError:
        p.join()
        return {"error": f"worker died without result (exitcode={p.exitcode}, "
                          f"likely OOM-killed or crashed)"}
    p.join()
    return out


# ---------------------------------------------------------------------------
# Core per-problem runner
# ---------------------------------------------------------------------------

def run_one(result: dict, problem: str, nc: int, alpha1: float, alpha2: float,
            tol: float, time_limit: float, max_iter: int, solvers: set,
            cooldown: float = 0.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses.

    Mutates `result` in place and calls `flush_cb()` (if given) right after
    each solver finishes, so partial results are persisted immediately
    instead of waiting for the whole sweep to complete.
    """
    n_disp = _n_display(nc)
    a2_str = f"{alpha2:.0e}" if alpha2 > 0 else "0"
    print(f"  {problem} nc={nc} (n={n_disp:.2e})  "
          f"alpha1={alpha1:.0e}  alpha2={a2_str}", flush=True)

    # --- SSN-PMM ---
    if "ssn-pmm" in solvers:
        ssn_out = _run_isolated(_worker_ssn,
                                (problem, nc, alpha1, alpha2, tol, time_limit, max_iter))
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
        qpalm_out = _run_isolated(_worker_qpalm,
                                  (problem, nc, alpha1, alpha2, tol, time_limit))
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
        osqp_out = _run_isolated(_worker_osqp,
                                 (problem, nc, alpha1, alpha2, tol, time_limit))
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
    "problem", "nc", "n_display", "alpha1", "alpha2",
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

def _run_vary_n(problem: str, nc_list: list[int], alpha1_list: list[float],
                alpha2: float, tol: float, time_limit: float,
                max_iter: int, result_dir: Path, solvers: set,
                cooldown: float = 0.0, name_prefix: str = "") -> list[dict]:
    label = f"{problem}_vary_n"
    csv_path = result_dir / f"{name_prefix}{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(alpha1_list)
    done = 0

    def _flush() -> None:
        _write_csv(csv_path, rows)

    for nc in nc_list:
        for alpha1 in alpha1_list:
            done += 1
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"problem": problem, "nc": nc, "n_display": _n_display(nc),
                   "alpha1": alpha1, "alpha2": alpha2}
            rows.append(row)
            run_one(row, problem, nc, alpha1, alpha2, tol, time_limit, max_iter, solvers,
                    cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_vary_a2(problem: str, nc: int, alpha1: float, alpha2_list: list[float],
                 tol: float, time_limit: float, max_iter: int,
                 result_dir: Path, solvers: set, cooldown: float = 0.0,
                 name_prefix: str = "") -> list[dict]:
    label = f"{problem}_vary_a2"
    csv_path = result_dir / f"{name_prefix}{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(alpha2_list)

    def _flush() -> None:
        _write_csv(csv_path, rows)

    for done, alpha2 in enumerate(alpha2_list, 1):
        print(f"\n[{label}  {done}/{n_total}]")
        row = {"problem": problem, "nc": nc, "n_display": _n_display(nc),
               "alpha1": alpha1, "alpha2": alpha2}
        rows.append(row)
        run_one(row, problem, nc, alpha1, alpha2, tol, time_limit, max_iter, solvers,
                cooldown, flush_cb=_flush)
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
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--nc",         type=int,   nargs="+", default=None,
                        help="Grid exponents for vary-n tables (overrides defaults)")
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
    name_prefix = f"{args.name}_" if args.name else ""

    # ---- poisson_vary_n ----------------------------------------------------
    if "poisson_vary_n" in tables:
        nc_list = args.nc or POISSON_VARY_N_NC
        rows = _run_vary_n("poisson", nc_list, POISSON_VARY_N_ALPHA1,
                           POISSON_VARY_N_ALPHA2, tol, time_limit, max_iter, result_dir, solvers, cooldown,
                           name_prefix)

    # ---- poisson_vary_a2 ---------------------------------------------------
    if "poisson_vary_a2" in tables:
        rows = _run_vary_a2("poisson", POISSON_VARY_A2_NC, POISSON_VARY_A2_ALPHA1,
                            POISSON_VARY_A2_ALPHA2, tol, time_limit, max_iter, result_dir, solvers, cooldown,
                            name_prefix)

    # ---- convdiff_vary_n ---------------------------------------------------
    if "convdiff_vary_n" in tables:
        nc_list = args.nc or CONVDIFF_VARY_N_NC
        rows = _run_vary_n("convdiff", nc_list, CONVDIFF_VARY_N_ALPHA1,
                           CONVDIFF_VARY_N_ALPHA2, tol, time_limit, max_iter, result_dir, solvers, cooldown,
                           name_prefix)

    # ---- convdiff_vary_a2 --------------------------------------------------
    if "convdiff_vary_a2" in tables:
        rows = _run_vary_a2("convdiff", CONVDIFF_VARY_A2_NC, CONVDIFF_VARY_A2_ALPHA1,
                            CONVDIFF_VARY_A2_ALPHA2, tol, time_limit, max_iter, result_dir, solvers, cooldown,
                            name_prefix)


if __name__ == "__main__":
    main()
