"""
Benchmark SSN-PMM vs QPALM vs OSQP on PDE-constrained QP problems.

Four named tables are produced:

  poisson_vary_n    Poisson control: varying grid size n and L1 reg alpha1
                    (alpha2 = 1e-2 fixed)

  poisson_vary_a2   Poisson control: varying L2 reg alpha2
                    ((n, alpha1) = (1.32e5, 1e-4) fixed)

  convdiff_vary_n   Conv-diff control: varying grid size n and L1 reg alpha1
                    (alpha2 = 1e-2 fixed)

  convdiff_vary_a2  Conv-diff control: varying L2 reg alpha2
                    ((n, alpha1) = (1.32e5, 1e-4) fixed)

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

Available table names: poisson_vary_n  poisson_vary_a2  convdiff_vary_n  convdiff_vary_a2

Settings: tol = 1e-6, time limit = 600 s by default.
"""

import sys
import os
import csv
import time
import argparse
import threading
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

try:
    import psutil
    _HAVE_PSUTIL = True
except ModuleNotFoundError:
    _HAVE_PSUTIL = False
    print("Warning: psutil not found — RAM tracking disabled. Install with: pip install psutil")


class _PeakRSS:
    """Polls process RSS in a background thread to capture peak memory usage."""
    def __init__(self, interval: float = 0.05):
        self._enabled = _HAVE_PSUTIL
        if self._enabled:
            self._proc = psutil.Process()
            self._interval = interval
            self._peak = self._proc.memory_info().rss
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if self._enabled:
            self._thread.start()
        return self

    def __exit__(self, *_):
        if self._enabled:
            self._stop.set()
            self._thread.join()

    def _run(self):
        while not self._stop.wait(self._interval):
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss

    @property
    def peak_mb(self) -> float:
        if not self._enabled:
            return float("nan")
        return self._peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

POISSON_VARY_N_NC     = [6, 7, 8, 9, 10]
POISSON_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6]
POISSON_VARY_N_ALPHA2 = 1e-2

POISSON_VARY_A2_NC     = 7
POISSON_VARY_A2_ALPHA1 = 1e-4
POISSON_VARY_A2_ALPHA2 = [1e-1, 1e-2, 1e-4, 1e-6, 0.0]

CONVDIFF_VARY_N_NC     = [6, 7, 8, 9, 10]
CONVDIFF_VARY_N_ALPHA1 = [1e-2, 1e-4, 1e-6]
CONVDIFF_VARY_N_ALPHA2 = 1e-2

CONVDIFF_VARY_A2_NC     = 8
CONVDIFF_VARY_A2_ALPHA1 = 1e-4
CONVDIFF_VARY_A2_ALPHA2 = [1e-1, 1e-2, 1e-4, 1e-6]

ALL_TABLES = ["poisson_vary_n", "poisson_vary_a2", "convdiff_vary_n", "convdiff_vary_a2"]

QPALM_SOLVED = qpalm.Info.SOLVED
OSQP_SOLVED  = 1


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _n_display(nc: int) -> int:
    """Paper's 'n' = 2*(2^nc+1)^2 (state + control, before u+/u- split)."""
    n1d = (1 << nc) + 1
    return 2 * n1d * n1d


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
    return f"{r['pmm_iter']}({r['ssn_iter']}){{{r['krylov_iter']}}}"


def _fmt_qpalm(r: dict) -> str:
    if r.get("qpalm_status") != QPALM_SOLVED:
        return "FAIL"
    return str(r["qpalm_iter"])


def _fmt_osqp(r: dict) -> str:
    if r.get("osqp_status") != OSQP_SOLVED:
        return "FAIL"
    return str(r["osqp_iter"])


def _fmt_time(r: dict, key: str, solved_key: str, solved_val) -> str:
    if r.get(solved_key) != solved_val:
        return "—"
    return f"{r[key]:.2f}"


def _fmt_ram(r: dict, key: str) -> str:
    v = r.get(key, float("nan"))
    if v != v:  # nan → psutil unavailable
        return "n/a"
    return f"{v:.0f}"


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

def run_qpalm(pd: dict, tol: float, time_limit: float) -> dict:
    Q_upper, q, C, bmin, bmax, n, m_total = pdpmm_to_qpalm(pd)

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

    t0 = time.perf_counter()
    solver = qpalm.Solver(data, settings)
    solver.solve()
    t1 = time.perf_counter()

    info = solver.info
    return {
        "status":       int(info.status_val),
        "obj_val":      float(info.objective),
        "solving_time": t1 - t0,
        "iter":         int(info.iter),
    }


def run_osqp(pd: dict, tol: float, time_limit: float) -> dict:
    Q_upper, q, C, bmin, bmax, *_ = pdpmm_to_qpalm(pd)

    prob = osqp.OSQP()
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()

    return {
        "status":       int(res.info.status_val),
        "obj_val":      float(res.info.obj_val),
        "solving_time": t1 - t0,
        "iter":         int(res.info.iter),
    }


# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn(problem, nc, alpha1, alpha2, tol, time_limit, max_iter, conn):
    result = {}
    with _PeakRSS() as mem:
        try:
            pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
            result["n_vars"] = pd_data["n"]
            t0 = time.perf_counter()
            r  = ssn_pmm_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
            t1 = time.perf_counter()
            r["solving_time"] = t1 - t0
            result["res"] = r
        except Exception as e:
            result["error"] = str(e)
        result["ram"] = mem.peak_mb
    conn.send(result)
    conn.close()


def _worker_qpalm(problem, nc, alpha1, alpha2, tol, time_limit, conn):
    result = {}
    with _PeakRSS() as mem:
        try:
            pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
            result["res"] = run_qpalm(pd_data, tol, time_limit)
        except Exception as e:
            result["error"] = str(e)
        result["ram"] = mem.peak_mb
    conn.send(result)
    conn.close()


def _worker_osqp(problem, nc, alpha1, alpha2, tol, time_limit, conn):
    result = {}
    with _PeakRSS() as mem:
        try:
            pd_data = ssn_pmm_bind.generate_pde_problem_params(problem, nc, alpha1, alpha2)
            result["res"] = run_osqp(pd_data, tol, time_limit)
        except Exception as e:
            result["error"] = str(e)
        result["ram"] = mem.peak_mb
    conn.send(result)
    conn.close()


def _run_isolated(target_func, args: tuple) -> dict:
    """Spawn a fresh process, run target_func(*args, conn), return the sent dict."""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=target_func, args=(*args, child_conn))
    p.start()
    child_conn.close()
    out = parent_conn.recv()
    p.join()
    return out


# ---------------------------------------------------------------------------
# Core per-problem runner
# ---------------------------------------------------------------------------

def run_one(problem: str, nc: int, alpha1: float, alpha2: float,
            tol: float, time_limit: float, max_iter: int) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses."""
    n_disp = _n_display(nc)
    a2_str = f"{alpha2:.0e}" if alpha2 > 0 else "0"
    print(f"  {problem} nc={nc} (n={n_disp:.2e})  "
          f"alpha1={alpha1:.0e}  alpha2={a2_str}", flush=True)

    result = {"problem": problem, "nc": nc, "n_display": n_disp,
              "alpha1": alpha1, "alpha2": alpha2}

    # --- SSN-PMM ---
    ssn_out = _run_isolated(_worker_ssn,
                            (problem, nc, alpha1, alpha2, tol, time_limit, max_iter))
    if "n_vars" in ssn_out:
        print(f"  n_vars={ssn_out['n_vars']}")
    if "error" in ssn_out:
        print(f"    SSN-PMM  ERROR — {ssn_out['error']}")
        result.update(ssn_status=-99, ssn_solved=0, ssn_time=float("inf"),
                      pmm_iter=-1, ssn_iter=-1, krylov_iter=-1, ssn_obj=float("nan"),
                      ssn_peak_ram_mb=ssn_out.get("ram", float("nan")))
    else:
        r = ssn_out["res"]
        result.update(
            ssn_status=r["status"], ssn_solved=int(r["status"] == 0),
            ssn_time=r["solving_time"],
            pmm_iter=r["pmm_iter"], ssn_iter=r["ssn_iter"], krylov_iter=r["krylov_iter"],
            ssn_obj=r["obj_val"], ssn_peak_ram_mb=ssn_out["ram"],
        )
        ok = "OK" if r["status"] == 0 else f"status={r['status']}"
        print(f"    SSN-PMM  {ok:8s}  t={r['solving_time']:.2f}s  "
              f"{r['pmm_iter']}({r['ssn_iter']}){{{r['krylov_iter']}}}  "
              f"RAM={ssn_out['ram']:.0f}MB")

    # --- QPALM ---
    qpalm_out = _run_isolated(_worker_qpalm,
                              (problem, nc, alpha1, alpha2, tol, time_limit))
    if "error" in qpalm_out:
        print(f"    QPALM    ERROR — {qpalm_out['error']}")
        result.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=float("inf"),
                      qpalm_iter=-1, qpalm_obj=float("nan"),
                      qpalm_peak_ram_mb=qpalm_out.get("ram", float("nan")))
    else:
        r = qpalm_out["res"]
        result.update(
            qpalm_status=r["status"], qpalm_solved=int(r["status"] == QPALM_SOLVED),
            qpalm_time=r["solving_time"], qpalm_iter=r["iter"], qpalm_obj=r["obj_val"],
            qpalm_peak_ram_mb=qpalm_out["ram"],
        )
        ok = "OK" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
        print(f"    QPALM    {ok:8s}  t={r['solving_time']:.2f}s  iter={r['iter']}  "
              f"RAM={qpalm_out['ram']:.0f}MB")

    # --- OSQP ---
    osqp_out = _run_isolated(_worker_osqp,
                             (problem, nc, alpha1, alpha2, tol, time_limit))
    if "error" in osqp_out:
        print(f"    OSQP     ERROR — {osqp_out['error']}")
        result.update(osqp_status=-99, osqp_solved=0, osqp_time=float("inf"),
                      osqp_iter=-1, osqp_obj=float("nan"),
                      osqp_peak_ram_mb=osqp_out.get("ram", float("nan")))
    else:
        r = osqp_out["res"]
        result.update(
            osqp_status=r["status"], osqp_solved=int(r["status"] == OSQP_SOLVED),
            osqp_time=r["solving_time"], osqp_iter=r["iter"], osqp_obj=r["obj_val"],
            osqp_peak_ram_mb=osqp_out["ram"],
        )
        ok = "OK" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
        print(f"    OSQP     {ok:8s}  t={r['solving_time']:.2f}s  iter={r['iter']}  "
              f"RAM={osqp_out['ram']:.0f}MB")

    return result


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

_W_ID   = 12
_W_A1   =  8
_W_ITER = 20
_W_T    = 10
_W_RAM  =  9

_HDR_VARY_N = (
    f"{'n':>{_W_ID}}  {'α₁':<{_W_A1}}  "
    f"{'PMM(SSN){Krylov}':<{_W_ITER}}  "
    f"{'QPALM iter':<{_W_ITER}}  "
    f"{'OSQP iter':<{_W_ITER}}  "
    f"{'PMM(s)':>{_W_T}}  {'QPALM(s)':>{_W_T}}  {'OSQP(s)':>{_W_T}}  "
    f"{'PMM MB':>{_W_RAM}}  {'QPALM MB':>{_W_RAM}}  {'OSQP MB':>{_W_RAM}}"
)

_HDR_VARY_A2 = (
    f"{'α₂':<{_W_A1}}  "
    f"{'PMM(SSN){Krylov}':<{_W_ITER}}  "
    f"{'QPALM iter':<{_W_ITER}}  "
    f"{'OSQP iter':<{_W_ITER}}  "
    f"{'PMM(s)':>{_W_T}}  {'QPALM(s)':>{_W_T}}  {'OSQP(s)':>{_W_T}}  "
    f"{'PMM MB':>{_W_RAM}}  {'QPALM MB':>{_W_RAM}}  {'OSQP MB':>{_W_RAM}}"
)


def _data_cols(r: dict) -> str:
    return (
        f"{_fmt_ssn(r):<{_W_ITER}}  "
        f"{_fmt_qpalm(r):<{_W_ITER}}  "
        f"{_fmt_osqp(r):<{_W_ITER}}  "
        f"{_fmt_time(r,'ssn_time','ssn_status',0):>{_W_T}}  "
        f"{_fmt_time(r,'qpalm_time','qpalm_status',QPALM_SOLVED):>{_W_T}}  "
        f"{_fmt_time(r,'osqp_time','osqp_status',OSQP_SOLVED):>{_W_T}}  "
        f"{_fmt_ram(r,'ssn_peak_ram_mb'):>{_W_RAM}}  "
        f"{_fmt_ram(r,'qpalm_peak_ram_mb'):>{_W_RAM}}  "
        f"{_fmt_ram(r,'osqp_peak_ram_mb'):>{_W_RAM}}"
    )


def _print_vary_n_table(rows: list[dict], title: str) -> None:
    sep = "-" * len(_HDR_VARY_N)
    print(f"\n{title}")
    print(sep)
    print(_HDR_VARY_N)
    print(sep)
    prev_n = None
    for r in rows:
        n_str = _fmt_n(r["n_display"]) if r["n_display"] != prev_n else ""
        prev_n = r["n_display"]
        print(f"{n_str:>{_W_ID}}  {_fmt_alpha(r['alpha1']):<{_W_A1}}  {_data_cols(r)}")
    print(sep)


def _print_vary_a2_table(rows: list[dict], title: str) -> None:
    sep = "-" * len(_HDR_VARY_A2)
    print(f"\n{title}")
    print(sep)
    print(_HDR_VARY_A2)
    print(sep)
    for r in rows:
        print(f"{_fmt_alpha(r['alpha2']):<{_W_A1}}  {_data_cols(r)}")
    print(sep)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "problem", "nc", "n_display", "alpha1", "alpha2",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter", "krylov_iter", "ssn_time", "ssn_obj", "ssn_peak_ram_mb",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_time", "qpalm_obj", "qpalm_peak_ram_mb",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_time",  "osqp_obj",  "osqp_peak_ram_mb",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Table runners
# ---------------------------------------------------------------------------

def _run_vary_n(problem: str, nc_list: list[int], alpha1_list: list[float],
                alpha2: float, tol: float, time_limit: float,
                max_iter: int, result_dir: Path) -> list[dict]:
    rows = []
    n_total = len(nc_list) * len(alpha1_list)
    done = 0
    label = f"{problem}_vary_n"
    for nc in nc_list:
        for alpha1 in alpha1_list:
            done += 1
            print(f"\n[{label}  {done}/{n_total}]")
            rows.append(run_one(problem, nc, alpha1, alpha2, tol, time_limit, max_iter))
    _write_csv(result_dir / f"{label}.csv", rows)
    return rows


def _run_vary_a2(problem: str, nc: int, alpha1: float, alpha2_list: list[float],
                 tol: float, time_limit: float, max_iter: int,
                 result_dir: Path) -> list[dict]:
    rows = []
    n_total = len(alpha2_list)
    label = f"{problem}_vary_a2"
    for done, alpha2 in enumerate(alpha2_list, 1):
        print(f"\n[{label}  {done}/{n_total}]")
        rows.append(run_one(problem, nc, alpha1, alpha2, tol, time_limit, max_iter))
    _write_csv(result_dir / f"{label}.csv", rows)
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
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol        = args.tol
    time_limit = args.time_limit
    max_iter   = 10_000_000_000
    tables     = set(args.table)

    # ---- poisson_vary_n ----------------------------------------------------
    if "poisson_vary_n" in tables:
        nc_list = args.nc or POISSON_VARY_N_NC
        rows = _run_vary_n("poisson", nc_list, POISSON_VARY_N_ALPHA1,
                           POISSON_VARY_N_ALPHA2, tol, time_limit, max_iter, result_dir)
        _print_vary_n_table(rows,
            "Poisson control: varying n and α₁  (α₂ = 1e-2)")

    # ---- poisson_vary_a2 ---------------------------------------------------
    if "poisson_vary_a2" in tables:
        rows = _run_vary_a2("poisson", POISSON_VARY_A2_NC, POISSON_VARY_A2_ALPHA1,
                            POISSON_VARY_A2_ALPHA2, tol, time_limit, max_iter, result_dir)
        n_str = _fmt_n(_n_display(POISSON_VARY_A2_NC))
        _print_vary_a2_table(rows,
            f"Poisson control: varying α₂  (n={n_str}, α₁=1e-4)")

    # ---- convdiff_vary_n ---------------------------------------------------
    if "convdiff_vary_n" in tables:
        nc_list = args.nc or CONVDIFF_VARY_N_NC
        rows = _run_vary_n("convdiff", nc_list, CONVDIFF_VARY_N_ALPHA1,
                           CONVDIFF_VARY_N_ALPHA2, tol, time_limit, max_iter, result_dir)
        _print_vary_n_table(rows,
            "Conv-diff control: varying n and α₁  (α₂ = 1e-2)")

    # ---- convdiff_vary_a2 --------------------------------------------------
    if "convdiff_vary_a2" in tables:
        rows = _run_vary_a2("convdiff", CONVDIFF_VARY_A2_NC, CONVDIFF_VARY_A2_ALPHA1,
                            CONVDIFF_VARY_A2_ALPHA2, tol, time_limit, max_iter, result_dir)
        n_str = _fmt_n(_n_display(CONVDIFF_VARY_A2_NC))
        _print_vary_a2_table(rows,
            f"Conv-diff control: varying α₂  (n={n_str}, α₁=1e-4)")


if __name__ == "__main__":
    main()
