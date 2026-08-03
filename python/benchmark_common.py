"""
Shared helpers for benchmark_*.py scripts (benchmark_mm.py, benchmark_pde.py, benchmark_smooth_pde.py):

  - QPALM / OSQP imports (with install hints) and status constants
  - PDPMMdata -> QPALM/OSQP conversion and solver wrappers
  - crash-tolerant subprocess isolation
  - the per-problem three-solver runner shared by the PDE benchmarks
  - CSV read/write helpers
"""

import sys
import csv
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _import_or_exit(module_name: str, hint: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        sys.exit(hint)


qpalm = _import_or_exit("qpalm", "Cannot find qpalm. Install it with: pip install qpalm")
osqp  = _import_or_exit("osqp",  "Cannot find osqp. Install it with: pip install osqp")

QPALM_SOLVED = qpalm.Info.SOLVED   # == 1
OSQP_SOLVED  = 1                   # osqp.constant("OSQP_SOLVED")


# ---------------------------------------------------------------------------
# Convert PDPMMdata dict (from ssn_pmm_bind) to QPALM/OSQP inputs.
#
# PDPMMdata form:  min ½ xᵀQx + cᵀx   s.t.  Ax = b,  lw ≤ Bx ≤ uw,  lx ≤ x ≤ ux
# QPALM/OSQP form: min ½ xᵀQx + qᵀx   s.t.  bmin ≤ Cx ≤ bmax
#
# Stacking:  C = [A; B; Iₙ],  bmin/bmax accordingly.
# ---------------------------------------------------------------------------

def _make_upper_triangular(Q):
    """Return upper-triangular part of a symmetric Q.

    Works regardless of whether Q is stored fully, as lower-triangular only,
    or as upper-triangular only. Uses COO deduplication to avoid double-counting.
    """
    Q_coo = Q.tocoo()
    entries: dict[tuple[int, int], float] = {}
    for r, c, v in zip(Q_coo.row.tolist(), Q_coo.col.tolist(), Q_coo.data.tolist()):
        key = (min(r, c), max(r, c))
        entries.setdefault(key, v)   # first occurrence wins
    if not entries:
        return sp.csc_matrix(Q.shape)
    rows, cols, vals = zip(*((k[0], k[1], v) for k, v in entries.items()))
    return sp.coo_matrix((vals, (rows, cols)), shape=Q.shape).tocsc()


def pdpmm_to_qpalm(pd: dict):
    """Build (Q_upper, q, C, bmin, bmax, n, m_total) from a PDPMMdata dict."""
    n, ell = pd["n"], pd["l"]
    INF = 1e30   # QPALM/OSQP treat values beyond this as infinite

    Q = sp.csc_matrix((pd["Q_data"], pd["Q_indices"], pd["Q_indptr"]), shape=pd["Q_shape"])
    A = sp.csc_matrix((pd["A_data"], pd["A_indices"], pd["A_indptr"]), shape=pd["A_shape"])
    B = sp.csc_matrix((pd["B_data"], pd["B_indices"], pd["B_indptr"]), shape=pd["B_shape"])

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

def run_qpalm(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0) -> dict:
    """Run QPALM on a problem already converted via pdpmm_to_qpalm."""
    Q_upper, q, C, bmin, bmax, n, m_total = qpalm_data

    data      = qpalm.Data(n, m_total)
    data.Q    = Q_upper
    data.q    = q
    data.A    = C
    data.bmin = bmin
    data.bmax = bmax

    settings              = qpalm.Settings()
    settings.eps_abs      = tol
    settings.eps_rel      = tol
    settings.max_iter     = 2_000_000_000   # effectively infinite
    settings.time_limit   = time_limit
    settings.verbose      = 0               # silent
    settings.scaling      = 10              # default Ruiz scaling passes

    solver = qpalm.Solver(data, settings)   # setup: Ruiz scaling + factorisation
    solver.solve()

    info = solver.info
    return {
        "status":       int(info.status_val),
        # QPALM's Data has no constant-term field; it only ever reports 0.5 x'Qx + q'x,
        # so obj_const is added back explicitly to compare against SSN-PMM's obj_val.
        "obj_val":      float(info.objective) + obj_const,
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter_out),
        "inner_iter":   int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }


def run_osqp(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0) -> dict:
    """Run OSQP on a problem already converted via pdpmm_to_qpalm."""
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data

    prob = osqp.OSQP()
    prob.setup(                       # setup: scaling + factorisation
        Q_upper, q, C, bmin, bmax,
        eps_abs    = tol,
        eps_rel    = tol,
        max_iter   = 2_000_000_000,   # effectively infinite
        time_limit = time_limit,
        verbose    = False,
        scaling    = 10,
    )
    res = prob.solve()

    info = res.info
    return {
        "status":       int(info.status_val),
        # OSQP's setup() has no constant-term argument so obj_const is added back explicitly.
        "obj_val":      float(info.obj_val) + obj_const,
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter),
        "inner_iter":   0,
        "tol_achieved": max(float(info.prim_res), float(info.dual_res)),
    }


# ---------------------------------------------------------------------------
# Crash-tolerant subprocess isolation
# ---------------------------------------------------------------------------

def _run_isolated(target_func, args: tuple) -> dict:
    """Spawn a fresh process, run target_func(*args, conn), return the sent dict.

    If the worker dies without sending a result (segfault, OOM-kill by the
    OS, etc.), returns an {"error": ...} dict instead of raising.
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
# Per-problem three-solver runner (shared by the PDE-constrained benchmarks)
# ---------------------------------------------------------------------------

def run_solvers(result: dict, worker_args: tuple,
                worker_ssn, worker_qpalm, worker_osqp,
                tol: float, time_limit: float, max_iter: int, solvers: set,
                cooldown: float = 0.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses.

    worker_args are the problem-defining positional args for worker_ssn/
    worker_qpalm/worker_osqp, passed ahead of (tol, time_limit[, max_iter], conn).
    """
    if "ssn-pmm" in solvers:
        ssn_out = _run_isolated(worker_ssn, (*worker_args, tol, time_limit, max_iter))
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

    if "qpalm" in solvers:
        qpalm_out = _run_isolated(worker_qpalm, (*worker_args, tol, time_limit))
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

    if "osqp" in solvers:
        osqp_out = _run_isolated(worker_osqp, (*worker_args, tol, time_limit))
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

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Rewrite the whole CSV from `rows`. Called after every solver finishes."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_rows(path: Path) -> list[dict]:
    """Load previously written rows so a rerun appends instead of overwriting."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
