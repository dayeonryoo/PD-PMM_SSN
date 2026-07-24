"""
Benchmark SSN-PMM vs QPALM vs OSQP on the Maros-Meszaros QP test set.

Outputs
-------
  results/comparison_mm.csv                  - per-problem timing, iteration counts, and status
  results/performance_profile_mm.pdf/png     - Dolan-Moré performance profile (solve time)
  results/performance_profile_mm_iters.pdf/png - Dolan-Moré performance profile (iterations)

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Install Python dependencies
-------------------------------------
  pip install qpalm osqp numpy scipy matplotlib pandas

Step 2 - Build the SSN-PMM Python binding
------------------------------------------
  All commands are run from the PD-PMM_SSN/python/ directory.

  mkdir build
  cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

  This produces ssn_pmm_bind.cpython-<tag>-darwin.so in python/.
  You only need to rebuild if the C++ solver source changes.

Step 3 - Run the benchmark
---------------------------
  python3 benchmark_mm.py

  Optional: override the project root if running from a different directory:
  python3 benchmark_mm.py --root /path/to/PD-PMM_SSN

Settings: tol = 1e-6, time limit = 600 s (10 min), max iterations = infinity.
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
import matplotlib
matplotlib.use("Agg")        # non-interactive backend; remove if running interactively
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Locate and import the pybind11 extension
# ---------------------------------------------------------------------------
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
# Maros-Meszaros problem list  (name → reference optimal objective)
# ---------------------------------------------------------------------------
QPS = {
    "AUG2D":     1.6874118e+06,
    "AUG2DC":    1.8183681e+06,
    "AUG2DCQP":  6.4981348e+06,
    "AUG2DQP":   6.2370121e+06,
    "AUG3D":     5.5406773e+02,
    "AUG3DC":    7.7126244e+02,
    "AUG3DCQP":  9.9336215e+02,
    "AUG3DQP":   6.7523767e+02,
    "BOYD1":    -6.1735220e+07,
    "BOYD2":     2.1256767e+01,
    "CONT-050": -4.5638509e+00,
    "CONT-100": -4.6443979e+00,
    "CONT-101":  1.9552733e-01,
    "CONT-200": -4.6848759e+00,
    "CONT-201":  1.9248337e-01,
    "CONT-300":  1.9151232e-01,
    "CVXQP1L":   1.0870480e+08,
    "CVXQP1M":   1.0875116e+06,
    "CVXQP1S":   1.1590718e+04,
    "CVXQP2L":   8.1842458e+07,
    "CVXQP2M":   8.2015543e+05,
    "CVXQP2S":   8.1209405e+03,
    "CVXQP3L":   1.1571110e+08,
    "CVXQP3M":   1.3628287e+06,
    "CVXQP3S":   1.1943432e+04,
    "DPKLO1":    3.7009622e-01,
    "DTOC3":     2.3526248e+02,
    "DUAL1":     3.5012966e-02,
    "DUAL2":     3.3733676e-02,
    "DUAL3":     1.3575584e-01,
    "DUAL4":     7.4609084e-01,
    "DUALC1":    6.1552508e+03,
    "DUALC2":    3.5513077e+03,
    "DUALC5":    4.2723233e+02,
    "DUALC8":    1.8309359e+04,
    "EXDATA":   -1.4184343e+02,
    "GENHS28":   9.2717369e-01,
    "GOULDQP2":  1.8427534e-04,
    "GOULDQP3":  2.0627840e+00,
    "HS118":     6.6482045e+02,
    "HS21":     -9.9960000e+01,
    "HS268":     5.7310705e-07,
    "HS35":      1.1111111e-01,
    "HS35MOD":   2.5000000e-01,
    "HS51":      8.8817842e-16,
    "HS52":      5.3266476e+00,
    "HS53":      4.0930233e+00,
    "HS76":     -4.6818182e+00,
    "HUES-MOD":  3.4824690e+07,
    "HUESTIS":   3.4824690e+11,
    "KSIP":      5.7579794e-01,
    "LASER":     2.4096014e+06,
    "LISWET1":   3.6122402e+01,
    "LISWET10":  4.9485785e+01,
    "LISWET11":  4.9523957e+01,
    "LISWET12":  1.7369274e+03,
    "LISWET2":   2.4998076e+01,
    "LISWET3":   2.5001220e+01,
    "LISWET4":   2.5000112e+01,
    "LISWET5":   2.5034253e+01,
    "LISWET6":   2.4995748e+01,
    "LISWET7":   4.9884089e+02,
    "LISWET8":   7.1447006e+03,
    "LISWET9":   1.9632513e+03,
    "LOTSCHD":   2.3984159e+03,
    "MOSARQP1": -9.5287544e+02,
    "MOSARQP2": -1.5974821e+03,
    "POWELL20":  5.2089583e+10,
    "PRIMAL1":  -3.5012965e-02,
    "PRIMAL2":  -3.3733676e-02,
    "PRIMAL3":  -1.3575584e-01,
    "PRIMAL4":  -7.4609083e-01,
    "PRIMALC1": -6.1552508e+03,
    "PRIMALC2": -3.5513077e+03,
    "PRIMALC5": -4.2723233e+02,
    "PRIMALC8": -1.8309430e+04,
    "Q25FV47":   1.3744448e+07,
    "QADLITTL":  4.8031886e+05,
    "QAFIRO":   -1.5907818e+00,
    "QBANDM":    1.6352342e+04,
    "QBEACONF":  1.6471206e+05,
    "QBORE3D":   3.1002008e+03,
    "QBRANDY":   2.8375115e+04,
    "QCAPRI":    6.6793293e+07,
    "QE226":     2.1265343e+02,
    "QETAMACR":  8.6760370e+04,
    "QFFFFF80":  8.7314747e+05,
    "QFORPLAN":  7.4566315e+09,
    "QGFRDXPN":  1.0079059e+11,
    "QGROW15":  -1.0169364e+08,
    "QGROW22":  -1.4962895e+08,
    "QGROW7":   -4.2798714e+07,
    "QISRAEL":   2.5347838e+07,
    "QPCBLEND": -7.8425409e-03,
    "QPCBOEI1":  1.1503914e+07,
    "QPCBOEI2":  8.1719623e+06,
    "QPCSTAIR":  6.2043875e+06,
    "QPILOTNO":  4.7285869e+06,
    "QPTEST":    4.3718750e+00,
    "QRECIPE":  -2.6661600e+02,
    "QSC205":   -5.8139518e-03,
    "QSCAGR25":  2.0173794e+08,
    "QSCAGR7":   2.6865949e+07,
    "QSCFXM1":   1.6882692e+07,
    "QSCFXM2":   2.7776162e+07,
    "QSCFXM3":   3.0816355e+07,
    "QSCORPIO":  1.8805096e+03,
    "QSCRS8":    9.0456001e+02,
    "QSCSD1":    8.6666667e+00,
    "QSCSD6":    5.0808214e+01,
    "QSCSD8":    9.4076357e+02,
    "QSCTAP1":   1.4158611e+03,
    "QSCTAP2":   1.7350265e+03,
    "QSCTAP3":   1.4387547e+03,
    "QSEBA":     8.1481801e+07,
    "QSHARE1B":  7.2007832e+05,
    "QSHARE2B":  1.1703692e+04,
    "QSHELL":    1.5726368e+12,
    "QSHIP04L":  2.4200155e+06,
    "QSHIP04S":  2.4249937e+06,
    "QSHIP08L":  2.3760406e+06,
    "QSHIP08S":  2.3857289e+06,
    "QSHIP12L":  3.0188766e+06,
    "QSHIP12S":  3.0569623e+06,
    "QSIERRA":   2.3750458e+07,
    "QSTAIR":    7.9854528e+06,
    "QSTANDAT":  6.4118384e+03,
    "S268":      5.7310705e-07,
    "STADAT1":  -2.8526864e+07,
    "STADAT2":  -3.2626665e+01,
    "STADAT3":  -3.5779453e+01,
    "STCQP1":    1.5514356e+05,
    "STCQP2":    2.2327313e+04,
    "TAME":      0.0000000e+00,
    "UBH1":      1.1160008e+00,
    "VALUES":   -1.3966211e+00,
    "YAO":       1.9770426e+02,
    "ZECEVIC2": -4.1250000e+00,
}

# ---------------------------------------------------------------------------
# Convert PDPMMdata dict (from ssn_pmm_bind.parse_sif) to QPALM inputs.
#
# PDPMMdata form:  min ½ xᵀQx + cᵀx   s.t.  Ax = b,  lw ≤ Bx ≤ uw,  lx ≤ x ≤ ux
# QPALM form:      min ½ xᵀQx + qᵀx   s.t.  bmin ≤ Cx ≤ bmax
#
# Stacking:  C = [A; B; Iₙ],  bmin/bmax accordingly.
# ---------------------------------------------------------------------------

def _make_upper_triangular(Q):
    """Return upper-triangular part of a symmetric Q.

    Works regardless of whether Q is stored fully, as lower-triangular only,
    or as upper-triangular only.  Uses COO deduplication to avoid double-counting.
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
    """Build (Q_upper, q, C, bmin, bmax, n, m_total) from a parse_sif dict."""
    n, ell = pd["n"], pd["l"]
    INF = 1e30   # QPALM treats values beyond this as infinite

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
# QPALM solver wrapper
# ---------------------------------------------------------------------------
QPALM_SOLVED = qpalm.Info.SOLVED   # == 1

def run_qpalm(qpalm_data: tuple, tol: float, time_limit: float) -> dict:
    """Run QPALM on a problem already converted via pdpmm_to_qpalm.

    Returns dict with: status, obj_val, run_time, outter_iter, inner_iter.
    """
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
        "obj_val":      float(info.objective),
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter_out),
        "inner_iter":   int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }


# ---------------------------------------------------------------------------
# OSQP solver wrapper
# ---------------------------------------------------------------------------
OSQP_SOLVED = 1   # osqp.constant("OSQP_SOLVED")

def run_osqp(qpalm_data: tuple, tol: float, time_limit: float) -> dict:
    """Run OSQP on a problem already converted via pdpmm_to_qpalm.

    Returns dict with: status, obj_val, run_time, outer_iter, inner_iter=0.
    """
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data

    prob = osqp.OSQP()
    prob.setup(                         # setup: scaling + factorisation
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
        "obj_val":      float(info.obj_val),
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter),
        "inner_iter":   int(0),
        "tol_achieved": max(float(info.prim_res), float(info.dual_res)),
    }


# ---------------------------------------------------------------------------
# Performance profile (Dolan-Moré)
# ---------------------------------------------------------------------------

def compute_performance_profile(times: np.ndarray, tau_vals: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    times : (n_problems, n_solvers) float array  – np.inf when unsolved
    tau_vals : sorted 1-D array of τ values

    Returns
    -------
    profiles : (n_solvers, len(tau_vals)) array – ρ_s(τ) values in [0, 1]
    """
    n_p, n_s = times.shape
    best     = times.min(axis=1, keepdims=True)           # (n_p, 1)
    # Problems where every solver failed are excluded from the denominator.
    active   = np.isfinite(best).ravel()
    n_active = int(active.sum())
    if n_active == 0:
        return np.zeros((n_s, len(tau_vals)))

    ratios  = np.full_like(times, np.inf)
    ratios[active] = times[active] / best[active]         # (n_active, n_s)

    profiles = np.zeros((n_s, len(tau_vals)))
    for s in range(n_s):
        r_s = ratios[:, s]
        for j, tau in enumerate(tau_vals):
            profiles[s, j] = float(np.sum(r_s[active] <= tau)) / n_active
    return profiles


def _fmt_tol(tol: float) -> str:
    exp = int(round(np.log10(tol)))
    return rf"$10^{{{exp}}}$"

def _fmt_limit(time_limit: float) -> str:
    if time_limit % 60 == 0:
        return f"{int(time_limit // 60)} min"
    return f"{time_limit:g} s"

def plot_performance_profile(csv_path: Path, out_prefix: Path,
                             tol: float = 1e-6, time_limit: float = 600.0,
                             solvers: set | None = None) -> None:
    if solvers is None:
        solvers = {"ssn-pmm", "qpalm", "osqp"}
    _meta = [
        ("ssn-pmm", "ssn_solved",   "ssn_time",   "SSN-PMM", "#1f77b4", "-"),
        ("qpalm",   "qpalm_solved", "qpalm_time", "QPALM",   "#ff7f0e", "--"),
        ("osqp",    "osqp_solved",  "osqp_time",  "OSQP",    "#2ca02c", "-."),
    ]
    df = pd.read_csv(csv_path)
    n_total = len(df)

    active = [(k, sc, tc, lbl, col, ls) for k, sc, tc, lbl, col, ls in _meta if k in solvers]

    times_list, plot_entries = [], []
    for k, solved_col, time_col, label, color, ls in active:
        t = np.where(df[solved_col].fillna(0).astype(bool), df[time_col].fillna(np.inf), np.inf)
        times_list.append(t)
        plot_entries.append((f"{label} ({int(np.isfinite(t).sum())}/{n_total} solved)", color, ls))

    times    = np.stack(times_list, axis=1)
    tau_vals = np.logspace(0, 3, 2000)
    profiles = compute_performance_profile(times, tau_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    for s, (label, color, ls) in enumerate(plot_entries):
        ax.semilogx(tau_vals, profiles[s], label=label,
                    color=color, linewidth=2, linestyle=ls)

    ax.set_xlabel(r"Performance ratio $\tau$", fontsize=13)
    ax.set_ylabel(r"Fraction of problems $\rho_s(\tau)$", fontsize=13)
    ax.set_title(
        "Performance profile — Maros-Meszaros QPs\n"
        f"(solve time,  tol = {_fmt_tol(tol)},  limit = {_fmt_limit(time_limit)})",
        fontsize=12,
    )
    ax.set_xlim([1.0, tau_vals[-1]])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_prefix.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


def plot_performance_profile_iters(csv_path: Path, out_prefix: Path,
                                   tol: float = 1e-6, time_limit: float = 600.0,
                                   solvers: set | None = None) -> None:
    """Dolan-Moré performance profile using iteration counts as the metric.

    SSN-PMM uses pmm_iter (total PMM iterations).
    QPALM and OSQP use their native iteration counters.
    """
    if solvers is None:
        solvers = {"ssn-pmm", "qpalm", "osqp"}
    _meta = [
        ("ssn-pmm", "ssn_solved",   "pmm_iter",   "SSN-PMM", "#1f77b4", "-"),
        ("qpalm",   "qpalm_solved", "qpalm_iter", "QPALM",   "#ff7f0e", "--"),
        ("osqp",    "osqp_solved",  "osqp_iter",  "OSQP",    "#2ca02c", "-."),
    ]
    df = pd.read_csv(csv_path)
    n_total = len(df)

    active = [(k, sc, ic, lbl, col, ls) for k, sc, ic, lbl, col, ls in _meta if k in solvers]

    iters_list, plot_entries = [], []
    for k, solved_col, iter_col, label, color, ls in active:
        it = np.where(df[solved_col].fillna(0).astype(bool), df[iter_col].fillna(np.inf), np.inf)
        iters_list.append(it)
        plot_entries.append((f"{label} ({int(np.isfinite(it).sum())}/{n_total} solved)", color, ls))

    iters    = np.stack(iters_list, axis=1)
    tau_vals = np.logspace(0, 4, 2000)
    profiles = compute_performance_profile(iters, tau_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    for s, (label, color, ls) in enumerate(plot_entries):
        ax.semilogx(tau_vals, profiles[s], label=label,
                    color=color, linewidth=2, linestyle=ls)

    ax.set_xlabel(r"Performance ratio $\tau$", fontsize=13)
    ax.set_ylabel(r"Fraction of problems $\rho_s(\tau)$", fontsize=13)
    ax.set_title(
        "Performance profile — Maros-Meszaros QPs\n"
        f"(iterations,  tol = {_fmt_tol(tol)},  limit = {_fmt_limit(time_limit)})",
        fontsize=12,
    )
    ax.set_xlim([1.0, tau_vals[-1]])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_prefix.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


def plot_performance_profile_inner_iters(csv_path: Path, out_prefix: Path,
                                         tol: float = 1e-6, time_limit: float = 600.0,
                                         solvers: set | None = None) -> None:
    """Dolan-Moré performance profile using inner iteration counts.

    SSN-PMM uses ssn_iter (total SSN Newton iterations).
    QPALM uses qpalm_inner_iter (total inner QPALM iterations).
    OSQP is excluded (no meaningful inner iterations).
    """
    if solvers is None:
        solvers = {"ssn-pmm", "qpalm"}
    solvers = solvers & {"ssn-pmm", "qpalm"}   # inner iters only defined for these two
    if not solvers:
        return

    _meta = [
        ("ssn-pmm", "ssn_solved",   "ssn_iter",         "SSN-PMM", "#1f77b4", "-"),
        ("qpalm",   "qpalm_solved", "qpalm_inner_iter", "QPALM",   "#ff7f0e", "--"),
    ]
    df = pd.read_csv(csv_path)
    n_total = len(df)

    active = [(k, sc, ic, lbl, col, ls) for k, sc, ic, lbl, col, ls in _meta if k in solvers]

    iters_list, plot_entries = [], []
    for k, solved_col, iter_col, label, color, ls in active:
        it = np.where(df[solved_col].fillna(0).astype(bool), df[iter_col].fillna(np.inf), np.inf)
        iters_list.append(it)
        plot_entries.append((f"{label} ({int(np.isfinite(it).sum())}/{n_total} solved)", color, ls))

    iters    = np.stack(iters_list, axis=1)
    tau_vals = np.logspace(0, 4, 2000)
    profiles = compute_performance_profile(iters, tau_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    for s, (label, color, ls) in enumerate(plot_entries):
        ax.semilogx(tau_vals, profiles[s], label=label,
                    color=color, linewidth=2, linestyle=ls)

    ax.set_xlabel(r"Performance ratio $\tau$", fontsize=13)
    ax.set_ylabel(r"Fraction of problems $\rho_s(\tau)$", fontsize=13)
    ax.set_title(
        "Performance profile — Maros-Meszaros QPs\n"
        f"(inner iterations,  tol = {_fmt_tol(tol)},  limit = {_fmt_limit(time_limit)})",
        fontsize=12,
    )
    ax.set_xlim([1.0, tau_vals[-1]])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_prefix.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn_mm(sif_path, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.parse_sif(sif_path)
        result["res"] = ssn_pmm_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_mm(sif_path, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.parse_sif(sif_path)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_mm(sif_path, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ssn_pmm_bind.parse_sif(sif_path)
        qpalm_data = pdpmm_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit)
    except Exception as e:
        result["error"] = str(e)
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
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(HERE.parent),
        help="Path to the PD-PMM_SSN project root (default: parent of this script)",
    )
    parser.add_argument(
        "--tol",        type=float, default=1e-6,  help="Primal-dual tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=600.0, help="Per-problem time limit in seconds (default: 600)"
    )
    parser.add_argument(
        "--name", default="", help="Prefix for output filenames (e.g. '0508' → '0508_comparison_mm.csv')"
    )
    parser.add_argument(
        "--solver", nargs="+", default=["ssn-pmm", "qpalm", "osqp"],
        choices=["ssn-pmm", "qpalm", "osqp"], metavar="SOLVER",
        help="Solvers to run (default: all three). Choices: ssn-pmm qpalm osqp",
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0,
        help="Seconds to sleep between problems to prevent CPU throttling (default: 3)",
    )
    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    solvers = set(args.solver)

    root      = Path(args.root).resolve()
    data_dir  = root / "data" / "maros_meszaros"
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000   # effectively infinite for SSN-PMM

    prefix = f"{args.name}_" if args.name else ""
    csv_path = result_dir / f"{prefix}comparison_mm.csv"
    fieldnames = [
        "name",
        "ssn_solved",   "ssn_status",   "pmm_iter", "ssn_iter",   "ssn_obj",   "pmm_tol_achieved",   "ssn_time",
        "qpalm_solved", "qpalm_status", "qpalm_iter", "qpalm_inner_iter", "qpalm_obj", "qpalm_tol_achieved", "qpalm_time",
        "osqp_solved",  "osqp_status",  "osqp_iter",                      "osqp_obj",  "osqp_tol_achieved",  "osqp_time",
    ]

    n_problems = len(QPS)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (name, _ref_obj) in enumerate(QPS.items(), 1):
            sif_path = str(data_dir / f"{name}.SIF")
            if not os.path.exists(sif_path):
                print(f"[{idx:3d}/{n_problems}] SKIP (file not found): {name}")
                continue

            print(f"\n[{idx:3d}/{n_problems}]  {name}")
            row: dict = {"name": name}

            # ---- SSN-PMM ------------------------------------------------
            if "ssn-pmm" in solvers:
                ssn_out = _run_isolated(_worker_ssn_mm, (sif_path, tol, time_limit, max_iter))
                if "error" in ssn_out:
                    print(f"  SSN-PMM : ERROR — {ssn_out['error']}")
                    row.update(ssn_status=-99, ssn_solved=0, ssn_time=np.inf,
                               pmm_iter=np.inf, ssn_iter=np.inf,
                               ssn_obj=np.nan, pmm_tol_achieved=np.nan)
                else:
                    r = ssn_out["res"]
                    row["ssn_status"]       = r["status"]
                    row["ssn_solved"]       = int(r["status"] == 0)
                    row["ssn_time"]         = r["run_time"]
                    row["pmm_iter"]         = r["pmm_iter"]
                    row["ssn_iter"]         = r["ssn_iter"]
                    row["ssn_obj"]          = r["obj_val"]
                    row["pmm_tol_achieved"] = r["pmm_tol_achieved"]
                    status_str = "OPTIMAL" if r["status"] == 0 else f"status={r['status']}"
                    print(f"  SSN-PMM : {status_str:12s}  t = {r['run_time']:.3f} s  "
                          f"pmm={r['pmm_iter']} ssn={r['ssn_iter']}  "
                          f"tol={r['pmm_tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
                if cooldown > 0:
                    time.sleep(cooldown)

            # ---- QPALM --------------------------------------------------
            if "qpalm" in solvers:
                qpalm_out = _run_isolated(_worker_qpalm_mm, (sif_path, tol, time_limit))
                if "error" in qpalm_out:
                    print(f"  QPALM   : ERROR — {qpalm_out['error']}")
                    row.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=np.inf,
                               qpalm_iter=np.inf, qpalm_inner_iter=np.inf,
                               qpalm_obj=np.nan, qpalm_tol_achieved=np.nan)
                else:
                    r = qpalm_out["res"]
                    row["qpalm_status"]        = r["status"]
                    row["qpalm_solved"]        = int(r["status"] == QPALM_SOLVED)
                    row["qpalm_time"]          = r["run_time"]
                    row["qpalm_iter"]          = r["outer_iter"]
                    row["qpalm_inner_iter"]    = r["inner_iter"]
                    row["qpalm_obj"]           = r["obj_val"]
                    row["qpalm_tol_achieved"]  = r["tol_achieved"]
                    status_str = "OPTIMAL" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
                    print(f"  QPALM   : {status_str:12s}  t = {r['run_time']:.3f} s  "
                          f"outer={r['outer_iter']} inner={r['inner_iter']}  "
                          f"tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
                if cooldown > 0:
                    time.sleep(cooldown)

            # ---- OSQP ---------------------------------------------------
            if "osqp" in solvers:
                osqp_out = _run_isolated(_worker_osqp_mm, (sif_path, tol, time_limit))
                if "error" in osqp_out:
                    print(f"  OSQP    : ERROR — {osqp_out['error']}")
                    row.update(osqp_status=-99, osqp_solved=0, osqp_time=np.inf, osqp_iter=np.inf,
                               osqp_obj=np.nan, osqp_tol_achieved=np.nan)
                else:
                    r = osqp_out["res"]
                    row["osqp_status"]       = r["status"]
                    row["osqp_solved"]       = int(r["status"] == OSQP_SOLVED)
                    row["osqp_time"]         = r["run_time"]
                    row["osqp_iter"]         = r["outer_iter"]
                    row["osqp_obj"]          = r["obj_val"]
                    row["osqp_tol_achieved"] = r["tol_achieved"]
                    status_str = "OPTIMAL" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
                    print(f"  OSQP    : {status_str:12s}  t = {r['run_time']:.3f} s  "
                          f"iter = {r['outer_iter']}  tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
                if cooldown > 0:
                    time.sleep(cooldown)

            writer.writerow(row)
            fh.flush()

    print(f"\nResults written to: {csv_path}")

    # ---- Performance profiles -------------------------------------------
    out_prefix = result_dir / f"{prefix}performance_profile_mm"
    plot_performance_profile(csv_path, out_prefix, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_iters = result_dir / f"{prefix}performance_profile_mm_iters"
    plot_performance_profile_iters(csv_path, out_prefix_iters, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_inner = result_dir / f"{prefix}performance_profile_mm_inner_iters"
    plot_performance_profile_inner_iters(csv_path, out_prefix_inner, tol=tol, time_limit=time_limit, solvers=solvers)


if __name__ == "__main__":
    main()
