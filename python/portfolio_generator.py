"""
Build CVaR / MAsD portfolio-selection QPs in KSPQPdata dict form (same format
as ksp_qp_bind.parse_sif()), so they can be passed directly to
ksp_qp_bind.solve_from_data() and benchmark_common.kspqp_to_qpalm().

Follows the mean-risk portfolio model in Pougkakiotis, Gondzio & Kalogerias,
"An Efficient Active-Set Method With Applications to Sparse Approximations
and Risk Minimization" (J. Sci. Comput. 2025), Section 5.1, reformulated as
a plain LP-in-QP via epigraph slack variables (Q = 0):

CVaR:
    min_{x,t,s}  t + 1/(l*alpha) * sum_i s_i
    s.t.         sum_i x_i = 1
                 mean_ret^T x >= r                      (optional)
                 -returns[i]^T x - t - s_i <= 0,  i=1..l
                 x in [0, au],  t free,  s >= 0

MAsD:
    min_{x,s}    1/l * sum_i s_i
    s.t.         sum_i x_i = 1
                 mean_ret^T x >= r                      (optional)
                 (mean_ret - returns[i])^T x - s_i <= 0, i=1..l
                 x in [0, au],  s >= 0

The active-set size at the optimum (# of s_i > 0, i.e. binding hinge rows)
scales with alpha for CVaR and stays near l/2 for MAsD -- unlike PDE-constrained
QPs where the active set saturates the full problem size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

INF = np.inf


# ---------------------------------------------------------------------------
# Return data
# ---------------------------------------------------------------------------

def synthetic_returns(n_assets: int, l_scenarios: int, seed: int = 0,
                       annual_vol: float = 0.20, annual_ret: float = 0.06,
                       avg_corr: float = 0.25) -> np.ndarray:
    """Generate a (l_scenarios, n_assets) matrix of weekly synthetic returns.

    Uses a one-factor correlation structure (common market factor + idiosyncratic
    noise) to give realistic cross-asset correlation, calibrated to plausible
    annualized vol/return. This is a stand-in for real historical data (e.g.
    the Bruni et al. 2016 datasets used in the reference paper) -- swap in
    load_returns_csv() once real data is available.
    """
    rng = np.random.default_rng(seed)
    weekly_vol = annual_vol / np.sqrt(52.0)
    weekly_ret = annual_ret / 52.0

    beta = rng.uniform(0.4, 1.2, size=n_assets)
    market = rng.normal(0.0, weekly_vol, size=l_scenarios)
    idio_vol = weekly_vol * np.sqrt(max(1e-6, 1.0 - avg_corr))
    idio = rng.normal(0.0, idio_vol, size=(l_scenarios, n_assets))

    returns = weekly_ret + np.outer(market, beta) + idio
    return returns


def bootstrap_resample(returns: np.ndarray, l_target: int, seed: int = 0,
                        block_size: int = 4) -> np.ndarray:
    """Block-bootstrap `returns` (l x n) up to l_target rows.

    Used to scale a fixed real (or synthetic) dataset up to large scenario
    counts while preserving its serial/cross-sectional correlation structure,
    to probe solver behaviour at scales beyond what the source dataset offers.
    """
    rng = np.random.default_rng(seed)
    l, n = returns.shape
    n_blocks = int(np.ceil(l_target / block_size))
    starts = rng.integers(0, max(1, l - block_size + 1), size=n_blocks)
    blocks = [returns[s:s + block_size] for s in starts]
    out = np.concatenate(blocks, axis=0)[:l_target]
    return out


def load_returns_csv(path: str) -> np.ndarray:
    """Load a (l_scenarios, n_assets) return matrix from CSV (no header, no index)."""
    return np.loadtxt(path, delimiter=",")


# ---------------------------------------------------------------------------
# Bruni, Cesarone, Scozzari & Tardella (2016) real-world portfolio datasets
# "Real-world datasets for portfolio selection and solutions of some
# stochastic dominance portfolio models", Data in Brief 8, 858-862.
# https://doi.org/10.1016/j.dib.2016.06.031 (open access, CC-BY)
#
# Converted once from the paper's supplementary Assets_Returns / Index_Returns
# .mat matrices (weekly returns) to CSV under data/portfolio/. Shapes match
# Table 1 of Pougkakiotis, Gondzio & Kalogerias (2025), Section 5.1.
# ---------------------------------------------------------------------------

BRUNI_DATASETS = ["DowJones", "NASDAQ100", "FTSE100", "FF49Industries", "SP500", "NASDAQComp"]

_PORTFOLIO_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "portfolio"


def load_bruni_dataset(name: str, data_dir=None) -> tuple[np.ndarray, float]:
    """Load a Bruni et al. dataset by name (see BRUNI_DATASETS).

    Returns (returns, r_benchmark):
      returns     -- (l_scenarios, n_assets) weekly asset returns
      r_benchmark -- mean weekly return of the dataset's market-index series
                     (uniform-allocation benchmark for FF49Industries, per the
                     paper), usable directly as the CVaR/MAsD model's expected
                     return threshold r in (5.1).
    """
    if name not in BRUNI_DATASETS:
        raise ValueError(f"Unknown Bruni dataset {name!r}; choices: {BRUNI_DATASETS}")
    root = Path(data_dir) if data_dir is not None else _PORTFOLIO_DATA_DIR
    returns = np.loadtxt(root / f"{name}_returns.csv", delimiter=",")
    index   = np.loadtxt(root / f"{name}_index.csv", delimiter=",")
    return returns, float(index.mean())


# ---------------------------------------------------------------------------
# KSPQPdata dict assembly
# ---------------------------------------------------------------------------

def _sparse_to_ksp(M: sp.spmatrix, key: str) -> dict:
    M = M.tocsc()
    M.sort_indices()
    return {
        f"{key}_data":    M.data.astype(np.float64),
        f"{key}_indices": M.indices.astype(np.int32),
        f"{key}_indptr":  M.indptr.astype(np.int32),
        f"{key}_shape":   M.shape,
    }


def _assemble(Q: sp.spmatrix, A: sp.spmatrix, B: sp.spmatrix,
              c: np.ndarray, b: np.ndarray,
              lx: np.ndarray, ux: np.ndarray,
              lw: np.ndarray, uw: np.ndarray) -> dict:
    n_z = Q.shape[0]
    out: dict = {"n": n_z, "m": int(A.shape[0]), "l": int(B.shape[0]), "obj_const": 0.0}
    out.update(_sparse_to_ksp(Q, "Q"))
    out.update(_sparse_to_ksp(A, "A"))
    out.update(_sparse_to_ksp(B, "B"))
    out["c"]  = c.astype(np.float64)
    out["b"]  = b.astype(np.float64)
    out["lx"] = lx.astype(np.float64)
    out["ux"] = ux.astype(np.float64)
    out["lw"] = lw.astype(np.float64)
    out["uw"] = uw.astype(np.float64)
    return out


def generate_cvar_qp(returns: np.ndarray, alpha: float, r: float | None = 0.0,
                      au: float = 1.0) -> dict:
    """Build the CVaR portfolio QP. Variables z = [x (n); t (1); s (l)]."""
    l, n = returns.shape
    mean_ret = returns.mean(axis=0)
    n_z = n + 1 + l
    t_idx = n

    c = np.zeros(n_z)
    c[t_idx] = 1.0
    c[t_idx + 1:] = 1.0 / (l * alpha)
    Q = sp.csc_matrix((n_z, n_z))

    # Equality: sum x_i = 1
    A_row = np.zeros(n_z)
    A_row[:n] = 1.0
    A = sp.csc_matrix(A_row.reshape(1, -1))
    b = np.array([1.0])

    # General inequalities
    rows, lw_list, uw_list = [], [], []
    if r is not None:
        row = np.zeros(n_z)
        row[:n] = mean_ret
        rows.append(row)
        lw_list.append(r)
        uw_list.append(INF)

    hinge = np.zeros((l, n_z))
    hinge[:, :n] = -returns
    hinge[:, t_idx] = -1.0
    hinge[np.arange(l), t_idx + 1 + np.arange(l)] = -1.0
    rows.extend(hinge)
    lw_list.extend([-INF] * l)
    uw_list.extend([0.0] * l)

    B = sp.csc_matrix(np.vstack(rows))
    lw = np.array(lw_list)
    uw = np.array(uw_list)

    lx = np.concatenate([np.zeros(n), [-INF], np.zeros(l)])
    ux = np.concatenate([np.full(n, au), [INF], np.full(l, INF)])

    return _assemble(Q, A, B, c, b, lx, ux, lw, uw)


def generate_masd_qp(returns: np.ndarray, r: float | None = 0.0,
                      au: float = 1.0) -> dict:
    """Build the MAsD portfolio QP. Variables z = [x (n); s (l)]."""
    l, n = returns.shape
    mean_ret = returns.mean(axis=0)
    n_z = n + l

    c = np.zeros(n_z)
    c[n:] = 1.0 / l
    Q = sp.csc_matrix((n_z, n_z))

    A_row = np.zeros(n_z)
    A_row[:n] = 1.0
    A = sp.csc_matrix(A_row.reshape(1, -1))
    b = np.array([1.0])

    rows, lw_list, uw_list = [], [], []
    if r is not None:
        row = np.zeros(n_z)
        row[:n] = mean_ret
        rows.append(row)
        lw_list.append(r)
        uw_list.append(INF)

    hinge = np.zeros((l, n_z))
    hinge[:, :n] = mean_ret - returns
    hinge[np.arange(l), n + np.arange(l)] = -1.0
    rows.extend(hinge)
    lw_list.extend([-INF] * l)
    uw_list.extend([0.0] * l)

    B = sp.csc_matrix(np.vstack(rows))
    lw = np.array(lw_list)
    uw = np.array(uw_list)

    lx = np.concatenate([np.zeros(n), np.zeros(l)])
    ux = np.concatenate([np.full(n, au), np.full(l, INF)])

    return _assemble(Q, A, B, c, b, lx, ux, lw, uw)
