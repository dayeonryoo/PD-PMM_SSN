"""
Build penalized quantile-regression and elastic-net linear-SVM QPs in
KSPQPdata dict form (same format as ksp_qp_bind.parse_sif()), following
Sections 5.3-5.4 of Pougkakiotis, Gondzio & Kalogerias (J. Sci. Comput. 2025).

Unlike the portfolio CVaR/MAsD reformulation (portfolio_generator.py), whose
constraint rows are dense because asset-return vectors are dense, these
problems use LIBSVM datasets whose feature vectors are natively sparse
(bag-of-words / TF-IDF text features, or few-dimensional numeric features) --
so the resulting KKT systems stay sparse, which is the property KSP-QP's
Schur-complement/ordering-selection machinery is actually designed around.

Quantile regression (5.5)-(5.6), using the pinball-loss identity
    ell_alpha(w) = (1-alpha) w_- + alpha w_+ = (w)_+ + (alpha - 1) w
(single hinge + linear term, verified numerically -- no second slack needed):

    min_{beta0,beta,s,v}  (1/l) sum_i s_i + (1-alpha) beta0
                          + (1-alpha) mean(X)^T beta + lam*tau*sum_j v_j
                          + lam*(1-tau)/2 * beta^T beta
    s.t.  s_i     >= y_i - beta0 - X[i]^T beta,           s_i >= 0
          v_j     >= beta_j,  v_j >= -beta_j (elastic-net L1 epigraph), v_j >= 0

Elastic-net linear SVM (5.7):

    min_{beta0,beta,s,v}  (1/l) sum_i s_i + lam*tau1*sum_j v_j
                          + lam*tau2/2 * beta^T beta
    s.t.  s_i     >= 1 - y_i*(X[i]^T beta - beta0),        s_i >= 0
          v_j     >= beta_j,  v_j >= -beta_j,              v_j >= 0

Variable layout for both: z = [beta0 (1); beta (d); s (l); v (d)].
"""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import scipy.sparse as sp

INF = np.inf


# ---------------------------------------------------------------------------
# LIBSVM-format parsing (dependency-free -- no scikit-learn in this venv)
# ---------------------------------------------------------------------------

def parse_libsvm(path: str, n_features: int | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    """Parse a LIBSVM-format file (plain text or .bz2) into (X, y).

    Each line: "label idx1:val1 idx2:val2 ...", 1-based ascending indices.
    Returns X as (n_samples, n_features) CSR, y as a 1D float array.
    """
    opener = bz2.open if str(path).endswith(".bz2") else open
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    labels: list[float] = []
    with opener(path, "rt") as fh:
        for i, line in enumerate(fh):
            parts = line.split()
            if not parts:
                continue
            labels.append(float(parts[0]))
            for tok in parts[1:]:
                idx_str, val_str = tok.split(":")
                cols.append(int(idx_str) - 1)
                vals.append(float(val_str))
                rows.append(i)
    n_samples = len(labels)
    max_col = (max(cols) + 1) if cols else 0
    if n_features is None:
        n_features = max_col
    else:
        n_features = max(n_features, max_col)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_features))
    y = np.array(labels, dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Feature standardization
#
# Raw LIBSVM regression features can span many orders of magnitude on
# different columns (e.g. space_ga mixes columns of O(10) with columns of
# O(1e7)); combined with a single scalar L2 penalty lam*(1-tau) on the whole
# beta block, this makes the reduced KKT system severely ill-conditioned
# regardless of solver. QPALM/OSQP mask this somewhat via their own internal
# Ruiz scaling; KSP-QP has no automatic problem scaling, so callers must
# standardize features themselves -- this is also just standard practice for
# any L1/L2-regularized linear model, since such penalties are scale-dependent.
#
# Mean-centering densifies a sparse matrix, so for large/sparse inputs (text
# features: TF-IDF, bag-of-words) we only rescale columns by their RMS norm
# (sparsity-preserving, no centering -- standard for such data anyway); for
# small/dense inputs we do full mean-centering + unit-variance scaling.
# ---------------------------------------------------------------------------

def standardize_features(X: sp.spmatrix, dense_entries_threshold: int = 5_000_000) -> sp.csr_matrix:
    l, d = X.shape
    if l * d <= dense_entries_threshold:
        Xd = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
        mu = Xd.mean(axis=0)
        sigma = Xd.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        return sp.csr_matrix((Xd - mu) / sigma)
    X = sp.csr_matrix(X, dtype=np.float64)
    col_sq_sum = np.asarray(X.multiply(X).sum(axis=0)).ravel()
    rms = np.sqrt(col_sq_sum / l)
    rms[rms < 1e-12] = 1.0
    return (X @ sp.diags(1.0 / rms)).tocsr()


# ---------------------------------------------------------------------------
# Shared sparse block-assembly helpers
# ---------------------------------------------------------------------------

def _l1_epigraph_block(d: int, n_z: int, beta_col_offset: int, v_col_offset: int) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Rows enforcing v_j >= |beta_j| via v_j >= beta_j and v_j >= -beta_j.

    Returns (B_l1 (2d x n_z), lw (2d,), uw (2d,)) with lw = 0, uw = +inf.
    """
    def _embed(cols_local: sp.spmatrix, col_offset: int) -> sp.csc_matrix:
        left  = sp.csc_matrix((d, col_offset))
        right = sp.csc_matrix((d, n_z - col_offset - cols_local.shape[1]))
        return sp.hstack([left, cols_local, right], format="csc")

    I_d = sp.eye(d, format="csc")
    row_a = _embed(-I_d, beta_col_offset) + _embed(I_d, v_col_offset)   # v_j - beta_j >= 0
    row_b = _embed(I_d, beta_col_offset)  + _embed(I_d, v_col_offset)   # v_j + beta_j >= 0
    B_l1 = sp.vstack([row_a, row_b], format="csc")
    lw = np.zeros(2 * d)
    uw = np.full(2 * d, INF)
    return B_l1, lw, uw


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
              lw: np.ndarray, uw: np.ndarray, obj_const: float = 0.0) -> dict:
    n_z = Q.shape[0]
    out: dict = {"n": n_z, "m": int(A.shape[0]), "l": int(B.shape[0]), "obj_const": float(obj_const)}
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


# ---------------------------------------------------------------------------
# Quantile regression QP
# ---------------------------------------------------------------------------

def generate_quantile_qp(X: sp.spmatrix, y: np.ndarray, alpha: float,
                         lam: float, tau: float, standardize: bool = True,
                         standardize_target: bool = True) -> dict:
    """Elastic-net penalized quantile regression, Section 5.3 of the reference paper.

    standardize=True (default) rescales X's columns before building the QP --
    see standardize_features(); without it, raw unnormalized features make the
    L1/L2 penalty's effective strength wildly inconsistent across coefficients
    and can leave the KKT system severely ill-conditioned for any solver.

    standardize_target=True (default) centers/scales y to zero mean, unit
    variance. Quantile regression is equivariant to affine transforms of y
    (the fit at level alpha just rescales along with it), so this changes
    nothing about which quantile is being estimated -- it only matters
    because some LIBSVM regression targets span O(1e5) (e.g. cadata), where
    requesting a fixed absolute tolerance like 1e-5 is otherwise meaningless
    for any solver (it would demand ~10 digits of relative precision).
    """
    X = standardize_features(X) if standardize else sp.csr_matrix(X)
    if standardize_target:
        y_std = y.std()
        y = (y - y.mean()) / (y_std if y_std > 1e-12 else 1.0)
    l, d = X.shape
    n_z = 1 + d + l + d
    beta0_idx, beta_off, s_off, v_off = 0, 1, 1 + d, 1 + d + l

    mean_X = np.asarray(X.mean(axis=0)).ravel()
    mean_y = float(y.mean())

    c = np.zeros(n_z)
    c[beta0_idx] = 1.0 - alpha
    c[beta_off:beta_off + d] = (1.0 - alpha) * mean_X
    c[s_off:s_off + l] = 1.0 / l
    c[v_off:v_off + d] = lam * tau
    obj_const = (alpha - 1.0) * mean_y

    Q_diag = np.zeros(n_z)
    Q_diag[beta_off:beta_off + d] = lam * (1.0 - tau)
    Q = sp.diags(Q_diag, format="csc")

    A = sp.csc_matrix((0, n_z))
    b = np.zeros(0)

    # Hinge rows: beta0 + X[i]^T beta + s_i >= y_i
    B_hinge = sp.hstack([
        np.ones((l, 1)),
        X,
        sp.eye(l, format="csc"),
        sp.csc_matrix((l, d)),
    ], format="csc")
    lw_hinge = y.copy()
    uw_hinge = np.full(l, INF)

    B_l1, lw_l1, uw_l1 = _l1_epigraph_block(d, n_z, beta_off, v_off)

    B = sp.vstack([B_hinge, B_l1], format="csc")
    lw = np.concatenate([lw_hinge, lw_l1])
    uw = np.concatenate([uw_hinge, uw_l1])

    lx = np.concatenate([[-INF], np.full(d, -INF), np.zeros(l), np.zeros(d)])
    ux = np.full(n_z, INF)

    return _assemble(Q, A, B, c, b, lx, ux, lw, uw, obj_const)


# ---------------------------------------------------------------------------
# Elastic-net linear SVM QP
# ---------------------------------------------------------------------------

def generate_svm_qp(X: sp.spmatrix, y: np.ndarray, lam: float,
                    tau1: float, tau2: float, standardize: bool = True) -> dict:
    """Elastic-net soft-margin linear SVM, Section 5.4 of the reference paper.

    y must be in {-1, +1}. See generate_quantile_qp() for why standardize
    defaults to True.
    """
    X = standardize_features(X) if standardize else sp.csr_matrix(X)
    l, d = X.shape
    n_z = 1 + d + l + d
    beta0_idx, beta_off, s_off, v_off = 0, 1, 1 + d, 1 + d + l

    c = np.zeros(n_z)
    c[s_off:s_off + l] = 1.0 / l
    c[v_off:v_off + d] = lam * tau1

    Q_diag = np.zeros(n_z)
    Q_diag[beta_off:beta_off + d] = lam * tau2
    Q = sp.diags(Q_diag, format="csc")

    A = sp.csc_matrix((0, n_z))
    b = np.zeros(0)

    # Hinge rows: -y_i*beta0 + y_i*X[i]^T beta + s_i >= 1
    y_diag = sp.diags(y, format="csc")
    B_hinge = sp.hstack([
        (-y).reshape(-1, 1),
        y_diag @ X,
        sp.eye(l, format="csc"),
        sp.csc_matrix((l, d)),
    ], format="csc")
    lw_hinge = np.ones(l)
    uw_hinge = np.full(l, INF)

    B_l1, lw_l1, uw_l1 = _l1_epigraph_block(d, n_z, beta_off, v_off)

    B = sp.vstack([B_hinge, B_l1], format="csc")
    lw = np.concatenate([lw_hinge, lw_l1])
    uw = np.concatenate([uw_hinge, uw_l1])

    lx = np.concatenate([[-INF], np.full(d, -INF), np.zeros(l), np.zeros(d)])
    ux = np.full(n_z, INF)

    return _assemble(Q, A, B, c, b, lx, ux, lw, uw)


# ---------------------------------------------------------------------------
# Dataset registry (paths relative to data/libsvm/, populated separately)
# ---------------------------------------------------------------------------

_LIBSVM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "libsvm"

QUANTILE_DATASETS = ["space_ga", "abalone", "cpusmall", "cadata", "E2006"]
SVM_DATASETS = ["rcv1", "real-sim", "news20"]


def load_libsvm_npz(name: str, data_dir=None) -> tuple[sp.csr_matrix, np.ndarray]:
    """Load a pre-converted dataset from data/libsvm/<name>.npz + <name>_y.npy."""
    root = Path(data_dir) if data_dir is not None else _LIBSVM_DATA_DIR
    X = sp.load_npz(root / f"{name}.npz")
    y = np.load(root / f"{name}_y.npy")
    return X, y
