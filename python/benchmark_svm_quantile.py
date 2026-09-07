"""
Benchmark KSP-QP vs QPALM vs OSQP on penalized quantile-regression and
elastic-net linear-SVM QPs (see python/quantile_svm_generator.py), following
Sections 5.3-5.4 of Pougkakiotis, Gondzio & Kalogerias (J. Sci. Comput. 2025),
extended to include QPALM (the paper only compared against IP-PMM here).

Motivation: the portfolio CVaR/MAsD benchmark (benchmark_portfolio.py) showed
KSP-QP losing badly, traced to genuinely DENSE constraint rows (asset-return
vectors touch every asset) defeating its sparse Schur-complement/ordering-
selection design -- independent of active-set size. Quantile regression and
linear SVM have the same hinge-loss shape, but on these LIBSVM datasets the
feature vectors are natively SPARSE (bag-of-words/TF-IDF text, or few-
dimensional numeric features), so this is the fairer test of the original
active-set hypothesis.

Data: LIBSVM datasets converted to data/libsvm/<name>.npz (+ _y.npy). See
quantile_svm_generator.parse_libsvm() / load_libsvm_npz(). Quantile targets:
space_ga, abalone, cpusmall, cadata, E2006. SVM targets: rcv1, real-sim, news20.

Outputs
-------
  results/svm_quantile_regression.csv
  results/svm_quantile_svm.csv

=== HOW TO RUN ===
  python3 benchmark_svm_quantile.py
  python3 benchmark_svm_quantile.py --task quantile --dataset space_ga abalone
  python3 benchmark_svm_quantile.py --task svm --alpha-or-tau ...  (see --help)
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    import ksp_qp_bind
except ModuleNotFoundError:
    sys.exit(
        "Cannot find ksp_qp_bind. "
        "Build it first:\n"
        "  cd python && mkdir build && cd build\n"
        "  cmake .. && cmake --build . --config Release"
    )

from benchmark_common import (
    kspqp_to_qpalm,
    run_qpalm,
    run_osqp,
    run_solvers,
    _write_csv,
    _load_existing_rows,
)
import quantile_svm_generator as qs

QUANTILE_DATASETS = ["space_ga", "abalone", "cpusmall", "cadata", "E2006"]
SVM_DATASETS = ["rcv1", "real-sim", "news20"]

# Paper defaults: Table 7 (alpha sweep, lam/tau fixed), Table 8 (tau/lam sweep, alpha fixed).
QUANTILE_ALPHA_DEFAULT = [0.05, 0.5, 0.9, 0.95]   # includes paper's values + extremes (0.05/0.95)
QUANTILE_LAM_DEFAULT = 1e-2
QUANTILE_TAU_DEFAULT = 0.5

# Paper Table 10: (tau1, tau2) pairs, lam fixed.
SVM_TAU_PAIRS_DEFAULT = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (5.0, 5.0)]
SVM_LAM_DEFAULT = 1e-2


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _build_quantile_pd(dataset, alpha, lam, tau):
    X, y = qs.load_libsvm_npz(dataset)
    return qs.generate_quantile_qp(X, y, alpha, lam, tau)


def _build_svm_pd(dataset, tau1, tau2, lam):
    X, y = qs.load_libsvm_npz(dataset)
    return qs.generate_svm_qp(X, y, lam, tau1, tau2)


def _worker_ssn_quantile(dataset, alpha, lam, tau, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _build_quantile_pd(dataset, alpha, lam, tau)
        result["n_vars"] = pd_data["n"]
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_quantile(dataset, alpha, lam, tau, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_quantile_pd(dataset, alpha, lam, tau)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_quantile(dataset, alpha, lam, tau, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_quantile_pd(dataset, alpha, lam, tau)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_ssn_svm(dataset, tau1, tau2, lam, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _build_svm_pd(dataset, tau1, tau2, lam)
        result["n_vars"] = pd_data["n"]
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_svm(dataset, tau1, tau2, lam, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_svm_pd(dataset, tau1, tau2, lam)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_svm(dataset, tau1, tau2, lam, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_svm_pd(dataset, tau1, tau2, lam)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# CSV fields
# ---------------------------------------------------------------------------

QUANTILE_CSV_FIELDS = [
    "dataset", "n_samples", "n_features", "alpha", "lam", "tau",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]

SVM_CSV_FIELDS = [
    "dataset", "n_samples", "n_features", "tau1", "tau2", "lam",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


def _dataset_shape(dataset: str) -> tuple[int, int]:
    import quantile_svm_generator as qs
    X, _ = qs.load_libsvm_npz(dataset)
    return X.shape


def run_quantile_table(datasets: list[str], alphas: list[float], lam: float, tau: float,
                       tol: float, time_limit: float, max_iter: int,
                       result_dir: Path, solvers: set, cooldown: float = 0.0,
                       name_prefix: str = "") -> None:
    csv_path = result_dir / f"{name_prefix}svm_quantile_regression.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(datasets) * len(alphas)
    done = 0

    def _flush() -> None:
        _write_csv(csv_path, rows, QUANTILE_CSV_FIELDS)

    for dataset in datasets:
        n_samples, n_features = _dataset_shape(dataset)
        for alpha in alphas:
            done += 1
            print(f"\n[quantile  {done}/{n_total}]  {dataset} alpha={alpha}  "
                  f"n={n_samples} d={n_features}", flush=True)
            row = {"dataset": dataset, "n_samples": n_samples, "n_features": n_features,
                   "alpha": alpha, "lam": lam, "tau": tau}
            rows.append(row)
            worker_args = (dataset, alpha, lam, tau)
            run_solvers(row, worker_args, _worker_ssn_quantile, _worker_qpalm_quantile, _worker_osqp_quantile,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")


def run_svm_table(datasets: list[str], tau_pairs: list[tuple[float, float]], lam: float,
                  tol: float, time_limit: float, max_iter: int,
                  result_dir: Path, solvers: set, cooldown: float = 0.0,
                  name_prefix: str = "") -> None:
    csv_path = result_dir / f"{name_prefix}svm_quantile_svm.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(datasets) * len(tau_pairs)
    done = 0

    def _flush() -> None:
        _write_csv(csv_path, rows, SVM_CSV_FIELDS)

    for dataset in datasets:
        n_samples, n_features = _dataset_shape(dataset)
        for tau1, tau2 in tau_pairs:
            done += 1
            print(f"\n[svm  {done}/{n_total}]  {dataset} tau1={tau1} tau2={tau2}  "
                  f"n={n_samples} d={n_features}", flush=True)
            row = {"dataset": dataset, "n_samples": n_samples, "n_features": n_features,
                   "tau1": tau1, "tau2": tau2, "lam": lam}
            rows.append(row)
            worker_args = (dataset, tau1, tau2, lam)
            run_solvers(row, worker_args, _worker_ssn_svm, _worker_qpalm_svm, _worker_osqp_svm,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",       default=str(HERE.parent))
    parser.add_argument("--tol",        type=float, default=1e-5)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--task",       nargs="+", default=["quantile", "svm"],
                        choices=["quantile", "svm"], metavar="TASK")
    parser.add_argument("--dataset",    nargs="+", default=None,
                        help="Datasets to run (default: all for the selected task(s))")
    parser.add_argument("--alpha",      type=float, nargs="+", default=QUANTILE_ALPHA_DEFAULT,
                        help="Quantile levels to sweep (default: 0.05 0.5 0.9 0.95)")
    parser.add_argument("--lam",        type=float, default=None,
                        help="Regularization strength (default: 1e-2 for both tasks)")
    parser.add_argument("--tau",        type=float, default=QUANTILE_TAU_DEFAULT,
                        help="Quantile task's elastic-net mixing parameter (default: 0.5)")
    parser.add_argument("--solver",     nargs="+", default=["ksp-qp", "qpalm", "osqp"],
                        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER")
    parser.add_argument("--cooldown",   type=float, default=0.0)
    parser.add_argument("--out",        default="")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol         = args.tol
    time_limit  = args.time_limit
    cooldown    = args.cooldown
    max_iter    = 10_000_000_000
    solvers     = set(args.solver)
    name_prefix = f"{args.out}_" if args.out else ""

    if "quantile" in args.task:
        datasets = args.dataset or QUANTILE_DATASETS
        lam = args.lam if args.lam is not None else QUANTILE_LAM_DEFAULT
        run_quantile_table(datasets, args.alpha, lam, args.tau, tol, time_limit, max_iter,
                          result_dir, solvers, cooldown, name_prefix)

    if "svm" in args.task:
        datasets = args.dataset or SVM_DATASETS
        lam = args.lam if args.lam is not None else SVM_LAM_DEFAULT
        run_svm_table(datasets, SVM_TAU_PAIRS_DEFAULT, lam, tol, time_limit, max_iter,
                     result_dir, solvers, cooldown, name_prefix)


if __name__ == "__main__":
    main()
