"""
Benchmark KSP-QP vs QPALM vs OSQP on CVaR / MAsD portfolio-selection QPs
(see python/portfolio_generator.py), extending the comparison in
Pougkakiotis, Gondzio & Kalogerias (J. Sci. Comput. 2025), Section 5.1, to
include QPALM (the paper only compared against OSQP for these instances).

Unlike the PDE-constrained benchmarks (benchmark_pde.py), where the active
set saturates the full problem size, CVaR/MAsD portfolio QPs have an
active-set size that stays a bounded fraction of the number of scenarios
(roughly alpha * l for CVaR, ~l/2 for MAsD) regardless of problem size --
this is the property that should favour KSP-QP's active-set/column-dropping
preconditioner over ADMM-style solvers (QPALM, OSQP).

Data: the 6 real Bruni et al. datasets (DowJones, NASDAQ100, FTSE100,
FF49Industries, SP500, NASDAQComp) converted to CSV under data/portfolio/
(see portfolio_generator.load_bruni_dataset), plus a synthetic "xlarge"
entry (see portfolio_generator.synthetic_returns) for scale-up beyond the
largest real dataset (NASDAQComp: 1203 assets / 685 scenarios).

Outputs
-------
  results/portfolio_cvar.csv
  results/portfolio_masd.csv

=== HOW TO RUN ===
  python3 benchmark_portfolio.py
  python3 benchmark_portfolio.py --dataset dowjones nasdaq100 --risk cvar --alpha 0.05 0.10
  python3 benchmark_portfolio.py --solver ksp-qp qpalm --time-limit 120
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
import portfolio_generator as pg

# ---------------------------------------------------------------------------
# Dataset registry
#
# kind="bruni": one of the 6 real-world datasets from Bruni, Cesarone,
#               Scozzari & Tardella (2016), Data in Brief 8, 858-862 -- the
#               same datasets used in Table 1 of Pougkakiotis, Gondzio &
#               Kalogerias (2025), Section 5.1. Loaded from data/portfolio/
#               (see portfolio_generator.load_bruni_dataset); r is each
#               dataset's own market-index mean weekly return (uniform-
#               allocation benchmark for FF49Industries, per the paper).
# kind="synthetic": (n_assets, l_scenarios) generated via pg.synthetic_returns,
#               for scale-up beyond the largest real dataset (NASDAQComp,
#               1203 assets / 685 scenarios).
# ---------------------------------------------------------------------------
DATASETS = {
    "dowjones":       dict(kind="bruni", name="DowJones",       n_assets=28,   l_scenarios=1_363),
    "nasdaq100":      dict(kind="bruni", name="NASDAQ100",      n_assets=82,   l_scenarios=596),
    "ftse100":        dict(kind="bruni", name="FTSE100",        n_assets=83,   l_scenarios=717),
    "ff49industries": dict(kind="bruni", name="FF49Industries", n_assets=49,   l_scenarios=2_325),
    "sp500":          dict(kind="bruni", name="SP500",          n_assets=442,  l_scenarios=595),
    "nasdaqcomp":     dict(kind="bruni", name="NASDAQComp",     n_assets=1203, l_scenarios=685),
    "xlarge":         dict(kind="synthetic", n_assets=1500, l_scenarios=40_000),
}

ALPHA_DEFAULT = [0.05, 0.10, 0.15]
SEED = 0


def _load_returns(dataset: str) -> tuple[np.ndarray, float]:
    """Return (returns, r_benchmark) for the named dataset."""
    spec = DATASETS[dataset]
    if spec["kind"] == "bruni":
        return pg.load_bruni_dataset(spec["name"])
    elif spec["kind"] == "synthetic":
        returns = pg.synthetic_returns(spec["n_assets"], spec["l_scenarios"], seed=SEED)
        return returns, 0.0
    raise ValueError(f"Unknown dataset kind: {spec['kind']}")


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _build_pd_data(dataset, risk, alpha):
    returns, r = _load_returns(dataset)
    if risk == "cvar":
        return pg.generate_cvar_qp(returns, alpha, r=r)
    return pg.generate_masd_qp(returns, r=r)


def _worker_ssn(dataset, risk, alpha, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _build_pd_data(dataset, risk, alpha)
        result["n_vars"] = pd_data["n"]
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(dataset, risk, alpha, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_pd_data(dataset, risk, alpha)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(dataset, risk, alpha, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_pd_data(dataset, risk, alpha)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# CSV fields
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "risk", "n_assets", "l_scenarios", "alpha",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


def run_one(result: dict, dataset: str, risk: str, alpha: float,
            tol: float, time_limit: float, max_iter: int, solvers: set,
            cooldown: float = 0.0, flush_cb=None) -> dict:
    spec = DATASETS[dataset]
    print(f"  {dataset} ({risk}"
          f"{f', alpha={alpha}' if risk == 'cvar' else ''})  "
          f"n_assets={spec.get('n_assets', '?')} l={spec.get('l_scenarios', '?')}", flush=True)
    worker_args = (dataset, risk, alpha)
    return run_solvers(result, worker_args, _worker_ssn, _worker_qpalm, _worker_osqp,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb)


def _run_table(risk: str, datasets: list[str], alphas: list[float],
              tol: float, time_limit: float, max_iter: int,
              result_dir: Path, solvers: set, cooldown: float = 0.0,
              name_prefix: str = "") -> None:
    csv_path = result_dir / f"{name_prefix}portfolio_{risk}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)

    alpha_list = alphas if risk == "cvar" else [None]
    n_total = len(datasets) * len(alpha_list)
    done = 0

    def _flush() -> None:
        _write_csv(csv_path, rows, CSV_FIELDS)

    for dataset in datasets:
        spec = DATASETS[dataset]
        for alpha in alpha_list:
            done += 1
            print(f"\n[{risk}  {done}/{n_total}]")
            row = {
                "dataset": dataset, "risk": risk,
                "n_assets": spec.get("n_assets"), "l_scenarios": spec.get("l_scenarios"),
                "alpha": alpha,
            }
            rows.append(row)
            run_one(row, dataset, risk, alpha, tol, time_limit, max_iter, solvers,
                    cooldown, flush_cb=_flush)
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
    parser.add_argument("--dataset",    nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()), metavar="DATASET")
    parser.add_argument("--risk",       nargs="+", default=["cvar", "masd"],
                        choices=["cvar", "masd"], metavar="RISK")
    parser.add_argument("--alpha",      type=float, nargs="+", default=ALPHA_DEFAULT,
                        help="CVaR confidence levels to sweep (default: 0.05 0.10 0.15)")
    parser.add_argument("--solver",     nargs="+", default=["ksp-qp", "qpalm", "osqp"],
                        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER")
    parser.add_argument("--cooldown",   type=float, default=0.0)
    parser.add_argument("--out",        default="",
                        help="Prefix for output filenames (e.g. '0508' -> '0508_portfolio_cvar.csv')")
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

    if "cvar" in args.risk:
        _run_table("cvar", args.dataset, args.alpha, tol, time_limit, max_iter,
                  result_dir, solvers, cooldown, name_prefix)

    if "masd" in args.risk:
        _run_table("masd", args.dataset, args.alpha, tol, time_limit, max_iter,
                  result_dir, solvers, cooldown, name_prefix)


if __name__ == "__main__":
    main()
