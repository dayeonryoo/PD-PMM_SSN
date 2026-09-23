"""
Benchmark KSP-QP vs QPALM vs OSQP on DC optimal power flow (DC-OPF) QPs
(see python/opf_generator.py).

DC-OPF's constraint matrix follows the power grid's own graph topology
(a weighted graph Laplacian), so this exercises KSP-QP's
ordering-selection/Schur-complement machinery on a genuinely different
sparsity pattern, at scale (PGLib-OPF cases run from ~10 buses to tens of
thousands).

Data: PGLib-OPF case files are not vendored in this repo -- see
opf_generator.py's module docstring for what to download and place under
data/pglib/. By default this script benchmarks whatever *.m files it finds
there, sorted small-to-large by file size; pass --case explicitly to select
a subset.

Outputs
-------
  results/opf_dcopf.csv
  results/performance_profile_opf.pdf/png             - Dolan-Moré performance profile (run time)
  results/performance_profile_opf_iters.pdf/png       - Dolan-Moré performance profile (iterations)
  results/performance_profile_opf_inner_iters.pdf/png - Dolan-Moré performance profile (inner iterations)

=== HOW TO RUN ===
  python3 benchmark_opf.py
  python3 benchmark_opf.py --case pglib_opf_case14_ieee pglib_opf_case118_ieee
  python3 benchmark_opf.py --solver ksp-qp qpalm --time-limit 120
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
    plot_performance_profile,
    plot_performance_profile_iters,
    plot_performance_profile_inner_iters,
)
import opf_generator as og


def _discover_cases() -> list[str]:
    """Every *.m file under data/pglib/, sorted small-to-large by file size
    (a cheap size proxy for bus/branch count, without parsing every file)."""
    if not og._PGLIB_DATA_DIR.exists():
        return []
    files = sorted(og._PGLIB_DATA_DIR.glob("*.m"), key=lambda p: p.stat().st_size)
    return [p.stem for p in files]


def _case_stats(case_name: str) -> dict:
    case = og.load_pglib_case(case_name)
    bus, branch, gen = case["bus"], case["branch"], case["gen"]
    branch_in_service = branch[branch[:, og.BR_STATUS] > 0]
    n_bus = bus.shape[0]
    n_gen = int(np.sum(gen[:, og.GEN_STATUS] > 0))
    n_lines_limited = int(np.sum(branch_in_service[:, og.RATE_A] > 0.0))
    return dict(
        n_bus=n_bus, n_gen=n_gen,
        n_branch_inservice=branch_in_service.shape[0],
        n_lines_limited=n_lines_limited,
        n=n_bus + n_gen, m=n_bus, l=n_lines_limited,
    )


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _build_pd_data(case_name: str) -> dict:
    case = og.load_pglib_case(case_name)
    return og.generate_dcopf_qp(case)


def _worker_ssn(case_name, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _build_pd_data(case_name)
        result["n_vars"] = pd_data["n"]
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(case_name, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_pd_data(case_name)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(case_name, tol, time_limit, conn):
    result = {}
    try:
        pd_data = _build_pd_data(case_name)
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
    "case", "n_bus", "n_gen", "n_branch_inservice", "n_lines_limited", "n", "m", "l",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


def run_one(result: dict, case_name: str, tol: float, time_limit: float, max_iter: int,
            solvers: set, cooldown: float = 0.0, flush_cb=None) -> dict:
    try:
        stats = _case_stats(case_name)
    except (ValueError, NotImplementedError) as e:
        print(f"  {case_name}  SKIP (unsupported by opf_generator): {e}", flush=True)
        if flush_cb is not None:
            flush_cb()
        return result
    result.update(stats)
    print(f"  {case_name}  n_bus={stats['n_bus']} n_gen={stats['n_gen']} "
          f"n_branch={stats['n_branch_inservice']} n_lines_limited={stats['n_lines_limited']}",
          flush=True)
    worker_args = (case_name,)
    return run_solvers(result, worker_args, _worker_ssn, _worker_qpalm, _worker_osqp,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mp.set_start_method("spawn", force=True)
    default_cases = _discover_cases()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",       default=str(HERE.parent))
    parser.add_argument("--tol",        type=float, default=1e-5)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--case",       nargs="+", default=default_cases, metavar="CASE")
    parser.add_argument("--solver",     nargs="+", default=["ksp-qp", "qpalm", "osqp"],
                        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER")
    parser.add_argument("--cooldown",   type=float, default=0.0)
    parser.add_argument("--out",        default="",
                        help="Prefix for output filename (e.g. '0508' -> '0508_opf_dcopf.csv')")
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

    cases = []
    for name in args.case:
        path = og._PGLIB_DATA_DIR / f"{name}.m"
        if not path.exists():
            print(f"  WARNING: {path} not found -- skipping {name!r}")
            continue
        cases.append(name)

    if not cases:
        sys.exit(
            f"No PGLib-OPF case files found under {og._PGLIB_DATA_DIR}. "
            f"Download plain pglib_opf_case*.m files from "
            f"github.com/power-grid-lib/pglib-opf and place them there "
            f"(see opf_generator.py's module docstring)."
        )

    csv_path = result_dir / f"{name_prefix}opf_dcopf.csv"
    rows: list[dict] = _load_existing_rows(csv_path)

    def _flush() -> None:
        _write_csv(csv_path, rows, CSV_FIELDS)

    for i, case_name in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}]")
        row = {"case": case_name}
        rows.append(row)
        run_one(row, case_name, tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)

    print(f"  Saved: {csv_path}")

    # ---- Performance profiles -------------------------------------------
    label = "DC-OPF (PGLib)"
    out_prefix = result_dir / f"{name_prefix}performance_profile_opf"
    plot_performance_profile(csv_path, out_prefix, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_iters = result_dir / f"{name_prefix}performance_profile_opf_iters"
    plot_performance_profile_iters(csv_path, out_prefix_iters, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_inner = result_dir / f"{name_prefix}performance_profile_opf_inner_iters"
    plot_performance_profile_inner_iters(csv_path, out_prefix_inner, label, tol=tol, time_limit=time_limit, solvers=solvers)


if __name__ == "__main__":
    main()
