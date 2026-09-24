"""
Benchmark KSP-QP vs QPALM vs OSQP on the Netlib LP and Kennington LP test sets.

Outputs (<set> is "netlib", "kennington", or "netlib_kennington" — see --set)
-------
  results/comparison_<set>.csv                    - per-problem timing, iteration counts, and status
  results/performance_profile_<set>.pdf/png       - Dolan-Moré performance profile (run time)
  results/performance_profile_<set>_iters.pdf/png - Dolan-Moré performance profile (iterations)

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Install Python dependencies
-------------------------------------
  pip install qpalm osqp numpy scipy matplotlib pandas

Step 2 - Build the KSP-QP Python binding
------------------------------------------
  All commands are run from the KSP-QP/python/ directory.

  mkdir build
  cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

  This produces ksp_qp_bind.cpython-<tag>-darwin.so in python/.
  You only need to rebuild if the C++ solver source changes.

Step 3 - Run the benchmark
---------------------------
  python3 benchmark_netlib.py

Settings: tol = 1e-6, time limit = 60 s, max iterations = infinity.
        --root:       to change the output directory (default: results/).
        --out:        to change the output file prefix (default: comparison_netlib).
        --set:        to select which test sets to run among netlib, kennington (default: both).
        --solver:     to select which solvers to run among ksp-qp, qpalm, osqp (default: all three).
        --tol:        to change the solver tolerance (default: 1e-6).
        --time-limit: to change the solver time limit in seconds (default: 60).
        --cooldown:   to change the cooldown time in seconds between solver runs (default: 0).
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate and import the pybind11 extension
# ---------------------------------------------------------------------------
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
    _run_isolated,
    QPALM_SOLVED,
    OSQP_SOLVED,
    _write_csv,
    _load_existing_rows,
    plot_performance_profile,
    plot_performance_profile_iters,
    plot_performance_profile_inner_iters,
)

# ---------------------------------------------------------------------------
# Netlib LP problem list  (name → reference optimal objective)
# ---------------------------------------------------------------------------
LPS = {
    "25FV47":    5.5018458883e+03,
    "80BAU3B":   9.8723216072e+05,
    "ADLITTLE":  2.2549496316e+05,
    "AFIRO":    -4.6475314286e+02,
    "AGG":      -3.5991767287e+07,
    "AGG2":     -2.0239252356e+07,
    "AGG3":      1.0312115935e+07,
    "BANDM":    -1.5862801845e+02,
    "BEACONFD":  3.3592485807e+04,
    "BLEND":    -3.0812149846e+01,
    "BNL1":      1.9776292856e+03,
    "BNL2":      1.8112365404e+03,
    "BOEING1":  -3.3521356751e+02,
    "BOEING2":  -3.1501872802e+02,
    "BORE3D":    1.3730803942e+03,
    "BRANDY":    1.5185098965e+03,
    "CAPRI":     2.6900129138e+03,
    "CYCLE":    -5.2263930249e+00,
    "CZPROB":    2.1851966989e+06,
    "D2Q06C":    1.2278423615e+05,
    "D6CUBE":    3.1549166667e+02,
    "DEGEN2":   -1.4351780000e+03,
    "DEGEN3":   -9.8729400000e+02,
    "DFL001":    1.12664e+07,
    "E226":     -1.8751929066e+01,
    "ETAMACRO": -7.5571521774e+02,
    "FFFFF800":  5.5567961165e+05,
    "FINNIS":    1.7279096547e+05,
    "FIT1D":    -9.1463780924e+03,
    "FIT1P":     9.1463780924e+03,
    "FIT2D":    -6.8464293294e+04,
    "FIT2P":     6.8464293232e+04,
    "FORPLAN":  -6.6421873953e+02,
    "GANGES":   -1.0958636356e+05,
    "GFRD-PNC":  6.9022359995e+06,
    "GREENBEA": -7.2462405908e+07,
    "GREENBEB": -4.3021476065e+06,
    "GROW15":   -1.0687094129e+08,
    "GROW22":   -1.6083433648e+08,
    "GROW7":    -4.7787811815e+07,
    "ISRAEL":   -8.9664482186e+05,
    "KB2":      -1.7499001299e+03,
    "LOTFI":    -2.5264706062e+01,
    "MAROS":    -5.8063743701e+04,
    "MAROS-R7":  1.4971851665e+06,
    "MODSZK1":   3.2061972906e+02,
    "NESM":      1.4076073035e+07,
    "PEROLD":   -9.3807580773e+03,
    "PILOT":    -5.5740430007e+02,
    "PILOT.JA": -6.1131344111e+03,
    "PILOT.WE": -2.7201027439e+06,
    "PILOT4":   -2.5811392641e+03,
    "PILOT87":   3.0171072827e+02,
    "PILOTNOV": -4.4972761882e+03,
    "QAP8":      2.0350000000e+02,
    "QAP12":     5.2289435056e+02,
    "QAP15":     1.0409940410e+03,
    "RECIPE":   -2.6661600000e+02,
    "SC105":    -5.2202061212e+01,
    "SC205":    -5.2202061212e+01,
    "SC50A":    -6.4575077059e+01,
    "SC50B":    -7.0000000000e+01,
    "SCAGR25":  -1.4753433061e+07,
    "SCAGR7":   -2.3313892548e+06,
    "SCFXM1":    1.8416759028e+04,
    "SCFXM2":    3.6660261565e+04,
    "SCFXM3":    5.4901254550e+04,
    "SCORPION":  1.8781248227e+03,
    "SCRS8":     9.0429998619e+02,
    "SCSD1":     8.6666666743e+00,
    "SCSD6":     5.0500000078e+01,
    "SCSD8":     9.0499999993e+02,
    "SCTAP1":    1.4122500000e+03,
    "SCTAP2":    1.7248071429e+03,
    "SCTAP3":    1.4240000000e+03,
    "SEBA":      1.5711600000e+04,
    "SHARE1B":  -7.6589318579e+04,
    "SHARE2B":  -4.1573224074e+02,
    "SHELL":     1.2088253460e+09,
    "SHIP04L":   1.7933245380e+06,
    "SHIP04S":   1.7987147004e+06,
    "SHIP08L":   1.9090552114e+06,
    "SHIP08S":   1.9200982105e+06,
    "SHIP12L":   1.4701879193e+06,
    "SHIP12S":   1.4892361344e+06,
    "SIERRA":    1.5394362184e+07,
    "STAIR":    -2.5126695119e+02,
    "STANDATA":  1.2576995000e+03,
    "STANDMPS":  1.4060175000e+03,
    "STOCFOR1": -4.1131976219e+04,
    "STOCFOR2": -3.9024408538e+04,
    "STOCFOR3": -3.9976661576e+04,
    "TRUSS":     4.5881584719e+05,
    "TUFF":      2.9214776509e-01,
    "VTP.BASE":  1.2983146246e+05,
    "WOOD1P":    1.4429024116e+00,
    "WOODW":     1.3044763331e+00,
}

# ---------------------------------------------------------------------------
# Kennington LP problem list  (name → reference optimal objective)
#
# The large Netlib "kennington" family, distributed separately from the main
# Netlib LP set and living in data/kennington/.  Reference objectives are the
# published Kennington optima; they are documentation only — nothing in this
# script reads them.
# ---------------------------------------------------------------------------
KENNINGTON = {
    "CRE-A":   2.9889732e+07,
    "CRE-B":   2.3129640e+07,
    "CRE-C":   2.5275116e+07,
    "CRE-D":   2.4454970e+07,
    "KEN-07": -6.7952044e+08,
    "KEN-11": -6.9723823e+09,
    "KEN-13": -1.0257395e+10,
    "KEN-18": -5.2217025e+10,
    "OSA-07":  5.3572252e+05,
    "OSA-14":  1.1064628e+06,
    "OSA-30":  2.1421399e+06,
    "OSA-60":  4.0440725e+06,
    "PDS-02":  2.8857862e+10,
    "PDS-06":  2.7761038e+10,
    "PDS-10":  2.6727094e+10,
    "PDS-20":  2.3821659e+10,
}

# Test set name → (problem dict, directory under data/, profile-plot label)
TEST_SETS = {
    "netlib":     (LPS,        "netlib",     "Netlib LPs"),
    "kennington": (KENNINGTON, "kennington", "Kennington LPs"),
}

# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn_netlib(mps_path, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_netlib(mps_path, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_netlib(mps_path, tol, time_limit, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(HERE.parent),
        help="Path to the KSP-QP project root (default: parent of this script)",
    )
    parser.add_argument(
        "--tol",        type=float, default=1e-6,  help="Primal-dual tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0, help="Per-problem time limit in seconds (default: 60)"
    )
    parser.add_argument(
        "--out", default="", help="Prefix for output filenames (e.g. '0508' → '0508_comparison_netlib.csv')"
    )
    parser.add_argument(
        "--set", nargs="+", dest="sets", default=["netlib", "kennington"],
        choices=["netlib", "kennington"], metavar="SET",
        help="Test sets to run (default: both). Choices: netlib kennington",
    )
    parser.add_argument(
        "--solver", nargs="+", default=["ksp-qp", "qpalm", "osqp"],
        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER",
        help="Solvers to run (default: all three). Choices: ksp-qp qpalm osqp",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds to sleep between problems to prevent CPU throttling (default: 0)",
    )
    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    solvers = set(args.solver)

    root      = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    # Selected test sets, always in the canonical order of TEST_SETS.
    set_names = [s for s in TEST_SETS if s in set(args.sets)]
    # (name, path) for every problem across the selected sets, in list order.
    problems: list[tuple[str, Path]] = []
    for s in set_names:
        lp_dict, subdir, _ = TEST_SETS[s]
        data_dir = root / "data" / subdir
        problems += [(name, data_dir / f"{name}.mps") for name in lp_dict]
    suffix = "_".join(set_names)
    label  = " + ".join(TEST_SETS[s][2] for s in set_names)

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000   # effectively infinite for KSP-QP

    prefix = f"{args.out}_" if args.out else ""
    csv_path = result_dir / f"{prefix}comparison_{suffix}.csv"
    fieldnames = [
        "name",
        "ssn_solved",   "ssn_status",   "pmm_iter", "ssn_iter",
        "krylov_iter",  "fact",         "smw_count",
        "ssn_obj",   "pmm_tol_achieved",   "ssn_time",
        "qpalm_solved", "qpalm_status", "qpalm_iter", "qpalm_inner_iter", "qpalm_obj", "qpalm_tol_achieved", "qpalm_time",
        "osqp_solved",  "osqp_status",  "osqp_iter",                      "osqp_obj",  "osqp_tol_achieved",  "osqp_time",
    ]

    n_problems = len(problems)
    rows: list[dict] = _load_existing_rows(csv_path)

    def _flush() -> None:
        """Rewrite the CSV from `rows` — called right after every solver finishes."""
        _write_csv(csv_path, rows, fieldnames)

    for idx, (name, path) in enumerate(problems, 1):
        mps_path = str(path)
        if not os.path.exists(mps_path):
            print(f"[{idx:3d}/{n_problems}] SKIP (file not found): {name}")
            continue

        print(f"\n[{idx:3d}/{n_problems}]  {name}")
        row: dict = {"name": name}
        rows.append(row)

        # ---- KSP-QP ------------------------------------------------
        if "ksp-qp" in solvers:
            ssn_out = _run_isolated(_worker_ssn_netlib, (mps_path, tol, time_limit, max_iter))
            if "error" in ssn_out:
                print(f"  KSP-QP : ERROR — {ssn_out['error']}")
                row.update(ssn_status=-99, ssn_solved=0, ssn_time=np.inf,
                           pmm_iter=np.inf, ssn_iter=np.inf,
                           krylov_iter=np.inf, fact=np.inf, smw_count=np.inf,
                           ssn_obj=np.nan, pmm_tol_achieved=np.nan)
            else:
                r = ssn_out["res"]
                row["ssn_status"]       = r["status"]
                row["ssn_solved"]       = int(r["status"] == 0)
                row["ssn_time"]         = r["run_time"]
                row["pmm_iter"]         = r["pmm_iter"]
                row["ssn_iter"]         = r["ssn_iter"]
                row["krylov_iter"]      = r["krylov_iter"]
                row["fact"]             = r["fact"]
                row["smw_count"]        = r["smw_count"]
                row["ssn_obj"]          = r["obj_val"]
                row["pmm_tol_achieved"] = r["pmm_tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == 0 else f"status={r['status']}"
                print(f"  KSP-QP : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"pmm={r['pmm_iter']} ssn={r['ssn_iter']} "
                      f"krylov={r['krylov_iter']} fact={r['fact']} smw={r['smw_count']}  "
                      f"tol={r['pmm_tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- QPALM --------------------------------------------------
        if "qpalm" in solvers:
            qpalm_out = _run_isolated(_worker_qpalm_netlib, (mps_path, tol, time_limit))
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
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- OSQP ---------------------------------------------------
        if "osqp" in solvers:
            osqp_out = _run_isolated(_worker_osqp_netlib, (mps_path, tol, time_limit))
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
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

    print(f"\nResults written to: {csv_path}")

    # ---- Performance profiles -------------------------------------------
    out_prefix = result_dir / f"{prefix}performance_profile_{suffix}"
    plot_performance_profile(csv_path, out_prefix, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_iters = result_dir / f"{prefix}performance_profile_{suffix}_iters"
    plot_performance_profile_iters(csv_path, out_prefix_iters, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_inner = result_dir / f"{prefix}performance_profile_{suffix}_inner_iters"
    plot_performance_profile_inner_iters(csv_path, out_prefix_inner, label, tol=tol, time_limit=time_limit, solvers=solvers)


if __name__ == "__main__":
    main()
