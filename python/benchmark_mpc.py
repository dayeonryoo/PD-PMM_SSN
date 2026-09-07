"""
Benchmark KSP-QP vs QPALM vs OSQP on linear Model Predictive Control (MPC)
QPs (see python/mpc_generator.py).

MPC is a native smooth QP: pure quadratic tracking cost, linear dynamics
equality constraints, simple box bounds. Its constraint matrix is genuinely
sparse.

Scope note (read before interpreting results): KSP-QP has NO warm-start
capability at any level yet. Every solve below is therefore a cold start,
for ALL THREE solvers -- QPALM's own warm_start() API is deliberately never
invoked, even though it exists, so the comparison stays fair. This benchmark
tests whether KSP-QP's native sparse/banded formulation solves efficiently
on a single cold-started QP, repeated across a realistic closed-loop
trajectory -- not whether warm-starting helps.

Trajectory design: two loop modes are available via --loop-mode:
  - "shared" (default): a single canonical state trajectory xbar_0..xbar_{T-1}
    is precomputed ONCE per (system, N, M) config via a reference OSQP solve,
    then every solver is benchmarked against the exact same sequence of T QP
    instances. This is the controlled comparison -- it isolates per-instance
    solve behaviour from any confound of solver-specific control choices
    compounding differently over a trajectory, and is what all figures in
    the paper draft are based on.
  - "own": each solver closes its OWN feedback loop, advancing the state
    with its own computed u_0 at every step. This is the more operationally
    realistic setting (as MPC is actually deployed) and additionally surfaces
    whether a solver's numerical error compounds differently over many steps
    -- a question "shared" mode cannot answer, since it removes that
    variable by construction. Both modes are valid; they answer different
    questions, and neither supersedes the other.

Outputs
-------
  results/mpc_rollout.csv

=== HOW TO RUN ===
  python3 benchmark_mpc.py
  python3 benchmark_mpc.py --system double_integrator --N 10 20 --T 10
  python3 benchmark_mpc.py --solver ksp-qp qpalm --time-limit 60
  python3 benchmark_mpc.py --loop-mode own --system cartpole
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import qpalm
import osqp

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
    QPALM_SOLVED,
    OSQP_SOLVED,
    _run_isolated,
    _write_csv,
    _load_existing_rows,
)
import mpc_generator as mg

DEFAULT_N = [10, 20, 50]
DEFAULT_M = [3, 5, 10, 20, 50, 100, 200, 300]
PLATOON_M_FIXED = 5
PLATOON_N_FIXED = 20
DEFAULT_T = {"double_integrator": 40, "cartpole": 100, "dcmotor": 80, "platoon": 40}


# ---------------------------------------------------------------------------
# Solver wrappers that also return x (needed for the active-set diagnostic;
# benchmark_common.run_qpalm/run_osqp discard it, so these are local variants
# rather than modifications to that shared, already-relied-upon module).
# ---------------------------------------------------------------------------

def _solve_qpalm_x(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0):
    Q_upper, q, C, bmin, bmax, n, m_total = qpalm_data
    data = qpalm.Data(n, m_total)
    data.Q, data.q, data.A, data.bmin, data.bmax = Q_upper, q, C, bmin, bmax

    settings = qpalm.Settings()
    settings.eps_abs, settings.eps_rel = tol, tol
    settings.max_iter = 2_000_000_000
    settings.time_limit = time_limit
    settings.verbose = 0
    settings.scaling = 10

    solver = qpalm.Solver(data, settings)
    solver.solve()
    info = solver.info
    result = {
        "status": int(info.status_val),
        "obj_val": float(info.objective) + obj_const,
        "run_time": float(info.run_time),
        "outer_iter": int(info.iter_out),
        "inner_iter": int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }
    return result, np.asarray(solver.solution.x, dtype=np.float64)


def _solve_osqp_x(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0):
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data
    prob = osqp.OSQP()
    prob.setup(Q_upper, q, C, bmin, bmax, eps_abs=tol, eps_rel=tol,
               max_iter=2_000_000_000, time_limit=time_limit, verbose=False, scaling=10)
    res = prob.solve()
    info = res.info
    result = {
        "status": int(info.status_val),
        "obj_val": float(info.obj_val) + obj_const,
        "run_time": float(info.run_time),
        "outer_iter": int(info.iter),
        "inner_iter": 0,
        "tol_achieved": max(float(info.prim_res), float(info.dual_res)),
    }
    return result, np.asarray(res.x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Trajectory precomputation (once per system/N/M config, via reference OSQP)
# ---------------------------------------------------------------------------

def generate_trajectory(system_name: str, N: int, M: int, T: int) -> list[np.ndarray]:
    sysm = mg.SYSTEM_BUILDERS[system_name](M=M)
    xbar = sysm.xbar0.copy()
    traj = [xbar.copy()]
    for _ in range(T - 1):
        pdd = mg.generate_mpc_qp(sysm, N, xbar)
        qd = kspqp_to_qpalm(pdd)
        _, x = _solve_osqp_x(qd, 1e-8, 30.0, pdd.get("obj_const", 0.0))
        u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
        xbar = mg.step_dynamics(sysm, xbar, u0)
        traj.append(xbar.copy())
    return traj


# ---------------------------------------------------------------------------
# Subprocess rollout workers -- each runs ALL T steps of ONE solver's rollout
# internally (not one subprocess per step: MPC QPs solve in milliseconds, so
# per-step spawn overhead would dominate wall time for no benefit -- crash/
# OOM isolation is still kept, just at (system,N,M,solver) granularity).
#
# Two loop modes (see module docstring):
#   "shared": xbar_source is the precomputed list of T states; every solver
#             sees the identical sequence, regardless of its own u_0.
#   "own":    xbar_source is the single initial xbar0; each solver advances
#             the state itself, using its own computed u_0 at every step.
# ---------------------------------------------------------------------------

_PREFIX = {"ksp-qp": "ssn", "qpalm": "qpalm", "osqp": "osqp"}


def _solve_one_step(solver_name, pdd, sysm, tol, time_limit, max_iter):
    """Solve one QP instance with the given solver. Returns (row_updates, u0).

    All per-solver-specific fields (including xbar_norm) are prefixed by
    solver -- required for --loop-mode own, where each solver compounds a
    DIFFERENT trajectory and a shared column would be silently overwritten by
    whichever solver runs last at a given step.
    """
    prefix = _PREFIX[solver_name]
    if solver_name == "ksp-qp":
        r = ksp_qp_bind.solve_from_data(pdd, tol, max_iter, time_limit)
        x = np.asarray(r["x"], dtype=np.float64)
        row = dict(ssn_status=r["status"], ssn_solved=int(r["status"] == 0),
                   pmm_iter=r["pmm_iter"], ssn_iter=r["ssn_iter"],
                   krylov_iter=r["krylov_iter"], fact=r["fact"], smw_count=r["smw_count"],
                   pmm_tol_achieved=r["pmm_tol_achieved"], ssn_time=r["run_time"],
                   ssn_obj=r["obj_val"])
    elif solver_name == "qpalm":
        qd = kspqp_to_qpalm(pdd)
        r, x = _solve_qpalm_x(qd, tol, time_limit, pdd.get("obj_const", 0.0))
        row = dict(qpalm_status=r["status"], qpalm_solved=int(r["status"] == QPALM_SOLVED),
                   qpalm_time=r["run_time"], qpalm_iter=r["outer_iter"],
                   qpalm_inner_iter=r["inner_iter"], qpalm_obj=r["obj_val"],
                   qpalm_tol_achieved=r["tol_achieved"])
    elif solver_name == "osqp":
        qd = kspqp_to_qpalm(pdd)
        r, x = _solve_osqp_x(qd, tol, time_limit, pdd.get("obj_const", 0.0))
        row = dict(osqp_status=r["status"], osqp_solved=int(r["status"] == OSQP_SOLVED),
                   osqp_time=r["run_time"], osqp_iter=r["outer_iter"], osqp_obj=r["obj_val"],
                   osqp_tol_achieved=r["tol_achieved"])
    else:
        raise ValueError(f"Unknown solver_name: {solver_name}")

    u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
    return row, u0


def _worker_rollout(solver_name, system_name, N, M, loop_mode, xbar_source, T,
                    tol, time_limit, max_iter, conn):
    rows = []
    try:
        sysm = mg.SYSTEM_BUILDERS[system_name](M=M)

        if loop_mode == "shared":
            xbar_traj = xbar_source
        else:
            xbar_traj = None
            xbar = xbar_source.copy()

        prefix = _PREFIX[solver_name]
        for t in range(T):
            xbar = xbar_traj[t] if loop_mode == "shared" else xbar
            pdd = mg.generate_mpc_qp(sysm, N, xbar)
            row = {"step": t, f"{prefix}_xbar_norm": float(np.linalg.norm(xbar))}
            updates, u0 = _solve_one_step(solver_name, pdd, sysm, tol, time_limit, max_iter)
            row.update(updates)
            rows.append(row)
            if loop_mode == "own" and t < T - 1:
                xbar = mg.step_dynamics(sysm, xbar, u0)
        conn.send({"rows": rows})
    except Exception as e:
        conn.send({"error": str(e)})
    conn.close()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "system", "N", "M", "loop_mode", "n_x", "n_u", "n_z", "m_eq", "l_ineq", "step",
    "ssn_xbar_norm", "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_xbar_norm", "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter",
    "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_xbar_norm", "osqp_status", "osqp_solved", "osqp_iter", "osqp_tol_achieved", "osqp_time", "osqp_obj",
]


def run_config(system_name: str, N: int, M: int, T: int, tol: float, time_limit: float,
              max_iter: int, solvers: set, rows: list, csv_path: Path,
              cooldown: float = 0.0, loop_mode: str = "shared") -> None:
    sysm = mg.SYSTEM_BUILDERS[system_name](M=M)
    print(f"  {system_name} N={N} M={M} n_x={sysm.n_x} n_u={sysm.n_u} T={T} loop_mode={loop_mode}",
          flush=True)

    if loop_mode == "shared":
        xbar_source = generate_trajectory(system_name, N, M, T)
    else:
        xbar_source = sysm.xbar0

    # step -> merged row across solvers
    merged = {t: {"system": system_name, "N": N, "M": M, "loop_mode": loop_mode,
                  "n_x": sysm.n_x, "n_u": sysm.n_u,
                  "n_z": (N + 1) * sysm.n_x + N * sysm.n_u, "m_eq": (N + 1) * sysm.n_x,
                  "l_ineq": 0}
              for t in range(T)}

    def _flush():
        _write_csv(csv_path, rows, CSV_FIELDS)

    for solver_name in ["ksp-qp", "qpalm", "osqp"]:
        if solver_name not in solvers:
            continue
        out = _run_isolated(_worker_rollout,
                            (solver_name, system_name, N, M, loop_mode, xbar_source, T,
                             tol, time_limit, max_iter))
        if "error" in out:
            print(f"    {solver_name:8s} ERROR — {out['error']}")
            continue
        n_solved = 0
        for r in out["rows"]:
            t = r["step"]
            merged[t].update(r)
            if solver_name == "ksp-qp":
                n_solved += int(r.get("ssn_solved", 0))
            elif solver_name == "qpalm":
                n_solved += int(r.get("qpalm_solved", 0))
            else:
                n_solved += int(r.get("osqp_solved", 0))
        total_time = sum(r.get(f"{'ssn' if solver_name == 'ksp-qp' else solver_name}_time", 0.0)
                          for r in out["rows"])
        print(f"    {solver_name:8s} solved={n_solved}/{T}  total_time={total_time:.2f}s")

    for t in range(T):
        rows.append(merged[t])
    _flush()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(HERE.parent))
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--system", nargs="+",
                        choices=["double_integrator", "cartpole", "dcmotor", "platoon"],
                        default=["double_integrator", "cartpole", "dcmotor", "platoon"])
    parser.add_argument("--N", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--M", type=int, nargs="+", default=DEFAULT_M,
                        help="Platoon vehicle counts (ignored for other systems)")
    parser.add_argument("--platoon-M-fixed", type=int, default=PLATOON_M_FIXED)
    parser.add_argument("--platoon-N-fixed", type=int, default=PLATOON_N_FIXED)
    parser.add_argument("--T", type=int, default=None,
                        help="Rollout length (default: per-system, see DEFAULT_T)")
    parser.add_argument("--solver", nargs="+", default=["ksp-qp", "qpalm", "osqp"],
                        choices=["ksp-qp", "qpalm", "osqp"])
    parser.add_argument("--loop-mode", default="shared", choices=["shared", "own"],
                        help="'shared' (default): every solver sees the identical "
                             "precomputed state trajectory -- the controlled comparison. "
                             "'own': each solver closes its own feedback loop using its "
                             "own computed u_0 -- the operationally realistic comparison. "
                             "See module docstring for the distinction.")
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol, time_limit, cooldown = args.tol, args.time_limit, args.cooldown
    max_iter = 10_000_000_000
    solvers = set(args.solver)
    name_prefix = f"{args.out}_" if args.out else ""
    csv_path = result_dir / f"{name_prefix}mpc_rollout.csv"
    rows: list[dict] = _load_existing_rows(csv_path)

    configs = []
    for system_name in args.system:
        T = args.T if args.T is not None else DEFAULT_T[system_name]
        if system_name == "platoon":
            for N in args.N:
                configs.append(("platoon", N, args.platoon_M_fixed, T))
            for M in args.M:
                if M == args.platoon_M_fixed:
                    continue
                configs.append(("platoon", args.platoon_N_fixed, M, T))
        else:
            for N in args.N:
                configs.append((system_name, N, 1, T))

    for i, (system_name, N, M, T) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        run_config(system_name, N, M, T, tol, time_limit, max_iter, solvers, rows, csv_path,
                  cooldown, args.loop_mode)

    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
