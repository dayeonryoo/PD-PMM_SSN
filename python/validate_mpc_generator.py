"""
Validation suite for mpc_generator.py. Run directly: python3 validate_mpc_generator.py

Checks, in order:
  1. Smoke tests: dimensions, banded-sparsity bound, platoon M=1 shape reduction,
     x_0 block bounds always +-inf.
  2. Objective cross-check: KSP-QP vs a raw OSQP solve, on the (feasible,
     near-bound) default xbar0 for each system.
  3. LQR exact ground-truth check: with the terminal weight P set to the exact
     DARE solution, the finite-horizon solution equals the infinite-horizon LQR
     solution EXACTLY for any N>=1 whenever no inequality constraint is active
     (P is then a fixed point of the backward Riccati recursion) -- not just
     asymptotically as N->inf. Checked two ways:
       - obj_val == xbar'*P*xbar + obj_const (checkable for ALL solvers,
         including KSP-QP, since it only needs obj_val)
       - u_0* == -K @ xbar (checkable directly against KSP-QP's OWN solution
         vector now that solve_from_data returns x, not just against a proxy
         solver's x as before)
  4. Feasibility check across the full (system, N, M) test matrix used by
     benchmark_mpc.py's default sweep.
"""

import sys
import numpy as np
import scipy.sparse as sp
import osqp
import ksp_qp_bind
import mpc_generator as mg

TOL = 1e-5
FAILURES = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def solve_raw_osqp(pdd, tol=1e-9):
    n = pdd["n"]
    Q = sp.csc_matrix((pdd["Q_data"], pdd["Q_indices"], pdd["Q_indptr"]), shape=pdd["Q_shape"])
    A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
    B = sp.csc_matrix((pdd["B_data"], pdd["B_indices"], pdd["B_indptr"]), shape=pdd["B_shape"])
    INF = 1e30
    C = sp.vstack([A, B, sp.eye(n)], format="csc")
    bmin = np.concatenate([pdd["b"], np.clip(pdd["lw"], -INF, INF), np.clip(pdd["lx"], -INF, INF)])
    bmax = np.concatenate([pdd["b"], np.clip(pdd["uw"], -INF, INF), np.clip(pdd["ux"], -INF, INF)])
    prob = osqp.OSQP()
    prob.setup(Q, pdd["c"], C, bmin, bmax, eps_abs=tol, eps_rel=tol, max_iter=200_000, verbose=False)
    res = prob.solve()
    return res.x, res.info.obj_val + pdd["obj_const"], res.info.status


def smoke_tests() -> None:
    print("\n=== 1. Smoke tests ===")
    for name, builder in mg.SYSTEM_BUILDERS.items():
        sysm = builder()
        N = 10
        pdd = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
        n_x, n_u = sysm.n_x, sysm.n_u

        _check(f"{name}: n matches (N+1)n_x+Nn_u", pdd["n"] == (N + 1) * n_x + N * n_u)
        _check(f"{name}: m matches (N+1)n_x", pdd["m"] == (N + 1) * n_x)
        _check(f"{name}: l == 0 (no B needed)", pdd["l"] == 0)

        A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
        max_row_nnz = int(np.diff(A.tocsr().indptr).max())
        _check(f"{name}: banded (max row nnz <= 2n_x+n_u)", max_row_nnz <= 2 * n_x + n_u,
               f"max_row_nnz={max_row_nnz}, limit={2 * n_x + n_u}")

        lx, ux = pdd["lx"], pdd["ux"]
        _check(f"{name}: x_0 block is +-inf (never finite)",
               bool(np.all(np.isinf(lx[:n_x])) and np.all(np.isinf(ux[:n_x]))))

    plat1 = mg.platoon_system(1)
    di = mg.double_integrator_system()
    _check("platoon M=1 shape matches double_integrator", plat1.n_x == di.n_x and plat1.n_u == di.n_u,
           f"platoon=({plat1.n_x},{plat1.n_u}) double_integrator=({di.n_x},{di.n_u})")


def objective_cross_check() -> None:
    print("\n=== 2. Objective cross-check: KSP-QP vs raw OSQP ===")
    for name, builder in mg.SYSTEM_BUILDERS.items():
        sysm = builder()
        pdd = mg.generate_mpc_qp(sysm, N=10, xbar=sysm.xbar0)
        res = ksp_qp_bind.solve_from_data(pdd, 1e-8, 1_000_000, 30.0)
        _, obj_osqp, status_osqp = solve_raw_osqp(pdd)
        rel_diff = abs(res["obj_val"] - obj_osqp) / max(1.0, abs(obj_osqp))
        _check(f"{name}: KSP-QP obj matches OSQP (rel_diff<1e-5)", rel_diff < 1e-5,
               f"ksp={res['obj_val']:.6f} osqp={obj_osqp:.6f} ({status_osqp}) rel_diff={rel_diff:.2e}")


def lqr_ground_truth_check() -> None:
    print("\n=== 3. LQR exact ground-truth check (relaxed xbar, N=10) ===")
    for name, builder in mg.SYSTEM_BUILDERS.items():
        sysm = builder()
        n_x, n_u = sysm.n_x, sysm.n_u
        xbar_small = sysm.xbar0 * 1e-3
        N = 10
        pdd = mg.generate_mpc_qp(sysm, N, xbar_small)
        res = ksp_qp_bind.solve_from_data(pdd, 1e-9, 1_000_000, 30.0)

        Ad, Bd = mg.discretize(sysm.Ac, sysm.Bc, sysm.Ts)
        P = mg.terminal_cost(Ad, Bd, sysm.Q, sysm.R)
        K = mg.lqr_gain(Ad, Bd, P, sysm.R)

        obj_lqr = xbar_small @ P @ xbar_small + pdd["obj_const"]
        _check(f"{name}: obj_val == xbar'Pxbar (all-solver check)",
               abs(res["obj_val"] - obj_lqr) < 1e-6,
               f"ksp={res['obj_val']:.10f} lqr={obj_lqr:.10f} diff={abs(res['obj_val'] - obj_lqr):.2e}")

        u0_ksp = res["x"][n_x:n_x + n_u]
        u0_lqr = -K @ xbar_small
        _check(f"{name}: u_0* == -K@xbar, checked against KSP-QP's OWN x",
               np.max(np.abs(u0_ksp - u0_lqr)) < 1e-5,
               f"diff={np.max(np.abs(u0_ksp - u0_lqr)):.2e}")


def feasibility_check() -> None:
    print("\n=== 4. Feasibility across the default benchmark_mpc.py test matrix ===")
    for name, builder in mg.SYSTEM_BUILDERS.items():
        if name != "platoon":
            sysm = builder()
            for N in [10, 20, 50]:
                pdd = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
                _, _, status = solve_raw_osqp(pdd, tol=1e-6)
                _check(f"{name} N={N}: feasible", status == "solved", status)
    for M in [1, 5, 20, 50]:
        sysm = mg.platoon_system(M)
        for N in [10, 20, 50]:
            pdd = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
            _, _, status = solve_raw_osqp(pdd, tol=1e-6)
            _check(f"platoon M={M} N={N}: feasible", status == "solved", status)


def main() -> None:
    smoke_tests()
    objective_cross_check()
    lqr_ground_truth_check()
    feasibility_check()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
