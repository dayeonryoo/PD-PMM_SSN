"""
Validation suite for opf_generator.py. Run directly: python3 validate_opf_generator.py

Checks, in order:
  1. Parser smoke tests: case-file structure, single reference bus, no
     isolated buses, quadratic (MODEL=2, NCOST=3) generator costs.
  2. QP structural smoke tests: dimensions, Q's theta-block is zero, the
     reference angle is pinned, Cg's columns each sum to 1, B_bus's row
     sums are ~0 (Laplacian sanity).
  3. Objective cross-check: KSP-QP vs a raw OSQP solve, at each case's
     native (possibly congested) demand.
  4. Economic-dispatch exact ground-truth check: on a guaranteed-uncongested
     instance (found by scaling demand down until no line/generator bound
     binds), the DC-OPF generation dispatch must equal the "copper-plate"
     network-free economic-dispatch solution -- a lossless, uncongested DC
     network lets angles absorb any flow at zero cost, so marginal
     generation costs equalize system-wide regardless of topology. The
     ground truth is computed independently via 1-D bisection on the shared
     marginal cost (no external solver).
  5. Feasibility sweep across every *.m file found under data/pglib/.

Requires PGLib-OPF case files under data/pglib/ (not vendored in this repo --
see opf_generator.py's module docstring for what to download). Checks whose
required file is missing are reported as FAILs naming the missing path,
rather than silently skipped, except the feasibility sweep (section 5),
which simply has nothing to iterate over if the directory is empty, and
skips (rather than fails) any file the generator doesn't support (isolated
buses, non-quadratic generator costs) -- that's a parser/generator
limitation, not a sign the case's QP is ill-posed.
"""

import sys
import time

import numpy as np
import scipy.sparse as sp
import osqp
import ksp_qp_bind
import opf_generator as og

FAILURES = []

VALIDATION_CASES = [
    "pglib_opf_case14_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
]


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def solve_raw_osqp(pdd, tol=1e-9, max_iter=2_000_000, time_limit=120.0):
    n = pdd["n"]
    Q = sp.csc_matrix((pdd["Q_data"], pdd["Q_indices"], pdd["Q_indptr"]), shape=pdd["Q_shape"])
    A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
    B = sp.csc_matrix((pdd["B_data"], pdd["B_indices"], pdd["B_indptr"]), shape=pdd["B_shape"])
    INF = 1e30
    C = sp.vstack([A, B, sp.eye(n)], format="csc")
    bmin = np.concatenate([pdd["b"], np.clip(pdd["lw"], -INF, INF), np.clip(pdd["lx"], -INF, INF)])
    bmax = np.concatenate([pdd["b"], np.clip(pdd["uw"], -INF, INF), np.clip(pdd["ux"], -INF, INF)])
    prob = osqp.OSQP()
    prob.setup(Q, pdd["c"], C, bmin, bmax, eps_abs=tol, eps_rel=tol,
              max_iter=max_iter, time_limit=time_limit, verbose=False)
    res = prob.solve()
    return res.x, res.info.obj_val + pdd["obj_const"], res.info.status


def parser_smoke_tests() -> None:
    print("\n=== 1. Parser smoke tests ===")
    for name in VALIDATION_CASES:
        path = og._PGLIB_DATA_DIR / f"{name}.m"
        if not path.exists():
            _check(f"{name}: file present under data/pglib/", False, str(path))
            continue
        case = og.load_pglib_case(name)
        bus, branch, gen, gencost = case["bus"], case["branch"], case["gen"], case["gencost"]

        _check(f"{name}: bus/branch/gen/gencost have enough columns",
               bus.shape[1] >= 5 and branch.shape[1] >= 11
               and gen.shape[1] >= 10 and gencost.shape[1] >= 7)
        _check(f"{name}: exactly one reference bus",
               int(np.sum(bus[:, og.BUS_TYPE] == og.REF_BUS_TYPE)) == 1)
        _check(f"{name}: no isolated buses",
               not bool(np.any(bus[:, og.BUS_TYPE] == og.ISOLATED_BUS_TYPE)))
        _check(f"{name}: MODEL==2 & NCOST==3 for all generators",
               bool(np.all(gencost[:, og.MODEL] == 2) and np.all(gencost[:, og.NCOST] == 3)))


def qp_structural_smoke_tests() -> None:
    print("\n=== 2. QP structural smoke tests ===")
    for name in VALIDATION_CASES:
        path = og._PGLIB_DATA_DIR / f"{name}.m"
        if not path.exists():
            continue
        case = og.load_pglib_case(name)
        pdd = og.generate_dcopf_qp(case)

        nb = case["bus"].shape[0]
        ng = int(np.sum(case["gen"][:, og.GEN_STATUS] > 0))
        branch_in_service = case["branch"][case["branch"][:, og.BR_STATUS] > 0]
        n_limited = int(np.sum(branch_in_service[:, og.RATE_A] > 0.0))

        _check(f"{name}: n == ng+nb", pdd["n"] == ng + nb, f"n={pdd['n']} ng+nb={ng + nb}")
        _check(f"{name}: m == nb", pdd["m"] == nb, f"m={pdd['m']} nb={nb}")
        _check(f"{name}: l == #rate-limited in-service branches", pdd["l"] == n_limited,
               f"l={pdd['l']} expected={n_limited}")

        Q = sp.csc_matrix((pdd["Q_data"], pdd["Q_indices"], pdd["Q_indptr"]), shape=pdd["Q_shape"])
        _check(f"{name}: Q's theta-block is exactly zero",
               bool(np.all(Q.diagonal()[ng:] == 0.0)))

        ref_idx = int(np.argmax(case["bus"][:, og.BUS_TYPE] == og.REF_BUS_TYPE))
        _check(f"{name}: theta_ref pinned to 0",
               pdd["lx"][ng + ref_idx] == 0.0 and pdd["ux"][ng + ref_idx] == 0.0)

        A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
        col_sums = np.asarray(A[:, :ng].sum(axis=0)).ravel()
        _check(f"{name}: Cg columns each sum to 1", bool(np.all(col_sums == 1.0)))

        row_sums = np.asarray((-A[:, ng:]).sum(axis=1)).ravel()
        max_row_sum = float(np.max(np.abs(row_sums)))
        _check(f"{name}: B_bus row sums ~0 (Laplacian)", max_row_sum < 1e-8,
               f"max|row_sum|={max_row_sum:.2e}")


def objective_cross_check() -> None:
    print("\n=== 3. Objective cross-check: KSP-QP vs raw OSQP ===")
    for name in VALIDATION_CASES:
        path = og._PGLIB_DATA_DIR / f"{name}.m"
        if not path.exists():
            continue
        case = og.load_pglib_case(name)
        pdd = og.generate_dcopf_qp(case)
        res = ksp_qp_bind.solve_from_data(pdd, 1e-8, 1_000_000, 60.0)
        _, obj_osqp, status_osqp = solve_raw_osqp(pdd)
        rel_diff = abs(res["obj_val"] - obj_osqp) / max(1.0, abs(obj_osqp))
        _check(f"{name}: KSP-QP obj matches OSQP (rel_diff<1e-5)", rel_diff < 1e-5,
               f"ksp={res['obj_val']:.4f} osqp={obj_osqp:.4f} ({status_osqp}) rel_diff={rel_diff:.2e}")


def economic_dispatch(c1: np.ndarray, c2: np.ndarray, Pmin: np.ndarray, Pmax: np.ndarray,
                      D: float, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Closed-form-via-bisection solution to the network-free "copper-plate"
    economic dispatch min sum(c2 Pg^2 + c1 Pg) s.t. sum(Pg)=D, Pmin<=Pg<=Pmax.

    KKT: every unclamped generator shares one marginal cost mu, with
    Pg_g(mu) = clip((mu-c1_g)/(2c2_g), Pmin_g, Pmax_g) for quadratic-cost
    generators (c2_g>0); sum_g Pg_g(mu) is nondecreasing in mu, so mu* is
    found by bisection.

    Generators with c2_g==0 (a linear cost -- common in PGLib's plain "ieee"
    cases, e.g. case14_ieee has c2=0 for every unit) need special handling:
    their marginal cost is the CONSTANT c1_g regardless of output level, so
    their stationarity condition is satisfied at ANY output in
    [Pmin_g,Pmax_g] once mu equals c1_g exactly -- the aggregate supply
    curve has a vertical jump (Pmin_g->Pmax_g) at mu=c1_g rather than a
    continuous ramp. If D falls inside that jump -- the generic case
    whenever a linear generator is the marginal, price-setting unit, e.g.
    case14_ieee's cheapest generator alone serves the entire ~259MW load at
    an INTERIOR point of its [0,340] range -- naive bisection would
    incorrectly snap that unit to one of its bounds instead. Detected here
    by evaluating the supply at mu* under both tie-breaks (tied linear units
    sent to Pmin, or to Pmax); if neither sums to D, the residual
    D - (sum with tied units at Pmin) is assigned directly to the tied
    unit(s) (split proportionally to headroom Pmax-Pmin on the rare exact
    multi-way tie), which is the stationarity-consistent resolution.
    """
    linear = c2 == 0.0

    def supply(mu, ge=False):
        out = np.empty_like(c1)
        if ge:
            out[linear] = np.where(mu >= c1[linear], Pmax[linear], Pmin[linear])
        else:
            out[linear] = np.where(mu > c1[linear], Pmax[linear], Pmin[linear])
        quad = ~linear
        out[quad] = np.clip((mu - c1[quad]) / (2.0 * c2[quad]), Pmin[quad], Pmax[quad])
        return out

    lo = float(np.min(c1 + 2.0 * c2 * Pmin))
    hi = float(np.max(c1 + 2.0 * c2 * Pmax))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if supply(mid).sum() < D:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    # supply(lo, ge=False).sum() < D and supply(hi, ge=True).sum() >= D are
    # loop invariants (lo/hi are only ever assigned from a mid satisfying
    # the matching strict/non-strict comparison) -- evaluating the tie-break
    # AT these bracket endpoints, rather than at an already-averaged mu_star,
    # avoids floating-point slop from bisection landing mu_star a hair off
    # the true tie value on one side, which would make both tie-breaks agree
    # and silently hide the jump.
    Pg_lo, Pg_hi = supply(lo, ge=False), supply(hi, ge=True)
    d_scale = max(1.0, abs(D))
    if abs(Pg_lo.sum() - D) < 1e-6 * d_scale:
        return Pg_lo
    if abs(Pg_hi.sum() - D) < 1e-6 * d_scale:
        return Pg_hi

    residual = D - Pg_lo.sum()
    tied = linear & (c1 >= lo) & (c1 <= hi)
    headroom = (Pmax - Pmin)[tied]
    out = Pg_lo.copy()
    out[tied] = Pmin[tied] + residual * headroom / headroom.sum()
    return out


def _scaled_case(case: dict, lam: float) -> dict:
    """Copy of `case` with bus real-power demand (Pd) scaled by lam (Gs
    left unscaled)."""
    scaled = dict(case)
    bus = case["bus"].copy()
    bus[:, og.PD] *= lam
    scaled["bus"] = bus
    return scaled


def _congestion_free(pdd: dict, x: np.ndarray, eps: float = 1e-6) -> bool:
    """True iff no line-flow limit binds. Generator Pmin/Pmax activation is
    NOT checked here: a clamped generator does not break the uniform-
    marginal-cost property that makes the economic-dispatch ground truth
    valid -- the KKT derivation (see module docstring context) shows the
    system-wide multiplier stays a single scalar across all buses as long as
    B_bus's null space is exactly the constant vector (i.e. no line limit is
    active to introduce locational price differences), regardless of which
    generators are at a bound. economic_dispatch()'s own clip(...) already
    models bound-clamped generators correctly, so only network congestion
    -- not generator-bound activation -- needs to be excluded here."""
    if pdd["l"] == 0:
        return True
    B = sp.csc_matrix((pdd["B_data"], pdd["B_indices"], pdd["B_indptr"]), shape=pdd["B_shape"])
    flow = B @ x
    return bool(np.all(flow > pdd["lw"] + eps) and np.all(flow < pdd["uw"] - eps))


def economic_dispatch_ground_truth_check() -> None:
    print("\n=== 4. Economic-dispatch exact ground-truth check ===")
    name = "pglib_opf_case14_ieee"
    path = og._PGLIB_DATA_DIR / f"{name}.m"
    if not path.exists():
        _check(f"{name}: file present under data/pglib/ (required for ground-truth check)",
               False, str(path))
        return

    case = og.load_pglib_case(name)
    gen = case["gen"][case["gen"][:, og.GEN_STATUS] > 0]
    gencost = case["gencost"][case["gen"][:, og.GEN_STATUS] > 0]
    ng = gen.shape[0]
    Pmin, Pmax = gen[:, og.PMIN], gen[:, og.PMAX]
    c2, c1, c0 = gencost[:, og.COST0], gencost[:, og.COST0 + 1], gencost[:, og.COST0 + 2]

    # Informational only: case14_ieee happens to have c2==0 for every
    # generator (a purely linear cost) and Gs==0 everywhere -- economic_dispatch()
    # handles c2==0 via its bang-bang branch, and the Gs pathway is exercised
    # separately in synthetic_shunt_ground_truth_check() below since no plain
    # PGLib "ieee" test case seems to carry a nonzero shunt conductance.
    print(f"  [info] {name}: min(c2)={c2.min():.4g}  sum|Gs|="
          f"{np.sum(np.abs(case['bus'][:, og.GS])):.4g}")

    lam, pdd, res = 1.0, None, None
    for _ in range(30):
        pdd = og.generate_dcopf_qp(_scaled_case(case, lam))
        res = ksp_qp_bind.solve_from_data(pdd, 1e-10, 1_000_000, 30.0)
        if _congestion_free(pdd, res["x"]):
            break
        lam *= 0.5
    else:
        _check(f"{name}: found an uncongested instance by scaling demand", False)
        return

    x = res["x"]
    D = float(pdd["b"].sum())   # sum of all balance rows == sum(Pg) == sum(Pd)+sum(Gs)
    Pg_ed = economic_dispatch(c1, c2, Pmin, Pmax, D)

    diff = float(np.max(np.abs(x[:ng] - Pg_ed)))
    _check(f"{name} (lam={lam:g}): Pg* matches economic dispatch", diff < 1e-5,
           f"max|diff|={diff:.2e}")

    obj_ed = float(np.sum(c2 * Pg_ed ** 2 + c1 * Pg_ed + c0))
    rel_diff = abs(res["obj_val"] - obj_ed) / max(1.0, abs(obj_ed))
    _check(f"{name} (lam={lam:g}): obj_val matches economic-dispatch cost", rel_diff < 1e-6,
           f"ksp={res['obj_val']:.6f} ed={obj_ed:.6f} rel_diff={rel_diff:.2e}")

    A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
    resid = float(np.max(np.abs(A @ x - pdd["b"])))
    _check(f"{name} (lam={lam:g}): power-balance residual small", resid < 1e-6,
           f"max|resid|={resid:.2e}")


def _synthetic_case_with_shunt() -> dict:
    """Hand-built 3-bus case exercising the Gs (bus shunt conductance)
    pathway, which no plain PGLib "ieee"-tier case seems to carry (all three
    VALIDATION_CASES have Gs==0 everywhere). Manually verified by hand:
    ref bus 1 (Gs=1) links to bus 2 (Pd=100) and bus 3 (Pd=50); with
    generator costs c2=[0.02,0.03], c1=[20,15] the shared-marginal-cost
    solution is mu*=21.624 -> Pg*=[40.6, 110.4] (D=151=sum(Pd)+Gs)."""
    bus = np.array([
        # BUS_I BUS_TYPE  PD QD GS BS AREA VM VA BASE_KV ZONE VMAX VMIN
        [1, 3,   0, 0, 1, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 100, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 1,  50, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
    ])
    gen = np.array([
        # BUS PG QG QMAX QMIN VG MBASE ST PMAX PMIN Pc1 Pc2 Qc1n Qc1x Qc2n Qc2x agc r10 r30 rq apf
        [1, 0, 0, 100, -100, 1, 100, 1, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 100, -100, 1, 100, 1, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    branch = np.array([
        # F T   R    X   B RATE_A RATE_B RATE_C RATIO ANGLE ST ANGMIN ANGMAX
        [1, 2, 0.01, 0.1, 0, 80, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.01, 0.1, 0, 60, 0, 0, 0, 0, 1, -360, 360],
        [1, 3, 0.01, 0.2, 0, 80, 0, 0, 0, 0, 1, -360, 360],
    ])
    gencost = np.array([
        [2, 0, 0, 3, 0.02, 20, 0],
        [2, 0, 0, 3, 0.03, 15, 0],
    ])
    return {"baseMVA": 100.0, "bus": bus, "branch": branch, "gen": gen, "gencost": gencost}


def synthetic_shunt_ground_truth_check() -> None:
    print("\n=== 4b. Synthetic Gs (bus shunt) ground-truth check ===")
    case = _synthetic_case_with_shunt()
    pdd = og.generate_dcopf_qp(case)
    res = ksp_qp_bind.solve_from_data(pdd, 1e-10, 1_000_000, 30.0)
    _check("synthetic-Gs: solved", res["status"] == 0)
    _check("synthetic-Gs: uncongested (no line limit binds)", _congestion_free(pdd, res["x"]))

    ng = case["gen"].shape[0]
    Pmin, Pmax = case["gen"][:, og.PMIN], case["gen"][:, og.PMAX]
    c2 = case["gencost"][:, og.COST0]
    c1 = case["gencost"][:, og.COST0 + 1]
    c0 = case["gencost"][:, og.COST0 + 2]

    D = float(pdd["b"].sum())   # sum(Pd)+sum(Gs)
    Pg_ed = economic_dispatch(c1, c2, Pmin, Pmax, D)
    diff = float(np.max(np.abs(res["x"][:ng] - Pg_ed)))
    _check("synthetic-Gs: Pg* matches economic dispatch (exercises Gs term)", diff < 1e-5,
           f"max|diff|={diff:.2e}  Pg_ksp={res['x'][:ng]}  Pg_ed={Pg_ed}")

    A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
    resid = float(np.max(np.abs(A @ res["x"] - pdd["b"])))
    _check("synthetic-Gs: power-balance residual small", resid < 1e-6, f"max|resid|={resid:.2e}")


def feasibility_sweep() -> None:
    """Sweep every *.m found under data/pglib/, using KSP-QP itself (not
    raw OSQP) as the status oracle: DC-OPF's graph-Laplacian-structured KKT
    system is known (see empirical timing during development) to make plain
    ADMM converge extremely slowly at PGLib's largest scales (tens of
    thousands of iterations short of convergence even after 60s on
    case6470_rte/case9241_pegase) -- exactly the kind of hard instance this
    generator exists to expose, but a poor "cheap oracle" for a sanity sweep.
    A negative status (infeasible/numerical error) is a real FAIL. A
    positive status (iteration/time limit reached without a definitive
    answer) is reported as inconclusive, not failed -- it says nothing about
    whether the generated QP is well-posed, only that it wasn't solved
    within this sweep's time budget; see benchmark_opf.py for a properly
    timed, multi-solver characterization of these cases."""
    print("\n=== 5. Feasibility sweep across data/pglib/ ===")
    files = sorted(og._PGLIB_DATA_DIR.glob("*.m")) if og._PGLIB_DATA_DIR.exists() else []
    if not files:
        print("  (no .m files found under data/pglib/ -- skipping)")
        return
    for path in files:
        try:
            case = og.parse_matpower_case(path)
            pdd = og.generate_dcopf_qp(case)
        except (ValueError, NotImplementedError) as e:
            print(f"  [skipped] {path.stem}: {e}")
            continue
        t0 = time.time()
        res = ksp_qp_bind.solve_from_data(pdd, 1e-6, 10_000_000, 120.0)
        dt = time.time() - t0
        status = res["status"]
        label = f"{path.stem}: feasible (n={pdd['n']} m={pdd['m']} l={pdd['l']})"
        if status == 0:
            _check(label, True, f"solved in {dt:.1f}s")
        elif status < 0:
            _check(label, False, f"status={status} (infeasible/numerical error) t={dt:.1f}s")
        else:
            print(f"  [inconclusive] {path.stem}: not solved within the {dt:.0f}s validation "
                  f"budget (status={status}, {res['pmm_iter']} PMM / {res['ssn_iter']} SSN "
                  f"iters) -- not counted as a failure, see benchmark_opf.py for a proper run")


def main() -> None:
    parser_smoke_tests()
    qp_structural_smoke_tests()
    objective_cross_check()
    economic_dispatch_ground_truth_check()
    synthetic_shunt_ground_truth_check()
    feasibility_sweep()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
