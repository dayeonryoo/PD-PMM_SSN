"""
Build DC optimal power flow (DC-OPF) QPs in KSPQPdata dict form (same format
as ksp_qp_bind.parse_sif()), from MATPOWER/PGLib-OPF .m case files.

DC-OPF is a lossless, linearized power-flow model: generator active-power
output is scheduled to meet demand subject to a linear (DC) approximation of
the AC power-flow equations, plus line-flow and generator limits. The
constraint matrix here follows the power grid's own graph topology -- a
weighted graph Laplacian (bus susceptance matrix) -- so it stress-tests
KSP-QP's ordering-selection/Schur-complement machinery on a genuinely
different sparsity pattern, at scale (PGLib-OPF cases run from ~10 buses to
tens of thousands).

Model (following MATPOWER's own dcopf.m formulation)
------------------------------------------------------
    min_{Pg,theta}  sum_g (c2_g Pg_g^2 + c1_g Pg_g + c0_g)
    s.t.            Cg'Pg - B_bus theta = Pd + Gs      (bus power balance)
                    -RATE_A_k <= (theta_f-theta_t)/x_k <= RATE_A_k
                                        (rate-limited in-service branches only)
                    Pmin <= Pg <= Pmax
                    theta_ref = 0

B_bus is the weighted graph-Laplacian bus-susceptance matrix (b_k=1/x_k per
in-service branch); Cg is the (ng x nb) generator-to-bus incidence matrix.
Gs (bus shunt conductance) is a fixed real-power draw at the DC-OPF
flat-voltage (V=1 p.u.) assumption, so it folds directly into the
fixed-demand RHS alongside Pd.

Reference-bus handling: theta_ref is fixed via the box bound lx=ux=0, NOT an
extra equality row -- ALL nb power-balance rows are kept in A. This differs
from the "drop the slack bus's row" folklore sometimes quoted for DC-OPF: A
(including the Cg' block) is actually full row rank nb even though B_bus
alone is a singular Laplacian, because every column of Cg' sums to 1 and no
nonzero vector in B_bus's null space (constant-per-component vectors) can
also be annihilated by Cg' -- provided every connected bus component has
>=1 in-service generator (checked explicitly below; violated only for an
islanded, generator-less sub-network). The real reason theta_ref needs
pinning is that the constant angle-shift direction is simultaneously in
ker(A) and ker(Q) (theta doesn't appear in the objective) -- a genuine flat
direction of the QP that only a box bound removes.

Variable stacking: z = [Pg (ng); theta (nb)], n = ng + nb.
Mapping to KSP-QP's min c'x + 0.5x'Qx s.t. Ax=b, lw<=Bx<=uw, lx<=x<=ux:
  - Q = diag(2*c2, 0_nb) (the factor of 2 reconciles c2*Pg^2 with 0.5*x'Qx)
  - c = [c1; 0_nb], obj_const = sum(c0)
  - A, b: the power-balance block above
  - B, lw, uw: line-flow limits, one row per branch with RATE_A>0 (RATE_A==0
    is MATPOWER's "unlimited" sentinel -- those branches are excluded from B
    entirely rather than given +-inf rows)
  - lx, ux: Pmin/Pmax on the Pg block, +-inf on theta except lx=ux=0 at the
    reference bus

Only the MODEL=2 (polynomial), NCOST=3 (quadratic: c2,c1,c0) generator cost
model is supported -- this holds for every plain pglib_opf_case*.m file (not
the __api/__sad/__FERC scenario variants, which can carry different cost
structures) -- and is checked explicitly at parse time.

Data: case files are not vendored in this repo. Download the plain
pglib_opf_case*.m files (not __api/__sad/__FERC variants) from
github.com/power-grid-lib/pglib-opf and place them under data/pglib/, e.g.
pglib_opf_case14_ieee.m, pglib_opf_case118_ieee.m, ..., up to the large
synthetic cases (case9241_pegase, case13659_pegase, ...) for stress testing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

INF = np.inf


# ---------------------------------------------------------------------------
# MATPOWER column indices (0-indexed, translated from MATPOWER's 1-indexed
# case-format docs).
# ---------------------------------------------------------------------------

BUS_I, BUS_TYPE, PD, QD, GS = 0, 1, 2, 3, 4
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, BR_STATUS = 0, 1, 2, 3, 4, 5, 10
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
MODEL, STARTUP, SHUTDOWN, NCOST, COST0 = 0, 1, 2, 3, 4

REF_BUS_TYPE = 3
ISOLATED_BUS_TYPE = 4
POLYNOMIAL_MODEL = 2

_PGLIB_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pglib"


# ---------------------------------------------------------------------------
# MATPOWER .m case-file parser
# ---------------------------------------------------------------------------

def parse_matpower_case(path) -> dict:
    """Parse mpc.baseMVA/bus/branch/gen/gencost out of a MATPOWER-format .m
    case file. Only the flat numeric matrices are extracted -- everything
    else in the file (bus/gen names, area/zone tables, version string, ...)
    is ignored.
    """
    text = Path(path).read_text()
    text = re.sub(r"%[^\n]*", "", text)   # strip comments before any block regex

    m = re.search(r"mpc\.baseMVA\s*=\s*([^\s;]+)\s*;", text)
    if m is None:
        raise ValueError(f"{path}: mpc.baseMVA not found")
    baseMVA = float(m.group(1))

    case = {"baseMVA": baseMVA}
    for field in ("bus", "branch", "gen", "gencost"):
        block = re.search(rf"mpc\.{field}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if block is None:
            raise ValueError(f"{path}: mpc.{field} not found")
        rows: list[list[float]] = []
        for i, raw_row in enumerate(block.group(1).split(";")):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            try:
                rows.append([float(t) for t in raw_row.split()])
            except ValueError as e:
                raise ValueError(f"{path}: mpc.{field} row {i}: {e}") from e
        n_cols = len(rows[0]) if rows else 0
        for i, row in enumerate(rows):
            if len(row) != n_cols:
                raise ValueError(
                    f"{path}: mpc.{field} row {i} has {len(row)} columns, expected {n_cols}")
        case[field] = np.array(rows, dtype=np.float64)

    _validate_case(case, path)
    return case


def _validate_case(case: dict, path) -> None:
    bus, gencost = case["bus"], case["gencost"]
    n_ref = int(np.sum(bus[:, BUS_TYPE] == REF_BUS_TYPE))
    if n_ref != 1:
        raise ValueError(f"{path}: expected exactly 1 reference bus, found {n_ref}")
    if np.any(bus[:, BUS_TYPE] == ISOLATED_BUS_TYPE):
        raise ValueError(f"{path}: isolated bus(es) present (BUS_TYPE==4); not supported")
    if not (np.all(gencost[:, MODEL] == POLYNOMIAL_MODEL) and np.all(gencost[:, NCOST] == 3)):
        raise NotImplementedError(
            f"{path}: only MODEL=2 (polynomial), NCOST=3 (quadratic c2,c1,c0) "
            f"generator costs are supported by this generator")


def load_pglib_case(name: str, data_dir=None) -> dict:
    """Load a PGLib-OPF case by name (with or without a trailing '.m'), from
    data/pglib/ by default (or `data_dir` if given)."""
    if name.endswith(".m"):
        name = name[:-2]
    root = Path(data_dir) if data_dir is not None else _PGLIB_DATA_DIR
    return parse_matpower_case(root / f"{name}.m")


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
              lw: np.ndarray, uw: np.ndarray, obj_const: float = 0.0) -> dict:
    n = Q.shape[0]
    out: dict = {"n": n, "m": int(A.shape[0]), "l": int(B.shape[0]), "obj_const": float(obj_const)}
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
# DC-OPF -> QP
# ---------------------------------------------------------------------------

def _build_bus_susceptance(branch: np.ndarray, in_service: np.ndarray,
                           bus_id_to_idx: dict, nb: int) -> sp.csc_matrix:
    """Weighted graph-Laplacian bus susceptance matrix B_bus (b_k=1/x_k per
    in-service branch), built via COO triplets -- never densified. Relies on
    scipy's default duplicate-triplet summing (coo -> csc) for parallel
    branches between the same bus pair."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for br in branch[in_service]:
        x = br[BR_X]
        if x == 0.0:
            raise ValueError("in-service branch with BR_X==0 (unsupported)")
        f = bus_id_to_idx[int(br[F_BUS])]
        t = bus_id_to_idx[int(br[T_BUS])]
        b = 1.0 / x
        rows += [f, t, f, t]
        cols += [f, t, t, f]
        vals += [b, b, -b, -b]
    return sp.coo_matrix((vals, (rows, cols)), shape=(nb, nb)).tocsc()


def _build_generator_incidence(gen_bus_idx: np.ndarray, nb: int) -> sp.csc_matrix:
    """(ng x nb) generator-to-bus incidence matrix Cg: one 1.0 per row."""
    ng = len(gen_bus_idx)
    return sp.csc_matrix((np.ones(ng), (np.arange(ng), gen_bus_idx)), shape=(ng, nb))


def _build_line_limit_block(branch: np.ndarray, in_service: np.ndarray,
                            bus_id_to_idx: dict, ng: int, nb: int):
    """General-inequality block for rate-limited in-service branches:
    -RATE_A <= (theta_f-theta_t)/x <= RATE_A. Branches with RATE_A==0
    (MATPOWER's "unlimited" sentinel) are excluded entirely, not given
    +-inf rows."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lw: list[float] = []
    uw: list[float] = []
    row_idx = 0
    for br in branch[in_service]:
        rate_a = br[RATE_A]
        if rate_a <= 0.0:
            continue
        x = br[BR_X]
        f = bus_id_to_idx[int(br[F_BUS])]
        t = bus_id_to_idx[int(br[T_BUS])]
        b = 1.0 / x
        rows += [row_idx, row_idx]
        cols += [ng + f, ng + t]
        vals += [b, -b]
        lw.append(-rate_a)
        uw.append(rate_a)
        row_idx += 1
    B = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, ng + nb)).tocsc()
    return B, np.array(lw), np.array(uw)


def generate_dcopf_qp(case: dict) -> dict:
    """Build the DC-OPF QP for a parsed MATPOWER/PGLib case dict (from
    parse_matpower_case / load_pglib_case). Variables z = [Pg (ng); theta (nb)]."""
    bus, branch, gen, gencost = case["bus"], case["branch"], case["gen"], case["gencost"]

    bus_ids = bus[:, BUS_I].astype(int)
    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    nb = len(bus_ids)

    gen_in_service = gen[:, GEN_STATUS] > 0
    gen, gencost = gen[gen_in_service], gencost[gen_in_service]
    ng = gen.shape[0]
    gen_bus_idx = np.array([bus_id_to_idx[int(g)] for g in gen[:, GEN_BUS]])

    branch_in_service = branch[:, BR_STATUS] > 0

    B_bus = _build_bus_susceptance(branch, branch_in_service, bus_id_to_idx, nb)
    Cg = _build_generator_incidence(gen_bus_idx, nb)

    n_comp, labels = connected_components(csgraph=B_bus, directed=False)
    gen_components = set(labels[gen_bus_idx].tolist())
    if len(gen_components) < n_comp:
        raise ValueError(
            f"{n_comp - len(gen_components)} connected bus component(s) have no "
            f"in-service generator -- DC-OPF power balance would be singular there")

    A = sp.hstack([Cg.T, -B_bus], format="csc")
    b = bus[:, PD] + bus[:, GS]

    B, lw, uw = _build_line_limit_block(branch, branch_in_service, bus_id_to_idx, ng, nb)

    c2 = gencost[:, COST0]
    c1 = gencost[:, COST0 + 1]
    c0 = gencost[:, COST0 + 2]
    Q = sp.diags(np.concatenate([2.0 * c2, np.zeros(nb)]), format="csc")
    c = np.concatenate([c1, np.zeros(nb)])
    obj_const = float(c0.sum())

    lx = np.concatenate([gen[:, PMIN], np.full(nb, -INF)])
    ux = np.concatenate([gen[:, PMAX], np.full(nb, INF)])
    ref_idx = int(np.argmax(bus[:, BUS_TYPE] == REF_BUS_TYPE))
    lx[ng + ref_idx] = 0.0
    ux[ng + ref_idx] = 0.0

    return _assemble(Q, A, B, c, b, lx, ux, lw, uw, obj_const)
