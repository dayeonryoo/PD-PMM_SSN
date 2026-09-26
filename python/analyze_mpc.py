"""
Turn an MPC sweep CSV (benchmark_mpc.py, schema_version 2) into figures and tables.

Reads the CSV and nothing else -- no solver imports, no solving -- so every figure
is reproducible from committed data and re-running is instant.

Why these figures
-----------------
The result is two-dimensional and monotone in BOTH axes: the KSP-QP/QPALM ratio
rises with the vehicle count M (band width) and with the horizon N (band length),
crossing parity along a diagonal. A 1-D scaling curve cannot show that -- and the
two 1-D slices the old benchmark swept (M=5, and N=20) are precisely the two
losing edges of the grid. So the headline is a regime MAP, not a curve:

  F1  ratio heatmap over (M, N)      -- the whole result, wins and losses together
  F2  outer iterations vs n_z        -- the mechanism, and it is hardware-free
  F3  ratio vs n_z                   -- size alone does not predict the outcome
  F4  absolute solve times           -- so a 0.46x at 5 ms is not read as a 4x at 13 s

Colour: KSP-QP blue / QPALM orange throughout, and F1's diverging map runs between
those same two hues through a neutral grey at parity -- so "blue means KSP-QP is
ahead" holds in every figure. The pair is CVD-validated (OKLab dE 35.7 normal,
24.6 protan, 34.3 deutan; target >= 8). The orange sits slightly under the 3:1
contrast target against paper white, which is why every series is also directly
labelled and the same numbers ship as a table.

These are print figures (PDF + PNG), so the interaction and dark-mode passes that
apply to on-screen charts do not apply here.

=== HOW TO RUN ===
  python3 analyze_mpc.py                       # results/0926_mpc_sweep.csv
  python3 analyze_mpc.py --csv results/x.csv --out-prefix x
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

KSP, QPALM = "ksp-qp", "qpalm"
C_KSP, C_QPALM = "#1f77b4", "#ff7f0e"
LABEL = {KSP: "KSP-QP", QPALM: "QPALM"}
COLOR = {KSP: C_KSP, QPALM: C_QPALM}
STYLE = {KSP: "-", QPALM: "--"}         # identity is never colour-alone
MARKER = {KSP: "o", QPALM: "s"}

# Diverging map for the ratio: QPALM-faster (orange) <- neutral grey -> KSP-faster
# (blue). Two hues plus a neutral midpoint, never a rainbow.
CMAP = LinearSegmentedColormap.from_list(
    "ksp_qpalm", ["#7f3d02", C_QPALM, "#f2f2f0", C_KSP, "#0d3c61"])

INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e3e3e0"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["solver"].isin([KSP, QPALM])].copy()
    for col in ("M", "N", "n_z", "rep", "step", "outer_iter", "inner_iter", "solved"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("run_time", "tol_achieved", "obj_val"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (M, N, solver).

    Timing uses the MINIMUM over repeats with repeat 0 discarded: repeat 0 pays
    allocator and page-fault warm-up, and for a deterministic single-threaded
    kernel the minimum is the least scheduler-contaminated estimate. Median is
    kept alongside so the spread is auditable.
    """
    timed = df[df["rep"] > 0]
    agg = (timed.groupby(["M", "N", "solver"])
           .agg(t_best=("run_time", "min"),
                t_med=("run_time", "median"),
                t_p90=("run_time", lambda s: float(np.percentile(s, 90))),
                n_obs=("run_time", "size"))
           .reset_index())
    first = (df[df["rep"] == 0].groupby(["M", "N", "solver"])
             .agg(n_z=("n_z", "first"),
                  outer=("outer_iter", "median"),
                  inner=("inner_iter", "median"),
                  tol_ach=("tol_achieved", "median"),
                  solved=("solved", "sum"),
                  n_inst=("solved", "size"))
             .reset_index())
    return agg.merge(first, on=["M", "N", "solver"])


def ratio_table(a: pd.DataFrame) -> pd.DataFrame:
    """QPALM / KSP-QP wall-time ratio per (M, N). >1 means KSP-QP is faster."""
    k = a[a["solver"] == KSP].set_index(["M", "N"])
    q = a[a["solver"] == QPALM].set_index(["M", "N"])
    idx = k.index.intersection(q.index)
    return pd.DataFrame({
        "M": [i[0] for i in idx],
        "N": [i[1] for i in idx],
        "n_z": k.loc[idx, "n_z"].values,
        "t_ksp": k.loc[idx, "t_best"].values,
        "t_qpalm": q.loc[idx, "t_best"].values,
        "ratio": (q.loc[idx, "t_best"] / k.loc[idx, "t_best"]).values,
        "it_ksp": k.loc[idx, "outer"].values,
        "it_qpalm": q.loc[idx, "outer"].values,
    }).sort_values(["M", "N"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# F1 - the regime map
# ---------------------------------------------------------------------------

def fig_heatmap(r: pd.DataFrame, out: Path) -> None:
    Ms = sorted(r["M"].unique())
    Ns = sorted(r["N"].unique())
    grid = np.full((len(Ms), len(Ns)), np.nan)
    for _, row in r.iterrows():
        grid[Ms.index(row["M"]), Ns.index(row["N"])] = row["ratio"]

    # Symmetric in log space, so 0.5x and 2x sit equally far from parity.
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    span = max(abs(np.log10(lo)), abs(np.log10(hi)))
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_axisbelow(False)
    ax.grid(False)
    im = ax.imshow(np.log10(grid), cmap=CMAP, norm=norm, aspect="auto", origin="upper")

    for i in range(len(Ms)):
        for j in range(len(Ns)):
            v = grid[i, j]
            if np.isnan(v):
                # Capped by --max-nz: large M and large N simultaneously.
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f7f7f5",
                                           edgecolor="white", hatch="///", lw=0))
                continue
            rgba = CMAP(norm(np.log10(v)))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                    color="white" if lum < 0.55 else INK,
                    fontsize=9, fontweight="medium")

    # 2px surface gap between cells
    ax.set_xticks(np.arange(len(Ns)) - .5, minor=True)
    ax.set_yticks(np.arange(len(Ms)) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(len(Ns)), [str(n) for n in Ns])
    ax.set_yticks(range(len(Ms)), [str(m) for m in Ms])
    ax.set_xlabel("horizon $N$   (band length)")
    ax.set_ylabel("vehicles $M$   (band width)")
    ax.set_title("KSP-QP vs QPALM on platoon MPC: where each solver wins\n"
                 "wall-time ratio QPALM / KSP-QP  ($>1$ = KSP-QP faster)",
                 loc="left", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    ticks = [t for t in (-1, -0.5, -0.301, 0, 0.301, 0.5, 1) if -span <= t <= span]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{10**t:.2f}×" for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("QPALM / KSP-QP", rotation=90)

    fig.text(0.015, 0.015,
             "Hatched = not run (capped at $n_z\\leq$65,000). Parity is the neutral band. "
             "Each solver's own reported convergence at tol $10^{-6}$; criteria differ.",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, out)


# ---------------------------------------------------------------------------
# F2 - the mechanism
# ---------------------------------------------------------------------------

def fig_iterations(a: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s in (KSP, QPALM):
        d = a[a["solver"] == s].sort_values("n_z")
        ax.plot(d["n_z"], d["outer"], marker=MARKER[s], ms=5, color=COLOR[s],
                label=LABEL[s], alpha=0.9, markeredgecolor="white",
                markeredgewidth=0.7, linestyle="none")

    k = a[a["solver"] == KSP]
    ax.axhspan(k["outer"].min(), k["outer"].max(), color=C_KSP, alpha=0.10, lw=0)
    ax.annotate(f"KSP-QP: {int(k['outer'].min())}–{int(k['outer'].max())} PMM iterations\n"
                f"across a {int(a['n_z'].max() / a['n_z'].min())}× range in size",
                xy=(a["n_z"].max() * 0.55, k["outer"].max() * 1.35),
                color=C_KSP, fontsize=8.5, ha="right", fontweight="medium")
    q = a[a["solver"] == QPALM]
    qmax = q.loc[q["outer"].idxmax()]
    ax.annotate(f"QPALM: grows to {int(qmax['outer'])}",
                xy=(qmax["n_z"], qmax["outer"]),
                xytext=(qmax["n_z"] * 0.30, qmax["outer"] * 1.15),
                color="#b35806", fontsize=8.5, fontweight="medium",
                arrowprops=dict(arrowstyle="-", color="#b35806", lw=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 50, 100, 200])
    ax.set_yticklabels(["10", "20", "50", "100", "200"])
    ax.set_xlabel("problem size $n_z$")
    ax.set_ylabel("outer iterations per solve")
    ax.set_title("Why: KSP-QP's outer iteration count is size-invariant",
                 loc="left", pad=10)
    ax.legend(loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.015, 0.015,
             "Each point is one (M, N) configuration; median over the rollout. "
             "Iteration counts are hardware- and compiler-independent.",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out)


# ---------------------------------------------------------------------------
# F3 - size alone does not predict the outcome
# ---------------------------------------------------------------------------

def fig_ratio_vs_size(r: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    Ms = sorted(r["M"].unique())

    # A faint spine per M, so the eye can follow one vehicle count across horizons;
    # each labelled at its LEFT end, where the series are well separated on log x.
    for M in Ms:
        d = r[r["M"] == M].sort_values("n_z")
        ax.plot(d["n_z"], d["ratio"], "-", color=GRID, lw=1.2, zorder=1)
        first = d.iloc[0]
        ax.annotate(f"$M$={M}", xy=(first["n_z"], first["ratio"]),
                    xytext=(-9, -1), textcoords="offset points",
                    fontsize=7.5, color=INK_MUTED, va="center", ha="right",
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.85))

    # Points carry the same meaning as F1's cells: blue = KSP-QP ahead.
    win = r["ratio"] > 1
    ax.scatter(r.loc[win, "n_z"], r.loc[win, "ratio"], s=46, color=C_KSP,
               edgecolor="white", linewidth=0.8, zorder=3, label="KSP-QP faster")
    ax.scatter(r.loc[~win, "n_z"], r.loc[~win, "ratio"], s=46, color=C_QPALM,
               marker="s", edgecolor="white", linewidth=0.8, zorder=3,
               label="QPALM faster")

    ax.axhline(1.0, color=INK_MUTED, lw=1, ls=":", zorder=2)

    # The pair that makes the point: near-identical n_z, opposite verdicts.
    # Prefer a parity-straddling pair; fall back to the widest spread.
    best = None
    rows = list(r.itertuples())
    for i, p in enumerate(rows):
        for q in rows[i + 1:]:
            if abs(np.log(p.n_z / q.n_z)) > np.log(1.08):
                continue
            straddles = (p.ratio - 1) * (q.ratio - 1) < 0
            spread = abs(np.log(p.ratio / q.ratio))
            score = (1 if straddles else 0, spread)
            if best is None or score > best[0]:
                best = (score, p, q)
    if best:
        _, p, q = best
        for pt, dy, ha in ((p, 26, "right"), (q, -30, "left")):
            ax.annotate(f"$M$={int(pt.M)}, $N$={int(pt.N)}\n"
                        f"$n_z$={int(pt.n_z):,}  →  {pt.ratio:.2f}×",
                        xy=(pt.n_z, pt.ratio), xytext=(-10 if ha == "right" else 10, dy),
                        textcoords="offset points", fontsize=7.5, ha=ha,
                        color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                             ec=GRID, lw=0.8),
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8),
                        zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4])
    ax.set_yticklabels(["0.5×", "1×", "2×", "4×"])
    ax.set_xlabel("problem size $n_z$")
    ax.set_ylabel("QPALM / KSP-QP")
    ax.set_title("Problem size alone does not predict the winner\n"
                 "the same $n_z$ lands on either side, depending on the $M$:$N$ mix",
                 loc="left", pad=10)
    ax.legend(loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.015, 0.015, "Each grey spine is one vehicle count $M$ across horizons. "
                           "This is why the sweep is a grid, not two 1-D slices.",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out)


# ---------------------------------------------------------------------------
# F4 - absolute times, so ratios are read in context
# ---------------------------------------------------------------------------

def fig_times(a: pd.DataFrame, out: Path) -> None:
    Ms = sorted(a["M"].unique())
    ncol = 4
    nrow = int(np.ceil(len(Ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, M in zip(axes, Ms):
        for s in (KSP, QPALM):
            d = a[(a["solver"] == s) & (a["M"] == M)].sort_values("N")
            ax.plot(d["N"], d["t_best"] * 1000, STYLE[s], marker=MARKER[s], ms=4.5,
                    lw=2, color=COLOR[s], label=LABEL[s],
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$M$ = {M}", loc="left", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(Ms):]:
        ax.set_visible(False)

    for ax in axes[:len(Ms)]:
        ax.set_xticks([20, 50, 100, 200, 400])
        ax.set_xticklabels(["20", "50", "100", "200", "400"])
    for i, ax in enumerate(axes[:len(Ms)]):
        if i % ncol == 0:
            ax.set_ylabel("solve time (ms)")
        if i >= len(Ms) - ncol:
            ax.set_xlabel("horizon $N$")

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Absolute cold-solve time per instance   (best of repeats, log-log)",
                 x=0.012, ha="left", fontsize=10.5)
    fig.text(0.012, 0.012,
             "Losses sit at the small end (a few ms either way); wins sit at the large "
             "end (seconds). Ratios should be read against these magnitudes.",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    save(fig, out)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(r: pd.DataFrame, a: pd.DataFrame, prefix: Path) -> None:
    md = ["| M | N | n_z | KSP-QP (ms) | QPALM (ms) | ratio | PMM it | QPALM it |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, x in r.iterrows():
        md.append(f"| {int(x['M'])} | {int(x['N'])} | {int(x['n_z']):,} | "
                  f"{x['t_ksp']*1000:.1f} | {x['t_qpalm']*1000:.1f} | "
                  f"{x['ratio']:.2f}× | {int(x['it_ksp'])} | {int(x['it_qpalm'])} |")
    (prefix.with_name(prefix.name + "_table.md")).write_text("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
           r"$M$ & $N$ & $n_z$ & KSP-QP (ms) & QPALM (ms) & ratio & PMM it. & QPALM it. \\",
           r"\midrule"]
    for _, x in r.iterrows():
        tex.append(f"{int(x['M'])} & {int(x['N'])} & {int(x['n_z']):,} & "
                   f"{x['t_ksp']*1000:.1f} & {x['t_qpalm']*1000:.1f} & "
                   f"{x['ratio']:.2f} & {int(x['it_ksp'])} & {int(x['it_qpalm'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (prefix.with_name(prefix.name + "_table.tex")).write_text("\n".join(tex) + "\n")


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.pdf').name} / {out.with_suffix('.png').name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT / "results" / "0926_mpc_sweep.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "figs"))
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"no such CSV: {csv_path}")
    prefix = args.out_prefix or csv_path.stem.replace("_mpc_sweep", "")
    out_dir = Path(args.out_dir)

    df = load(csv_path)
    a = aggregate(df)
    r = ratio_table(a)

    unsolved = int((a["solved"] < a["n_inst"]).sum())
    gm = float(np.exp(np.mean(np.log(r["ratio"]))))
    print(f"{csv_path.name}: {len(df)} rows, {len(r)} configurations, "
          f"{unsolved} with any unsolved instance")
    print(f"  KSP-QP faster in {int((r['ratio'] > 1).sum())}/{len(r)}  "
          f"| max {r['ratio'].max():.2f}x  min {r['ratio'].min():.2f}x  geomean {gm:.2f}x")
    print(f"  reported tol_achieved  KSP-QP median {a[a.solver==KSP]['tol_ach'].median():.2e}"
          f"  QPALM median {a[a.solver==QPALM]['tol_ach'].median():.2e}")

    fig_heatmap(r, out_dir / f"{prefix}_f1_regime_map")
    fig_iterations(a, out_dir / f"{prefix}_f2_iterations")
    fig_ratio_vs_size(r, out_dir / f"{prefix}_f3_ratio_vs_size")
    fig_times(a, out_dir / f"{prefix}_f4_times")
    write_tables(r, a, out_dir / f"{prefix}")
    print(f"  wrote {prefix}_table.md / {prefix}_table.tex")


if __name__ == "__main__":
    main()
