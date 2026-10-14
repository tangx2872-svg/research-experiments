#!/usr/bin/env python
"""E19 paper convergence -- regenerate publication figures, tables and notes.

CPU-only. Reads ONLY frozen E16/E17/E18 artifacts; writes ONLY under results/e19/.
Deterministic: a rerun reproduces byte-identical CSV/MD outputs
(asserted by scripts/e19_verify.py).

Read-only sources
-----------------
results/e16/e16_partial.json            full-precision F0/R1 metrics + diagnostics
results/e17/e17_partial.json            full-precision BETA-sweep configs (C0-C4, seed0)
results/e17/final_method.json           frozen advancement gates (authoritative deltas)
results/e18/main_table.csv              final benchmark main table (full precision)
results/e18/illum_breakdown.csv         illumination 01-10 breakdown (24 sources x 11 groups)
results/e18/bootstrap_f0_vs_ours.json   paired bootstrap CIs (F0 vs Ours)

Writes -> results/e19/{tables,figures,notes}/
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
E16P = RES / "e16" / "e16_partial.json"
E17R = RES / "e17" / "e17_partial.json"
E17F = RES / "e17" / "final_method.json"
E18D = RES / "e18"
OUT = RES / "e19"
TAB = OUT / "tables"
FIG = OUT / "figures"
NOTES = OUT / "notes"

CATS = ["Bird", "Car", "Cube"]
CAT_COLOR = {"Bird": "#1b9e77", "Car": "#d95f02", "Cube": "#7570b3"}
BETA_OF = {"C0": 0.1, "C1": 0.0, "C2": 0.05, "C3": 0.2, "C4": 0.5}
CFG_ORDER = ["C1", "C2", "C0", "C3", "C4"]  # sorted by beta
CFG_COLOR = {"C0": "#111111", "C1": "#d95f02", "C2": "#7570b3", "C3": "#1b9e77", "C4": "#e7298a"}
F0_COLOR = "#7f7f7f"
OURS_COLOR = "#1f77b4"

METHOD_ORDER = ["PatchCore-emb + unified scoring", "AnomalyDINO", "SoftPatch",
                "EfficientAD", "F0", "Ours (R1-ILA)"]
METHOD_SHORT = {
    "PatchCore-emb + unified scoring": "PatchCore\n(emb, unified)",
    "AnomalyDINO": "AnomalyDINO",
    "SoftPatch": "SoftPatch",
    "EfficientAD": "EfficientAD",
    "F0": "F0\n(DINOv2-B)",
    "Ours (R1-ILA)": "Ours\n(R1-ILA)",
}
METHOD_COLOR = {
    "PatchCore-emb + unified scoring": "#c2c2c2",
    "AnomalyDINO": "#66c2a5",
    "SoftPatch": "#fc8d62",
    "EfficientAD": "#8da0cb",
    "F0": F0_COLOR,
    "Ours (R1-ILA)": OURS_COLOR,
}
CTX_COLOR = {
    "PatchCore-emb(unified)": "#c2c2c2",
    "PatchCore-official": "#999999",
    "AnomalyDINO": "#66c2a5",
    "SoftPatch": "#fc8d62",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fnum(x):
    """Shortest round-trip repr -> deterministic CSV/MD output."""
    if x is None:
        return ""
    if isinstance(x, float):
        return repr(x)
    return str(x)


def wcsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fnum(v) for v in r])


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fget(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# E19-1  BETA ablation (source: E17 sweep, E16 frozen reference)
# --------------------------------------------------------------------------
def build_beta():
    e16 = load_json(E16P)["cells"]
    e17 = load_json(E17R)["cells"]
    fm = load_json(E17F)
    gates = fm["advancement_gates"]
    f0 = {c: e16[f"{c}|0"]["f0"] for c in CATS}

    data = {}          # (cat, cfg) -> dict of full-precision metrics
    rows = []
    for c in CATS:
        cc = e17[f"{c}|0"]["configs"]
        c0 = cc["C0"]
        for cfg in ["C0", "C1", "C2", "C3", "C4"]:
            v = cc[cfg]
            rec = {
                "category": c, "config": cfg, "beta": v["beta"],
                "auroc": v["auroc"], "ap": v["ap"], "d_prime": v["d_prime"],
                "robustness_feat": v["robustness_feat"],
                "robustness_score": v["robustness_score"],
                "illum_shift": v["illum_shift"], "steps_used": int(v["steps_used"]),
                "std_ratio": v["std_ratio"], "pairwise_cos": v["pairwise_cos"],
                "collapsed": bool(v["collapsed"]), "loss_finite": bool(v["loss_finite"]),
                "runtime_s": v["runtime_s"],
                "dAUROC_vs_F0": v["auroc"] - f0[c]["auroc"],
                "dAP_vs_F0": v["ap"] - f0[c]["ap"],
                "dAUROC_vs_C0": v["auroc"] - c0["auroc"],
                "dAP_vs_C0": v["ap"] - c0["ap"],
                "d_rob_vs_C0": v["robustness_score"] - c0["robustness_score"],
                "d_shift_vs_C0": v["illum_shift"] - c0["illum_shift"],
            }
            data[(c, cfg)] = rec
            if cfg == "C0":
                rec["dAUROC_vs_C0"] = 0.0
                rec["dAP_vs_C0"] = 0.0
                rec["d_rob_vs_C0"] = 0.0
                rec["d_shift_vs_C0"] = 0.0
            rows.append(rec)

    header = ["category", "config", "beta", "auroc", "ap", "d_prime", "robustness_feat",
              "robustness_score", "illum_shift", "steps_used", "std_ratio", "pairwise_cos",
              "collapsed", "loss_finite", "runtime_s", "dAUROC_vs_F0", "dAP_vs_F0",
              "dAUROC_vs_C0", "dAP_vs_C0", "d_rob_vs_C0", "d_shift_vs_C0",
              "source_files"]
    src = "results/e17/e17_partial.json + results/e16/e16_partial.json"
    wcsv(TAB / "beta_ablation.csv", header,
         [[r[k] for k in header[:-1]] + [src] for r in rows])

    # ---- figure 1: AUROC vs BETA with frozen F0 reference ----
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [BETA_OF[cfg] for cfg in CFG_ORDER]
    for c in CATS:
        ys = [data[(c, cfg)]["auroc"] for cfg in CFG_ORDER]
        ax.plot(xs, ys, "o-", color=CAT_COLOR[c], lw=1.4, ms=5, label=c)
        ax.axhline(f0[c]["auroc"], color=CAT_COLOR[c], ls=":", lw=1.0, alpha=0.65)
        y0 = data[(c, "C0")]["auroc"]
        ax.plot([0.1], [y0], marker="*", ms=13, color=CAT_COLOR[c],
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.5"])
    ax.set_xlim(-0.005, 0.62)
    ax.set_xlabel("BETA  (0 = pure illumination consistency, no frozen anchor)")
    ax.set_ylabel("Image AUROC (seed 0)")
    ax.set_title("BETA sweep: AUROC vs BETA\n(dotted = frozen F0; star = frozen BETA=0.1; red ring = collapsed)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", framealpha=0.9)
    save(fig, "fig_beta_auroc")

    # ---- figure 2: decision view (deltas + trade-off) ----
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    w = 0.2
    xcat = np.arange(3)
    for j, cfg in enumerate(["C1", "C2", "C3", "C4"]):
        vals = [data[(c, cfg)]["dAUROC_vs_C0"] for c in CATS]
        ax.bar(xcat + (j - 1.5) * w, vals, w, label=f"{cfg} (BETA={BETA_OF[cfg]})",
               color=CFG_COLOR[cfg])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.003, color="green", ls="--", lw=0.9)
    ax.axhline(-0.003, color="red", ls="--", lw=0.9)
    ax.text(2.42, 0.0035, "advance gate +0.003", color="green", fontsize=7, ha="right")
    ax.text(2.42, -0.0055, "Cube safety -0.003", color="red", fontsize=7, ha="right")
    ax.set_xticks(xcat)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("delta AUROC vs frozen BETA=0.1 (seed 0)")
    ax.set_title("Challenger deltas (full precision; gates closed at 1e-5)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for c in CATS:
        xs2 = [data[(c, cfg)]["robustness_score"] for cfg in CFG_ORDER]
        ys2 = [data[(c, cfg)]["auroc"] for cfg in CFG_ORDER]
        ax.plot(xs2, ys2, "o-", color=CAT_COLOR[c], lw=1.3, ms=4.5, label=c)
        for cfg, xx, yy in zip(CFG_ORDER, xs2, ys2):
            ax.annotate(cfg, (xx, yy), fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points")
        cx, cy = data[(c, "C1")]["robustness_score"], data[(c, "C1")]["auroc"]
        ax.plot([cx], [cy], "o", ms=10, mfc="none", mec="red", mew=1.4)
    ax.set_xlabel("robustness_score (lower = fewer illumination-induced score shifts)")
    ax.set_ylabel("Image AUROC (seed 0)")
    ax.set_title("Detection vs illumination-consistency trade-off\n(red ring = BETA=0, Cube collapsed)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_beta_decision")

    # ---- figure 3: collapse diagnostics ----
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    panels = [("pairwise_cos", "mean pairwise cosine of adapted features\n(collapse flag at 0.98)", 0.98),
              ("std_ratio", "feature std ratio (adapted / frozen)", None),
              ("steps_used", "optimization steps used (max 300)", None)]
    for ax, (key, ttl, ref) in zip(axes, panels):
        for c in CATS:
            ys3 = [data[(c, cfg)][key] for cfg in CFG_ORDER]
            ax.plot(xs, ys3, "o-", color=CAT_COLOR[c], lw=1.3, ms=4.5, label=c)
        if ref is not None:
            ax.axhline(ref, color="red", ls="--", lw=0.9)
            ax.text(0.41, ref + 0.004, "collapse threshold", color="red", fontsize=7)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks([0, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.5"])
        ax.set_xlim(-0.005, 0.62)
        ax.set_xlabel("BETA")
        ax.set_title(ttl)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_beta_collapse")

    # ---- notes ----
    def g(cfg, key):
        if key in ("bc_dAUROC", "bc_dAP", "bc_drob"):
            m = {"bc_dAUROC": "bird_car_mean_dAUROC_vs_C0",
                 "bc_dAP": "bird_car_mean_dAP_vs_C0",
                 "bc_drob": "bird_car_mean_d_rob_vs_C0"}[key]
            return gates[cfg][m]
        if key == "cube":
            return gates[cfg]["cube_dAUROC_vs_C0"]
        return None

    gate_rows = []
    for cfg in ["C1", "C2", "C3", "C4"]:
        gd = gates[cfg]
        ok = all([gd["gate1_ge_+0.003"], gd["gate2_cube_ge_-0.003"],
                  gd["gate3_dAP_ge_-0.002"], gd["gate4_rob_not_worse"],
                  gd["gate5_no_collapse"]])
        gate_rows.append(
            f"| {cfg} | {BETA_OF[cfg]:g} | {gd['bird_car_mean_dAUROC_vs_C0']:+.5f} | "
            f"{gd['cube_dAUROC_vs_C0']:+.5f} | {gd['bird_car_mean_dAP_vs_C0']:+.5f} | "
            f"{gd['bird_car_mean_d_rob_vs_C0']:+.5f} | {str(gd['collapsed'])} | "
            f"{'PASS' if ok else 'FAIL'} |")
    t = "\n".join(gate_rows)
    notes = f"""# E19-1 · BETA ablation — notes

**Source (read-only)**: `results/e17/e17_partial.json` (full-precision sweep configs, seed 0),
`results/e16/e16_partial.json` (full-precision F0 reference), `results/e17/final_method.json`
(frozen advancement gates). **No new GPU run.**

**Claim served**: R1-ILA's only explicit loss parameter BETA is a monotone detection ↔
illumination-consistency knob; the frozen BETA=0.1 is an aggregate interior optimum, and
BETA=0 (pure consistency, no frozen anchor) degrades the representation (Cube collapse flag;
lowest d' in all three categories).

**Figures**
- `figures/fig_beta_auroc.png/.pdf` — AUROC vs BETA per category (seed 0); dotted = frozen F0.
- `figures/fig_beta_decision.png/.pdf` — (left) challenger delta vs frozen BETA=0.1 with the
  pre-registered +0.003 advance gate and -0.003 Cube safety line; (right) detection vs
  robustness_score trade-off, red ring = BETA=0.
- `figures/fig_beta_collapse.png/.pdf` — collapse diagnostics (pairwise cosine, std ratio,
  steps used) vs BETA.

**Facts (full precision, seed 0)** — read the AUROC and d' statements together:
- BETA=0 (C1) is the **worst challenger by AUROC on Bird and Car**
  ({data[('Bird','C1')]['auroc']:.4f} / {data[('Car','C1')]['auroc']:.4f}) and the **lowest by d' in all
  three categories** ({data[('Bird','C1')]['d_prime']:.4f} / {data[('Car','C1')]['d_prime']:.4f} /
  {data[('Cube','C1')]['d_prime']:.4f}).
- On **Cube**, BETA=0 has the nominally **highest** AUROC among the sweep
  ({data[('Cube','C1')]['auroc']:.4f}) — but it is the only configuration flagged `collapsed=True`
  (pairwise cosine {data[('Cube','C1')]['pairwise_cos']:.4f} > 0.98; std ratio
  {data[('Cube','C1')]['std_ratio']:.2f}; steps_used {data[('Cube','C1')]['steps_used']} = 300/300).
  The elevated Cube AUROC therefore comes together with a degenerate (near-constant) adapted
  representation and fails frozen gate 5 (`no_collapse`); it is **not** accepted as a gain.
  Bird/Car pairwise cosines are also at the threshold edge (0.979 / 0.976).
- Lower BETA monotonically lowers robustness_score and illum_shift (more illumination-consistent);
  higher BETA raises AUROC on Bird but not on Car/Cube.

> Documentation note (fact, recorded 2026-10-15): `results/e17/README.md` states "BETA=0 is the
> worst detector on all three categories". By AUROC this holds for Bird/Car but **not** for Cube
> (where d' is instead the lowest metric). The E17 verdict itself is unchanged. The E17 document
> was NOT modified by E19; this note only aligns the E19 wording with the E17 artifact data.

**Frozen advancement gates (authoritative values, `final_method.json`)**

| cfg | BETA | Bird+Car mean dAUROC vs C0 | Cube dAUROC vs C0 | Bird+Car mean dAP vs C0 | Bird+Car mean d_rob vs C0 | collapsed | verdict |
|---|---|---|---|---|---|---|---|
{t}

No challenger passes; E17 verdict = `FREEZE_CURRENT_R1` and Round 2 was NOT run.

**Precision audit note (recorded, not a defect)**: the frozen gates were computed from
`results/e17/round1.csv`, whose values are recorded to 5 decimals. An independent recompute from
`results/e17/e17_partial.json` (full precision) differs by at most **9.2e-6** (see
`../verification.csv`, group BETA). This is far below every gate margin (smallest relevant margin:
C3 Bird+Car mean dAUROC +0.00173 vs the +0.003 bar), so no gate verdict is affected.

**Boundary**: BETA sweep covers one seed (seed 0) x three categories; it is a controlled
ablation (same backbone, same scoring, same bank protocol). It does not by itself establish
cross-seed stability — that is E16's 3-seed result. `robustness_score` / `illum_shift` are
feature- and score-level diagnostics frozen in E16 (`scripts/e16_qualify.py`), not causal
measurements.
"""
    (NOTES / "e19_beta_ablation_notes.md").write_text(notes, encoding="utf-8")
    return data, f0


# --------------------------------------------------------------------------
# E19-2  illumination groups 01-10 (source: E18 breakdown)
# --------------------------------------------------------------------------
def build_illum():
    raw = read_rows(E18D / "illum_breakdown.csv")
    # per-source ALL reference
    all_ref = {}
    for r in raw:
        if r["illumination"] == "ALL":
            all_ref[r["source"]] = {"auroc": float(r["auroc"]), "good_mean": float(r["good_mean"])}
    rows = []
    for r in raw:
        method, cat, seed = r["source"].split("|")
        ref = all_ref[r["source"]]
        rel = (float(r["good_mean"]) / ref["good_mean"]) if ref["good_mean"] else float("nan")
        rows.append({
            "source": r["source"], "method": method, "category": cat, "seed": seed,
            "illumination": r["illumination"], "n_ok": int(r["n_ok"]), "n_ng": int(r["n_ng"]),
            "auroc": float(r["auroc"]), "ap": float(r["ap"]),
            "good_mean": float(r["good_mean"]), "good_std": float(r["good_std"]),
            "auroc_minus_all": float(r["auroc"]) - ref["auroc"],
            "rel_good_mean": rel,
            "role": "primary" if method in ("F0", "Ours") else "context",
            "source_file": "results/e18/illum_breakdown.csv",
        })
    header = list(rows[0].keys())
    wcsv(TAB / "illum_groups.csv", header, [[r[k] for k in header] for r in rows])

    d = {}
    for r in rows:
        d[(r["source"], r["illumination"])] = r
    groups = [f"{i:02d}" for i in range(1, 11)]

    # ---- figure: AUROC per illumination group, F0 vs Ours ----
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    xg = np.arange(1, 11)
    for ax, c in zip(axes, CATS):
        for meth, col, mk in (("F0", F0_COLOR, "o"), ("Ours", OURS_COLOR, "s")):
            src = f"{meth}|{c}|seed0"
            ys = [d[(src, g)]["auroc"] for g in groups]
            ax.plot(xg, ys, marker=mk, color=col, lw=1.3, ms=4, label=meth)
            ax.axhline(d[(src, "ALL")]["auroc"], color=col, ls=":", lw=1.0, alpha=0.7)
        n_ok = d[(f"F0|{c}|seed0", "ALL")]["n_ok"]
        n_ng = d[(f"F0|{c}|seed0", "ALL")]["n_ng"]
        ax.set_xticks(xg)
        ax.set_xlabel("illumination group (01-10)")
        ax.set_ylim(0.55, 0.95)
        ax.set_title(f"{c}  (ALL: n_ok={n_ok}, n_ng={n_ng})")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Image AUROC (seed 0)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-illumination AUROC, descriptive only (small per-group n; dotted = ALL)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_illum_auroc")

    # ---- figure: drift + spread ----
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    for c in CATS:
        for meth, ls, col in (("F0", "--", CAT_COLOR[c]), ("Ours", "-", CAT_COLOR[c])):
            src = f"{meth}|{c}|seed0"
            ys = [d[(src, g)]["rel_good_mean"] - 1.0 for g in groups]
            ax.plot(xg, (np.array(ys) * 100.0), ls, color=col, lw=1.4,
                    marker="o" if meth == "F0" else "s", ms=3.5,
                    label=f"{c} {meth}")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xg)
    ax.set_xlabel("illumination group (01-10)")
    ax.set_ylabel("relative good-image score drift (%)")
    ax.set_title("Good-image score drift per illumination\n(relative to ALL; only spread/shape interpretable)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    spreads = {}
    for c in CATS:
        for meth in ("F0", "Ours"):
            src = f"{meth}|{c}|seed0"
            gm = [d[(src, g)]["good_mean"] for g in groups]
            spreads[(c, meth)] = max(gm) - min(gm)
    xc = np.arange(3)
    w = 0.34
    ax.bar(xc - w / 2, [spreads[(c, "F0")] for c in CATS], w, label="F0", color=F0_COLOR)
    ax.bar(xc + w / 2, [spreads[(c, "Ours")] for c in CATS], w, label="Ours (R1-ILA)", color=OURS_COLOR)
    for i, c in enumerate(CATS):
        ax.text(i - w / 2, spreads[(c, "F0")] + 0.0015, f"{spreads[(c, 'F0')]:.4f}",
                ha="center", fontsize=7)
        ax.text(i + w / 2, spreads[(c, "Ours")] + 0.0015, f"{spreads[(c, 'Ours')]:.4f}",
                ha="center", fontsize=7)
    ax.set_xticks(xc)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("spread of good-image score across groups (max - min)")
    ax.set_title("Cross-illumination spread of good-image scores\n(lower = smaller illumination-induced score drift)")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_illum_drift")

    # ---- figure: context sources ----
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    ax = axes[0]
    for meth in ["Ours", "F0", "PatchCore-emb(unified)", "SoftPatch", "AnomalyDINO"]:
        src = f"{meth}|Cube|seed0"
        ys = [d[(src, g)]["auroc"] for g in groups]
        col = OURS_COLOR if meth == "Ours" else (F0_COLOR if meth == "F0" else CTX_COLOR[meth])
        ax.plot(xg, ys, "o-", color=col, lw=1.3, ms=3.5, label=meth)
        ax.axhline(d[(src, "ALL")]["auroc"], color=col, ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(xg)
    ax.set_xlabel("illumination group (01-10)")
    ax.set_ylabel("Image AUROC (seed 0)")
    ax.set_title("Cube: per-illumination AUROC, all available sources")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7)
    ax = axes[1]
    for meth in ["Ours", "F0", "PatchCore-emb(unified)", "PatchCore-official"]:
        src = f"{meth}|Bird|seed0"
        ys = [d[(src, g)]["auroc"] for g in groups]
        col = OURS_COLOR if meth == "Ours" else (F0_COLOR if meth == "F0" else CTX_COLOR[meth])
        ax.plot(xg, ys, "o-", color=col, lw=1.3, ms=3.5, label=meth)
        ax.axhline(d[(src, "ALL")]["auroc"], color=col, ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(xg)
    ax.set_xlabel("illumination group (01-10)")
    ax.set_title("Bird: per-illumination AUROC, all available sources")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7)
    fig.suptitle("Appendix context (single seed, small per-group n; NOT a ranking claim)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig_illum_context")

    # ---- notes ----
    sp_txt = "\n".join(
        f"| {c} | {spreads[(c, 'F0')]:.4f} | {spreads[(c, 'Ours')]:.4f} | "
        f"{(spreads[(c, 'Ours')] / spreads[(c, 'F0')] - 1) * 100:+.1f}% |" for c in CATS)
    notes = f"""# E19-2 · Illumination groups 01-10 — notes

**Source (read-only)**: `results/e18/illum_breakdown.csv` (24 sources x 11 groups).
**No new GPU run, no model re-executed.**

**Claim served**: R1-ILA is *consistent with* reducing illumination-induced score drift:
the cross-illumination spread of good-image scores shrinks in the same direction in all three
categories. This is a descriptive, downstream (score-level) observation — **not a causal proof**.

**Figures**
- `figures/fig_illum_auroc.png/.pdf` — per-group AUROC, F0 vs Ours, three categories
  (descriptive; dotted = ALL).
- `figures/fig_illum_drift.png/.pdf` — (left) good-score drift relative to ALL; (right) spread
  max-min of good-image score across groups.
- `figures/fig_illum_context.png/.pdf` — appendix: every available source (Cube / Bird), context only.

**Facts — spread of good-image score across groups (seed 0)**

| category | F0 spread | Ours spread | change |
|---|---|---|---|
{sp_txt}

- Ours = R1-ILA (BETA=0.1), F0 = frozen DINOv2-B 1-NN-max.
- The spread decreases in all three categories; per-group AUROC differences are noisy
  (per-group n_ok ~ 32 for Bird and n_ok ~ 26 for Car/Cube) and are reported descriptively only.
- Largest single-group drift: Car group 07, {(d[('F0|Car|seed0', '07')]['rel_good_mean'] - 1) * 100:+.1f}%
  (F0) vs {(d[('Ours|Car|seed0', '07')]['rel_good_mean'] - 1) * 100:+.1f}% (Ours); that group has
  n_ok=32, n_ng=29. Note this single-group excursion is **larger under Ours**: the spread reduction
  above is a distribution-level property, **not** the removal of this peak. Recorded as a data
  fact; no mechanism is claimed for it.

**Reference (E18)**: identical values are reported in `results/e18/statistical_analysis.md` §B.1
(cross-checked in `../verification.csv`, group ILLUM).

**Interpretation boundary (mandatory wording)**
- Only the **spread / shape** across groups is interpretable. Absolute score levels are NOT
  comparable between F0 and Ours (R1-ILA changes the feature scale).
- This is reported as **"consistent with"** the E16 illumination-shift account. The correlation
  between the training objective (feature-level L_cons) and this downstream score-level quantity
  does **not** establish causality.
- Context sources (PatchCore-emb unified, PatchCore-official, AnomalyDINO, SoftPatch) are
  single-seed and have small per-group n; they are shown for context, never as a ranking.
"""
    (NOTES / "e19_illum_groups_notes.md").write_text(notes, encoding="utf-8")
    return rows, spreads


# --------------------------------------------------------------------------
# E19-3  main results table + figures (source: E18 main table + bootstrap)
# --------------------------------------------------------------------------
def build_main():
    mt = read_rows(E18D / "main_table.csv")
    bs = load_json(E18D / "bootstrap_f0_vs_ours.json")
    nat = read_rows(E18D / "native_scoring_appendix.csv")

    recs = {}
    for r in mt:
        recs[r["method"]] = r

    status_note = {
        "PatchCore-emb + unified scoring":
            "representation-only variant under the unified scoring; NOT the same configuration as PatchCore official",
        "AnomalyDINO": "official scoring; Cube filled in E18-P1 (Bird/Car from E14)",
        "SoftPatch": "official scoring; Cube filled in E18-P1 (Bird/Car from E14)",
        "EfficientAD": "Bird-only: E15 pre-registered Stage-A STOP; Car/Cube NOT RUN (no imputation)",
        "F0": "frozen DINOv2-B baseline; 3 seeds available (E16)",
        "Ours (R1-ILA)": "final frozen method, BETA=0.1; 3 seeds available (E16)",
    }
    header = ["method", "scoring", "n_seeds", "complete_3cat",
              "Bird_AUROC", "Bird_AP", "Car_AUROC", "Car_AP", "Cube_AUROC", "Cube_AP",
              "status_note", "source_file"]
    rows = []
    for m in METHOD_ORDER:
        r = recs[m]
        vals = [fget(r, f"{c}_{k}") for c in CATS for k in ("AUROC", "AP")]
        rows.append([m, r["scoring"], int(r["n_seeds"]), r["complete_3cat"]] +
                    [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]] +
                    [status_note[m], "results/e18/main_table.csv"])
    wcsv(TAB / "main_results.csv", header, rows)

    # markdown version
    md = ["# E19 Main results — M²AD view120 / seed 0 / 300-bank / official image protocol",
          "",
          "AUROC / AP at full precision; empty cells are `not run` (NOT estimated).",
          "F0 vs Ours additionally carry 3 seeds (E16); all baselines are single-seed.",
          "",
          "| method | Bird AUROC | Bird AP | Car AUROC | Car AP | Cube AUROC | Cube AP | complete | seeds |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{v:.5f}" if isinstance(v, float) else ("—" if v is None or v == "" else str(v))
                 for v in r[4:10]]
        md.append(f"| {r[0]} | " + " | ".join(cells) + f" | {r[3]} | {r[2]} |")
    md += ["", "Notes:",
           "- PatchCore official scoring (native) is Bird-only: Car/Cube `BLOCKED` "
           "(raw 1024px inputs unavailable); see `results/e18/native_scoring_appendix.csv`.",
           "- EfficientAD Car/Cube: `EARLY_STOP` (E15 pre-registered Stage-A STOP).",
           "- Blank cells are gaps, never imputed. Values verified against "
           "`results/e18/main_table.csv` at 1e-12 (`../verification.csv`, group MAIN).",
           "- All methods keep their native scoring; the main table is not a single-scoring leaderboard.",
           ""]
    (TAB / "main_results.md").write_text("\n".join(md), encoding="utf-8")

    # 3-seed extension
    h3 = ["category", "metric", "F0_mean", "F0_std", "Ours_mean", "Ours_std",
          "delta_seedmean", "delta_seed0", "delta_seed1", "delta_seed2",
          "ci_lo", "ci_hi", "p_gt0", "source_file"]
    rows3 = []
    rf0 = recs["F0"]
    rou = recs["Ours (R1-ILA)"]
    for c in CATS:
        for metric, tag in (("AUROC", "auroc"), ("AP", "ap")):
            rows3.append([
                c, metric,
                float(rf0[f"{c}_{metric}_3seed_mean"]), float(rf0[f"{c}_{metric}_3seed_std"]),
                float(rou[f"{c}_{metric}_3seed_mean"]), float(rou[f"{c}_{metric}_3seed_std"]),
                bs[f"{c}|seedmean"][f"d_{tag}"],
                bs[f"{c}|seed0"][f"d_{tag}"], bs[f"{c}|seed1"][f"d_{tag}"],
                bs[f"{c}|seed2"][f"d_{tag}"],
                bs[f"{c}|seedmean"][f"d_{tag}_ci"][0], bs[f"{c}|seedmean"][f"d_{tag}_ci"][1],
                bs[f"{c}|seedmean"][f"p_{tag}_gt0"],
                "results/e18/main_table.csv + results/e18/bootstrap_f0_vs_ours.json",
            ])
    wcsv(TAB / "main_results_3seed.csv", h3, rows3)

    # ---- figure: main AUROC bars ----
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    for ax, c in zip(axes, CATS):
        xs = np.arange(len(METHOD_ORDER))
        vals, cols, miss = [], [], []
        for m in METHOD_ORDER:
            v = fget(recs[m], f"{c}_AUROC")
            vals.append(v if v is not None else 0.0)
            cols.append(METHOD_COLOR[m])
            miss.append(v is None)
        for i, (v, mm) in enumerate(zip(vals, miss)):
            if mm:
                ax.add_patch(plt.Rectangle((i - 0.31, 0.455), 0.62, 0.46, fill=False,
                                           edgecolor="#999999", ls="--", lw=0.8))
                ax.text(i, 0.685, "not run\n(E15 stop)", ha="center", va="center",
                        fontsize=6.2, color="#777777", rotation=90, linespacing=1.4)
            else:
                ax.bar(xs[i], v, 0.62, color=cols[i], edgecolor="black", linewidth=0.5)
                ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=6.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER], fontsize=6.6, rotation=30, ha="right")
        ax.set_ylim(0.45, 0.94)
        ax.set_title(c)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0].set_ylabel("Image AUROC (seed 0)")
    fig.suptitle("Main results: image AUROC per category (M²AD view120, seed 0, 300-bank, official protocol)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig_main_auroc")

    # ---- figure: forest plot (paired bootstrap deltas) ----
    order_cells = [(c, s) for c in CATS for s in ["seed0", "seed1", "seed2", "seedmean"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, tag, ttl in ((axes[0], "auroc", "delta AUROC (Ours - F0)"),
                         (axes[1], "ap", "delta AP (Ours - F0)")):
        ypos = np.arange(len(order_cells))[::-1]
        for y, (c, s) in zip(ypos, order_cells):
            e = bs[f"{c}|{s}"]
            d, ci = e[f"d_{tag}"], e[f"d_{tag}_ci"]
            col = CAT_COLOR[c]
            star = s == "seedmean"
            ax.plot([ci[0], ci[1]], [y, y], color=col, lw=2.2 if star else 1.4,
                    alpha=1.0 if star else 0.75)
            ax.plot([d], [y], marker="D" if star else "o", ms=6 if star else 4.5, color=col,
                    markeredgecolor="black" if star else col, markeredgewidth=0.6)
            ax.text(0.058, y, f"{d:+.4f}", fontsize=6.4, va="center", ha="right", color=col)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{c} {s.replace('seedmean', 'seed-mean')}" for c, s in order_cells],
                           fontsize=7)
        ax.set_xlim(-0.038, 0.078)
        ax.set_xlabel(ttl)
        ax.set_title(f"{ttl}\n(paired bootstrap 95% CI, test-image resampling)")
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    fig.tight_layout()
    save(fig, "fig_main_forest")

    # ---- figure: 3-seed extension ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, metric in zip(axes, ["AUROC", "AP"]):
        xs = np.arange(3)
        w = 0.34
        f0m = [float(rf0[f"{c}_{metric}_3seed_mean"]) for c in CATS]
        f0s = [float(rf0[f"{c}_{metric}_3seed_std"]) for c in CATS]
        oum = [float(rou[f"{c}_{metric}_3seed_mean"]) for c in CATS]
        ous = [float(rou[f"{c}_{metric}_3seed_std"]) for c in CATS]
        ax.bar(xs - w / 2, f0m, w, yerr=f0s, capsize=3, label="F0 (DINOv2-B)",
               color=F0_COLOR, error_kw={"lw": 1.0})
        ax.bar(xs + w / 2, oum, w, yerr=ous, capsize=3, label="Ours (R1-ILA)",
               color=OURS_COLOR, error_kw={"lw": 1.0})
        ax.set_xticks(xs)
        ax.set_xticklabels(CATS)
        ax.set_ylim(0.70, 0.96)
        ax.set_ylabel(f"Image {metric} (mean ± std, 3 seeds)")
        ax.set_title(f"{metric}, 3-seed extension (F0 / Ours only)")
        ax.grid(alpha=0.25, lw=0.5, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_main_3seed")

    # ---- notes ----
    def cell(m, c, k):
        v = fget(recs[m], f"{c}_{k}")
        return "BLOCKED/not run" if v is None else f"{v:.5f}"

    dom = "\n".join(
        f"| {c} | {cell('Ours (R1-ILA)', c, 'AUROC')} | {cell('F0', c, 'AUROC')} | "
        f"{bs[f'{c}|seedmean']['d_auroc']:+.5f} "
        f"[{bs[f'{c}|seedmean']['d_auroc_ci'][0]:+.5f}, {bs[f'{c}|seedmean']['d_auroc_ci'][1]:+.5f}] | "
        f"{bs[f'{c}|seedmean']['p_auroc_gt0']:.3f} |" for c in CATS)

    notes = f"""# E19-3 · Main results table + figures — notes

**Source (read-only)**: `results/e18/main_table.csv` (final main table, full precision),
`results/e18/bootstrap_f0_vs_ours.json` (paired bootstrap), `results/e18/native_scoring_appendix.csv`.
**No new GPU run.**

**Claim served**: under the M²AD view120 / seed 0 / 300-bank / official image protocol,
R1-ILA (BETA=0.1) is above every comparable baseline in all three categories, and the
F0-vs-Ours gap is supported by a paired per-image bootstrap on the two methods that share
3 seeds and an identical bank/feature pipeline.

**Tables**
- `tables/main_results.csv` / `main_results.md` — final 6-row main table (full precision).
- `tables/main_results_3seed.csv` — 3-seed extension (F0 / Ours only) + per-seed deltas + CIs.

**Figures**
- `figures/fig_main_auroc.png/.pdf` — per-category AUROC bars, all methods; missing cells are
  marked with a neutral dashed frame and labelled (never imputed).
- `figures/fig_main_forest.png/.pdf` — paired bootstrap deltas (Ours − F0), AUROC and AP.
- `figures/fig_main_3seed.png/.pdf` — 3-seed mean ± std for F0 / Ours.

**Facts — seed-mean delta AUROC (Ours − F0), paired bootstrap**

| category | Ours AUROC (seed 0) | F0 AUROC (seed 0) | delta seed-mean [95% CI] | P(delta>0) |
|---|---|---|---|---|
{dom}

**Mandatory boundaries**
1. Baselines are **single-seed**; only F0 / Ours have 3 seeds. No claim that baselines are
   seed-stable.
2. The per-image bootstrap quantifies **test-set sampling uncertainty only** — it is NOT
   cross-seed uncertainty (seed variance is the ± std over 3 independent banks).
3. Empty cells are explicit gaps (`BLOCKED` / `EARLY_STOP`); no imputation, and the main table
   is not a single-scoring leaderboard (each method keeps its native scoring).
4. The main-table protocol on view120 uses official labels `image_anomaly` + `detectable`;
   legacy `object_anomaly` numbers (e.g. PatchCore Bird 0.78089) are excluded and quarantined
   in `results/e18/native_scoring_appendix.csv`.
5. Results are limited to M²AD Bird/Car/Cube, view120, 300-image bank; no cross-dataset or
   cross-view generalization is claimed here.
"""
    (NOTES / "e19_main_results_notes.md").write_text(notes, encoding="utf-8")
    return rows, rows3


def main():
    for p in (TAB, FIG, NOTES):
        p.mkdir(parents=True, exist_ok=True)
    data, f0 = build_beta()
    ill, spreads = build_illum()
    mrows, rows3 = build_main()
    print(json.dumps({
        "beta_rows": len(data), "illum_rows": len(ill), "main_rows": len(mrows),
        "spread": {f"{k[0]}|{k[1]}": v for k, v in spreads.items()},
        "out": str(OUT),
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
