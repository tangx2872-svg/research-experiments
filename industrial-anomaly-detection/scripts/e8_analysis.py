#!/usr/bin/env python
"""E8 - stability aggregation, verdict, tables and figures (protocol s7 / s3.5)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import e8_common as E8  # noqa: E402

FIG = E8.OUT / "figures"
CELLS = ["image_auroc", "image_aupr", "d_prime", "pixel_auroc", "aupro"]


def load_units() -> dict:
    out = {}
    for f in sorted(E8.RAW.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("status") == "OK":
            out[(d["method"], d["category"], int(d["seed"]))] = d
    return out


def per_cell(units: dict) -> dict:
    cells = {}
    for (m, c, s), d in units.items():
        for v, met in d.get("per_view", {}).items():
            cells[(m, c, s, v)] = met
    return cells


def main() -> int:
    units = load_units()
    cells = per_cell(units)
    if not cells:
        print("[E8 analysis] no finished units yet", flush=True)
        return 1
    methods = sorted({k[0] for k in cells})
    cats = sorted({k[1] for k in cells})
    seeds = sorted({k[2] for k in cells})
    views = sorted({k[3] for k in cells})
    base = "P00"
    print(f"[E8 analysis] {len(units)} units, {len(cells)} cells | methods={methods} "
          f"cats={cats} seeds={seeds} views={views}", flush=True)

    # ---- absolute per-category means -------------------------------------------
    abs_table = []
    for c in cats:
        for m in methods:
            for metric in CELLS:
                vals = [cells[(m, c, s, v)][metric] for s in seeds for v in views
                        if (m, c, s, v) in cells and metric in cells[(m, c, s, v)]]
                vals = [x for x in vals if x is not None and np.isfinite(x)]
                if vals:
                    abs_table.append({"category": c, "method": m, "metric": metric,
                                      "n": len(vals), "mean": float(np.mean(vals)),
                                      "median": float(np.median(vals)),
                                      "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
    E8.save_json(E8.OUT / "abs_table.json", abs_table)

    # ---- deltas vs P00 ---------------------------------------------------------
    delta_table, per_cat, per_seed, per_view = [], {}, {}, {}
    pos_cell = {m: [0, 0] for m in methods}
    pos_cat_seed = {m: [0, 0] for m in methods}
    for m in methods:
        if m == base:
            continue
        for c in cats:
            for s in seeds:
                cs = []
                for v in views:
                    a, b = cells.get((m, c, s, v)), cells.get((base, c, s, v))
                    if not a or not b:
                        continue
                    d = {k: (a[k] - b[k]) if (k in a and k in b and a[k] is not None and b[k] is not None
                                              and np.isfinite(a[k]) and np.isfinite(b[k])) else None
                         for k in CELLS}
                    delta_table.append({"method": m, "category": c, "seed": s, "view": v, **d})
                    per_cat.setdefault((m, c), []).append(d["image_auroc"])
                    per_seed.setdefault((m, s), []).append(d["image_auroc"])
                    per_view.setdefault((m, v), []).append(d["image_auroc"])
                    if d["image_auroc"] is not None:
                        pos_cell[m][1] += 1
                        pos_cell[m][0] += int(d["image_auroc"] > 0)
                        cs.append(d["image_auroc"])
                if cs:
                    pos_cat_seed[m][1] += 1
                    pos_cat_seed[m][0] += int(float(np.mean(cs)) > 0)

    E8.save_json(E8.OUT / "delta_table.json", delta_table)

    cat_tbl = []
    for (m, c), v in sorted(per_cat.items()):
        a = E8.agg(v)
        # absolute means for the same scope
        ab = {mm: E8.agg([cells[(mm, c, s, vv)]["image_auroc"] for s in seeds for vv in views
                          if (mm, c, s, vv) in cells])["mean"] for mm in methods}
        cat_tbl.append({"category": c, "method": m, "n": a["n"], "mean_d_auroc": a["mean"],
                        "median_d_auroc": a["median"], "std_d_auroc": a["std"],
                        "worst_cell": a["min"], "best_cell": a["max"],
                        "positive_views": sum(1 for vv in views
                                              if (m, c, 0, vv) in cells
                                              and (base, c, 0, vv) in cells
                                              and cells[(m, c, 0, vv)]["image_auroc"]
                                              > cells[(base, c, 0, vv)]["image_auroc"]),
                        "n_views": len(views), **{f"abs_{k}": val for k, val in ab.items()}})
    E8.save_json(E8.OUT / "per_category.json", cat_tbl)

    seed_tbl = [{"method": m, "seed": s, "n": (a := E8.agg(v))["n"], "mean_d_auroc": a["mean"],
                 "median_d_auroc": a["median"], "std_d_auroc": a["std"]}
                for (m, s), v in sorted(per_seed.items())]
    E8.save_json(E8.OUT / "per_seed.json", seed_tbl)

    view_tbl = [{"method": m, "view": v, "n": (a := E8.agg(x))["n"], "mean_d_auroc": a["mean"],
                 "median_d_auroc": a["median"], "std_d_auroc": a["std"]}
                for (m, v), x in sorted(per_view.items())]
    E8.save_json(E8.OUT / "per_view.json", view_tbl)

    # ---- global verdict --------------------------------------------------------
    cv = E8.core_verdict(cells)
    cv["positive_cells"] = {m: {"pos": pos_cell[m][0], "tot": pos_cell[m][1],
                                "rate": (pos_cell[m][0] / pos_cell[m][1]) if pos_cell[m][1] else None}
                            for m in methods if m != base}
    cv["positive_category_seed"] = {m: {"pos": pos_cat_seed[m][0], "tot": pos_cat_seed[m][1],
                                        "rate": (pos_cat_seed[m][0] / pos_cat_seed[m][1])
                                        if pos_cat_seed[m][1] else None}
                                    for m in methods if m != base}
    cv["n_units"] = len(units)
    cv["categories"] = cats
    cv["seeds"] = seeds
    cv["views"] = views
    cv["methods"] = methods
    E8.save_json(E8.OUT / "verdict_final.json", cv)
    print(json.dumps({m: {"mean_dAUROC": cv["per_method"][m]["cells"]["mean"],
                          "ci": cv["per_method"][m]["bootstrap_ci"],
                          "verdict": cv["per_method"][m]["verdict"],
                          "pos_cats": f"{cv['per_method'][m]['positive_categories']}/"
                                      f"{cv['per_method'][m]['n_categories']}",
                          "mean_dd_prime": cv["per_method"][m]["mean_d_dprime"]}
                      for m in cv["per_method"]}, indent=2, default=float), flush=True)
    print(f"[E8 analysis] E8 VERDICT = {cv['E8']} (primary {cv['primary']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------------ figures
def _amean(cells, m, c, seeds, views, metric="image_auroc"):
    v = [cells[(m, c, s, vv)][metric] for s in seeds for vv in views
         if (m, c, s, vv) in cells and metric in cells[(m, c, s, vv)]]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


COLORS = {"P00": "#4c4c4c", "P02": "#2e86de", "P03": "#e8590c"}


def fig1(methods, cats, seeds, views, cells):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        xs = np.arange(len(cats)) + (i - (len(methods) - 1) / 2) * w
        ys = [_amean(cells, m, c, seeds, views) for c in cats]
        ax.bar(xs, ys, width=w, color=COLORS.get(m), edgecolor="black", linewidth=0.4,
               label=f"{m} {E8.METHOD_DESC.get(m, m)}")
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats)
    ax.set_ylabel("mean image AUROC")
    ax.set_title("E8 Fig 1 - per-category performance (seeds x views pooled)")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    p = FIG / "fig1_per_category_auroc.png"; fig.tight_layout(); fig.savefig(p, dpi=160)
    plt.close(fig); return p


def fig2(cands, cats, seeds, views, cells, base="P00"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cands), figsize=(5.2 * len(cands), 3.6), squeeze=False)
    for ax, m in zip(axes[0], cands):
        M = np.full((len(cats), len(views)), np.nan)
        for i, c in enumerate(cats):
            for j, v in enumerate(views):
                d = [cells[(m, c, s, v)]["image_auroc"] - cells[(base, c, s, v)]["image_auroc"]
                     for s in seeds if (m, c, s, v) in cells and (base, c, s, v) in cells]
                if d:
                    M[i, j] = float(np.mean(d))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.06, vmax=0.06, aspect="auto")
        ax.set_xticks(range(len(views))); ax.set_xticklabels(views)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
        ax.set_title(f"{m} $\\Delta$AUROC (view x category)")
        for i in range(len(cats)):
            for j in range(len(views)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("E8 Fig 2 - CLAHE gain by illumination view (negative cells NOT hidden)", fontsize=10)
    p = FIG / "fig2_delta_heatmap.png"; fig.tight_layout(); fig.savefig(p, dpi=160)
    plt.close(fig); return p


def fig3(cands, cats, seeds, views, cells, base="P00"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mks = ["o", "s", "^", "D", "v"]
    for mi, m in enumerate(cands):
        for ci, c in enumerate(cats):
            for si, s in enumerate(seeds):
                d = [cells[(m, c, s, v)]["image_auroc"] - cells[(base, c, s, v)]["image_auroc"]
                     for v in views if (m, c, s, v) in cells and (base, c, s, v) in cells]
                if not d:
                    continue
                x = ci + (mi - (len(cands) - 1) / 2) * 0.16
                ax.scatter([x], [float(np.mean(d))], marker=mks[mi % len(mks)], s=46,
                           color=COLORS.get(m), alpha=0.85,
                           label=m if (ci == 0 and si == 0) else None)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
    ax.set_ylabel("mean $\\Delta$AUROC of one seed")
    ax.set_title("E8 Fig 3 - seed consistency (one point per category x seed)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = FIG / "fig3_seed_stability.png"; fig.tight_layout(); fig.savefig(p, dpi=160)
    plt.close(fig); return p


def fig4(cands, cats, seeds, views, cells, base="P00"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    mks = ["o", "s", "^", "D", "v"]
    for mi, m in enumerate(cands):
        for c in cats:
            for s in seeds:
                da = [cells[(m, c, s, v)]["image_auroc"] - cells[(base, c, s, v)]["image_auroc"]
                      for v in views if (m, c, s, v) in cells and (base, c, s, v) in cells]
                dd = []
                for v in views:
                    if (m, c, s, v) in cells and (base, c, s, v) in cells:
                        x, y = cells[(m, c, s, v)].get("d_prime"), cells[(base, c, s, v)].get("d_prime")
                        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                            dd.append(x - y)
                if not da:
                    continue
                x = float(np.mean(da)); y = float(np.mean(dd)) if dd else float("nan")
                ax.scatter([x], [y], marker=mks[mi % len(mks)], s=54, color=COLORS.get(m),
                           alpha=0.85, label=m if (c == cats[0] and s == seeds[0]) else None)
                ax.annotate(f"{c[:4]}\ns{s}", (x, y), fontsize=5.5, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("$\\Delta$AUROC (robustness)"); ax.set_ylabel("$\\Delta d'$ (defect preservation)")
    ax.set_title("E8 Fig 4 - robustness vs preservation (one point per category x seed)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = FIG / "fig4_robustness_vs_preservation.png"; fig.tight_layout(); fig.savefig(p, dpi=160)
    plt.close(fig); return p


def fig5(cands, cats, seeds, views, cells, base="P00"):
    import matplotlib.pyplot as plt
    ncol = len(seeds) * len(views)
    fig, axes = plt.subplots(1, len(cands), figsize=(4.6 * len(cands), 0.5 * len(cats) + 2.2),
                             squeeze=False)
    labels = [f"s{s}:{v}" for s in seeds for v in views]
    for ax, m in zip(axes[0], cands):
        M = np.full((len(cats), ncol), np.nan)
        for i, c in enumerate(cats):
            for k, s in enumerate(seeds):
                for j, v in enumerate(views):
                    if (m, c, s, v) in cells and (base, c, s, v) in cells:
                        M[i, k * len(views) + j] = (cells[(m, c, s, v)]["image_auroc"]
                                                    - cells[(base, c, s, v)]["image_auroc"])
        im = ax.imshow(M, cmap="RdYlGn", vmin=-0.06, vmax=0.06, aspect="auto")
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=8)
        ax.set_xticks(range(ncol)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"{m} win/loss (seed:view)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"E8 Fig 5 - win/loss across {len(cats)} categories (negative cells kept)",
                 fontsize=10)
    p = FIG / "fig5_winloss_heatmap.png"; fig.tight_layout(); fig.savefig(p, dpi=160)
    plt.close(fig); return p


def make_figures(units: dict, cells: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    methods = sorted({k[0] for k in cells})
    cats = sorted({k[1] for k in cells})
    seeds = sorted({k[2] for k in cells})
    views = sorted({k[3] for k in cells})
    cands = [m for m in methods if m != "P00"]
    FIG.mkdir(parents=True, exist_ok=True)
    made = [fig1(methods, cats, seeds, views, cells),
            fig2(cands, cats, seeds, views, cells),
            fig3(cands, cats, seeds, views, cells),
            fig4(cands, cats, seeds, views, cells)]
    if len(cats) >= 4:
        made.append(fig5(cands, cats, seeds, views, cells))
    return made


def main2() -> int:
    units = load_units()
    cells = per_cell(units)
    if not cells:
        return 1
    for p in make_figures(units, cells):
        print(f"[E8 analysis] figure {p}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        main2()
    raise SystemExit(rc)
