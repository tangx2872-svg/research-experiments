"""Experiment 1J-A — Feature Geometry Probe 分析。

输入：
  results/experiment_1j/per_image/experiment1j_geometry_per_image.csv   （逐 image 几何统计）
  results/experiment_1j/nn/experiment1j_nn_per_image.csv                （逐 image 分层 NN 统计）
  results/experiment_1hs/analysis/dispersion_layer_summary.csv          （score dispersion，已有）
  results/experiment_1g/analysis/defect_summary.csv                     （post_concat NN，seed0，historical ref）

产出（results/experiment_1j/analysis/）：
  experiment1j_geometry_per_seed.csv
  experiment1j_geometry_per_defect.csv
  experiment1j_family_summary.csv
  experiment1j_cross_chain_summary.csv
  figures/ figure1..figure5

Primary metric = GAI_radius = R_radius_L3 - R_radius_L2（within-layer log-ratio）。
判据（冻结，见 README）CASE A/B/C。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
J_OUT = PROJECT_ROOT / "results" / "experiment_1j"
PI_CSV = J_OUT / "per_image" / "experiment1j_geometry_per_image.csv"
NN_CSV = J_OUT / "nn" / "experiment1j_nn_per_image.csv"
SEL_CSV = PROJECT_ROOT / "results" / "experiment_1g" / "selection.csv"
HS_SUMMARY = PROJECT_ROOT / "results" / "experiment_1hs" / "analysis" / "dispersion_layer_summary.csv"
G_SUMMARY = PROJECT_ROOT / "results" / "experiment_1g" / "analysis" / "defect_summary.csv"

ANALYSIS_DIR = J_OUT / "analysis"
FIG_DIR = J_OUT / "figures"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
LAYERS = ["layer2", "layer3"]
ALPHAS = [0.0, 1.0]

FROZEN_PRIMARY = {
    "shrink": [("cable", "bent_wire"), ("hazelnut", "print"), ("bottle", "contamination")],
    "neutral": [("screw", "manipulated_front"), ("screw", "thread_side"), ("screw", "thread_top")],
    "expand": [("grid", "glue"), ("grid", "metal_contamination"), ("grid", "thread")],
}
FAMILY_OF = {}
for fam, lst in FROZEN_PRIMARY.items():
    for cat, dt in lst:
        FAMILY_OF[(cat, dt)] = fam


def load_pi() -> list[dict]:
    rows = list(csv.DictReader(open(PI_CSV, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha"] = float(r["alpha"])
        for k in ["mdc", "rms_radius", "normalized_pr", "pca1_ratio", "feature_norm"]:
            r[k] = float(r[k])
    return rows


def load_nn() -> list[dict]:
    if not NN_CSV.exists():
        return []
    rows = list(csv.DictReader(open(NN_CSV, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha"] = float(r["alpha"])
        for k in ["mean_nn_distance", "std_nn_distance", "median_nn_distance"]:
            r[k] = float(r[k])
    return rows


def load_hs_summary() -> dict[tuple[str, str, str], dict]:
    """{(category, defect_type, location): {delta_std_mean, ...}}"""
    out = {}
    for r in csv.DictReader(open(HS_SUMMARY, encoding="utf-8")):
        key = (r["category"], r["defect_type"], r["location"])
        entry = {}
        for k, v in r.items():
            if k in ("category", "defect_type", "location", "seeds"):
                entry[k] = v  # 非数值字段原样保留
            else:
                try:
                    entry[k] = float(v)
                except ValueError:
                    entry[k] = v
        out[key] = entry
    return out


def load_g_nn() -> dict[tuple[str, str], dict]:
    """{(category, defect_type): {delta_nn_std, delta_dispersion}}，post_concat seed0 ref"""
    out = {}
    for r in csv.DictReader(open(G_SUMMARY, encoding="utf-8")):
        out[(r["category"], r["defect_type"])] = {
            "delta_nn_std": float(r["delta_nn_std"]),
            "delta_dispersion": float(r["delta_dispersion"]),
        }
    return out


# ---------------------------------------------------------------------------
# per-image -> per-seed 聚合
# ---------------------------------------------------------------------------
def aggregate_per_seed(pi_rows: list[dict]) -> list[dict]:
    """按 (category, defect, seed, layer, alpha) 对 image-level metric 求 mean。

    返回 per-seed 行，含 l2_a0/l2_a1/l3_a0/l3_a1 及 response / gai。
    """
    # 先按 (cat,dt,seed,layer,alpha) 聚合 image mean
    agg = defaultdict(list)
    for r in pi_rows:
        if (r["category"], r["defect_type"]) not in FAMILY_OF:
            continue
        agg[(r["category"], r["defect_type"], r["seed"], r["layer"], r["alpha"])].append(r)

    # 每个 (cat,dt,seed) 一个输出行
    by_def_seed = defaultdict(dict)
    for (cat, dt, seed, layer, alpha), rs in agg.items():
        for metric in ["mdc", "rms_radius", "normalized_pr", "pca1_ratio", "feature_norm"]:
            vals = [r[metric] for r in rs]
            by_def_seed[(cat, dt, seed)][(layer, alpha, metric)] = float(np.mean(vals))

    out = []
    for (cat, dt, seed), d in sorted(by_def_seed.items()):
        fam = FAMILY_OF[(cat, dt)]
        def get(layer, alpha, metric):
            return d.get((layer, alpha, metric), float("nan"))

        row = {
            "category": cat, "defect_type": dt, "family": fam, "seed": seed,
            "mdc_l2_a0": get("layer2", 0.0, "mdc"), "mdc_l2_a1": get("layer2", 1.0, "mdc"),
            "mdc_l3_a0": get("layer3", 0.0, "mdc"), "mdc_l3_a1": get("layer3", 1.0, "mdc"),
            "radius_l2_a0": get("layer2", 0.0, "rms_radius"), "radius_l2_a1": get("layer2", 1.0, "rms_radius"),
            "radius_l3_a0": get("layer3", 0.0, "rms_radius"), "radius_l3_a1": get("layer3", 1.0, "rms_radius"),
            "pr_l2_a0": get("layer2", 0.0, "normalized_pr"), "pr_l2_a1": get("layer2", 1.0, "normalized_pr"),
            "pr_l3_a0": get("layer3", 0.0, "normalized_pr"), "pr_l3_a1": get("layer3", 1.0, "normalized_pr"),
            "pca1_l2_a0": get("layer2", 0.0, "pca1_ratio"), "pca1_l2_a1": get("layer2", 1.0, "pca1_ratio"),
            "pca1_l3_a0": get("layer3", 0.0, "pca1_ratio"), "pca1_l3_a1": get("layer3", 1.0, "pca1_ratio"),
        }
        # response（log-ratio）
        def log_ratio(a1, a0):
            if np.isnan(a1) or np.isnan(a0):
                return float("nan")
            return float(np.log((a1 + EPS) / (a0 + EPS)))

        row["radius_response_l2"] = log_ratio(row["radius_l2_a1"], row["radius_l2_a0"])
        row["radius_response_l3"] = log_ratio(row["radius_l3_a1"], row["radius_l3_a0"])
        row["gai_radius"] = row["radius_response_l3"] - row["radius_response_l2"]
        row["mdc_response_l2"] = log_ratio(row["mdc_l2_a1"], row["mdc_l2_a0"])
        row["mdc_response_l3"] = log_ratio(row["mdc_l3_a1"], row["mdc_l3_a0"])
        row["gai_mdc"] = row["mdc_response_l3"] - row["mdc_response_l2"]
        row["pr_response_l2"] = log_ratio(row["pr_l2_a1"], row["pr_l2_a0"])
        row["pr_response_l3"] = log_ratio(row["pr_l3_a1"], row["pr_l3_a0"])
        row["pca1_response_l2"] = row["pca1_l2_a1"] - row["pca1_l2_a0"]
        row["pca1_response_l3"] = row["pca1_l3_a1"] - row["pca1_l3_a0"]
        out.append(row)
    return out


def aggregate_per_defect(per_seed: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in per_seed:
        by[(r["category"], r["defect_type"])].append(r)
    out = []
    for (cat, dt), rs in sorted(by.items()):
        fam = FAMILY_OF[(cat, dt)]
        row = {"category": cat, "defect_type": dt, "family": fam, "n_seed": len(rs)}
        for metric in ["radius_response_l2", "radius_response_l3", "gai_radius",
                       "mdc_response_l2", "mdc_response_l3", "gai_mdc",
                       "pr_response_l2", "pr_response_l3",
                       "pca1_response_l2", "pca1_response_l3"]:
            vals = [r[metric] for r in rs if not np.isnan(r[metric])]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_seeds"] = ",".join(f"{r[metric]:.4f}" for r in rs if not np.isnan(r[metric]))
        # sign consistency of gai_radius
        gais = [r["gai_radius"] for r in rs if not np.isnan(r["gai_radius"])]
        if gais:
            pos = sum(1 for g in gais if g > 0)
            neg = sum(1 for g in gais if g < 0)
            row["gai_sign_consistency"] = max(pos, neg) / len(gais)
        else:
            row["gai_sign_consistency"] = float("nan")
        out.append(row)
    return out


def family_summary(per_defect: list[dict]) -> list[dict]:
    out = []
    for fam in ["shrink", "neutral", "expand"]:
        rs = [r for r in per_defect if r["family"] == fam]
        row = {"family": fam, "n_defect": len(rs)}
        for metric in ["gai_radius", "gai_mdc", "radius_response_l2", "radius_response_l3"]:
            vals = [r[f"{metric}_mean"] for r in rs if not np.isnan(r[f"{metric}_mean"])]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        gais = [r["gai_radius_mean"] for r in rs if not np.isnan(r["gai_radius_mean"])]
        if gais:
            row["gai_all_positive"] = int(all(g > 0 for g in gais))
            row["gai_all_negative"] = int(all(g < 0 for g in gais))
        else:
            row["gai_all_positive"] = row["gai_all_negative"] = 0
        out.append(row)
    return out


def cross_chain(per_seed: list[dict], nn_rows: list[dict],
                hs: dict, g_nn: dict) -> list[dict]:
    """对齐 geometry / NN dispersion / score dispersion / LSI。"""
    # 分层 NN response per (cat,dt,seed,layer)
    nn_agg = defaultdict(list)
    for r in nn_rows:
        if (r["category"], r["defect_type"]) not in FAMILY_OF:
            continue
        nn_agg[(r["category"], r["defect_type"], r["seed"], r["layer"])].append(r)
    nn_resp = {}
    for (cat, dt, seed, layer), rs in nn_agg.items():
        by_a = {}
        for r in rs:
            by_a[r["alpha"]] = r["std_nn_distance"]
        if 0.0 in by_a and 1.0 in by_a:
            nn_resp[(cat, dt, seed, layer)] = by_a[1.0] - by_a[0.0]

    # 每 defect 聚合（跨 seed mean）
    by_def = defaultdict(list)
    for r in per_seed:
        by_def[(r["category"], r["defect_type"])].append(r)

    out = []
    for (cat, dt), rs in sorted(by_def.items()):
        fam = FAMILY_OF[(cat, dt)]
        def mmean(metric):
            vals = [r[metric] for r in rs if not np.isnan(r[metric])]
            return float(np.mean(vals)) if vals else float("nan")

        row = {
            "category": cat, "defect_type": dt, "family": fam,
            "geometry_radius_response_l2": mmean("radius_response_l2"),
            "geometry_radius_response_l3": mmean("radius_response_l3"),
            "gai_radius": mmean("gai_radius"),
            "geometry_mdc_response_l2": mmean("mdc_response_l2"),
            "geometry_mdc_response_l3": mmean("mdc_response_l3"),
            "gai_mdc": mmean("gai_mdc"),
            "effective_rank_response_l2": mmean("pr_response_l2"),
            "effective_rank_response_l3": mmean("pr_response_l3"),
            "pca1_response_l2": mmean("pca1_response_l2"),
            "pca1_response_l3": mmean("pca1_response_l3"),
        }
        # 分层 NN response（跨 seed mean）
        for layer in LAYERS:
            vals = [nn_resp[(cat, dt, s, layer)] for s in [0, 1, 2]
                    if (cat, dt, s, layer) in nn_resp]
            row[f"nn_dispersion_response_{layer}"] = float(np.mean(vals)) if vals else float("nan")
        # score dispersion（1H-S）
        for layer in LAYERS:
            key = (cat, dt, layer)
            row[f"score_dispersion_response_{layer}"] = hs.get(key, {}).get("delta_std_mean", float("nan"))
        # LSI_std（1H-S 的 lsi_std.csv 没有直接存这里，用 delta_std 重算）
        d2 = row["score_dispersion_response_layer2"]
        d3 = row["score_dispersion_response_layer3"]
        if not (np.isnan(d2) or np.isnan(d3)):
            row["lsi_std_existing"] = (abs(d3) - abs(d2)) / (abs(d3) + abs(d2) + EPS)
        else:
            row["lsi_std_existing"] = float("nan")
        # 1G post_concat NN（seed0，historical ref）
        gn = g_nn.get((cat, dt), {})
        row["1G_postconcat_seed0_delta_nn_std"] = gn.get("delta_nn_std", float("nan"))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_figures(per_defect: list[dict], cross: list[dict], per_seed: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_color = {"shrink": "#1f77b4", "neutral": "#7f7f7f", "expand": "#d62728"}
    fam_order = ["shrink", "neutral", "expand"]

    def order9():
        labels, cats = [], []
        for fam in fam_order:
            for cat, dt in FROZEN_PRIMARY[fam]:
                labels.append(f"{dt}")
                cats.append(fam)
        return labels, cats

    # Figure 1 — Radius Response by Defect（paired L2/L3）
    labels, cats = order9()
    pd_map = {(r["category"], r["defect_type"]): r for r in per_defect}
    x = np.arange(9)
    l2 = [pd_map[(c, d)]["radius_response_l2_mean"] for c, d in
          [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    l3 = [pd_map[(c, d)]["radius_response_l3_mean"] for c, d in
          [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fam in enumerate(cats):
        ax.axvspan(i - 0.4, i + 0.4, color=fam_color[fam], alpha=0.08)
    ax.plot(x, l2, "o-", color="#555", label="Layer2", markersize=6)
    ax.plot(x, l3, "s-", color="#d62728", label="Layer3", markersize=6)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("log radius response  R = log(r_a1 / r_a0)")
    ax.set_title("Figure 1 — Radius Response by Defect (paired L2 vs L3)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure1_radius_response.png", dpi=150); plt.close(fig)

    # Figure 2 — GAI_radius by defect
    gai = [pd_map[(c, d)]["gai_radius_mean"] for c, d in
           [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (g, fam) in enumerate(zip(gai, cats)):
        ax.bar(i, g, color=fam_color[fam], alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("GAI_radius = R_L3 - R_L2")
    ax.set_title("Figure 2 — Geometry Amplification Index (GAI_radius)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure2_gai.png", dpi=150); plt.close(fig)

    # Figure 3 — Geometry Components (effective rank / PCA1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pr_l2 = [pd_map[(c, d)]["pr_response_l2_mean"] for c, d in
             [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    pr_l3 = [pd_map[(c, d)]["pr_response_l3_mean"] for c, d in
             [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    axes[0].plot(x, pr_l2, "o-", color="#555", label="Layer2")
    axes[0].plot(x, pr_l3, "s-", color="#d62728", label="Layer3")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("normalized-PR log response"); axes[0].set_title("Effective rank response")
    axes[0].legend()
    p1_l2 = [pd_map[(c, d)]["pca1_response_l2_mean"] for c, d in
             [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    p1_l3 = [pd_map[(c, d)]["pca1_response_l3_mean"] for c, d in
             [item for fam in fam_order for item in FROZEN_PRIMARY[fam]]]
    axes[1].plot(x, p1_l2, "o-", color="#555", label="Layer2")
    axes[1].plot(x, p1_l3, "s-", color="#d62728", label="Layer3")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("PCA1 ratio Δ (a1 - a0)"); axes[1].set_title("PCA1 (anisotropy) response")
    axes[1].legend()
    fig.suptitle("Figure 3 — Geometry Components (spread vs anisotropy)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure3_components.png", dpi=150); plt.close(fig)

    # Figure 4 — Geometry -> NN alignment（layer-matched）
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in fam_order:
        xs, ys, ds = [], [], []
        for cat, dt in FROZEN_PRIMARY[fam]:
            r = [c for c in cross if c["category"] == cat and c["defect_type"] == dt][0]
            if np.isnan(r["gai_radius"]) or np.isnan(r.get("nn_dispersion_response_layer2")) or np.isnan(r.get("nn_dispersion_response_layer3")):
                continue
            nn_gai = r["nn_dispersion_response_layer3"] - r["nn_dispersion_response_layer2"]
            xs.append(r["gai_radius"]); ys.append(nn_gai); ds.append(dt)
        ax.scatter(xs, ys, color=fam_color[fam], label=fam, s=80)
        for xi, yi, di in zip(xs, ys, ds):
            ax.annotate(di, (xi, yi), fontsize=8, textcoords="offset points", xytext=(4, 4))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("GAI_radius (geometry)")
    ax.set_ylabel("NN_response_L3 - NN_response_L2")
    ax.set_title("Figure 4 — Geometry -> NN alignment (descriptive)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure4_geometry_nn.png", dpi=150); plt.close(fig)

    # Figure 5 — Cross-chain heatmap（标准化后）
    metrics = ["geometry_radius_response_l2", "geometry_radius_response_l3", "gai_radius",
               "nn_dispersion_response_layer2", "nn_dispersion_response_layer3",
               "score_dispersion_response_layer2", "score_dispersion_response_layer3"]
    mat = np.full((9, len(metrics)), np.nan)
    for i, (fam) in enumerate(fam_order):
        for cat, dt in FROZEN_PRIMARY[fam]:
            r = [c for c in cross if c["category"] == cat and c["defect_type"] == dt][0]
            row_idx = [item for f2 in fam_order for item in FROZEN_PRIMARY[f2]].index((cat, dt))
            for j, m in enumerate(metrics):
                mat[row_idx, j] = r.get(m, np.nan)
    # 每列 z-score（用 9 defect）
    mat_z = np.full_like(mat, np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mu, sd = np.nanmean(col), np.nanstd(col)
        if sd > 0:
            mat_z[:, j] = (col - mu) / sd
        else:
            mat_z[:, j] = 0.0
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat_z, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(9):
        for j in range(len(metrics)):
            v = mat_z[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Figure 5 — Cross-chain overview (z-scored per column)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure5_cross_chain.png", dpi=150); plt.close(fig)


def main():
    pi_rows = load_pi()
    nn_rows = load_nn()
    hs = load_hs_summary()
    g_nn = load_g_nn()
    print(f"per_image rows: {len(pi_rows)}")
    print(f"nn rows: {len(nn_rows)}")
    print(f"hs summary entries: {len(hs)}")

    per_seed = aggregate_per_seed(pi_rows)
    per_defect = aggregate_per_defect(per_seed)
    fam_sum = family_summary(per_defect)
    cross = cross_chain(per_seed, nn_rows, hs, g_nn)

    # 写 CSV
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    seed_fields = list(per_seed[0].keys()) if per_seed else []
    write_csv(ANALYSIS_DIR / "experiment1j_geometry_per_seed.csv", per_seed, seed_fields)
    defect_fields = list(per_defect[0].keys()) if per_defect else []
    write_csv(ANALYSIS_DIR / "experiment1j_geometry_per_defect.csv", per_defect, defect_fields)
    fam_fields = list(fam_sum[0].keys()) if fam_sum else []
    write_csv(ANALYSIS_DIR / "experiment1j_family_summary.csv", fam_sum, fam_fields)
    cross_fields = list(cross[0].keys()) if cross else []
    write_csv(ANALYSIS_DIR / "experiment1j_cross_chain_summary.csv", cross, cross_fields)

    make_figures(per_defect, cross, per_seed)

    # 打印核心结果
    print("\n===== 9 primary defects GAI_radius =====")
    for fam in ["shrink", "neutral", "expand"]:
        for cat, dt in FROZEN_PRIMARY[fam]:
            r = [x for x in per_defect if x["category"] == cat and x["defect_type"] == dt][0]
            print(f"  [{fam:7s}] {cat}/{dt}: GAI={r['gai_radius_mean']:+.4f} "
                  f"(R_L2={r['radius_response_l2_mean']:+.4f}, R_L3={r['radius_response_l3_mean']:+.4f}, "
                  f"consist={r['gai_sign_consistency']:.2f})")

    print("\n===== family summary =====")
    for r in fam_sum:
        print(f"  {r['family']:7s}: GAI_mean={r['gai_radius_mean']:+.4f} "
              f"all_pos={r['gai_all_positive']} all_neg={r['gai_all_negative']}")

    print("\n===== cross-chain (expand) =====")
    for r in cross:
        if r["family"] == "expand":
            print(f"  {r['category']}/{r['defect_type']}: gai={r['gai_radius']:+.4f} "
                  f"nn_L2={r.get('nn_dispersion_response_layer2', float('nan')):+.4f} "
                  f"nn_L3={r.get('nn_dispersion_response_layer3', float('nan')):+.4f} "
                  f"score_L2={r['score_dispersion_response_layer2']:+.4f} "
                  f"score_L3={r['score_dispersion_response_layer3']:+.4f}")

    print("\n[DONE] analysis complete.")


if __name__ == "__main__":
    main()
