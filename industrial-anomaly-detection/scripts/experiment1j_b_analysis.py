"""Experiment 1J-B — Transmission Intervention 分析。

输入：results/experiment_1j_b/raw/per_image.csv
产出：analysis/{per_defect,family_summary,transmission_summary}.csv + figures/5 张
Primary：β dose-response（radius → NN std → score proxy）
TRR 为 descriptive。
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_OUT = ROOT / "results" / "experiment_1j_b"
PI_CSV = B_OUT / "raw" / "per_image.csv"
ANALYSIS_DIR = B_OUT / "analysis"
FIG_DIR = B_OUT / "figures"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
BETAS = [0.0, 0.5, 1.0]

FROZEN_PRIMARY = {
    "shrink": [("cable", "bent_wire"), ("hazelnut", "print"), ("bottle", "contamination")],
    "neutral": [("screw", "manipulated_front"), ("screw", "thread_side"), ("screw", "thread_top")],
    "expand": [("grid", "glue"), ("grid", "metal_contamination"), ("grid", "thread")],
}
FAMILY_OF = {}
for fam, lst in FROZEN_PRIMARY.items():
    for cat, dt in lst:
        FAMILY_OF[(cat, dt)] = fam


def load() -> list[dict]:
    rows = list(csv.DictReader(open(PI_CSV, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["beta"] = float(r["beta"])
        for k in ["radius", "mean_nn", "std_nn", "median_nn", "score_proxy_max", "score_proxy_std"]:
            r[k] = float(r[k])
    return rows


def agg_per_defect(rows: list[dict]) -> list[dict]:
    """按 (cat,dt,seed,beta) 对 image 求 mean，再跨 seed 求 mean/sd。"""
    # 第一步：image -> (cat,dt,seed,beta)
    by = defaultdict(list)
    for r in rows:
        by[(r["category"], r["defect_type"], r["seed"], r["beta"])].append(r)
    seed_level = {}
    for (cat, dt, seed, beta), rs in by.items():
        seed_level[(cat, dt, seed, beta)] = {
            k: float(np.mean([r[k] for r in rs])) for k in
            ["radius", "mean_nn", "std_nn", "median_nn", "score_proxy_max", "score_proxy_std"]
        }
    # 第二步：跨 seed 聚合到 (cat,dt,beta)
    out = []
    for (cat, dt), _ in [((c, d), None) for c, d in FAMILY_OF.keys()]:
        for beta in BETAS:
            vals = [seed_level[(cat, dt, s, beta)] for s in [0, 1, 2] if (cat, dt, s, beta) in seed_level]
            if not vals:
                continue
            row = {"category": cat, "defect_type": dt, "family": FAMILY_OF[(cat, dt)], "beta": beta}
            for k in ["radius", "mean_nn", "std_nn", "median_nn", "score_proxy_max", "score_proxy_std"]:
                v = [x[k] for x in vals]
                row[f"{k}_mean"] = float(np.mean(v))
                row[f"{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            out.append(row)
    return out


def transmission_summary(per_defect: list[dict]) -> list[dict]:
    """每个 defect 计算 β=0 vs β=1 的 dose response 与 TRR。"""
    # 需要 α=0 / α=1 的原始值来算 TRR 分母：从 1J-A 的 geometry/NN csv 拿
    # 这里先用 β=0（≈α=0 radius）与 β=1（≈α=1 radius）作为 radius 基准；
    # NN 的原始 α 值从 1J-A nn csv 读。
    j_nn = ROOT / "results" / "experiment_1j" / "nn" / "experiment1j_nn_per_image.csv"
    nn_alpha = {}
    if j_nn.exists():
        for r in csv.DictReader(open(j_nn, encoding="utf-8")):
            if r["layer"] != "layer3":
                continue
            nn_alpha[(r["category"], r["defect_type"], int(r["seed"]), float(r["alpha"]))] = float(r["std_nn_distance"])

    out = []
    by_def = defaultdict(list)
    for r in per_defect:
        by_def[(r["category"], r["defect_type"])].append(r)

    for (cat, dt), rs in sorted(by_def.items()):
        fam = FAMILY_OF[(cat, dt)]
        d = {r["beta"]: r for r in rs}
        if not all(b in d for b in BETAS):
            continue
        row = {"category": cat, "defect_type": dt, "family": fam}
        for k in ["radius", "std_nn", "score_proxy_max", "score_proxy_std"]:
            row[f"{k}_b0"] = d[0.0][f"{k}_mean"]
            row[f"{k}_b05"] = d[0.5][f"{k}_mean"]
            row[f"{k}_b1"] = d[1.0][f"{k}_mean"]
        # dose：β=0 -> β=1 的绝对变化
        row["delta_radius_0to1"] = row["radius_b1"] - row["radius_b0"]
        row["delta_nnstd_0to1"] = row["std_nn_b1"] - row["std_nn_b0"]
        row["delta_score_0to1"] = row["score_proxy_max_b1"] - row["score_proxy_max_b0"]

        # TRR（descriptive）：分母用 1J-A 的 α=1 - α=0（原始 NN）
        trr_nn = float("nan")
        trr_score = float("nan")
        a0 = [nn_alpha.get((cat, dt, s, 0.0)) for s in [0, 1, 2]]
        a1 = [nn_alpha.get((cat, dt, s, 1.0)) for s in [0, 1, 2]]
        a0 = [x for x in a0 if x is not None]
        a1 = [x for x in a1 if x is not None]
        if a0 and a1:
            denom = np.mean(a1) - np.mean(a0)
            if abs(denom) > EPS:
                trr_nn = row["delta_nnstd_0to1"] / denom
        row["trr_nn"] = trr_nn
        row["trr_score"] = trr_score  # score 无独立 α 基准（score_proxy 仅本实验），留 nan
        out.append(row)
    return out


def family_summary(trans: list[dict]) -> list[dict]:
    out = []
    for fam in ["shrink", "neutral", "expand"]:
        rs = [r for r in trans if r["family"] == fam]
        row = {"family": fam, "n_defect": len(rs)}
        for k in ["delta_radius_0to1", "delta_nnstd_0to1", "delta_score_0to1", "trr_nn"]:
            vals = [r[k] for r in rs if not np.isnan(r[k])]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{k}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # 单调性计数：NN std 是否随 β 单调升
        mono_nn = sum(1 for r in rs if r["std_nn_b0"] <= r["std_nn_b05"] <= r["std_nn_b1"])
        row["n_monotonic_nn"] = mono_nn
        out.append(row)
    return out


def make_figures(trans: list[dict], per_defect: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_color = {"shrink": "#1f77b4", "neutral": "#7f7f7f", "expand": "#d62728"}
    fam_order = ["shrink", "neutral", "expand"]
    tmap = {(r["category"], r["defect_type"]): r for r in trans}

    def order9():
        labels = []
        for fam in fam_order:
            for cat, dt in FROZEN_PRIMARY[fam]:
                labels.append(dt)
        return labels

    labels = order9()

    # Figure 1 — expand dose-response（radius / NN std / score）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k, name in [
        (axes[0], "radius", "radius"),
        (axes[1], "std_nn", "NN std"),
        (axes[2], "score_proxy_max", "score proxy (max NN)"),
    ]:
        for fam in ["expand"]:
            for cat, dt in FROZEN_PRIMARY[fam]:
                r = tmap[(cat, dt)]
                ys = [r[f"{k}_b0"], r[f"{k}_b05"], r[f"{k}_b1"]]
                ax.plot([0, 0.5, 1], ys, "o-", label=dt, markersize=6)
        ax.set_xlabel("β"); ax.set_ylabel(name); ax.set_title(f"expand dose-response: {name}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Figure 1 — Expand intervention dose-response")
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure1_expand_dose_response.png", dpi=150); plt.close(fig)

    # Figure 2 — NN transmission（ΔNN_0to1 by defect，按 family 着色）
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(9)
    dnn = [tmap[(c, d)]["delta_nnstd_0to1"] for fam in fam_order for c, d in FROZEN_PRIMARY[fam]]
    cols = [fam_color[fam] for fam in fam_order for c, d in FROZEN_PRIMARY[fam]]
    ax.bar(x, dnn, color=cols, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ NN std (β=0 → β=1)")
    ax.set_title("Figure 2 — NN dispersion transmission")
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure2_nn_transmission.png", dpi=150); plt.close(fig)

    # Figure 3 — Score transmission
    fig, ax = plt.subplots(figsize=(10, 5))
    dsc = [tmap[(c, d)]["delta_score_0to1"] for fam in fam_order for c, d in FROZEN_PRIMARY[fam]]
    ax.bar(x, dsc, color=cols, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ score proxy (β=0 → β=1)")
    ax.set_title("Figure 3 — Score (proxy) transmission")
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure3_score_transmission.png", dpi=150); plt.close(fig)

    # Figure 4 — Family control（radius Δ vs NN Δ 散点）
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in fam_order:
        xs, ys, ds = [], [], []
        for cat, dt in FROZEN_PRIMARY[fam]:
            r = tmap[(cat, dt)]
            xs.append(r["delta_radius_0to1"]); ys.append(r["delta_nnstd_0to1"]); ds.append(dt)
        ax.scatter(xs, ys, color=fam_color[fam], label=fam, s=90)
        for xi, yi, di in zip(xs, ys, ds):
            ax.annotate(di, (xi, yi), fontsize=8, textcoords="offset points", xytext=(4, 4))
    ax.axhline(0, color="gray", lw=0.8, ls="--"); ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Δ radius (β=0 → β=1)"); ax.set_ylabel("Δ NN std (β=0 → β=1)")
    ax.set_title("Figure 4 — Family control: radius change vs NN change")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure4_family_control.png", dpi=150); plt.close(fig)

    # Figure 5 — Final chain（列内归一化的热力图 + 真实 Δ 标注）
    fig, ax = plt.subplots(figsize=(9, 4.5))
    stages = ["Layer3 geometry\n(Δradius)", "NN dispersion\n(Δstd)", "Score (proxy)\n(ΔmaxNN)"]
    fam_order5 = ["expand", "neutral", "shrink"]
    mat = np.zeros((3, 3))
    texts = [[""]*3 for _ in range(3)]
    for i, fam in enumerate(fam_order5):
        rs = [r for r in trans if r["family"] == fam]
        if not rs:
            continue
        vals = [np.mean([r["delta_radius_0to1"] for r in rs]),
                np.mean([r["delta_nnstd_0to1"] for r in rs]),
                np.mean([r["delta_score_0to1"] for r in rs])]
        mat[i, :] = vals
        texts[i] = [f"{v:+.2f}" for v in vals]
    # 列内归一化（每列除以该列最大绝对值）——只用于颜色，标注保持真实值
    mat_norm = mat / (np.max(np.abs(mat), axis=0, keepdims=True) + EPS)
    im = ax.imshow(mat_norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(fam_order5, fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5 — Final transmission chain\n(color: column-normalized; text: real Δ, β=0→β=1)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "figure5_final_chain.png", dpi=150); plt.close(fig)


def main():
    rows = load()
    print(f"per_image rows: {len(rows)}")
    per_defect = agg_per_defect(rows)
    trans = transmission_summary(per_defect)
    fam_sum = family_summary(trans)

    def write_csv(path, data, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write_csv(ANALYSIS_DIR / "per_defect.csv", per_defect,
              list(per_defect[0].keys()) if per_defect else [])
    write_csv(ANALYSIS_DIR / "transmission_summary.csv", trans,
              list(trans[0].keys()) if trans else [])
    write_csv(ANALYSIS_DIR / "family_summary.csv", fam_sum,
              list(fam_sum[0].keys()) if fam_sum else [])

    make_figures(trans, per_defect)

    # 打印核心
    print("\n===== transmission summary (Δ β0→β1) =====")
    for fam in ["shrink", "neutral", "expand"]:
        for cat, dt in FROZEN_PRIMARY[fam]:
            r = [x for x in trans if x["category"] == cat and x["defect_type"] == dt]
            if not r:
                continue
            r = r[0]
            print(f"  [{fam:7s}] {cat}/{dt}: Δradius={r['delta_radius_0to1']:+.3f} "
                  f"ΔNNstd={r['delta_nnstd_0to1']:+.3f} Δscore={r['delta_score_0to1']:+.3f} "
                  f"TRR_NN={r['trr_nn']:+.3f}")

    print("\n===== family summary =====")
    for r in fam_sum:
        print(f"  {r['family']:7s}: Δradius={r['delta_radius_0to1_mean']:+.3f} "
              f"ΔNNstd={r['delta_nnstd_0to1_mean']:+.3f} Δscore={r['delta_score_0to1_mean']:+.3f} "
              f"monoNN={r['n_monotonic_nn']}/{r['n_defect']}")

    print("\n[DONE] 1J-B analysis complete.")


if __name__ == "__main__":
    main()
