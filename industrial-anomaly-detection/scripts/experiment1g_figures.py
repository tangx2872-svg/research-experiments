"""
Experiment 1G — 4 张核心图（协议 figures 字段）
Figure 1: 1E Δdefect_std vs 1G ΔNN-distance std（核心关联，CASE 判定的 LEVEL 1）
Figure 2: shrink/neutral/expand 三组的 NN-dispersion α 轨迹
Figure 3: feature geometry（dispersion / norm / channel var 的 Δ 与 1E 的对应）
Figure 4: defect vs background 的 NN-dispersion 对比
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "experiment_1g" / "analysis"
FIG_DIR = ROOT / "results" / "experiment_1g" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATS = ["bottle", "grid", "cable", "screw", "hazelnut"]
CAT_COLOR = {"bottle": "#1f77b4", "grid": "#d62728", "cable": "#2ca02c",
             "screw": "#9467bd", "hazelnut": "#ff7f0e"}
CAT_MARKER = {"bottle": "o", "grid": "s", "cable": "^", "screw": "D", "hazelnut": "v"}
FAM_COLOR = {"shrink": "#d62728", "neutral": "#7f7f7f", "expand": "#2ca02c"}
FAM_MARKER = {"shrink": "v", "neutral": "o", "expand": "^"}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_summary() -> list[dict]:
    return list(csv.DictReader(open(ANALYSIS / "defect_summary.csv", encoding="utf-8")))


def load_1e_response() -> dict:
    resp = {}
    for r in csv.DictReader(open(
            ROOT / "results/experiment_1e/analysis/defect_response_signatures_seed_summary.csv",
            encoding="utf-8")):
        resp[(r["category"], r["defect_type"])] = float(r["delta_defect_std_mean"])
    return resp


def fig1(summary: list[dict], resp: dict) -> None:
    """核心：1E Δdefect_std (x) vs 1G ΔNN std matched (y)。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, field, title in [
        (axes[0], "delta_nn_std", "matched bank Mα (system-level response)"),
        (axes[1], "delta_nn_std_frozen", "frozen bank M0 (defect representation drift)"),
    ]:
        xs, ys = [], []
        for r in summary:
            key = (r["category"], r["defect_type"])
            d1e = resp.get(key, np.nan)
            dy = float(r[field])
            if np.isnan(d1e) or np.isnan(dy):
                continue
            xs.append(d1e)
            ys.append(dy)
            fam = r["family"]
            ax.scatter(d1e, dy, c=FAM_COLOR[fam], marker=FAM_MARKER[fam], s=90,
                       edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3)
            ax.annotate(f"{r['category']}/{r['defect_type']}", (d1e, dy),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points", alpha=0.7)
        xs = np.array(xs); ys = np.array(ys)
        if len(xs) >= 5:
            rho, p = stats.spearmanr(xs, ys)
            ax.set_title(f"{title}\nSpearman ρ={rho:+.3f} (p={p:.3f}, n={len(xs)})", fontsize=11)
            # 拟合线
            z = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xline, np.polyval(z, xline), color="gray", lw=1, ls="--", alpha=0.7)
        else:
            ax.set_title(f"{title}\n(n={len(xs)} insufficient)", fontsize=11)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("1E Δdefect_std (anomaly-score dispersion)")
        ax.set_ylabel("1G ΔNN-distance std (defect patches)")
        ax.grid(alpha=0.25)

    handles = [plt.Line2D([0], [0], color=FAM_COLOR[f], marker=FAM_MARKER[f], ls="",
                          markeredgecolor="black", markersize=9, label=f) for f in FAM_COLOR]
    fig.legend(handles=handles, ncol=3, loc="lower center", fontsize=10, frameon=False)
    fig.suptitle("Experiment 1G — LEVEL 1: does 1E dispersion shrink/expand trace to NN-distance?",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "figure1_1e_vs_1g_core.png", dpi=150)
    plt.close(fig)
    print("figure1 saved")


def fig2(summary: list[dict]) -> None:
    """三组 NN-dispersion α 轨迹（需要 image_level.csv 的 per-alpha 数据）。"""
    img = list(csv.DictReader(open(ANALYSIS / "image_level.csv", encoding="utf-8")))
    # 按 (cat, dt, alpha) 聚合 defect_nn_std
    agg: dict[tuple, dict] = {}
    for r in img:
        key = (r["category"], r["defect_type"], r["family"])
        a = float(r["alpha"])
        agg.setdefault(key, {})[a] = agg.get(key, {}).get(a, []) + [float(r["defect_nn_std"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, fam in zip(axes, ["shrink", "neutral", "expand"]):
        for (cat, dt, f), ad in agg.items():
            if f != fam:
                continue
            xs = sorted(ad.keys())
            ys = [np.mean(ad[a]) for a in xs]
            # 相对 α=0 归一化，突出变化方向
            ys = np.array(ys) - ys[0]
            ax.plot(xs, ys, marker=CAT_MARKER[cat], color=CAT_COLOR[cat],
                    markersize=6, lw=1.5, label=f"{cat}/{dt}")
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{fam} family (n={sum(1 for k,v in agg.items() if k[2]==fam)})", fontsize=11)
        ax.set_xlabel("α")
        ax.set_ylabel("Δ NN-dispersion std (rel. α=0)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Experiment 1G — NN-dispersion α trajectory by family", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "figure2_trajectory_3group.png", dpi=150)
    plt.close(fig)
    print("figure2 saved")


def fig3(summary: list[dict], resp: dict) -> None:
    """feature geometry：Δdispersion / Δnorm_std / Δchannel_var vs 1E Δdefect_std。"""
    fields = [("delta_dispersion", "Δ feature dispersion (to centroid)"),
              ("delta_norm_std", "Δ feature norm std"),
              ("delta_channel_var", "Δ channel variance (mean)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (field, title) in zip(axes, fields):
        xs, ys = [], []
        for r in summary:
            key = (r["category"], r["defect_type"])
            d1e = resp.get(key, np.nan)
            dy = float(r[field])
            if np.isnan(d1e) or np.isnan(dy):
                continue
            xs.append(d1e); ys.append(dy)
            ax.scatter(d1e, dy, c=FAM_COLOR[r["family"]], marker=FAM_MARKER[r["family"]],
                       s=80, edgecolors="black", linewidths=0.5, alpha=0.85)
        xs = np.array(xs); ys = np.array(ys)
        if len(xs) >= 5:
            rho, p = stats.spearmanr(xs, ys)
            title = f"{title}\nSpearman ρ={rho:+.3f} (p={p:.3f})"
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("1E Δdefect_std")
        ax.grid(alpha=0.25)
    fig.suptitle("Experiment 1G — LEVEL 2/3: feature geometry vs 1E dispersion", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "figure3_feature_geometry.png", dpi=150)
    plt.close(fig)
    print("figure3 saved")


def fig4(summary: list[dict]) -> None:
    """defect vs background：NN-dispersion 的 α 响应对比（需要 image_level 的 bg 字段）。"""
    img = list(csv.DictReader(open(ANALYSIS / "image_level.csv", encoding="utf-8")))
    # 按 (cat, dt, family) 聚合 defect_nn_std 和 bg_nn_std 的 α=0 与 α=1
    agg: dict[tuple, dict] = {}
    for r in img:
        key = (r["category"], r["defect_type"], r["family"])
        a = float(r["alpha"])
        slot = agg.setdefault(key, {"d0": [], "d1": [], "b0": [], "b1": []})
        if a == 0.0:
            slot["d0"].append(float(r["defect_nn_std"]))
            slot["b0"].append(float(r["bg_nn_std"]))
        elif a == 1.0:
            slot["d1"].append(float(r["defect_nn_std"]))
            slot["b1"].append(float(r["bg_nn_std"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, fam in zip(axes, ["shrink", "neutral", "expand"]):
        pts = []
        for (cat, dt, f), s in agg.items():
            if f != fam or not (s["d0"] and s["d1"] and s["b0"] and s["b1"]):
                continue
            dd = np.mean(s["d1"]) - np.mean(s["d0"])
            db = np.mean(s["b1"]) - np.mean(s["b0"])
            pts.append((dd, db, cat, dt))
            ax.scatter(dd, db, c=CAT_COLOR[cat], marker=CAT_MARKER[cat], s=90,
                       edgecolors="black", linewidths=0.5, alpha=0.9)
            ax.annotate(f"{cat}/{dt}", (dd, db), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.plot([-10, 10], [-10, 10], color="gray", lw=0.8, ls=":", alpha=0.4)
        ax.set_title(f"{fam} family\n(n={len(pts)})", fontsize=11)
        ax.set_xlabel("Δ defect NN-std (α0→α1)")
        ax.set_ylabel("Δ background NN-std (α0→α1)")
        ax.grid(alpha=0.25)
    fig.suptitle("Experiment 1G — defect vs background NN-dispersion response (diagonal = coupled)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "figure4_defect_vs_background.png", dpi=150)
    plt.close(fig)
    print("figure4 saved")


def main() -> None:
    summary = load_summary()
    resp = load_1e_response()
    fig1(summary, resp)
    fig2(summary)
    fig3(summary, resp)
    fig4(summary)


if __name__ == "__main__":
    main()
