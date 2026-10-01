"""Experiment 1 补充分析：Q6 少数样本驱动检查 + 跨 α 可比性处理。

关键方法论说明（写入 README 的依据）：
  不同 α 下 InstanceNorm 改变了特征尺度，因此跨 α 的绝对 anomaly score
  不可直接比较。跨 α 可比的量只有：
    1. 同一 α 内的配对差值（perturbed - original，同图同模型）；
    2. 秩相关指标（AUROC / AUPR，对单调变换不敏感）；
    3. defect 与 normal 的分离度（同一 α 内）。

本脚本输出：
  - 正常图 perturbation delta 的逐样本符号比例、top-3 样本贡献占比（Q6）
  - 每个 α 的 defect-normal 分离度 margin（同 α 内，附尺度混淆警告）
  - Wilcoxon signed-rank：各 α 下 perturbation delta 是否显著 > 0
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "experiment1_illumination_tradeoff"
RAW_CSV = RESULTS_ROOT / "raw_results.csv"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]
COND_ORDER = ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]


def cond_key(r: dict) -> str:
    if r["illumination_type"] == "identity":
        return "original"
    return f"{r['illumination_type']}_{r['illumination_level']:g}"


def load_raw() -> list[dict]:
    rows = []
    with open(RAW_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["alpha"] = float(r["alpha"])
            r["pred_score"] = float(r["pred_score"])
            r["illumination_level"] = float(r["illumination_level"])
            rows.append(r)
    return rows


def main() -> None:
    rows = load_raw()
    normal = [r for r in rows if r["gt_type"] == "good"]
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]

    # 原图分数索引
    orig = {}
    for r in normal:
        if cond_key(r) == "original":
            orig[(r["image_path"], r["alpha"])] = r["pred_score"]

    out: dict = {"sample_fraction": {}, "top3_contribution": {}, "wilcoxon": {}, "margin": {}}

    print("=" * 78)
    print("[Q6] 逐样本 delta 分析：正样本比例 + top-3 贡献占比 + Wilcoxon")
    print("=" * 78)
    for ck in COND_ORDER:
        out["sample_fraction"][ck] = {}
        out["top3_contribution"][ck] = {}
        out["wilcoxon"][ck] = {}
        for alpha in ALPHAS:
            deltas = []
            for r in normal:
                if r["alpha"] == alpha and cond_key(r) == ck:
                    key = (r["image_path"], alpha)
                    if key in orig:
                        deltas.append(r["pred_score"] - orig[key])
            d = np.array(deltas)
            n = len(d)
            pos = int((d > 0).sum())
            neg = int((d < 0).sum())
            # top-3 正向样本对总正增量的贡献
            pos_sum = d[d > 0].sum()
            top3 = np.sort(d)[::-1][:3].sum()
            frac_top3 = float(top3 / pos_sum) if pos_sum > 0 else None
            out["sample_fraction"][ck][f"a{alpha:g}"] = {
                "n": n, "n_pos": pos, "n_neg": neg, "frac_pos": round(pos / n, 3),
            }
            out["top3_contribution"][ck][f"a{alpha:g}"] = round(frac_top3, 3) if frac_top3 is not None else None
            # Wilcoxon（配对 vs 0）
            try:
                w = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
                p = float(w.pvalue)
            except Exception:
                p = None
            out["wilcoxon"][ck][f"a{alpha:g}"] = p
            print(
                f"  {ck:>16} a={alpha:g}: mean_delta={d.mean():+.3f}  pos={pos}/{n}  "
                f"top3贡献={frac_top3:.0%}  wilcoxon_p={p if p is None else f'{p:.2e}'}"
            )

    print()
    print("=" * 78)
    print("[跨α可比] defect-normal 分离度 margin（同一 α 内，绝对分数跨α不可比）")
    print("=" * 78)
    for alpha in ALPHAS:
        n_scores = [r["pred_score"] for r in normal if r["alpha"] == alpha and cond_key(r) == "original"]
        n_mean = float(np.mean(n_scores))
        n_std = float(np.std(n_scores, ddof=1))
        line = f"  a={alpha:g}: normal mean={n_mean:.2f} std={n_std:.2f}"
        out["margin"][f"a{alpha:g}"] = {"normal_mean": n_mean, "normal_std": n_std}
        for dt in DEFECT_TYPES:
            d_scores = [r["pred_score"] for r in defect if r["alpha"] == alpha and r["gt_type"] == dt]
            d_mean = float(np.mean(d_scores))
            margin = d_mean - n_mean
            # 标准化分离度（类似 d'）：margin / normal_std
            sep = margin / n_std
            out["margin"][f"a{alpha:g}"][dt] = {"defect_mean": d_mean, "margin": margin, "standardized_sep": sep}
            line += f" | {dt}: margin={margin:+.2f} (d'={sep:.1f})"
        print(line)

    with open(RESULTS_ROOT / "supplementary_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {RESULTS_ROOT / 'supplementary_analysis.json'}")


if __name__ == "__main__":
    main()
