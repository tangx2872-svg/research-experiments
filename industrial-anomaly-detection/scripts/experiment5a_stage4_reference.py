"""Experiment 5A — Stage ④ geometry reference 汇编（纯 CPU，无 GPU）。

从 Stage ④ 冻结的权威 CSV 中读取真实数值，生成：

  results/experiment_5a/stage4_geometry_reference.csv

每行 = (defect, layer)，字段：
  category, defect_type, family, layer
  geometry_response        —— 1J-A radius_response（log-ratio, α0→α1）
  gai_radius               —— 1J-A GAI_radius
  nn_dispersion_response   —— 1J-A cross-chain layer-specific NN std response
  score_dispersion_response—— 1J-A cross-chain layer-specific score response
  intervention_response_*  —— 1J-B β-intervention（仅 layer3；layer2 行为 na）

并计算 layer sensitivity（family 级 mean |radius_response|，即 1J-A
family_summary 的聚合口径）与由此派生的 geometry-guided α rule 常数，
打印并写入 results/experiment_5a/geometry_rule.json（运行前冻结）。

数据来源（只读，禁止修改）：
  results/experiment_1j/analysis/experiment1j_cross_chain_summary.csv
  results/experiment_1j/analysis/experiment1j_family_summary.csv
  results/experiment_1j_b/analysis/transmission_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiment_5a"

CROSS_CHAIN = ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_cross_chain_summary.csv"
FAMILY_SUMMARY = ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_family_summary.csv"
TRANSMISSION = ROOT / "results" / "experiment_1j_b" / "analysis" / "transmission_summary.csv"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cc = load_csv(CROSS_CHAIN)
    fam = load_csv(FAMILY_SUMMARY)
    tr = {(r["category"], r["defect_type"]): r for r in load_csv(TRANSMISSION)}

    # ---- 1) per (defect, layer) 长表 -------------------------------------------------
    LAYER_KEY = {"layer2": "l2", "layer3": "l3"}
    rows: list[dict] = []
    for r in cc:
        cat, dt, family = r["category"], r["defect_type"], r["family"]
        t = tr.get((cat, dt), {})
        for layer in ("layer2", "layer3"):
            lk = LAYER_KEY[layer]
            row = {
                "category": cat,
                "defect_type": dt,
                "family": family,
                "layer": layer,
                "geometry_response": r[f"geometry_radius_response_{lk}"],
                "gai_radius": r["gai_radius"],
                "nn_dispersion_response": r[f"nn_dispersion_response_{layer}"],
                "score_dispersion_response": r[f"score_dispersion_response_{layer}"],
                # 1J-B β intervention 只作用于 Layer3，layer2 无对应干预（不伪造）
                "intervention_response_delta_radius": t.get("delta_radius_0to1", "na") if layer == "layer3" else "na",
                "intervention_response_delta_nnstd": t.get("delta_nnstd_0to1", "na") if layer == "layer3" else "na",
                "intervention_response_delta_score": t.get("delta_score_0to1", "na") if layer == "layer3" else "na",
                "intervention_response_trr_nn": t.get("trr_nn", "na") if layer == "layer3" else "na",
            }
            rows.append(row)

    out_csv = OUT_DIR / "stage4_geometry_reference.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_csv} ({len(rows)} rows = 9 defects x 2 layers)")

    # ---- 2) layer sensitivity（family 级 mean |radius_response|，1J-A family_summary 口径）----
    s = {}
    for layer in ("l2", "l3"):
        vals = [abs(float(r[f"radius_response_{layer}_mean"])) for r in fam]
        s[layer] = sum(vals) / len(vals)

    # defect 级口径（敏感性方向核对用，不作为 primary rule）
    s_defect = {}
    for layer in ("l2", "l3"):
        vals = [abs(float(r[f"geometry_radius_response_{layer}"])) for r in cc]
        s_defect[layer] = sum(vals) / len(vals)

    # ---- 3) geometry-guided α rule（方案 A：inverse sensitivity，max-rescale）---------
    # 敏感层少Normalize，不敏感层可承受更强 IN：
    #   α_sensitive  = A * (s_insensitive / s_sensitive)
    #   α_insensitive = A
    # family 级口径：s_l2 > s_l3 → α_l2 = A * s_l3/s_l2, α_l3 = A
    ratio = s["l3"] / s["l2"]
    rule = {
        "rule": "inverse-sensitivity max-rescale (方案A)",
        "sensitivity_source": "1J-A family_summary: mean over 3 families of |radius_response_layer|",
        "sensitivity_l2_family_level": round(s["l2"], 6),
        "sensitivity_l3_family_level": round(s["l3"], 6),
        "sensitivity_l2_defect_level_check": round(s_defect["l2"], 6),
        "sensitivity_l3_defect_level_check": round(s_defect["l3"], 6),
        "sensitivity_ordering_agrees": (s["l2"] > s["l3"]) == (s_defect["l2"] > s_defect["l3"]),
        "alpha_ratio_l2_over_l3": round(ratio, 6),
        "formula": "alpha_l3 = A; alpha_l2 = A * (s_l3 / s_l2)",
        "guided_levels": {
            f"A={A:g}": {
                "alpha_l2": round(A * ratio, 4),
                "alpha_l3": round(A, 4),
                "effective_mean_alpha_simple": round(A * (1 + ratio) / 2, 4),
                "effective_mean_alpha_channel_weighted": round((512 * A * ratio + 1024 * A) / 1536, 4),
            }
            for A in (0.5, 0.75, 1.0)
        },
        "frozen_before_run": True,
    }
    out_json = OUT_DIR / "geometry_rule.json"
    with open(out_json, "w") as f:
        json.dump(rule, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {out_json}")
    print(json.dumps(rule, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
