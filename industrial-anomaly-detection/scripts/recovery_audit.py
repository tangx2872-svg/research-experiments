"""Research Recovery Audit — README coverage 审计（只读；不修改任何历史数据）。

对 Exp1B–16 逐个检查 9 个维度（Why / Question / Protocol / Results / Negative /
Interpretation / PaperRole / Exit / Next），用关键词启发式 + artifact 存在性检查。
判据：COMPLETE（章节+数字/路径齐全）/ PARTIAL（部分可恢复）/ MISSING（不可恢复）/ N/A（该实验不适用）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (实验 ID, README 候选路径列表, results 产物目录)
EXPS = [
    ("1B", ["results/experiment_1b/README.md",
            "experiments/2026-09-30_illumination_sensitivity_exploration/README.md"], "results/experiment_1b"),
    ("1C", ["experiments/exp1c_multiseed/README.md"], "results/experiment_1c"),
    ("1D", ["experiments/exp1d_size_control/README.md"], "results/experiment_1d"),
    ("1E", ["experiments/exp1e_cross_category/README.md"], "results/experiment_1e"),
    ("1F", ["experiments/exp1f_mechanism_screening/README.md"], "results/experiment_1f"),
    ("1G", ["experiments/exp1g_feature_space/README.md"], "results/experiment_1g"),
    ("1H", ["experiments/exp1h_layer_selectivity/README.md"], "results/experiment_1h"),
    ("1H-S", ["experiments/exp1hs_dispersion_layer/README.md"], "results/experiment_1hs"),
    ("1I", ["experiments/experiment1i_spatial_statistics_control/README.md"], "results/experiment_1i"),
    ("1J", ["experiments/experiment1j_feature_geometry/README.md"], "results/experiment_1j"),
    ("1J-B", ["experiments/experiment1j_b_transmission/README.md"], "results/experiment_1j_b"),
    ("5A", ["experiments/experiment5a/README.md"], "results/experiment_5a"),
    ("5A-H", ["experiments/experiment5a_h/README.md"], "results/experiment_5a_h"),
    ("5B", ["experiments/experiment5b/README.md"], "results/experiment_5b"),
    ("5B-final", ["experiments/experiment5b/README.md"], "results/experiment_5b_final"),
    ("5C", ["experiments/experiment5c/README.md"], "results/experiment_5c"),
    ("5D", ["experiments/experiment5d/README.md"], "results/experiment_5d"),
    ("6A", ["experiments/experiment6a/README.md"], "results/experiment_6a"),
    ("6B", ["experiments/experiment6b/README.md"], "results/experiment_6b"),
    ("7A-O", ["experiments/experiment7a_o/README.md"], "results/experiment_7a_o"),
    ("7A-O-Q2", ["results/experiment_7a_o_q2/README.md"], "results/experiment_7a_o_q2"),
    ("7A-O-Q3", ["results/experiment_7a_o_q3/README.md"], "results/experiment_7a_o_q3"),
    ("7A-O-Q4", ["results/experiment_7a_o_q4/README.md"], "results/experiment_7a_o_q4"),
    ("8B", ["experiments/experiment8b/README.md"], "results/experiment8b"),
    ("9A", ["results/experiment_9a_screening/README.md"], "results/experiment_9a_screening"),
    ("9B", ["results/experiment_9b_screening/README.md"], "results/experiment_9b_screening"),
    ("9C", ["results/experiment_9c_rng_calibration/README.md"], "results/experiment_9c_rng_calibration"),
    ("9B-R", ["results/experiment_9b_r_strict_replay/README.md"], "results/experiment_9b_r_strict_replay"),
    ("10", ["results/experiment_10_preservation_recovery/README.md"], "results/experiment_10_preservation_recovery"),
    ("11", ["results/experiment_11_c2_adaptive/README.md"], "results/experiment_11_c2_adaptive"),
    ("12", ["results/experiment_12/README.md"], "results/experiment_12"),
    ("13-P", ["results/experiment_13p/README.md"], "results/experiment_13p"),
    ("14", ["results/experiment_14/README.md"], "results/experiment_14"),
    ("15", ["results/experiment_15/README.md"], "results/experiment_15"),
    ("16", ["results/experiment_16_final_validation/README.md"], "results/experiment_16_final_validation"),
]

# 9 维度的关键词（中英混合，覆盖两种写作风格）
DIMS = {
    "Why": [r"为什么", r"动机", r"motivation", r"背景", r"purpose", r"##\s*0", r"问题"],
    "Question": [r"问题", r"question", r"假设", r"hypothesis", r"研究问题", r"核心"],
    "Protocol": [r"冻结", r"frozen", r"protocol", r"协议", r"threshold", r"判据", r"seed", r"metric"],
    "Results": [r"结果", r"result", r"PASS\s*\d", r"\d+/\d+", r"Δd", r"d[′']", r"\.csv", r"table"],
    "Negative": [r"negative", r"否定", r"淘汰", r"STOP", r"失败", r"not\s+promot", r"eliminat", r"CASE_[BC]"],
    "Interpretation": [r"判定", r"verdict", r"interpretation", r"结论", r"CASE_[A-D]", r"解读"],
    "PaperRole": [r"论文", r"paper", r"证据链", r"导航", r"stage", r"§"],
    "Exit": [r"verdict", r"判定", r"ADVANCE", r"HOLD", r"STOP", r"FREEZE", r"淘汰", r"正式"],
    "Next": [r"下一步", r"next", r"后续", r"进入"],
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def main() -> None:
    rows = []
    for eid, cands, rdir in EXPS:
        readme = next((ROOT / c for c in cands if (ROOT / c).exists()), None)
        rd = ROOT / rdir
        n_files = sum(1 for _ in rd.rglob("*") if _.is_file()) if rd.exists() else 0
        n_csv = sum(1 for _ in rd.rglob("*.csv")) if rd.exists() else 0
        n_json = sum(1 for _ in rd.rglob("*.json")) if rd.exists() else 0
        n_fig = sum(1 for _ in rd.rglob("*.png")) if rd.exists() else 0
        if readme is None:
            rows.append({"exp": eid, "readme_path": "—", "readme_lines": 0, "artifacts": n_files,
                         "csv": n_csv, "json": n_json, "png": n_fig, "dims": {}, "coverage": "MISSING",
                         "note": "README not found in either convention directory"})
            continue
        txt = norm(readme.read_text(errors="ignore"))
        dims = {k: any(re.search(p, txt, re.I) for p in pats) for k, pats in DIMS.items()}
        hit = sum(dims.values())
        coverage = "COMPLETE" if hit >= 8 else ("PARTIAL" if hit >= 5 else "MISSING")
        rows.append({"exp": eid, "readme_path": str(readme.relative_to(ROOT)),
                     "readme_lines": len(readme.read_text(errors="ignore").splitlines()),
                     "artifacts": n_files, "csv": n_csv, "json": n_json, "png": n_fig,
                     "dims": dims, "coverage": coverage, "note": ""})
    out = ROOT / "results" / "recovery_audit_raw.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("%-9s%-8s%-52s%6s%5s%5s%5s  %s" % ("exp", "cover", "readme_path", "lines", "csv", "json", "png", "missing dims"))
    for r in rows:
        miss = [k for k, v in r["dims"].items() if not v]
        print("%-9s%-8s%-52s%6d%5d%5d%5d  %s" % (r["exp"], r["coverage"], r["readme_path"],
                                                  r["readme_lines"], r["csv"], r["json"], r["png"],
                                                  ",".join(miss) if miss else "none"))
    from collections import Counter
    print("\ncoverage:", dict(Counter(r["coverage"] for r in rows)))
    print("raw ->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
