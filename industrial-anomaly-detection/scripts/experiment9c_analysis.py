"""Experiment 9C — Matched-RNG Calibration analysis（CPU-only，零 GPU）。

对照三方：
  V1      : 无 replay（9B raw 复用：SMOKE_uniform_a025 vs SMOKE_gate_const025）
  V2.0    : 仅 coreset 入口 replay（9C round "v2"，暴露出第二个 replay 点）
  V2.1    : coreset 入口 + Engine.fit 入口 replay（9C round "v2b"，目标协议）

用法：python -u scripts/experiment9c_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment5a_h_analysis as h5  # noqa: E402

EXP = ROOT / "results" / "experiment_9c_rng_calibration"
SUM, FIG = EXP / "summary", EXP / "figures"
NINE_B = ROOT / "results" / "experiment_9b_screening" / "raw"

V1_A = NINE_B / "round0/bottle/seed_0/config_SMOKE_uniform_a025"
V1_B = NINE_B / "smoke/bottle/seed_0/config_SMOKE_gate_const025"
V20 = EXP / "raw/v2/bottle/seed_0"
V21 = EXP / "raw/v2b/bottle/seed_0"
CFG_A, CFG_B = "C1_uniform_a025", "C2_gate_const025"


def wcsv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ent(p: Path):
    info = json.loads((p / "info.json").read_text())
    rows = list(csv.DictReader(open(p / "per_image.csv")))
    sub = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]), 0.0, 0.0))
    return {"sub": sub, "info": info, "rows": rows}


def rowmap(p: Path):
    return {(r["subset"], r["shift"], r["defect_type"], r["image_path"]): float(r["score"])
            for r in csv.DictReader(open(p / "per_image.csv"))}


def rng_ev(unit_dir: Path, cfg: str) -> dict:
    p = unit_dir / f"config_{cfg}" / f"rng_evidence_bottle_0_{cfg}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fit_ev(unit_dir: Path, cfg: str) -> dict:
    p = unit_dir / f"config_{cfg}" / f"fit_replay_evidence_bottle_0_{cfg}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def bank_ev(unit_dir: Path, cfg: str) -> dict:
    p = unit_dir / f"config_{cfg}" / "bank_evidence.json"
    return json.loads(p.read_text()) if p.exists() else {}


def compare(a_dir: Path, b_dir: Path, cfg_a: str, cfg_b: str) -> dict:
    ea, eb = rng_ev(a_dir, cfg_a), rng_ev(b_dir, cfg_b)
    ca = (ea.get("select_calls") or [{}])[0]
    cb = (eb.get("select_calls") or [{}])[0]
    fa, fb = fit_ev(a_dir, cfg_a), fit_ev(b_dir, cfg_b)
    fca = (fa.get("calls") or [{}])[0]
    fcb = (fb.get("calls") or [{}])[0]
    ba, bb = bank_ev(a_dir, cfg_a), bank_ev(b_dir, cfg_b)
    sa, sb = rowmap(a_dir / f"config_{cfg_a}"), rowmap(b_dir / f"config_{cfg_b}")
    ks = [k for k in sa if k in sb]
    d = [abs(sa[k] - sb[k]) for k in ks] if ks else [float("nan")]
    m = h5.unit_metrics({("bottle", 0, "A"): ent(a_dir / f"config_{cfg_a}"),
                         ("bottle", 0, "B"): ent(b_dir / f"config_{cfg_b}")})
    ma, mb = m[("bottle", 0, "A")], m[("bottle", 0, "B")]
    return {
        "embedding_sum_abs": {"A": ca.get("embedding_stats", {}).get("sum_abs"),
                              "B": cb.get("embedding_stats", {}).get("sum_abs")},
        "embedding_identical": (ca.get("embedding_stats", {}).get("sum_abs") ==
                                cb.get("embedding_stats", {}).get("sum_abs")),
        "rng_hash_at_real_sampling": {"A": ca.get("rng_hash_at_real_sampling"),
                                      "B": cb.get("rng_hash_at_real_sampling")},
        "rng_state_identical": (ca.get("rng_hash_at_real_sampling") ==
                                cb.get("rng_hash_at_real_sampling")),
        "fit_entry_rng_hash_after_replay": {"A": fca.get("rng_hash_after_fit_replay"),
                                            "B": fcb.get("rng_hash_after_fit_replay")},
        "fit_entry_rng_identical": (fca.get("rng_hash_after_fit_replay") ==
                                    fcb.get("rng_hash_after_fit_replay")),
        "coreset_first20": {"A": ca.get("coreset", {}).get("first20"),
                            "B": cb.get("coreset", {}).get("first20")},
        "coreset_sha256": {"A": ca.get("coreset", {}).get("sha256"),
                           "B": cb.get("coreset", {}).get("sha256")},
        "coreset_identical": (ca.get("coreset", {}).get("sha256") ==
                              cb.get("coreset", {}).get("sha256")),
        "coreset_n": {"A": ca.get("coreset", {}).get("n"), "B": cb.get("coreset", {}).get("n")},
        "memory_bank_sha256": {"A": ba.get("memory_bank_sha256"), "B": bb.get("memory_bank_sha256")},
        "memory_bank_identical": (ba.get("memory_bank_sha256") == bb.get("memory_bank_sha256")),
        "tau": {"A": ba.get("tau_val") if ba.get("tau_val") is not None
                else json.loads((a_dir / f"config_{cfg_a}" / "info.json").read_text()).get("tau_val"),
                "B": bb.get("tau_val") if bb.get("tau_val") is not None
                else json.loads((b_dir / f"config_{cfg_b}" / "info.json").read_text()).get("tau_val")},
        "tau_identical": (
            (ba.get("tau_val") if ba.get("tau_val") is not None
             else json.loads((a_dir / f"config_{cfg_a}" / "info.json").read_text()).get("tau_val"))
            == (bb.get("tau_val") if bb.get("tau_val") is not None
                else json.loads((b_dir / f"config_{cfg_b}" / "info.json").read_text()).get("tau_val"))),
        "n_score_rows": len(ks),
        "max_abs_delta_score": float(max(d)),
        "score_bit_exact": bool(max(d) == 0.0),
        "metrics": {"A": {k: float(ma[k]) for k in ("mean_abs_delta_z", "mean_dprime",
                                                    "image_auroc", "pixel_auroc", "aupro")},
                    "B": {k: float(mb[k]) for k in ("mean_abs_delta_z", "mean_dprime",
                                                    "image_auroc", "pixel_auroc", "aupro")}},
        "NF_dz": abs(ma["mean_abs_delta_z"] - mb["mean_abs_delta_z"]),
        "NF_dprime": abs(ma["mean_dprime"] - mb["mean_dprime"]),
    }


def main() -> None:
    SUM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    v1 = compare(V1_A.parent, V1_B.parent, "SMOKE_uniform_a025", "SMOKE_gate_const025")
    v20 = compare(V20, V20, CFG_A, CFG_B)
    v21 = compare(V21, V21, CFG_A, CFG_B)
    # V1 没有 9C 的 instrumentation -> None==None 会伪真；改用 9B 的独立证据显式标注
    v1["rng_state_identical"] = False
    v1["rng_state_evidence"] = ("9B rng_diagnostics: torch.rand 0.9509 vs 0.5424, "
                                "randint 78637 vs 3666（两条路径进入 fit 前 RNG state 不同）")
    v1["coreset_identical"] = v1["tau_identical"]   # tau 不同 => coreset 不同（V1 未 instrument indices）
    v1["coreset_evidence"] = ("tau_val 19.013771057128906 vs 19.00691032409668（9B info.json）"
                              " -> coreset 不同（V1 未 instrument indices）")
    cases = {"V1": v1, "V2.0_coreset_replay_only": v20, "V2.1_coreset_plus_fit_replay": v21}

    # ---- CASE 判定（按冻结标准 A–E） ----
    def case_of(c: dict) -> str:
        if c["rng_state_identical"] is False:
            return "B (RNG state 仍不同)"
        if c["coreset_identical"] is False:
            return "B (RNG 相同但 coreset 不同 -> 行序等其他 RNG 消费者)"
        if c["score_bit_exact"] is not True:
            return "C (coreset 相同但 score 不同)"
        if c["embedding_identical"] is False:
            return "D (representation 实际不等价)"
        return "A (完全一致: embedding/RNG/coreset/bank/score 全部一致)"

    verdicts = {k: case_of(c) for k, c in cases.items()}

    (SUM / "rng_equivalence.json").write_text(json.dumps({
        "experiment": "9C", "comparison": "mathematically equivalent pair (uniform alpha=0.25 vs const-gate 0.25)",
        "category": "bottle", "seed": 0,
        "H1": "不同代码路径的 score 差异主要来自 fit 前 RNG state 不一致",
        "H2": "在 coreset selection 前恢复相同 RNG state => 相同 coreset / 相同 score",
        "H3": "Matched-RNG 不改变 representation 本身",
        "V1_H1_supported": v1["rng_state_identical"] is False and v1["coreset_identical"] is False,
        "V2.1_H2_supported": bool(v21["rng_state_identical"] and v21["coreset_identical"]
                                  and v21["score_bit_exact"]),
        "H3_check": {"V2.1_embedding_identical_to_each_other": v21["embedding_identical"],
                     "note": "V2.1 的两个实现 embedding 逐位一致（sum_abs 相等）"},
        "cases": {k: {"case": verdicts[k], "rng_state_identical": c["rng_state_identical"],
                      "coreset_identical": c["coreset_identical"],
                      "score_bit_exact": c["score_bit_exact"],
                      "embedding_identical": c["embedding_identical"]} for k, c in cases.items()},
    }, indent=2, ensure_ascii=False))

    (SUM / "rng_state_hashes.json").write_text(json.dumps({
        "canonical_state_definition": "torch.manual_seed(seed)+torch.cuda.manual_seed_all(seed)+numpy.random.seed(seed)+random.seed(seed) 之后捕获",
        "replay_points": {"R1": "Engine.fit 入口（固定 train DataLoader shuffle 行序）",
                          "R2": "KCenterGreedy.select_coreset_idxs 入口（固定投影矩阵 + 初始点采样）"},
        "streams_consumed_by_coreset_audit": json.loads((SUM / "rng_audit.json").read_text()) if (SUM / "rng_audit.json").exists() else None,
        "per_case": {k: {"fit_entry_rng_hash_after_replay": c["fit_entry_rng_hash_after_replay"],
                         "pre_coreset_rng_hash_at_real_sampling": c["rng_hash_at_real_sampling"],
                         "fit_entry_rng_identical": c["fit_entry_rng_identical"],
                         "rng_state_identical": c["rng_state_identical"]} for k, c in cases.items()},
    }, indent=2, ensure_ascii=False))

    (SUM / "coreset_equivalence.json").write_text(json.dumps({
        "per_case": {k: {"coreset_sha256": c["coreset_sha256"], "coreset_first20": c["coreset_first20"],
                         "coreset_n": c["coreset_n"], "coreset_identical": c["coreset_identical"],
                         "memory_bank_sha256": c["memory_bank_sha256"],
                         "memory_bank_identical": c["memory_bank_identical"],
                         "tau": c["tau"], "tau_identical": c["tau_identical"]} for k, c in cases.items()},
        "diagnosis": ("V2.0 中 RNG state 已一致、coreset 首个索引相同（51413），但后续索引分叉。"
                      "主因已定位：train DataLoader(shuffle=True，无 generator) 的行序仍由全局 RNG 决定，"
                      "行序不同 -> embedding 行置换（aggregate sum_abs 差 32/290M ≈ 1.1e-7，"
                      "无法区分行序归约舍入 vs 数值本身差异）-> k-center-greedy 轨迹不同。"
                      "V2.1 增加 Engine.fit 入口 replay 后 embedding/coreset/memory bank 逐位一致，"
                      "证明主因确实是行序（CASE B）。"),
    }, indent=2, ensure_ascii=False))

    (SUM / "score_equivalence.json").write_text(json.dumps({
        "per_case": {k: {"n_rows": c["n_score_rows"], "max_abs_delta_score": c["max_abs_delta_score"],
                         "score_bit_exact": c["score_bit_exact"], "metrics": c["metrics"],
                         "NF_dz": c["NF_dz"], "NF_dprime": c["NF_dprime"]} for k, c in cases.items()},
        "criteria_frozen": {"A_embedding": "max|dF| <= 1e-7", "B_coreset": "exact",
                            "C_score": "max|dscore| <= 1e-7", "D_robustness": "|d| <= 1e-6",
                            "E_preservation": "|d| <= 1e-6"},
        "V2.1_meets_all_criteria": bool(v21["score_bit_exact"] and v21["coreset_identical"]
                                        and v21["NF_dz"] <= 1e-6 and v21["NF_dprime"] <= 1e-6),
    }, indent=2, ensure_ascii=False))

    wcsv(SUM / "noise_floor_v1_vs_v2.csv", [
        {"protocol": "V1 (no replay)", "pair": "uniform a=0.25 vs const-gate 0.25",
         "NF_absz": round(v1["NF_dz"], 6), "NF_dprime": round(v1["NF_dprime"], 6),
         "max_abs_delta_score": round(v1["max_abs_delta_score"], 6),
         "coreset_identical": v1["coreset_identical"], "rng_state_identical": v1["rng_state_identical"],
         "source": "9B frozen raw (reused)"},
        {"protocol": "V2.0 (coreset replay only)", "pair": "uniform a=0.25 vs const-gate 0.25",
         "NF_absz": round(v20["NF_dz"], 6), "NF_dprime": round(v20["NF_dprime"], 6),
         "max_abs_delta_score": round(v20["max_abs_delta_score"], 6),
         "coreset_identical": v20["coreset_identical"], "rng_state_identical": v20["rng_state_identical"],
         "source": "9C round v2"},
        {"protocol": "V2.1 (coreset + fit replay)", "pair": "uniform a=0.25 vs const-gate 0.25",
         "NF_absz": round(v21["NF_dz"], 6), "NF_dprime": round(v21["NF_dprime"], 6),
         "max_abs_delta_score": round(v21["max_abs_delta_score"], 6),
         "coreset_identical": v21["coreset_identical"], "rng_state_identical": v21["rng_state_identical"],
         "source": "9C round v2b"},
    ])

    rt = []
    for rd in ("v2", "v2b"):
        for ip in sorted((EXP / "raw" / rd).glob("bottle/seed_0/config_*/module_meta.json")):
            mm = json.loads(ip.read_text())
            rt.append({"round": rd, "unit": mm["unit"], "config": mm["config"],
                       "runtime_seconds": mm.get("runtime_seconds"),
                       "embedding_dim": mm.get("embedding_dim"),
                       "memory_bank_size": mm.get("memory_bank_size"),
                       "peak_vram_mb": mm.get("peak_gpu_memory_allocated_mb"),
                       "status": mm.get("status")})
    (SUM / "runtime_summary.json").write_text(json.dumps(
        {"experiment": "9C", "gpu_units": rt, "gpu_units_count": len(rt),
         "gpu_seconds_total": round(sum((r["runtime_seconds"] or 0) for r in rt), 1),
         "budget_frozen": {"gpu_main_units": 3, "gpu_diagnostic_units_max": 2},
         "note": "V1 对照复用 9B raw，0 GPU"}, indent=2, ensure_ascii=False))


    # ---- figure: V1 vs V2 noise floor ----
    labels = ["V1\n(no replay)", "V2.0\n(coreset replay)", "V2.1\n(coreset+fit replay)"]
    dz = [v1["NF_dz"], v20["NF_dz"], v21["NF_dz"]]
    dp = [v1["NF_dprime"], v20["NF_dprime"], v21["NF_dprime"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, vals, name, col in ((axes[0], dz, "NF(|dNormalScore_z|)", "#4C72B0"),
                                (axes[1], dp, "NF(d')", "#C44E52")):
        bars = ax.bar(labels, vals, color=col, alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_title(name, fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        for b, val in zip(bars, vals):
            txt = "%.6f" % val if val > 0 else "0.0 (bit-exact)"
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, max(val, max(vals) * 0.02)),
                        ha="center", va="bottom", fontsize=9)
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, top * 1.28)
    fig.suptitle("Experiment 9C — coreset-RNG noise floor: equivalent implementation pair "
                 "(uniform alpha=0.25 vs constant-gate 0.25, bottle seed 0)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "noise_floor_v1_vs_v2.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- console ----
    print("")
    print("=" * 120)
    print("Experiment 9C - MATCHED-RNG CALIBRATION (bottle seed0; uniform alpha=0.25 vs const-gate 0.25)")
    print("=" * 120)
    print("%-30s%10s%14s%16s%14s%16s" % ("protocol", "RNG==", "coreset==", "score bit-exact",
                                         "NF(|dz|)", "NF(d')"))
    for k, c in cases.items():
        print("%-30s%10s%14s%16s%14.6f%16.6f" % (
            k, c["rng_state_identical"], c["coreset_identical"], c["score_bit_exact"],
            c["NF_dz"], c["NF_dprime"]))
    print("-" * 120)
    for k in cases:
        print("  CASE %-32s -> %s" % (k, verdicts[k]))
    print("  >>> 最终 CASE:", verdicts["V2.1_coreset_plus_fit_replay"])
    print("=" * 120)
    print("  artifacts -> %s | %s" % (SUM, FIG))


if __name__ == "__main__":
    main()
