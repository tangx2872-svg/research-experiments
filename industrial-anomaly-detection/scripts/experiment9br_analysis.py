"""Experiment 9B-R — analysis（CPU-only，零 GPU）。

对照三方：
  Old 9B    : 9B (V1) 冻结结果（ranking.csv）
  Strict REF: 本轮 V2 下的 Original α=0（r0_ref，兼作 Control 1）
  Strict Cx : 本轮 V2 下的候选（r1）
判据：见 config/config.json 的 verdict_rules_frozen（禁止事后修改）。
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

EXP = ROOT / "results" / "experiment_9b_r_strict_replay"
ANA, FIG = EXP / "analysis", EXP / "figures"
RAW = EXP / "raw"
NINE_B_RANK = ROOT / "results" / "experiment_9b_screening" / "summary" / "ranking.csv"
EPS_DP, EPS_RZ = 0.10, 0.02

CAND = [("T1_M7_L2a000_L3a025", "M7_L2a000_L3a025", "NEW(9B)"),
        ("T2_M10_concat_g100", "M10_concat_g100", "REUSE(7A-O B1_g100)"),
        ("T3_M7_L2a025_L3a000", "M7_L2a025_L3a000", "NEW(9B)")]
REF_DIRS = [RAW / "r0_ref/bottle/seed_0/config_REF_original",
            RAW / "r0_ref/bottle/seed_0/config_REF_original_repeat"]


def wcsv(p, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ent(d: Path):
    info = json.loads((d / "info.json").read_text())
    rows = list(csv.DictReader(open(d / "per_image.csv")))
    sub = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]), 0.0, 0.0))
    return {"sub": sub, "info": info, "rows": rows}


def jload(d: Path, pat: str):
    fs = sorted(d.glob(pat))
    return json.loads(fs[0].read_text()) if fs else {}


def old_9b() -> dict:
    out = {}
    for r in csv.DictReader(open(NINE_B_RANK)):
        out[r["method"]] = {"bottle_dprime": float(r["bottle_dprime"]),
                            "bottle_absdz": float(r["bottle_absdz"]),
                            "verdict": r["verdict"], "rank": r["rank"],
                            "beyond_gain_bottle": r.get("beyond_gain_bottle")}
    return out


def main() -> None:
    ANA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    OB = old_9b()

    # ---------- P3 sanity ----------
    checks = []
    ea = [jload(d, "embedding_evidence_*.json") for d in REF_DIRS]
    ba = [jload(d, "bank_evidence_*.json") for d in REF_DIRS]
    ra = [jload(d, "rng_evidence_*.json") for d in REF_DIRS]
    me = [ent(d) for d in REF_DIRS]
    m_ref = h5.unit_metrics({("bottle", 0, f"ref{i}"): me[i] for i in range(2)})
    import csv as _csv
    rm = [{ (r["subset"], r["shift"], r["defect_type"], r["image_path"]): float(r["score"])
            for r in _csv.DictReader(open(d / "per_image.csv"))} for d in REF_DIRS]
    ks = [k for k in rm[0] if k in rm[1]]
    dmax = max(abs(rm[0][k] - rm[1][k]) for k in ks)
    checks.append(("C1_representation_bit_identical_repeat",
                   ea[0]["embedding_sha256"] == ea[1]["embedding_sha256"],
                   f"embedding sha256 {ea[0]['embedding_sha256'][:16]} (2 independent processes)"))
    checks.append(("C1_coreset_indices_identical_repeat",
                   ra[0]["select_calls"][0]["coreset"]["sha256"] == ra[1]["select_calls"][0]["coreset"]["sha256"],
                   f"coreset sha {ra[0]['select_calls'][0]['coreset']['sha256'][:16]}"))
    checks.append(("C1_memory_bank_sha256_identical_repeat",
                   ba[0]["memory_bank_sha256"] == ba[1]["memory_bank_sha256"],
                   f"bank sha {ba[0]['memory_bank_sha256'][:16]}"))
    checks.append(("C1_max_abs_delta_score_le_1e-7", dmax <= 1e-7, f"max|dscore|={dmax:.3e} (n={len(ks)})"))
    eq = jload(ANA, "equivalence_evidence.json")
    c2 = json.loads((ROOT / "results/experiment_9c_rng_calibration/summary/coreset_equivalence.json").read_text())
    s2 = json.loads((ROOT / "results/experiment_9c_rng_calibration/summary/score_equivalence.json").read_text())
    v21c = c2["per_case"]["V2.1_coreset_plus_fit_replay"]
    v21s = s2["per_case"]["V2.1_coreset_plus_fit_replay"]
    checks.append(("C2_elementwise_max_abs_dF", eq.get("elementwise_max_abs_dF") == 0.0,
                   f"max|dF|={eq.get('elementwise_max_abs_dF')} "
                   f"n_diff={eq.get('n_diff_elements')}/{eq.get('n_elements_total')} (full 209-image train set)"))
    checks.append(("C2_coreset_indices_identical", v21c["coreset_identical"],
                   f"9C-v2b coreset sha {v21c['coreset_sha256']['A'][:16]}"))
    checks.append(("C2_memory_bank_identical", v21c["memory_bank_identical"],
                   f"9C-v2b bank sha {v21c['memory_bank_sha256']['A'][:16]}"))
    checks.append(("C2_max_abs_delta_score", v21s["score_bit_exact"],
                   f"9C-v2b max|dscore|={v21s['max_abs_delta_score']:.3e}"))
    checks.append(("C2_metrics_identical_robustness_preservation",
                   v21s["NF_dz"] <= 1e-6 and v21s["NF_dprime"] <= 1e-6,
                   f"NF_strict(|dz|)={v21s['NF_dz']:.3e} NF_strict(d')={v21s['NF_dprime']:.3e}"))
    sanity_pass = all(ok for _, ok, _ in checks)

    b0 = OB.get("B0_original_a000", {})
    ref = m_ref[("bottle", 0, "ref0")]
    ref2 = m_ref[("bottle", 0, "ref1")]
    nf_strict_dz = abs(ref["mean_abs_delta_z"] - ref2["mean_abs_delta_z"])
    nf_strict_dp = abs(ref["mean_dprime"] - ref2["mean_dprime"])
    checks.append(("NF_strict_measured_by_Control1", nf_strict_dz <= 1e-6 and nf_strict_dp <= 1e-6,
                   f"NF_strict(|dz|)={nf_strict_dz:.3e} NF_strict(d')={nf_strict_dp:.3e}"))


    # ---------- 候选指标 ----------
    def unit(cfg: str):
        d = RAW / "r1/bottle/seed_0" / f"config_{cfg}"
        return ent(d), d

    rows, ranking = [], []
    for tag, old_name, kind in CAND:
        e, d = unit(tag)
        m = h5.unit_metrics({("bottle", 0, tag): e})[("bottle", 0, tag)]
        info = json.loads((d / "info.json").read_text())
        be = jload(d, "bank_evidence_*.json")
        ee = jload(d, "embedding_evidence_*.json")
        re_ = jload(d, "rng_evidence_*.json")
        o = OB.get(old_name, {})
        dz_d, dp_d = m["mean_abs_delta_z"] - ref["mean_abs_delta_z"], m["mean_dprime"] - ref["mean_dprime"]
        old_dz, old_dp = o.get("bottle_absdz"), o.get("bottle_dprime")
        rec = {"candidate": tag, "old_9B_name": old_name, "impl": kind,
               "bottle_absdz_strict": round(m["mean_abs_delta_z"], 6),
               "bottle_dprime_strict": round(m["mean_dprime"], 6),
               "image_auroc": round(m["image_auroc"], 6), "pixel_auroc": round(m["pixel_auroc"], 6),
               "aupro": round(m["aupro"], 6), "tau": m["tau_val"],
               "delta_absz_vs_strict_original": round(dz_d, 6),
               "delta_dprime_vs_strict_original": round(dp_d, 6),
               "old_9B_bottle_absdz": old_dz, "old_9B_bottle_dprime": old_dp,
               "old_9B_verdict": o.get("verdict"), "old_9B_rank": o.get("rank"),
               "old_9B_beyond_gain_bottle": o.get("beyond_gain_bottle"),
               "improvement_survives_robustness": bool(dz_d <= -EPS_RZ),
               "improvement_survives_direction_9B": bool(dz_d < 0),
               "runtime_seconds": info.get("runtime_seconds"),
               "memory_bank_size": info.get("coreset_size"),
               "embedding_sha256": (ee.get("embedding_sha256") or "")[:16],
               "memory_bank_sha256": (be.get("memory_bank_sha256") or "")[:16],
               "coreset_sha256": ((re_.get("select_calls") or [{}])[0].get("coreset", {}) or {}).get("sha256", "")[:16],
               "v1_delta_absz_vs_B0": (round(o.get("bottle_absdz") - b0.get("bottle_absdz"), 6)
                                        if o.get("bottle_absdz") is not None else None),
               "v1_delta_dprime_vs_B0": (round(o.get("bottle_dprime") - b0.get("bottle_dprime"), 6)
                                         if o.get("bottle_dprime") is not None else None),
               "v2_delta_absz_vs_original": round(dz_d, 6),
               "v2_delta_dprime_vs_original": round(dp_d, 6),
               "improvement_sign_consistent_V1_vs_V2": bool(
                   (o.get("bottle_absdz") is not None and b0.get("bottle_absdz") is not None
                    and np.sign(o["bottle_absdz"] - b0["bottle_absdz"]) == np.sign(dz_d))),
               "strict_vs_9B_same_coreset": False,
               "note": "V2 coreset 与 9B(V1) 不同（跨协议不可逐位比较）"}
        rows.append(rec)

        rob_improve = dz_d <= -EPS_RZ
        pres_not_worse = dp_d >= -EPS_DP
        pres_improve = dp_d >= EPS_DP
        rob_not_worse = dz_d <= EPS_RZ
        strong = (rob_improve and pres_not_worse) or (pres_improve and rob_not_worse)
        both_worse = (dp_d <= -EPS_DP) and (dz_d >= EPS_RZ)
        no_improve = (dz_d > -EPS_RZ) and (dp_d < EPS_DP)
        if both_worse or no_improve:
            vd = "STOP"
            why = ("strict replay 后两指标均未优于 Original（robustness %+.4f, preservation %+.4f）→ improvement 消失"
                   % (dz_d, dp_d)) if no_improve else "两指标同时恶化"
        elif strong:
            vd = "ADVANCE"
            why = "strict replay 后仍满足预注册强条件（robustness %+.4f / preservation %+.4f）" % (dz_d, dp_d)
        else:
            vd = "HOLD"
            why = "improvement 存在但不满足强条件（robustness %+.4f / preservation %+.4f）" % (dz_d, dp_d)
        ranking.append({"candidate": tag, "impl": kind, "old_9B_rank": o.get("rank"),
                        "old_9B_verdict": o.get("verdict"), "old_9B_beyond_gain_bottle": o.get("beyond_gain_bottle"),
                        "bottle_absdz": rec["bottle_absdz_strict"],
                        "bottle_dprime": rec["bottle_dprime_strict"],
                        "delta_absz_vs_original": rec["delta_absz_vs_strict_original"],
                        "delta_dprime_vs_original": rec["delta_dprime_vs_strict_original"],
                        "old_9B_absdz": old_dz, "old_9B_dprime": old_dp,
                        "verdict": vd, "verdict_reason": why})
    order = {"ADVANCE": 0, "HOLD": 1, "STOP": 2}
    ranking.sort(key=lambda r: (order.get(r["verdict"], 3), r["delta_absz_vs_original"]))


    # ---------- 落盘 ----------
    wcsv(ANA / "sanity_checks.csv",
         [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in checks])
    wcsv(ANA / "candidate_ranking.csv", ranking)
    wcsv(ANA / "strict_vs_9b.csv", rows)
    (ANA / "cross_process_equivalence.json").write_text(json.dumps(
        {"control1_two_independent_processes": {
            "embedding_sha256": [ea[0]["embedding_sha256"], ea[1]["embedding_sha256"]],
            "coreset_sha256": [ra[0]["select_calls"][0]["coreset"]["sha256"],
                               ra[1]["select_calls"][0]["coreset"]["sha256"]],
            "memory_bank_sha256": [ba[0]["memory_bank_sha256"], ba[1]["memory_bank_sha256"]],
            "max_abs_delta_score": dmax, "n_score_rows": len(ks),
            "tau": [json.loads((d / "info.json").read_text())["tau_val"] for d in REF_DIRS],
            "runtime_seconds": [json.loads((d / "info.json").read_text())["runtime_seconds"]
                                for d in REF_DIRS]},
         "control2_elementwise": {k: eq.get(k) for k in
                                  ("elementwise_max_abs_dF", "elementwise_identical",
                                   "n_diff_elements", "n_elements_total", "scope")},
         "control2_cross_process_from_9C_v2b": {"coreset_identical": v21c["coreset_identical"],
                                                "memory_bank_identical": v21c["memory_bank_identical"],
                                                "max_abs_delta_score": v21s["max_abs_delta_score"],
                                                "NF_dz": v21s["NF_dz"], "NF_dprime": v21s["NF_dprime"]},
         "NF_strict": {"dz": nf_strict_dz, "dprime": nf_strict_dp},
         "reference_original_strict": {"absdz": ref["mean_abs_delta_z"], "dprime": ref["mean_dprime"],
                                       "tau": ref["tau_val"], "image_auroc": ref["image_auroc"],
                                       "pixel_auroc": ref["pixel_auroc"], "aupro": ref["aupro"]}},
        indent=2, ensure_ascii=False))

    import platform
    import torch
    units = []
    for p in sorted(RAW.glob("*/*/seed_0/config_*/module_meta.json")):
        mm = json.loads(p.read_text())
        units.append({"round": p.parents[3].name, "unit": mm.get("unit"), "config": mm["config"],
                      "runtime_seconds": mm.get("runtime_seconds"),
                      "memory_bank_size": mm.get("memory_bank_size"),
                      "peak_vram_mb": mm.get("peak_gpu_memory_allocated_mb"),
                      "status": mm.get("status")})
    (ANA / "runtime_summary.json").write_text(json.dumps(
        {"gpu_units": units, "gpu_units_count": len(units),
         "gpu_seconds_total": round(sum((u["runtime_seconds"] or 0) for u in units), 1),
         "reused_zero_gpu": "Control 2 pair (uniform a=0.25 vs const-gate 0.25) 复用 9C-v2b raw；9B (V1) 结果仅作对照",
         "env": {"python": platform.python_version(), "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "git_head": __import__("subprocess").check_output(
                     ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()}},
        indent=2, ensure_ascii=False))

    (ANA / "final_verdict.json").write_text(json.dumps(
        {"sanity_pass": sanity_pass, "ranking": ranking,
         "advaned": [r["candidate"] for r in ranking if r["verdict"] == "ADVANCE"],
         "held": [r["candidate"] for r in ranking if r["verdict"] == "HOLD"],
         "stopped": [r["candidate"] for r in ranking if r["verdict"] == "STOP"]},
        indent=2, ensure_ascii=False))


    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    names = [r["candidate"].split("_")[0] for r in ranking]
    x = np.arange(len(names))
    axes[0].axhline(ref["mean_abs_delta_z"], ls="--", c="#333333", lw=1.4,
                    label="Strict Original (alpha=0)")
    axes[0].bar(x, [r["bottle_absdz"] for r in ranking], color="#4C72B0", alpha=0.85,
                edgecolor="black", lw=0.8, label="candidate (strict V2)")
    axes[0].plot(x, [r["old_9B_absdz"] for r in ranking], "o", ms=9, color="#C44E52",
                 label="Old 9B (V1)", zorder=5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_title("Robustness cost  mean|dNormalScore_z| (lower=better)")
    axes[0].set_ylabel("mean |dNormalScore_z|"); axes[0].grid(alpha=0.3, axis="y")
    axes[1].axhline(ref["mean_dprime"], ls="--", c="#333333", lw=1.4,
                    label="Strict Original (alpha=0)")
    axes[1].bar(x, [r["bottle_dprime"] for r in ranking], color="#55A868", alpha=0.85,
                edgecolor="black", lw=0.8, label="candidate (strict V2)")
    axes[1].plot(x, [r["old_9B_dprime"] for r in ranking], "o", ms=9, color="#C44E52",
                 label="Old 9B (V1)", zorder=5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_title("Preservation  mean defect d' (higher=better)")
    axes[1].set_ylabel("mean defect d'"); axes[1].grid(alpha=0.3, axis="y")
    # 放大 d' 轴到相关区间（所有值都在 7.9-8.6，全轴 0 起点不可读）
    dvals = [r["bottle_dprime"] for r in ranking] + [r["old_9B_dprime"] for r in ranking] + [ref["mean_dprime"]]
    axes[1].set_ylim(min(vals for vals in dvals) - 0.45, max(dvals) + 0.35)
    axes[1].legend(fontsize=8, loc="lower right")
    avals = [r["bottle_absdz"] for r in ranking] + [r["old_9B_absdz"] for r in ranking] + [ref["mean_abs_delta_z"]]
    axes[0].set_ylim(0, max(avals) * 1.22)
    for ax in axes:
        for i, r in enumerate(ranking):
            ax.annotate(r["verdict"], (i, ax.get_ylim()[1]), ha="center", va="top", fontsize=8)
    fig.suptitle("Experiment 9B-R — strict replay (Matched-RNG Protocol V2) vs old 9B (V1), bottle seed0",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "strict_vs_9b_bottle.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- console ----------
    print("")
    print("=" * 128)
    print("Experiment 9B-R — STRICT REPLAY RE-EVALUATION (bottle seed0, Protocol V2)")
    print("=" * 128)
    print("Strict Original (alpha=0): |dz| = %.4f | d' = %.4f | tau = %.4f | imgAUROC = %.4f"
          % (ref["mean_abs_delta_z"], ref["mean_dprime"], ref["tau_val"], ref["image_auroc"]))
    print("NF_strict (Control 1, 2 independent processes) = %.3e (|dz|) / %.3e (d')"
          % (nf_strict_dz, nf_strict_dp))
    print("Old 9B (V1) B0 reference: |dz| = %.4f | d' = %.4f  (跨协议，coreset 不同)"
          % (b0.get("bottle_absdz", float("nan")), b0.get("bottle_dprime", float("nan"))))
    print("-" * 128)
    print("%-24s%9s%9s%10s%10s%9s%9s%10s%10s%9s" % (
        "candidate", "|dz|V2", "d'V2", "d|dz|V2", "dd'V2", "d|dz|V1", "dd'V1", "rank9B", "sign", "verdict"))
    for r in ranking:
        row = next(x for x in rows if x["candidate"] == r["candidate"])
        print("%-24s%9.4f%9.4f%+10.4f%+10.4f%+9.4f%+9.4f%10s%10s%9s" % (
            r["candidate"], r["bottle_absdz"], r["bottle_dprime"],
            r["delta_absz_vs_original"], r["delta_dprime_vs_original"],
            row["v1_delta_absz_vs_B0"], row["v1_delta_dprime_vs_B0"],
            r["old_9B_rank"] or "-",
            "yes" if row["improvement_sign_consistent_V1_vs_V2"] else "NO", r["verdict"]))
    print("-" * 128)
    for c, ok, d in checks:
        print("  %-5s%-52s%s" % ("PASS" if ok else "FAIL", c, d[:62]))
    print("  -> sanity %d/%d PASS | ADVANCE=%d HOLD=%d STOP=%d"
          % (sum(1 for _, ok, _ in checks if ok), len(checks),
             sum(1 for r in ranking if r["verdict"] == "ADVANCE"),
             sum(1 for r in ranking if r["verdict"] == "HOLD"),
             sum(1 for r in ranking if r["verdict"] == "STOP")))
    print("  artifacts -> %s | %s" % (ANA, FIG))


if __name__ == "__main__":
    main()
