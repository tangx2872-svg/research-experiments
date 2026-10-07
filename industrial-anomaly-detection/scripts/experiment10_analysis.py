"""Experiment 10 — 统一分析（CPU-only，零 GPU；可增量重复运行）。

输出：
  protocol_audit/protocol_audit.{json,csv}      Stage 0.5（209 vs 189；CASE P-A/P-B）
  reference_frontier/frontier.{csv,json}        Stage 1 corrected V2 frontier
  candidates/candidate_ranking.csv              Stage 2A tournament 判定
  candidates/equivalence_dedup.json             Family A 等价性（CPU 证明）
  final_ranking.csv                             全阶段总排名
  figures/fig1_frontier.png  fig2_tournament.png  fig3_cross_category.png
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
import experiment10_candidates as c10  # noqa: E402

EXP = ROOT / "results" / "experiment_10_preservation_recovery"
ANA, FIG = EXP / "analysis", EXP / "figures"
NINE_B_R = ROOT / "results" / "experiment_9b_r_strict_replay" / "raw"
EPS_DP, EPS_RZ = 0.10, 0.02
CATS_ALL = ["bottle", "cable", "hazelnut", "screw", "grid"]


def wcsv(p, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p.parent.mkdir(parents=True, exist_ok=True)
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


def find(cat: str, seed: int, cfg: str, roots=None):
    """在 exp10 raw（所有 stage）与指定历史 root 中定位 unit 目录。"""
    for r in (roots or [EXP / "raw", NINE_B_R]):
        for p in sorted(r.glob(f"*/{cat}/seed_{seed}/config_{cfg}")):
            if (p / "info.json").exists():
                return p
    return None


def metrics(cat: str, seed: int, cfg: str, roots=None):
    d = find(cat, seed, cfg, roots)
    if d is None:
        return None, None
    return h5.unit_metrics({(cat, seed, cfg): ent(d)})[(cat, seed, cfg)], d


def jl(d: Path, pat: str):
    fs = sorted(d.glob(pat))
    return json.loads(fs[0].read_text()) if fs else {}


def _rowmap(d: Path):
    return {(r["subset"], r["shift"], r["defect_type"], r["image_path"]): float(r["score"])
            for r in csv.DictReader(open(d / "per_image.csv"))}


def stage05_audit() -> dict:
    """Stage 0.5：corrected-189 vs historical-209（bottle seed0）。"""
    out, rows = {}, []
    pairs = [("Original", "P10_ORIG_a000", "REF_original"),
             ("T1", "P10_T1_L2a000_L3a025", "T1_M7_L2a000_L3a025")]
    m = {}
    for tag, cfg189, cfg209 in pairs:
        m189, d189 = metrics("bottle", 0, cfg189, [EXP / "raw"])
        m209, d209 = metrics("bottle", 0, cfg209, [NINE_B_R])
        m[tag] = (m189, m209)
        ev189 = jl(d189, "train_filter_evidence_*.json") if d189 else {}
        ee189, ee209 = jl(d189, "embedding_evidence_*.json") if d189 else {}, jl(d209, "embedding_evidence_*.json") if d209 else {}
        re189, re209 = jl(d189, "rng_evidence_*.json") if d189 else {}, jl(d209, "rng_evidence_*.json") if d209 else {}
        rows.append({
            "method": tag,
            "absdz_209": round(m209["mean_abs_delta_z"], 6), "dprime_209": round(m209["mean_dprime"], 6),
            "absdz_189": round(m189["mean_abs_delta_z"], 6), "dprime_189": round(m189["mean_dprime"], 6),
            "delta_absdz_189_minus_209": round(m189["mean_abs_delta_z"] - m209["mean_abs_delta_z"], 6),
            "delta_dprime_189_minus_209": round(m189["mean_dprime"] - m209["mean_dprime"], 6),
            "images_embedded_189": ev189.get("images_embedded_at_fit"),
            "filter_log_189": str([(x["n_before"], x["n_after"]) for x in ev189.get("filter_log", [])]),
            "embedding_shape_209": ee209.get("embedding_shape"), "embedding_shape_189": ee189.get("embedding_shape"),
            "n_observations_209": ee209.get("n_observations"), "n_observations_189": ee189.get("n_observations"),
            "coreset_sha_209": ((re209.get("select_calls") or [{}])[0].get("coreset", {}) or {}).get("sha256", "")[:16],
            "coreset_sha_189": ((re189.get("select_calls") or [{}])[0].get("coreset", {}) or {}).get("sha256", "")[:16],
            "tau_209": m209["tau_val"], "tau_189": m189["tau_val"],
            "image_auroc_189": round(m189["image_auroc"], 6), "pixel_auroc_189": round(m189["pixel_auroc"], 6),
            "aupro_189": round(m189["aupro"], 6),
        })
    (m_o189, m_o209), (m_t189, m_t209) = m["Original"], m["T1"]
    d209 = {"absdz": m_t209["mean_abs_delta_z"] - m_o209["mean_abs_delta_z"],
            "dprime": m_t209["mean_dprime"] - m_o209["mean_dprime"]}
    d189 = {"absdz": m_t189["mean_abs_delta_z"] - m_o189["mean_abs_delta_z"],
            "dprime": m_t189["mean_dprime"] - m_o189["mean_dprime"]}
    same_dir = (np.sign(d209["absdz"]) == np.sign(d189["absdz"])
                and np.sign(d209["dprime"]) == np.sign(d189["dprime"]))
    tradeoff_same = bool((d189["absdz"] <= -EPS_RZ) and (d189["dprime"] <= -EPS_DP))
    case = "P-A" if (same_dir and tradeoff_same) else "P-B"
    out = {"bottle_seed0": {"units": rows,
                            "T1_minus_Original_209": {k: round(v, 6) for k, v in d209.items()},
                            "T1_minus_Original_189": {k: round(v, 6) for k, v in d189.items()},
                            "direction_consistent": bool(same_dir),
                            "tradeoff_judgement_unchanged": tradeoff_same,
                            "CASE": case,
                            "criterion": ("CASE P-A: 209->189 改变绝对值但 T1 vs Original 的 robustness/preservation "
                                          "增量同方向且 trade-off 判断不变 -> 从 Exp10 起冻结 corrected-189；"
                                          "CASE P-B: trade-off 判断改变 -> 必须重建 corrected-189 baseline/frontier")}}
    wcsv(EXP / "protocol_audit" / "protocol_audit.csv", rows)
    (EXP / "protocol_audit" / "protocol_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    (EXP / "protocol_audit" / "README.md").write_text(
        "# Stage 0.5 — Protocol Audit（209 vs 189）\n\n"
        "9B-R §11 发现 `Engine.fit` 内部 `dm.setup(\"fit\")` 会重置 runner 的 train 过滤，"
        "使 memory bank 实际含 209 张（含 20 张 val）。本阶段用 **corrected-189**（在 Lightning 再次 setup 后"
        "重新施加过滤，并用 `Patchcore.training_step` 计数证明实际嵌入图像数 = 189）对比 historical-209。\n\n"
        f"**CASE: {case}**（方向一致={same_dir}，trade-off 判断不变={tradeoff_same}）\n\n"
        "详情见 `protocol_audit.json` / `protocol_audit.csv`。\n")
    return out


def frontier_pts(cat: str, seed: int):
    """corrected V2 frontier：{alpha: (absdz, dprime, cfg, src_dir)}"""
    pts = {}
    for a in c10.ALPHA_GRID:
        # α=0 与 Original 数学等价（同一 config 值），直接复用 Original 单元（0 GPU）
        cfg = "P10_ORIG_a000" if a == 0.0 else "P10_uniform_a%s" % c10._an(a)
        m, d = metrics(cat, seed, cfg, [EXP / "raw"])
        if m is None:
            continue
        pts[a] = (m["mean_abs_delta_z"], m["mean_dprime"], cfg, d)
    return pts


def envelope(pts):
    fr = []
    for x, y in sorted((v[0], v[1]) for v in pts.values()):
        if not fr or y > fr[-1][1]:
            fr.append((x, y))
    return fr


def frontier_interp(fr, x):
    if not fr:
        return float("nan")
    if x <= fr[0][0]:
        return fr[0][1]
    if x >= fr[-1][0]:
        return fr[-1][1]
    for i in range(len(fr) - 1):
        x0, y0 = fr[i]
        x1, y1 = fr[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-12)
    return float("nan")


def stage1_frontier(cat: str = "bottle", seed: int = 0) -> dict:
    o, _ = metrics(cat, seed, "P10_ORIG_a000", [EXP / "raw"])
    pts = frontier_pts(cat, seed)
    rows = [{"alpha": a, "absdz": round(v[0], 6), "dprime": round(v[1], 6),
             "delta_absdz_vs_original": round(v[0] - o["mean_abs_delta_z"], 6),
             "delta_dprime_vs_original": round(v[1] - o["mean_dprime"], 6),
             "cfg": v[2]} for a, v in sorted(pts.items())]
    fr = envelope(pts)
    out = {"category": cat, "seed": seed,
           "original_189": {"absdz": o["mean_abs_delta_z"], "dprime": o["mean_dprime"],
                            "tau": o["tau_val"]},
           "workpoints": rows, "envelope": [{"absdz": round(x, 6), "dprime": round(y, 6)} for x, y in fr]}
    wcsv(EXP / "reference_frontier" / "frontier.csv", rows)
    (EXP / "reference_frontier" / "frontier.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def judge(dz: float, dp: float, dz_o: float, dp_o: float) -> tuple:
    """冻结判据（config 之外单独记录，读取结果前已写死）。"""
    dR, dP = dz - dz_o, dp - dp_o
    rob_ok = dR <= -EPS_RZ
    pres_ok = dP >= -EPS_DP
    pres_better = dP >= EPS_DP
    rob_worse = dR >= EPS_RZ
    if rob_ok and pres_ok:
        return "ADVANCE", "robustness gain 保留 且 preservation loss <= 0.10", dR, dP
    if pres_better and not rob_worse:
        return "ADVANCE", "preservation 改善且 robustness 未恶化", dR, dP
    if rob_ok:
        return "HOLD", "robustness gain 保留但 preservation loss > %.2f（未达今晚目标）" % EPS_DP, dR, dP
    return "STOP", "robustness gain 消失或两指标均无优势", dR, dP


def stage2_tournament(cat: str = "bottle", seed: int = 0) -> dict:
    o, _ = metrics(cat, seed, "P10_ORIG_a000", [EXP / "raw"])
    dz_o, dp_o = o["mean_abs_delta_z"], o["mean_dprime"]
    t1, _ = metrics(cat, seed, "P10_T1_L2a000_L3a025", [EXP / "raw"])
    f = stage1_frontier(cat, seed)
    env = [(e["absdz"], e["dprime"]) for e in f["envelope"]]
    rows, det = [], {"category": cat, "seed": seed,
                     "original": {"absdz": dz_o, "dprime": dp_o},
                     "T1": {"absdz": t1["mean_abs_delta_z"], "dprime": t1["mean_dprime"]}}
    for c in c10.CANDIDATES:
        if not c["gpu"]:
            continue
        m, d = metrics(cat, seed, c["id"], [EXP / "raw"])
        if m is None:
            continue
        dz, dp = m["mean_abs_delta_z"], m["mean_dprime"]
        vd, why, dR, dP = judge(dz, dp, dz_o, dp_o)
        interp = frontier_interp(env, dz)
        beyond = dp - interp
        be = jl(d, "bank_evidence_*.json")
        rows.append({"candidate": c["id"], "family": c["family"], "defn": c["defn"],
                     "absdz": round(dz, 6), "dprime": round(dp, 6),
                     "delta_absdz_vs_original": round(dR, 6), "delta_dprime_vs_original": round(dP, 6),
                     "delta_absdz_vs_T1": round(dz - t1["mean_abs_delta_z"], 6),
                     "delta_dprime_vs_T1": round(dp - t1["mean_dprime"], 6),
                     "frontier_interp_dprime": round(interp, 6), "beyond_frontier": round(beyond, 6),
                     "frontier_break": bool(beyond > 0),
                     "image_auroc": round(m["image_auroc"], 6), "pixel_auroc": round(m["pixel_auroc"], 6),
                     "aupro": round(m["aupro"], 6), "tau": m["tau_val"],
                     "memory_bank_size": be.get("memory_bank_shape"),
                     "verdict": vd, "verdict_reason": why})
    order = {"ADVANCE": 0, "HOLD": 1, "STOP": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], -r["beyond_frontier"]))
    det["candidates"] = rows
    det["n_advance"] = sum(1 for r in rows if r["verdict"] == "ADVANCE")
    det["n_frontier_break"] = sum(1 for r in rows if r["frontier_break"])
    wcsv(EXP / "candidates" / "candidate_ranking.csv", rows)
    (EXP / "candidates" / "equivalence_dedup.json").write_text(json.dumps(
        {"family_A_collapsed": c10.EQUIVALENTS,
         "note": ("Family A（F_R+λ(F_O−F_R)）代数上等价于 per-layer α（α'=α(1−λ)），"
                  "即已探索过的 M7 α-family -> 标记 EQUIVALENT，不消耗 GPU"),
         "other_dedup": [{"id": "P10_B1/B2", "note": "与 7A-O Family B / M10 的 cat([F, g*N_hat]) **不同**："
                                                     "M10 拼的是标准化副本，这里拼的是 robust 分支（含 Original block）"}],
         "n_unique_gpu": det.get("n_candidates", len(rows))}, indent=2, ensure_ascii=False))
    (EXP / "candidates" / "tournament.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))
    return det


# --- Stage 2B 聚合规则（在读取 seeds 1/2 结果之前写定；判据带 EPS_RZ/EPS_DP 沿用 Stage 2A 冻结值）---
def verdict_multiseed(rows: list) -> tuple:
    """rows: 每 seed 的 {dR, dP}。规则（冻结）：
    rob_ok: dR <= -EPS_RZ；pres_ok: dP >= -EPS_DP。
    ADVANCE: rob_ok 在 3/3 seeds 且 pres_ok 在 >=2/3 seeds 且 mean(dP) >= -EPS_DP 且无 seed 出现 dP <= -0.25。
    HOLD   : 方向一致但未达上述；STOP: rob_ok 在 <=1/3 seeds（direction 不一致/消失）。
    """
    rob = [r["dR"] <= -EPS_RZ for r in rows]
    pres = [r["dP"] >= -EPS_DP for r in rows]
    mdP = float(np.mean([r["dP"] for r in rows]))
    worst = min(r["dP"] for r in rows)
    if all(rob) and sum(pres) >= max(2, (2 * len(rows) + 2) // 3) and mdP >= -EPS_DP and worst > -0.25:
        return "ADVANCE", "rob_ok 3/3 且 pres_ok>=2/3 且 mean(dP)=%.4f >= -0.10" % mdP
    if sum(rob) <= 1:
        return "STOP", "robustness gain 在多数 seed 消失（rob_ok %d/%d）" % (sum(rob), len(rob))
    return "HOLD", "方向一致但未达 ADVANCE（rob_ok %d/%d, pres_ok %d/%d, mean dP=%+.4f, worst %+.4f）" % (
        sum(rob), len(rob), sum(pres), len(pres), mdP, worst)


def stage2b_multiseed(cands=None) -> dict:
    cands = cands or ["P10_C3_L2dual_L3a025", "P10_C2_L2resid025_L3a025", "P10_D3_L3m8z_ab025_b050"]
    det, rows = {"candidates": []}, []
    env = [(e["absdz"], e["dprime"]) for e in stage1_frontier()["envelope"]]
    for cid in cands:
        per, rs = [], []
        for s in (0, 1, 2):
            o, _ = metrics("bottle", s, "P10_ORIG_a000", [EXP / "raw"])
            t1, _ = metrics("bottle", s, "P10_T1_L2a000_L3a025", [EXP / "raw"])
            m, d = metrics("bottle", s, cid, [EXP / "raw"])
            if o is None or m is None:
                continue
            dR = m["mean_abs_delta_z"] - o["mean_abs_delta_z"]
            dP = m["mean_dprime"] - o["mean_dprime"]
            per.append({"seed": s, "absdz": m["mean_abs_delta_z"], "dprime": m["mean_dprime"],
                        "dR": dR, "dP": dP,
                        "orig_absdz": o["mean_abs_delta_z"], "orig_dprime": o["mean_dprime"],
                        "dP_vs_T1": (m["mean_dprime"] - t1["mean_dprime"]) if t1 else None,
                        "dR_vs_T1": (m["mean_abs_delta_z"] - t1["mean_abs_delta_z"]) if t1 else None})
            rs.append({"dR": dR, "dP": dP})
        if not per:
            continue
        vd, why = verdict_multiseed(rs)
        beyond = (per[0]["dprime"] - frontier_interp(env, per[0]["absdz"])) if env else float("nan")
        rec = {"candidate": cid, "per_seed": per,
               "mean_dR": float(np.mean([p["dR"] for p in per])),
               "std_dR": float(np.std([p["dR"] for p in per])),
               "mean_dP": float(np.mean([p["dP"] for p in per])),
               "std_dP": float(np.std([p["dP"] for p in per])),
               "worst_dP": min(p["dP"] for p in per),
               "worst_dR": max(p["dR"] for p in per),
               "seeds_same_direction": bool(all(p["dR"] < 0 for p in per) and all(p["dP"] < 0 for p in per)),
               "beyond_frontier_seed0": beyond, "verdict": vd, "verdict_reason": why}
        det["candidates"].append(rec)
        for p in per:
            rows.append({"candidate": cid, **{k: (round(v, 6) if isinstance(v, float) else v)
                                              for k, v in p.items()}, "verdict": vd})
    order = {"ADVANCE": 0, "HOLD": 1, "STOP": 2}
    det["candidates"].sort(key=lambda r: (order[r["verdict"]], -r["mean_dP"]))
    det["n_advance"] = sum(1 for c in det["candidates"] if c["verdict"] == "ADVANCE")
    wcsv(EXP / "candidates" / "multiseed.csv", rows)
    (EXP / "candidates" / "multiseed.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))
    return det


def _fig1(f: dict, tour: dict) -> None:
    env = f["envelope"]
    o = f["original_189"]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot([e["absdz"] for e in env], [e["dprime"] for e in env], "--o", color="#888888",
            lw=1.3, ms=5, label="corrected uniform-α frontier (existing, frozen grid)")
    for w in f["workpoints"]:
        ax.annotate("α=%g" % w["alpha"], (w["absdz"], w["dprime"]), fontsize=7, color="#777777",
                    textcoords="offset points", xytext=(4, -10))
    ax.scatter([o["absdz"]], [o["dprime"]], s=230, marker="*", color="#222222",
               edgecolors="black", zorder=6, label="Original α=0 (189)")
    t1, _ = metrics("bottle", 0, "P10_T1_L2a000_L3a025", [EXP / "raw"])
    ax.scatter([t1["mean_abs_delta_z"]], [t1["mean_dprime"]], s=150, marker="s", color="#DD8452",
               edgecolors="black", zorder=6, label="T1 (previous best)")
    mk = {"ADVANCE": ("*", "#55A868", 330), "HOLD": ("^", "#4C72B0", 130), "STOP": ("x", "#C44E52", 110)}
    for c in tour["candidates"]:
        m, s, sz = mk[c["verdict"]]
        ax.scatter([c["absdz"]], [c["dprime"]], s=sz, marker=m, color=s, edgecolors="black",
                   linewidths=0.8, zorder=7, label=("%s (%s)" % (c["verdict"], c["family"]))
                   if c["verdict"] not in [h.get_label() for h in []] else None)
        if c["verdict"] == "ADVANCE":
            ax.annotate(c["candidate"].split("_")[1], (c["absdz"], c["dprime"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=8, fontweight="bold")
    h, l = ax.get_legend_handles_labels()
    dd = dict(zip(l, h))
    ax.legend(dd.values(), dd.keys(), fontsize=8, loc="lower left")
    ax.set_xlabel("robustness cost  mean |ΔNormalScore_z|  (lower = better)")
    ax.set_ylabel("preservation  mean defect d′  (higher = better)")
    ax.set_title("Experiment 10 — corrected V2 (189) robustness–preservation plane, bottle seed0")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_frontier.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig2(tour: dict) -> None:
    ms = stage2b_multiseed()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for c in tour["candidates"]:
        col = {"ADVANCE": "#55A868", "HOLD": "#4C72B0", "STOP": "#C44E52"}[c["verdict"]]
        ax.scatter([c["delta_absdz_vs_original"]], [c["delta_dprime_vs_original"]], s=110,
                   color=col, edgecolors="black", zorder=5)
        ax.annotate(c["candidate"].replace("P10_", ""), (c["delta_absdz_vs_original"],
                    c["delta_dprime_vs_original"]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    for c in ms["candidates"]:
        col = {"ADVANCE": "#55A868", "HOLD": "#4C72B0", "STOP": "#C44E52"}[c["verdict"]]
        ax.scatter([c["mean_dR"]], [c["mean_dP"]], s=330, marker="*", color=col,
                   edgecolors="black", zorder=6)
        ax.errorbar([c["mean_dR"]], [c["mean_dP"]], yerr=[c["std_dP"]], fmt="none",
                    ecolor=col, capsize=4, zorder=4)
    ax.axhline(-EPS_DP, ls="--", c="#333333", lw=1.2, label="preservation tolerance −0.10")
    ax.axvline(-EPS_RZ, ls=":", c="#333333", lw=1.2, label="robustness tolerance −0.02")
    ax.axhline(0, c="#AAAAAA", lw=0.8); ax.axvline(0, c="#AAAAAA", lw=0.8)
    ax.set_xlabel("Δ robustness vs corrected Original  (more negative = better)")
    ax.set_ylabel("Δ preservation vs corrected Original")
    ax.set_title("Experiment 10 — candidate tournament (bars = 3-seed std for top-3; ★ = mean over seeds 0/1/2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_tournament.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def stage3_cross_category(cands=None) -> dict:
    """Top1-2 × 5 categories × seed0（bottle 复用 Stage 2B seed0）。冻结判据：
    每类 pass = (dR <= -EPS_RZ) and (dP >= -EPS_DP)；catastrophic = dP <= -0.25。
    ADVANCE: wins >= 3/5 且无 catastrophic。
    """
    cands = cands or ["P10_C3_L2dual_L3a025", "P10_C2_L2resid025_L3a025"]
    det = {"candidates": []}
    table, rows = [], []
    for cid in cands:
        per = {}
        for cat in CATS_ALL:
            o, _ = metrics(cat, 0, "P10_ORIG_a000", [EXP / "raw"])
            m, d = metrics(cat, 0, cid, [EXP / "raw"])
            if o is None or m is None:
                per[cat] = None
                continue
            dR = m["mean_abs_delta_z"] - o["mean_abs_delta_z"]
            dP = m["mean_dprime"] - o["mean_dprime"]
            per[cat] = {"absdz": m["mean_abs_delta_z"], "dprime": m["mean_dprime"],
                        "orig_absdz": o["mean_abs_delta_z"], "orig_dprime": o["mean_dprime"],
                        "dR": dR, "dP": dP,
                        "pass": bool(dR <= -EPS_RZ and dP >= -EPS_DP),
                        "catastrophic": bool(dP <= -0.25),
                        "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"]}
        got = {k: v for k, v in per.items() if v}
        wins = sum(1 for v in got.values() if v["pass"])
        cats = sum(1 for v in got.values() if v["catastrophic"])
        if wins >= 3 and cats == 0 and len(got) == len(CATS_ALL):
            vd, why = "ADVANCE", "wins %d/5 且无 catastrophic collapse" % wins
        elif wins >= 3:
            vd, why = "HOLD", "wins %d/5 但存在 catastrophic collapse (%d cats)" % (wins, cats)
        elif wins >= 2:
            vd, why = "HOLD", "wins %d/5（未达 3/5）" % wins
        else:
            vd, why = "STOP", "wins %d/5（跨类别不成立）" % wins
        rec = {"candidate": cid, "per_category": per, "wins": wins,
               "catastrophic_cats": cats, "verdict": vd, "verdict_reason": why,
               "mean_dR_5cat": float(np.mean([v["dR"] for v in got.values()])),
               "mean_dP_5cat": float(np.mean([v["dP"] for v in got.values()]))}
        det["candidates"].append(rec)
        table.append({"Method": cid.replace("P10_", ""), **{
            c: (("%+.4f / %+.4f" % (per[c]["dR"], per[c]["dP"])) if per[c] else "n/a")
            for c in CATS_ALL}, "Wins": "%d/5" % wins, "Verdict": vd})
        for c, v in per.items():
            if v:
                rows.append({"candidate": cid, "category": c, **{k: (round(x, 6) if isinstance(x, float) else x)
                                                                 for k, x in v.items()},
                             "verdict": vd})
    det["table"] = table
    order = {"ADVANCE": 0, "HOLD": 1, "STOP": 2}
    det["candidates"].sort(key=lambda r: (order[r["verdict"]], -r["wins"]))
    det["n_advance"] = sum(1 for c in det["candidates"] if c["verdict"] == "ADVANCE")
    wcsv(EXP / "analysis" / "cross_category.csv", rows)
    (EXP / "analysis" / "cross_category.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))
    return det


def stage4_final(cand: str = "P10_C2_L2resid025_L3a025") -> dict:
    """Top1 × 5 categories × seeds {0,1,2}（bottle 复用 Stage2B；其余由 Stage4 补齐）。"""
    per, missing = {}, []
    for cat in CATS_ALL:
        for s in (0, 1, 2):
            o, _ = metrics(cat, s, "P10_ORIG_a000", [EXP / "raw"])
            m, _ = metrics(cat, s, cand, [EXP / "raw"])
            if o is None or m is None:
                missing.append(f"{cat}:{s}")
                continue
            per.setdefault(cat, {})[s] = {
                "absdz": m["mean_abs_delta_z"], "dprime": m["mean_dprime"],
                "dR": m["mean_abs_delta_z"] - o["mean_abs_delta_z"],
                "dP": m["mean_dprime"] - o["mean_dprime"],
                "pass": bool(m["mean_abs_delta_z"] - o["mean_abs_delta_z"] <= -EPS_RZ
                             and m["mean_dprime"] - o["mean_dprime"] >= -EPS_DP),
                "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"]}
    rows = []
    for cat, ss in per.items():
        for s, v in ss.items():
            rows.append({"candidate": cand, "category": cat, "seed": s,
                         **{k: (round(x, 6) if isinstance(x, float) else x) for k, x in v.items()}})
    cat_summary = {cat: {"mean_dR": float(np.mean([v["dR"] for v in ss.values()])),
                         "mean_dP": float(np.mean([v["dP"] for v in ss.values()])),
                         "n_seeds": len(ss),
                         "all_seeds_pass": bool(all(v["pass"] for v in ss.values())),
                         "worst_dP": min(v["dP"] for v in ss.values())}
                   for cat, ss in per.items()}
    allv = [v for ss in per.values() for v in ss.values()]
    wins_cat = sum(1 for c, v in cat_summary.items()
                   if v["mean_dR"] <= -EPS_RZ and v["mean_dP"] >= -EPS_DP)
    det = {"candidate": cand, "missing": missing,
           "overall": {"n_units": len(allv),
                       "mean_dR": float(np.mean([v["dR"] for v in allv])),
                       "mean_dP": float(np.mean([v["dP"] for v in allv])),
                       "worst_dP": min(v["dP"] for v in allv),
                       "worst_dR": max(v["dR"] for v in allv),
                       "cats_pass": wins_cat, "n_cats": len(cat_summary),
                       "catastrophic": [c for c, v in cat_summary.items() if v["worst_dP"] <= -0.25]},
           "per_category": cat_summary}
    wcsv(EXP / "analysis" / "final_5x3.csv", rows)
    (EXP / "analysis" / "final_5x3.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))
    return det


def _fig3(cross: dict) -> None:
    cats = CATS_ALL
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    mk = {"P10_C2_L2resid025_L3a025": ("o", "#C44E52"), "P10_C3_L2dual_L3a025": ("^", "#4C72B0")}
    for r in cross["candidates"]:
        m, col = mk[r["candidate"]]
        xs = [r["per_category"][c]["dR"] if r["per_category"].get(c) else np.nan for c in cats]
        ys = [r["per_category"][c]["dP"] if r["per_category"].get(c) else np.nan for c in cats]
        ax.scatter(xs, ys, s=140, marker=m, color=col, edgecolors="black", zorder=5,
                   label=r["candidate"].replace("P10_", "") + " (%s)" % r["verdict"])
        for c, x, y in zip(cats, xs, ys):
            if np.isfinite(x):
                ax.annotate(c[:4], (x, y), textcoords="offset points", xytext=(6, 3), fontsize=7.5)
    ax.axhline(-EPS_DP, ls="--", c="#333333", lw=1.2)
    ax.axvline(-EPS_RZ, ls=":", c="#333333", lw=1.2)
    ax.axhline(0, c="#AAAAAA", lw=0.8); ax.axvline(0, c="#AAAAAA", lw=0.8)
    ys_all = [v for r in cross["candidates"] for v in
              ([r["per_category"][c]["dP"] for c in cats if r["per_category"].get(c)])]
    ax.set_ylim(min(ys_all) - 0.12, max(ys_all) + 0.12)
    ax.set_xlabel("Δ robustness vs category-matched corrected Original")
    ax.set_ylabel("Δ preservation vs category-matched corrected Original")
    ax.set_title("Experiment 10 — Stage 3 cross-category consistency (5 cats × seed0); dashed = −0.10 / −0.02 bands")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_cross_category.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_all() -> dict:
    a05 = stage05_audit()
    f = stage1_frontier()
    t = stage2_tournament()
    ms = stage2b_multiseed()
    cc = stage3_cross_category()
    s4 = stage4_final()
    _fig1(f, t)
    _fig2(t)
    _fig3(cc)

    ms_by = {c["candidate"]: c for c in ms["candidates"]}
    cc_by = {c["candidate"]: c for c in cc["candidates"]}
    rows = []
    for r in t["candidates"]:
        cid = r["candidate"]
        m = ms_by.get(cid, {})
        c = cc_by.get(cid, {})
        seeds = "3" if m else "1 (seed0)"
        cats = "%d/5" % c["wins"] if c else "1 (bottle)"
        rows.append({"rank": 0, "method": cid.replace("P10_", ""), "family": r["family"],
                     "stage2A_verdict": r["verdict"], "stage2B_verdict": m.get("verdict", "-"),
                     "stage3_verdict": c.get("verdict", "-"),
                     "bottle_absdz": r["absdz"], "bottle_dprime": r["dprime"],
                     "delta_absz_vs_original": r["delta_absdz_vs_original"],
                     "delta_dprime_vs_original": r["delta_dprime_vs_original"],
                     "mean_delta_dprime_3seed": (round(m["mean_dP"], 4) if m else None),
                     "worst_delta_dprime_3seed": (round(m["worst_dP"], 4) if m else None),
                     "mean_delta_absz_3seed": (round(m["mean_dR"], 4) if m else None),
                     "beyond_frontier_bottle_s0": r["beyond_frontier"],
                     "frontier_break": r["frontier_break"],
                     "seeds_evaluated": seeds, "categories_evaluated": cats,
                     "final_verdict": (c.get("verdict") if c else m.get("verdict", r["verdict"]))})
    order = {"ADVANCE": 0, "HOLD": 1, "STOP": 2}
    for r in rows:
        v = r["final_verdict"] if r["final_verdict"] in order else "HOLD"
        r["final_verdict"] = v
    rows.sort(key=lambda r: (order[r["final_verdict"]], -r["frontier_break"], r["delta_dprime_vs_original"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    wcsv(EXP / "final_ranking.csv", rows)

    reg = c10.registry()
    reg["resolved"] = [{"id": r["method"], "family": r["family"], "final_verdict": r["final_verdict"],
                        "seeds_evaluated": r["seeds_evaluated"],
                        "categories_evaluated": r["categories_evaluated"]} for r in rows]
    reg["family_A_note"] = "Family A (residual recovery) 代数等价于 per-layer alpha -> 未跑 GPU"
    (EXP / "candidate_registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False))

    (EXP / "analysis" / "summary.json").write_text(json.dumps(
        {"stage05": {k: v for k, v in a05["bottle_seed0"].items() if k != "units"},
         "frontier": f["envelope"], "original_189": f["original_189"],
         "stage2A": {"n_advance": t["n_advance"], "n_frontier_break": t["n_frontier_break"]},
         "stage2B": {"n_advance": ms["n_advance"]},
         "stage3": {"n_advance": cc["n_advance"], "table": cc["table"]},
         "stage4": s4["overall"],
         "final_ranking": rows}, indent=2, ensure_ascii=False))
    return {"a05": a05, "frontier": f, "tour": t, "ms": ms, "cc": cc, "s4": s4, "ranking": rows}


if __name__ == "__main__":
    res = run_all()
    print("=" * 132)
    print("Experiment 10 — Preservation Recovery Tournament | FINAL RANKING")
    print("=" * 132)
    _t1, _ = metrics("bottle", 0, "P10_T1_L2a000_L3a025", [EXP / "raw"])
    _env = [(e["absdz"], e["dprime"]) for e in res["frontier"]["envelope"]]
    print("corrected Original-189: |dz|=%.4f d'=%.4f | T1-189: |dz|=%.4f d'=%.4f | T1 beyond frontier=%+.4f"
          % (res["frontier"]["original_189"]["absdz"], res["frontier"]["original_189"]["dprime"],
             _t1["mean_abs_delta_z"], _t1["mean_dprime"],
             _t1["mean_dprime"] - frontier_interp(_env, _t1["mean_abs_delta_z"])))
    print("%-4s%-30s%-6s%10s%10s%10s%10s%8s%9s%9s" % ("#", "method", "fam", "|dz|", "d'", "d|dz|", "dd'",
                                                      "3seed dP", "seeds", "cats"))
    for r in res["ranking"]:
        print("%-4d%-30s%-6s%10.4f%10.4f%+10.4f%+10.4f%8s%9s%9s  %s" % (
            r["rank"], r["method"], r["family"], r["bottle_absdz"], r["bottle_dprime"],
            r["delta_absz_vs_original"], r["delta_dprime_vs_original"],
            (("%+.4f" % r["mean_delta_dprime_3seed"]) if r["mean_delta_dprime_3seed"] is not None else "-"),
            r["seeds_evaluated"], r["categories_evaluated"], r["final_verdict"]))
    s4 = res["s4"]["overall"]
    print("-" * 132)
    print("Stage 4 (Top1=C2, 5 cats × 3 seeds): n=%d mean dR=%+.4f mean dP=%+.4f worst dP=%+.4f "
          "cats_pass=%d/%d catastrophic=%s missing=%s"
          % (s4["n_units"], s4["mean_dR"], s4["mean_dP"], s4["worst_dP"], s4["cats_pass"],
             s4["n_cats"], s4["catastrophic"], res["s4"]["missing"]))
    print("artifacts -> %s | %s" % (ANA, FIG))
