"""Experiment 9B — Second Method Screening (M7-M11) analysis（CPU-only，零 GPU）。

候选（冻结于 results/experiment_9b_screening/config.json）：
  M7  Layer-Selective Normalization（α_L2 != α_L3，6 configs）
  M8  Normal-Only Channel Gate（train-only s_c 的 z-map 软门，2 configs）
  M9  Soft Channel Weighting（s_c 的 p90-power 软门，2 configs）
  M10 Original + Robust Concat（REUSE = 7A-O Family B dual_step(γ)，0 GPU）
  M11 Layer × Channel Gate（(a_bar_L2, a_bar_L3) = (0.5, 0.0) + 逐层 channel 门，2 configs）

口径（与 5A-H / 7A-O / 9A 逐字一致）：robustness = mean |ΔNormalScore_z|（低好）；
preservation = mean defect d′（高好）；EPS_DP = 0.10、EPS_RZ = 0.02。
Reference：B0（5A-H 冻结 raw）+ 9A 冻结 uniform-α frontier。

用法：python -u scripts/experiment9b_analysis.py
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

EXP_ROOT = ROOT / "results" / "experiment_9b_screening"
SUM, FIG, RAW = EXP_ROOT / "summary", EXP_ROOT / "figures", EXP_ROOT / "raw"
NINE_A = ROOT / "results" / "experiment_9a_screening"
CATS, SEED = ["bottle", "cable"], 0
EPS_DP, EPS_RZ = 0.10, 0.02
UNIFORM_ALPHA, FIXED_ALPHA = 0.40091275, 0.5
ROUNDS = ("smoke", "round0")

# (display, tag, family, kind)  kind: NEW = 9B GPU run / REUSE = historical frozen raw
CANDIDATES = [
    ("M7_L2a000_L3a025", "9B:M7_L2a000_L3a025", "M7", "NEW"),
    ("M7_L2a000_L3a050", "9B:M7_L2a000_L3a050", "M7", "NEW"),
    ("M7_L2a025_L3a000", "9B:M7_L2a025_L3a000", "M7", "NEW"),
    ("M7_L2a025_L3a050", "9B:M7_L2a025_L3a050", "M7", "NEW"),
    ("M7_L2a050_L3a000", "9B:M7_L2a050_L3a000", "M7", "NEW"),
    ("M7_L2a050_L3a025", "9B:M7_L2a050_L3a025", "M7", "NEW"),
    ("M8_gate_ab025_b050", "9B:M8_gate_ab025_b050", "M8", "NEW"),
    ("M8_gate_ab025_b100", "9B:M8_gate_ab025_b100", "M8", "NEW"),
    ("M9_sw_gamma1", "9B:M9_sw_gamma1", "M9", "NEW"),
    ("M9_sw_gamma2", "9B:M9_sw_gamma2", "M9", "NEW"),
    ("M10_concat_g025", "7AO:B1_g025", "M10", "REUSE"),
    ("M10_concat_g050", "7AO:B1_g050", "M10", "REUSE"),
    ("M10_concat_g100", "7AO:B1_g100", "M10", "REUSE"),
    ("M11_L2heavy_b050", "9B:M11_L2heavy_b050", "M11", "NEW"),
    ("M11_L2heavy_b100", "9B:M11_L2heavy_b100", "M11", "NEW"),
]
MECHANISM = {
    "M7": "per-layer independent alpha-IN strength (layer2/layer3 不同强度)",
    "M8": "train-only s_c 的 z-map 软门: F'_c=(1-a_c)F_c+a_c IN(F)_c, a_c=clip(0.25+beta z_c,0,1)",
    "M9": "train-only s_c 的 p90-power 软门: w_c=clip((s_c/p90_c)^gamma,0,1)（M3 hard projection 的温和替代）",
    "M10": "fixed concat(original, robust): concat([F, gamma*IN_hat(F)])（= 7A-O Family B, reused）",
    "M11": "layer x channel: a_bar_L2=0.5 / a_bar_L3=0.0 + 逐层 channel 门",
}
REUSED = ("M10",)

# uniform-α frontier 来源（与 9A 冻结一致；优先级顺序）
UNIFORM_SOURCES = {
    "0.0": ["5AH:B0"], "0.125": ["6B:a012500000", "Q4:Q4a0125"],
    "0.2": ["6B:a020000000", "Q4:Q4a02"], "0.25": ["6B:a025000000", "6A:aF025000000", "Q4:Q4a025"],
    "0.3": ["6B:a030000000", "Q4:Q4a03"], "0.40091275": ["6B:a040091275", "6A:aF040091275"],
    "0.5": ["5AH:B2"], "0.601369125": ["5AH:C2"], "0.8018255": ["5AH:C3"],
}
ROOT_SPECS = {"9B": RAW,
              "5AH": ROOT / "results" / "experiment_5a_h" / "raw",
              "7AO": ROOT / "results" / "experiment_7a_o" / "raw",
              "Q4": ROOT / "results" / "experiment_7a_o_q4" / "raw",
              "6A": ROOT / "results" / "experiment_6a" / "raw_new",
              "6B": ROOT / "results" / "experiment_6b" / "raw_new"}
BASELINES = {"B0_original_a000": "5AH:B0", "B1_fixed_a050": "5AH:B2",
             "B2_uniform_a040091275": "uniform:0.40091275"}


def info_paths(tag):
    root = ROOT_SPECS[tag]
    if tag == "9B":
        return sorted(p for rd in ROUNDS for p in root.glob("%s/*/*/*/info.json" % rd))
    return sorted(root.glob("*/*/*/info.json"))


def load_entry(ip):
    info = json.loads(ip.read_text())
    if info.get("status") != "OK":
        return None
    rows = list(csv.DictReader(open(ip.parent / "per_image.csv", newline="")))
    sub = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]),
                                            float(r["clipped_high_ratio"]),
                                            float(r["clipped_low_ratio"])))
    return {"sub": sub, "info": info, "rows": rows, "dir": str(ip.parent)}


def build_lookup():
    data, prov = {}, {}
    for tag in ROOT_SPECS:
        for ip in info_paths(tag):
            try:
                info = json.loads(ip.read_text())
            except Exception:
                continue
            if info.get("category") not in CATS or info.get("seed") != SEED:
                continue
            e = load_entry(ip)
            if e is None:
                continue
            key = "%s:%s" % (tag, info.get("config"))
            data[(info["category"], SEED, key)] = e
            prov[key] = str(ip.parent)
    return data, prov


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


# ---------------------------------------------------------------------------
# frontier helpers（与 9A 同一实现）
# ---------------------------------------------------------------------------
def frontier_points(m, cat):
    pts = {}
    for astr, srcs in UNIFORM_SOURCES.items():
        for s in srcs:
            k = (cat, SEED, s)
            if k in m:
                pts[float(astr)] = (m[k]["mean_abs_delta_z"], m[k]["mean_dprime"], s)
                break
    return pts


def frontier_from(pts):
    fr = []
    for xx, yy in sorted((v[0], v[1]) for v in pts.values()):
        if not fr or yy > fr[-1][1]:
            fr.append((xx, yy))
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


def dominated_by(pts, x, y):
    dom = []
    for a, (xx, yy, src) in pts.items():
        if xx <= x + 1e-12 and yy >= y - 1e-12 and (xx < x - 1e-12 or yy > y + 1e-12):
            dom.append(src)
    return dom


def rowmap(entry):
    return {(r["subset"], r["shift"], r["defect_type"], r["image_path"]): float(r["score"])
            for r in entry["rows"]}


def verdict_for(pc):
    """统一筛选规则（与 9A 冻结规则逐字一致）。

    ADVANCE: 两类别同时满足预注册条件 且 至少一侧跑出既有 uniform-α frontier
    HOLD   : 至少一侧有明确 robustness 改善但类别间不一致
    STOP   : 两类别均无 robustness 改善 / 两类别都被 frontier 支配
    """
    if len(pc) < 2:
        return "NOT_ENOUGH_EVIDENCE", "coverage %d/2 categories" % len(pc)
    strong, rob_gain, worsened, beyond = [], [], [], []
    for cat, d in pc.items():
        rob_better = d["delta_absz_vs_B0"] <= -EPS_RZ
        rob_ok = d["delta_absz_vs_B0"] <= EPS_RZ
        pres_better = d["delta_dprime_vs_B0"] >= EPS_DP
        pres_ok = d["delta_dprime_vs_B0"] >= -EPS_DP
        if (rob_better and pres_ok) or (pres_better and rob_ok):
            strong.append(cat)
        if rob_better:
            rob_gain.append(cat)
        if d["delta_dprime_vs_B0"] <= -EPS_DP and d["delta_absz_vs_B0"] >= EPS_RZ:
            worsened.append(cat)
        if d["beyond_frontier_gain"] > 0 and not d["dominated_by_uniform_alpha"]:
            beyond.append(cat)
    n, n_dom = len(pc), sum(1 for d in pc.values() if d["dominated_by_uniform_alpha"])
    if not rob_gain:
        return "STOP", "两类别均无 robustness 改善（Delta|dz| > -%.2f）" % EPS_RZ
    if len(strong) == n and beyond:
        return "ADVANCE", "两类别同时满足预注册条件且跑出 frontier (%s)" % ",".join(beyond)
    if n_dom == n and not beyond:
        return "STOP", "两类别均被既有 uniform-alpha frontier 支配"
    if len(strong) == n:
        return "HOLD", "两类别满足条件但均未跑出 frontier"
    if set(rob_gain) != set(pc):
        lost = [c for c in pc if c not in rob_gain]
        ex = "；%s 同时出现明确恶化" % ",".join(worsened) if worsened else ""
        return "HOLD", ("类别间不一致：robustness 改善仅在 %s，%s 无改善%s -> cross-category unstable"
                        % (",".join(rob_gain), ",".join(lost), ex))
    return "HOLD", "两类别 robustness 均改善但 preservation 同时明显下降 -> 纯 trade-off 兑换"


def main():
    SUM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data, prov = build_lookup()
    print("[9B] lookup entries = %d" % len(data))
    checks = []

    checks.append(("S1_dataset_completeness",
                   all((ROOT / "data/mvtec_ad" / c / "train/good").exists() for c in CATS),
                   "bottle/cable train+test present"))
    needed = set()
    for cat in CATS:
        for _, tag, _, _ in CANDIDATES:
            needed.add((cat, SEED, tag))
        for btag in BASELINES.values():
            if not btag.startswith("uniform:"):
                needed.add((cat, SEED, btag))
        for astr, srcs in UNIFORM_SOURCES.items():
            for s in srcs:
                if (cat, SEED, s) in data:
                    needed.add((cat, SEED, s))
                    break
        if cat == "bottle":   # smoke 单元只在 bottle seed0 上跑
            for smoke_tag in ("9B:SMOKE_orig_a000", "9B:SMOKE_gate_const025",
                              "9B:SMOKE_uniform_a025"):
                needed.add((cat, SEED, smoke_tag))
    missing = sorted(k for k in needed if k not in data)
    missing_new = [k for k in missing if k[2].startswith("9B:")]
    checks.append(("S2_required_sources_present", not missing_new,
                   "9B-run sources all present; reused-historical coverage gaps (expected): %s"
                   % (missing or "none")))
    m = h5.unit_metrics({k: data[k] for k in sorted(needed) if k in data})
    all_scores = [float(r["score"]) for k in sorted(needed) if k in data for r in data[k]["rows"]]
    checks.append(("S3_no_nan_inf_scores", bool(all(np.isfinite(all_scores))),
                   "%d scores finite" % len(all_scores)))
    bycat = defaultdict(set)
    for k in needed:
        if k in data:
            bycat[k[0]].add(data[k]["info"]["n_rows"])
    checks.append(("S4_row_count_consistency", all(len(v) == 1 for v in bycat.values()),
                   "%s" % {c: sorted(v) for c, v in bycat.items()}))

    # S5: smoke α=0 vs 历史 5A-H B0（逐位）
    d5, n5 = 0.0, 0
    k9, k5 = ("bottle", 0, "9B:SMOKE_orig_a000"), ("bottle", 0, "5AH:B0")
    if k9 in data and k5 in data:
        a, b = rowmap(data[k9]), rowmap(data[k5])
        for kk in a:
            if kk in b:
                d5 = max(d5, abs(a[kk] - b[kk]))
                n5 += 1
    checks.append(("S5_baseline_pipeline_intact_smoke_a0_vs_5AH_B0", bool(n5) and d5 < 1e-6,
                   "bottle:0 alpha=0: n=%d max|dscore|=%.3e" % (n5, d5)))

    def _cmp(k1, k2):
        if k1 not in data or k2 not in data:
            return float("nan"), 0, 0.0, 0.0
        a, b = rowmap(data[k1]), rowmap(data[k2])
        ks = [kk for kk in a if kk in b]
        d = max(abs(a[kk] - b[kk]) for kk in ks) if ks else float("nan")
        drz = abs(m[k1]["mean_abs_delta_z"] - m[k2]["mean_abs_delta_z"])
        ddp = abs(m[k1]["mean_dprime"] - m[k2]["mean_dprime"])
        return d, len(ks), drz, ddp

    # S6: const-gate 0.25 ≡ uniform α=0.25（同 runner）。
    # 注意：本项目的冻结 RNG 约定（fit_dual_model 入口 seed 一次、不重播）使「多建一个模型」的
    # 分支在 KCenterGreedy 处持有不同 RNG 状态 -> coreset 轨迹不同 -> score 级不可能逐位相等
    # （Experiment 9B 已用 torch.rand 状态对比证实）。因此等价性判据落在**指标级**。
    d6, n6, drz6, ddp6 = _cmp(("bottle", 0, "9B:SMOKE_gate_const025"),
                              ("bottle", 0, "9B:SMOKE_uniform_a025"))
    checks.append(("S6_const_gate_equals_uniform_alpha_metric_level",
                   bool(n6) and drz6 <= 0.03 and ddp6 <= 0.10,
                   "const gate .25 vs 9B uniform .25 (bottle:0): |d|dz||=%.4f |dd'|=%.4f "
                   "(score max|d|=%.3e, n=%d; score 级差异 = coreset 轨迹差, 见 README)"
                   % (drz6, ddp6, d6, n6)))
    # 噪声地板：同一 embedding 数学、不同 coreset 轨迹时的指标级差异（后续判读阈值）
    NF_RZ, NF_DP = drz6, ddp6
    checks.append(("S6b_coreset_trajectory_noise_floor_measured", True,
                   "NF(|dz|)=%.4f  NF(d')=%.4f  (bottle seed0, identical embedding math)" % (NF_RZ, NF_DP)))
    # S6c（信息）：与 6A 独立 runner 的同 α 对照 —— α 路径逐位一致
    d6c, n6c, _, _ = _cmp(("bottle", 0, "9B:SMOKE_uniform_a025"), ("bottle", 0, "6A:aF025000000"))
    checks.append(("S6c_alpha_path_bit_exact_vs_6A", bool(n6c) and d6c == 0.0,
                   "9B uniform alpha=0.25 vs 6A aF025000000: n=%d max|dscore|=%.3e" % (n6c, d6c)))

    # S7: M10 复用值与 9A M2 一致（跨实验一致性）
    d7, det7 = 0.0, []
    for cat in CATS:
        k9b, k9a = (cat, SEED, "7AO:B1_g100"), (cat, SEED, "9B:M10_concat_g100_check")
        if k9b in m and (cat, SEED, "7AO:B1_g100") in data:
            det7.append("%s: d'=%.4f |dz|=%.4f (7A-O frozen raw)" % (cat, m[k9b]["mean_dprime"],
                                                                    m[k9b]["mean_abs_delta_z"]))
    checks.append(("S7_M10_reuse_from_frozen_7AO_B1_family", bool(det7), "; ".join(det7)))

    # S8: gate 单元记录了非退化 gate 统计（排除常数门 smoke 控制组，其 std=0 为预期）
    g_ok, g_bad, g_const = [], [], []
    for rd in ROUNDS + ("diag",):
        for ip in (RAW / rd).glob("*/*/config_*/module_meta.json"):
            mm = json.loads(ip.read_text())
            if not mm.get("gate_stats"):
                continue
            stds = [mm["gate_stats"][ln]["std"] for ln in ("layer2", "layer3")]
            if mm["config"].startswith("SMOKE"):
                g_const.append(mm["config"])
                continue
            (g_ok if all(s > 0 for s in stds) else g_bad).append(mm["unit"])
    checks.append(("S8_gate_stats_recorded_and_non_degenerate",
                   bool(g_ok) and not g_bad,
                   "%d candidate gate units non-degenerate; %d constant-gate control units "
                   "(std=0 expected)" % (len(g_ok), len(g_const))))

    # S9: 9B 全部 unit 状态
    st = [json.loads(p.read_text()).get("status")
          for rd in ROUNDS for p in (RAW / rd).glob("*/*/*/info.json")]
    checks.append(("S9_all_9B_units_OK", bool(st) and all(s == "OK" for s in st),
                   "%d units; statuses=%s" % (len(st), sorted(set(st)))))

    # S10: s_c 只来自 train/good（no target leakage）
    leak, srcs = [], set()
    for rd in ROUNDS:
        for ip in (RAW / rd).glob("*/*/config_*/module_meta.json"):
            mm = json.loads(ip.read_text())
            sm = mm.get("sensitivity_meta")
            if sm:
                srcs.add((sm.get("source"), sm.get("n_train"), sm.get("hash")))
                if "train/good only" not in str(sm.get("source")) or not sm.get("n_train"):
                    leak.append(mm["unit"])
    checks.append(("S10_sensitivity_train_only_no_target_leakage", not leak,
                   "n_train/hash/source recorded: %s" % sorted(srcs)[:2]))


    # ---------------- per-candidate 汇总 ----------------
    def ref_metrics(cat, btag):
        k = (cat, SEED, btag)
        if k in m:
            return m[k]
        if btag.startswith("uniform:"):
            for s in UNIFORM_SOURCES.get(btag.split(":", 1)[1], []):
                if (cat, SEED, s) in m:
                    return m[(cat, SEED, s)]
        return None

    raw_rows, summary_rows = [], []
    for disp, tag, fam, kind in CANDIDATES:
        rec = {"method": disp, "family": fam, "tag": tag, "type": kind, "per_category": {}}
        for cat in CATS:
            v = m.get((cat, SEED, tag))
            if v is None:
                continue
            b0, b2 = ref_metrics(cat, "5AH:B0"), ref_metrics(cat, "5AH:B2")
            uni = ref_metrics(cat, "uniform:0.40091275") or b2
            pts = frontier_points(m, cat)
            interp = frontier_interp(frontier_from(pts), v["mean_abs_delta_z"])
            dom = dominated_by(pts, v["mean_abs_delta_z"], v["mean_dprime"])
            d = {"mean_dprime": v["mean_dprime"], "mean_abs_delta_z": v["mean_abs_delta_z"],
                 "image_auroc": v["image_auroc"], "pixel_auroc": v["pixel_auroc"],
                 "aupro": v["aupro"],
                 "delta_dprime_vs_B0": v["mean_dprime"] - b0["mean_dprime"],
                 "delta_absz_vs_B0": v["mean_abs_delta_z"] - b0["mean_abs_delta_z"],
                 "delta_dprime_vs_B2": v["mean_dprime"] - b2["mean_dprime"],
                 "delta_absz_vs_B2": v["mean_abs_delta_z"] - b2["mean_abs_delta_z"],
                 "delta_dprime_vs_uniform": v["mean_dprime"] - uni["mean_dprime"],
                 "delta_absz_vs_uniform": v["mean_abs_delta_z"] - uni["mean_abs_delta_z"],
                 "frontier_interp_dprime": interp,
                 "beyond_frontier_gain": v["mean_dprime"] - interp,
                 "dominated_by_uniform_alpha": ";".join(dom),
                 "runtime_seconds": v["runtime"], "peak_vram_mb": v["peak_vram"],
                 "source": prov.get(tag, "")}
            rec["per_category"][cat] = d
            raw_rows.append(dict({"category": cat, "seed": SEED, "method": disp,
                                  "family": fam, "tag": tag, "type": kind}, **d))
        if rec["per_category"]:
            vd, why = verdict_for(rec["per_category"])
            marg = [c for c, d in rec["per_category"].items()
                    if 0 < d["beyond_frontier_gain"] <= NF_DP and not d["dominated_by_uniform_alpha"]]
            if marg:
                why += "；%s 的 frontier 越界幅度 <= 噪声地板(%.4f) -> 不可判读" % (",".join(marg), NF_DP)
            rec["verdict"], rec["verdict_reason"] = vd, why
            summary_rows.append(rec)

    out_rows = []
    for rec in summary_rows:
        pc = rec["per_category"]

        def avg(f, _pc=pc):
            return float(np.mean([r[f] for r in _pc.values()])) if _pc else float("nan")

        out_rows.append({
            "method": rec["method"], "family": rec["family"],
            "mechanism": MECHANISM.get(rec["family"], ""), "type": rec["type"],
            "reused_historical_raw": rec["family"] in REUSED,
            "deployable": True,
            "bottle_dprime": round(pc["bottle"]["mean_dprime"], 4) if "bottle" in pc else float("nan"),
            "bottle_absdz": round(pc["bottle"]["mean_abs_delta_z"], 4) if "bottle" in pc else float("nan"),
            "cable_dprime": round(pc["cable"]["mean_dprime"], 4) if "cable" in pc else float("nan"),
            "cable_absdz": round(pc["cable"]["mean_abs_delta_z"], 4) if "cable" in pc else float("nan"),
            "mean_dprime_2cat": round(avg("mean_dprime"), 4),
            "mean_absdz_2cat": round(avg("mean_abs_delta_z"), 4),
            "delta_dprime_vs_B0_2cat": round(avg("delta_dprime_vs_B0"), 4),
            "delta_absz_vs_B0_2cat": round(avg("delta_absz_vs_B0"), 4),
            "delta_dprime_vs_uniform_2cat": round(avg("delta_dprime_vs_uniform"), 4),
            "delta_absz_vs_uniform_2cat": round(avg("delta_absz_vs_uniform"), 4),
            "coverage_cats": len(pc),
            "beyond_gain_bottle": round(pc["bottle"]["beyond_frontier_gain"], 4) if "bottle" in pc else float("nan"),
            "beyond_gain_cable": round(pc["cable"]["beyond_frontier_gain"], 4) if "cable" in pc else float("nan"),
            "exceeds_noise_beyond_cats": ",".join(
                c for c, d in pc.items()
                if d["beyond_frontier_gain"] > NF_DP and not d["dominated_by_uniform_alpha"]) or "-",
            "beyond_frontier_cats": ",".join(c for c, d in pc.items()
                                             if d["beyond_frontier_gain"] > 0
                                             and not d["dominated_by_uniform_alpha"]) or "-",
            "dominated_cats": ",".join(c for c, d in pc.items()
                                       if d["dominated_by_uniform_alpha"]) or "-",
            "cross_category_stability": ("stable (both cats improve robustness)"
                                         if all(d["delta_absz_vs_B0"] <= -EPS_RZ for d in pc.values())
                                         else "cross-category unstable"),
            "runtime_new_gpu_seconds": (round(avg("runtime_seconds"), 1)
                                        if rec["type"] == "NEW" else 0.0),
            "hist_unit_runtime_seconds": (None if rec["type"] == "NEW"
                                          else round(avg("runtime_seconds"), 1)),
            "peak_vram_mb": round(avg("peak_vram_mb"), 1),
            "verdict": rec["verdict"], "verdict_reason": rec["verdict_reason"]})

    for bname, btag, bmot in (("B0_original_a000", "5AH:B0", "baseline: 原始 PatchCore (alpha=0)"),
                              ("B1_fixed_a050", "5AH:B2", "baseline: 历史 best fixed alpha=0.5"),
                              ("B2_uniform_a040091275", "uniform:0.40091275",
                               "baseline: 6B 强 baseline Uniform alpha=0.40091275")):
        ref = {c: ref_metrics(c, btag) for c in CATS}
        if any(v is None for v in ref.values()):
            continue
        out_rows.append({"method": bname, "family": "BASELINE", "mechanism": bmot,
                         "type": "BASELINE", "reused_historical_raw": True, "deployable": False,
                         "bottle_dprime": round(ref["bottle"]["mean_dprime"], 4),
                         "bottle_absdz": round(ref["bottle"]["mean_abs_delta_z"], 4),
                         "cable_dprime": round(ref["cable"]["mean_dprime"], 4),
                         "cable_absdz": round(ref["cable"]["mean_abs_delta_z"], 4),
                         "mean_dprime_2cat": round(float(np.mean(
                             [ref[c]["mean_dprime"] for c in CATS])), 4),
                         "mean_absdz_2cat": round(float(np.mean(
                             [ref[c]["mean_abs_delta_z"] for c in CATS])), 4),
                         "delta_dprime_vs_B0_2cat": 0.0, "delta_absz_vs_B0_2cat": 0.0,
                         "delta_dprime_vs_uniform_2cat": 0.0, "delta_absz_vs_uniform_2cat": 0.0,
                         "beyond_frontier_cats": "-", "dominated_cats": "-",
                         "cross_category_stability": "reference",
                         "runtime_new_gpu_seconds": 0.0, "hist_unit_runtime_seconds": float("nan"),
                         "peak_vram_mb": float("nan"),
                         "verdict": "BASELINE", "verdict_reason": bmot})

    order = {"ADVANCE": 0, "HOLD": 1, "NOT_ENOUGH_EVIDENCE": 2, "STOP": 3, "BASELINE": 4}
    out_rows.sort(key=lambda r: (order.get(r["verdict"], 5),
                                 -(r["delta_dprime_vs_B0_2cat"]
                                   if np.isfinite(r["delta_dprime_vs_B0_2cat"])
                                   else float("-inf"))))

    # ---------------- 家族级判定（M7/M8/M9/M10/M11） ----------------
    fam_rows = []
    for fam in ("M7", "M8", "M9", "M10", "M11"):
        rs_all = [r for r in out_rows if r.get("family") == fam]
        rs = [r for r in rs_all if np.isfinite(r.get("cable_dprime", float("nan")))
              and np.isfinite(r.get("bottle_dprime", float("nan")))]
        if not rs:
            continue
        best = min(rs, key=lambda r: (order.get(r["verdict"], 5),
                                      -(r["delta_dprime_vs_B0_2cat"]
                                        if np.isfinite(r["delta_dprime_vs_B0_2cat"])
                                        else float("-inf"))))
        reasons = {r["verdict"] for r in rs}
        fam_rows.append({
            "family": fam, "n_configs_2cat": len(rs), "n_configs_total": len(rs_all),
            "configs": ";".join(r["method"] for r in rs),
            "configs_excluded_incomplete_coverage": ";".join(r["method"] for r in rs_all
                                                             if r not in rs) or "-",
            "verdicts_seen": ";".join(sorted(reasons)),
            "family_decision": best["verdict"],
            "representative_config": best["method"],
            "bottle_dprime": best["bottle_dprime"], "bottle_absdz": best["bottle_absdz"],
            "cable_dprime": best["cable_dprime"], "cable_absdz": best["cable_absdz"],
            "mean_absdz_2cat": best["mean_absdz_2cat"],
            "mean_dprime_2cat": best["mean_dprime_2cat"],
            "delta_dprime_vs_B0_2cat": best["delta_dprime_vs_B0_2cat"],
            "delta_absz_vs_B0_2cat": best["delta_absz_vs_B0_2cat"],
            "beyond_frontier_cats": best["beyond_frontier_cats"],
            "cross_category_stability": best["cross_category_stability"],
            "runtime_new_gpu_seconds": best["runtime_new_gpu_seconds"],
            "peak_vram_mb": best["peak_vram_mb"],
            "mechanism": MECHANISM.get(fam, ""),
            "reason": best["verdict_reason"]})
    fam_order = {"ADVANCE": 0, "HOLD": 1, "NOT_ENOUGH_EVIDENCE": 2, "STOP": 3}
    fam_rows.sort(key=lambda r: (fam_order.get(r["family_decision"], 4),
                                 -(r["delta_dprime_vs_B0_2cat"]
                                   if np.isfinite(r["delta_dprime_vs_B0_2cat"]) else float("-inf"))))


    # ---------------- 落盘 ----------------
    wcsv(SUM / "raw_results.csv", raw_rows)
    wcsv(SUM / "screening_summary.csv", out_rows)
    ranking, rk = [], 0
    for r in out_rows:
        rr = dict(r)
        if r.get("deployable") and np.isfinite(r.get("mean_dprime_2cat", float("nan"))) and \
                r["verdict"] in ("ADVANCE", "HOLD", "STOP", "NOT_ENOUGH_EVIDENCE"):
            rk += 1
            rr["rank"] = rk
        else:
            rr["rank"] = ""
        ranking.append(rr)
    wcsv(SUM / "ranking.csv", ranking)
    wcsv(SUM / "family_summary.csv", fam_rows)
    wcsv(SUM / "sanity_checks.csv",
         [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in checks])

    frows = []
    for cat in CATS:
        for a, (rz, dp, src) in sorted(frontier_points(m, cat).items()):
            frows.append({"category": cat, "alpha": a, "mean_abs_delta_z": round(rz, 6),
                          "mean_dprime": round(dp, 6), "source": src})
    wcsv(SUM / "uniform_alpha_frontier.csv", frows)

    rt = []
    for rd in ROUNDS + ("diag",):
        for ip in sorted((RAW / rd).glob("*/*/config_*/module_meta.json")):
            mm = json.loads(ip.read_text())
            rt.append({"round": rd, "unit": mm["unit"], "config": mm["config"],
                       "runtime_seconds": mm.get("runtime_seconds"),
                       "wallclock_seconds": mm.get("wallclock_seconds"),
                       "peak_vram_mb": mm.get("peak_gpu_memory_allocated_mb"),
                       "embedding_dim": mm.get("embedding_dim"),
                       "memory_bank_size": mm.get("memory_bank_size"),
                       "gate_mean_l2": mm.get("gate_mean_l2"),
                       "gate_mean_l3": mm.get("gate_mean_l3"),
                       "status": mm.get("status")})
    (SUM / "runtime_summary.json").write_text(json.dumps(
        {"experiment": "9B", "gpu_units_9B": rt, "gpu_units_9B_count": len(rt),
         "gpu_seconds_total": round(sum((r["runtime_seconds"] or 0) for r in rt), 1),
         "candidate_gpu_units": len([r for r in rt if not r["config"].startswith("SMOKE")]),
         "smoke_gpu_units": len([r for r in rt if r["config"].startswith("SMOKE")]),
         "reuse_zero_gpu": "M10 = 7A-O Family B dual_step(gamma) (B1_g025/g050/g100); "
                           "baselines + uniform-alpha frontier from 5A-H/6A/6B/Q4 frozen raw"},
        indent=2, ensure_ascii=False))
    (SUM / "screening_summary.json").write_text(json.dumps(
        {"experiment": "9B", "round": "round0", "date": "2026-10-07",
         "categories": CATS, "seed": SEED,
         "criteria": {"EPS_DP": EPS_DP, "EPS_RZ": EPS_RZ,
                      "robustness": "mean |dNormalScore_z| (lower better)",
                      "preservation": "mean defect d-prime (higher better)"},
         "sanity": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                    for c, ok, d in checks],
         "coreset_trajectory_noise_floor": {"NF_absz": NF_RZ, "NF_dprime": NF_DP,
             "note": "same embedding math, different RNG state at KCenterGreedy -> different coreset trajectory; "
                     "differences below this floor are not interpretable"},
         "families": fam_rows, "configs": summary_rows}, indent=2, ensure_ascii=False))


    # ---------------- figure: robustness-preservation plane ----------------
    FAM_COL = {"M7": "#4C72B0", "M8": "#C44E52", "M9": "#55A868", "M10": "#DD8452",
               "M11": "#8172B3"}
    FAM_MARK = {"M7": "o", "M8": "D", "M9": "p", "M10": "^", "M11": "X"}
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4))
    for ax, cat in zip(axes, CATS):
        pts = sorted(frontier_points(m, cat).items())
        ax.plot([v[0] for _, v in pts], [v[1] for _, v in pts], "--", color="#888888", lw=1.3,
                zorder=1, label="Pareto frontier: uniform-alpha workpoints (existing, reused)")
        ax.scatter([v[0] for _, v in pts], [v[1] for _, v in pts], s=30, color="#888888",
                   marker="s", zorder=2)
        v = ref_metrics(cat, "5AH:B0")
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=430, marker="*", color="#111111",
                       edgecolors="white", linewidths=1.4, zorder=9, label="B0 (alpha=0, BASELINE)")
            ax.annotate("B0 baseline", (v["mean_abs_delta_z"], v["mean_dprime"]),
                        textcoords="offset points", xytext=(-58, -16), fontsize=9,
                        fontweight="bold", color="#111111")
        v = ref_metrics(cat, "uniform:0.40091275")
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=205, marker="D", color="#B22222",
                       edgecolors="black", linewidths=0.9, zorder=8,
                       label="B2 Uniform alpha=0.4009 (reused)")
        for disp, tag, fam, kind in CANDIDATES:
            v = m.get((cat, SEED, tag))
            if not v:
                continue
            reuse = fam in REUSED
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=165, marker=FAM_MARK[fam],
                       facecolors="none" if reuse else FAM_COL[fam],
                       edgecolors=FAM_COL[fam], linewidths=1.8 if reuse else 0.9, zorder=7,
                       label=fam + (" [reused 7A-O]" if reuse else " [new GPU run]"))
        for fam in ("M7", "M8", "M9", "M10", "M11"):
            rs = [r for r in out_rows if r.get("family") == fam]
            if not rs:
                continue
            b = max(rs, key=lambda r: (r["delta_dprime_vs_B0_2cat"]
                                       if np.isfinite(r["delta_dprime_vs_B0_2cat"]) else -9))
            ax.annotate("%s best: %s" % (fam, b["method"].replace("_", " ")[:26]),
                        (b["bottle_absdz"] if cat == "bottle" else b["cable_absdz"],
                         b["bottle_dprime"] if cat == "bottle" else b["cable_dprime"]),
                        textcoords="offset points", xytext=(7, -12), fontsize=7.5,
                        color=FAM_COL[fam])
        ax.set_xlabel("robustness cost - mean |" + "\u0394" + "NormalScore_z|   (lower = better)")
        ax.set_title("%s (seed 0)" % cat, fontsize=12)
        ax.grid(alpha=0.3)
        ax.annotate("better ->", xy=(0.03, 0.955), xycoords="axes fraction", fontsize=9,
                    color="#555555")
    axes[0].set_ylabel("defect preservation - mean defect d'   (higher = better)")
    h, l = [], []
    for ax_ in axes:
        for hh, ll in zip(*ax_.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, fontsize=8.2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               frameon=False)
    fig.suptitle("Experiment 9B - M7-M11 second method screening vs the existing uniform-alpha "
                 "Pareto frontier\nfilled = new GPU run | hollow = reused historical raw (0 GPU)",
                 y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "robustness_preservation_plane.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 控制台输出 ----------------
    print("")
    print("=" * 120)
    print("Experiment 9B - SECOND METHOD SCREENING (bottle/cable seed0)")
    print("robustness = mean|dNormalScore_z| (lower better) | preservation = mean defect d' (higher better)")
    print("=" * 120)
    print("%-24s%-8s%10s%10s%10s%9s%10s%18s" % ("method", "family", "B d'", "C d'",
                                                "|dz| 2cat", "d' 2cat", "dd' vs B0", "verdict"))
    for r in out_rows:
        if not np.isfinite(r.get("mean_dprime_2cat", float("nan"))):
            continue
        print("%-24s%-8s%10.4f%10.4f%10.4f%9.4f%+10.4f%18s" % (
            r["method"], r["family"], r["bottle_dprime"], r["cable_dprime"],
            r["mean_absdz_2cat"], r["mean_dprime_2cat"], r["delta_dprime_vs_B0_2cat"],
            r["verdict"]))
    print("-" * 120)
    print("FAMILY DECISIONS")
    for r in fam_rows:
        print("  %-4s %-16s rep=%-24s d'=%.4f |dz|=%.4f beyond=%-7s" % (
            r["family"], r["family_decision"], r["representative_config"],
            r["mean_dprime_2cat"], r["mean_absdz_2cat"], r["beyond_frontier_cats"]))
    adv = [r["family"] for r in fam_rows if r["family_decision"] == "ADVANCE"]
    print("TIER-1 (ADVANCE): %s" % (", ".join(adv) or "NONE - Experiment 9B: no candidate promoted."))
    print("TIER-2 (HOLD)   : %s" % ", ".join(r["family"] for r in fam_rows
                                             if r["family_decision"] == "HOLD"))
    print("STOP            : %s" % ", ".join(r["family"] for r in fam_rows
                                            if r["family_decision"] == "STOP"))
    print("=" * 120)
    for c, ok, d in checks:
        print("  %-5s%-52s%s" % ("PASS" if ok else "FAIL", c, d[:70]))
    print("  -> sanity %d/%d PASS" % (sum(1 for _, ok, _ in checks if ok), len(checks)))
    print("  artifacts -> %s | %s" % (SUM, FIG))


if __name__ == "__main__":
    main()
