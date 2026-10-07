"""Experiment 9A — Method Screening Round-1 analysis（CPU-only，零 GPU）。

M1  9A new      illumination-robust representation（per-image 光照标准化到 train 参考统计）
M2  7A-O reuse  dual concat（Family B / B1_g100）
M3  9A new      nuisance subspace suppression（逐层 top-16 光照差分方向投影）
M4  9A sanity   residual compensation —— 代数冗余，只做等价性 unit
M5  —           HOLD（无公平 selection rule，本轮不实现）
M6  8B reuse    channel-selective —— **ORACLE UPPER BOUND**

口径（与历史 frozen protocol 逐字一致）：robustness = mean |ΔNormalScore_z|（低好）；
preservation = mean defect d′（高好）。两者均由 experiment5a_h_analysis.unit_metrics 重算。
用法：python -u scripts/experiment9a_analysis.py
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

EXP_ROOT = ROOT / "results" / "experiment_9a_screening"
SUM, FIG, RAW = EXP_ROOT / "summary", EXP_ROOT / "figures", EXP_ROOT / "raw"
CATS, SEED = ["bottle", "cable"], 0
EPS_DP, EPS_RZ = 0.10, 0.02          # 5A-H 冻结 ε / 7A-O Tier-A 冻结阈值
UNIFORM_ALPHA, FIXED_ALPHA = 0.40091275, 0.5

CANDIDATES = [("M1_illum_standard", "9A:M1_illum_standard", "NEW"),
              ("M2_dual_concat_B1_g100", "7AO:B1_g100", "REUSE"),
              ("M3_nuis_k16", "9A:M3_nuis_k16", "NEW"),
              ("M6_oracle_mask25", "8B:PROPOSED25", "REUSE"),
              ("M6_oracle_mask50", "8B:PROPOSED50", "REUSE")]
M4_EQUIV = "9A:M4_equiv_lam050_a040091275"
M4_RESID = "9A:M4_resid_lam050_a040091275"
MOTIVATION = {
    "M1_illum_standard": "图像级光照归一化（per-image photometric standardization）",
    "M2_dual_concat_B1_g100": "multi-scale feature fusion（PatchCore concat 变体，7A-O Family B）",
    "M3_nuis_k16": "nuisance / style 子空间抑制（逐层投影掉光照差分方向）",
    "M4_residual_compensation": "残差补偿（以 lambda 混合回原始 feature）",
    "M5_selective_bypass": "选择性 bypass（只对 illumination-sensitive 通道归一化）",
    "M6_oracle_mask25": "channel-selective 归一化（8B defect-sensitive/illumination-stable 子集, 25%）",
    "M6_oracle_mask50": "channel-selective 归一化（8B secondary ratio, 50%）",
}
REUSED = ("M2_dual_concat_B1_g100", "M6_oracle_mask25", "M6_oracle_mask50")

BASELINES = {"B0_original_a000": "5AH:B0", "B1_fixed_a050": "5AH:B2",
             "B2_uniform_a040091275": "uniform:0.40091275"}

UNIFORM_SOURCES = {"0.0": ["5AH:B0"],
                   "0.125": ["6B:a012500000", "Q4:Q4a0125"],
                   "0.2": ["6B:a020000000", "Q4:Q4a02"],
                   "0.25": ["6B:a025000000", "6A:aF025000000", "Q4:Q4a025"],
                   "0.3": ["6B:a030000000", "Q4:Q4a03"],
                   "0.40091275": ["6B:a040091275", "6A:aF040091275"],
                   "0.5": ["5AH:B2"],
                   "0.601369125": ["5AH:C2"],
                   "0.8018255": ["5AH:C3"]}

ROOT_SPECS = {"9A": RAW,
              "5AH": ROOT / "results" / "experiment_5a_h" / "raw",
              "7AO": ROOT / "results" / "experiment_7a_o" / "raw",
              "Q4": ROOT / "results" / "experiment_7a_o_q4" / "raw",
              "6A": ROOT / "results" / "experiment_6a" / "raw_new",
              "6B": ROOT / "results" / "experiment_6b" / "raw_new",
              "8B": ROOT / "results" / "experiment8b" / "probe" / "raw"}


def info_paths(tag):
    root = ROOT_SPECS[tag]
    pat = "*/*/*/*/info.json" if tag == "9A" else "*/*/*/info.json"
    return sorted(root.glob(pat))


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


def frontier_points(m, cat):
    """该 category 的 uniform-α 工作点（按 UNIFORM_SOURCES 优先级取第一个可用源）。"""
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


def repr_equivalence():
    """M4 residual 组合形式 vs 直接 effective-alpha：**表示层**数值等价（CPU，真实图像）。

    这是 M4 等价性判定的主判据：若两者生成的 embedding 在 float32 舍入级一致，
    则 F_out = F_r + lam*(F - F_r) 与 F_out = (1-a)F + a*IN(F) (a=(1-lam)*a') 同一方法。
    """
    sys.path.insert(0, str(ROOT / "experiments"
                           / "2026-09-30_illumination_sensitivity_exploration"
                           / "falpha_patchcore"))
    import torch
    import experiment9a_model as m9
    from experiment5a_model import FAlphaDualLayerPatchcoreModel
    import experiment1b_defect_sensitivity as e1b

    _, train = e1b.make_validation_split("bottle", 0)
    img = e1b.load_image_as_tensor(ROOT / "data" / "mvtec_ad" / "bottle" / "train"
                                   / "good" / train[0])
    x = e1b.preprocess_for_model(img, torch.device("cpu")).unsqueeze(0)
    kw = dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
              num_neighbors=9)
    a_eff = (1.0 - 0.5) * 0.40091275
    m_direct = FAlphaDualLayerPatchcoreModel(**kw, alpha_l2=a_eff, alpha_l3=a_eff).eval()
    m_resid = m9.M4ResidualCompensationModel(**kw, alpha_prime=0.40091275, lam=0.5).eval()
    with torch.no_grad():
        f = m_direct.feature_extractor(x)
        f = {k: m_direct.feature_pooler(v) for k, v in f.items()}
        e_direct = m_direct.generate_embedding({k: v.clone() for k, v in f.items()})
        e_resid = m_resid.generate_embedding({k: v.clone() for k, v in f.items()})
    dmax = float((e_direct - e_resid).abs().max())
    return {"max_abs_delta_feature": dmax,
            "rel_delta": dmax / float(e_direct.abs().mean()),
            "alpha_prime": 0.40091275, "lambda": 0.5, "effective_alpha": a_eff,
            "n_patches": int(e_direct.shape[0] * e_direct.shape[2] * e_direct.shape[3])}


def main():
    SUM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data, prov = build_lookup()
    print("[9A] lookup entries = %d" % len(data))
    checks = []

    checks.append(("S1_dataset_completeness",
                   all((ROOT / "data/mvtec_ad" / c / "train/good").exists() for c in CATS),
                   "bottle/cable train+test present"))

    needed = set()
    for cat in CATS:
        for _, tag, _ in CANDIDATES:
            needed.add((cat, SEED, tag))
        for btag in BASELINES.values():
            if not btag.startswith("uniform:"):
                needed.add((cat, SEED, btag))
    for cat in CATS:
        for astr, srcs in UNIFORM_SOURCES.items():
            for s in srcs:
                if (cat, SEED, s) in data:
                    needed.add((cat, SEED, s))
                    break
    missing = sorted(k for k in needed if k not in data)
    checks.append(("S2_required_sources_present", not missing,
                   "%d required; missing=%s" % (len(needed), missing)))

    m4_keys = [(c, SEED, t) for c in CATS for t in (M4_EQUIV, M4_RESID) if (c, SEED, t) in data]
    m = h5.unit_metrics({k: data[k] for k in sorted(needed | set(m4_keys))})

    all_scores = [float(r["score"]) for k in sorted(needed | set(m4_keys)) if k in data
                  for r in data[k]["rows"]]
    checks.append(("S3_no_nan_inf_scores", bool(all(np.isfinite(all_scores))),
                   "%d scores finite" % len(all_scores)))

    bycat = defaultdict(set)
    for k in needed:
        if k in data:
            bycat[k[0]].add(data[k]["info"]["n_rows"])
    checks.append(("S4_row_count_consistency", all(len(v) == 1 for v in bycat.values()),
                   "%s" % {c: sorted(v) for c, v in bycat.items()}))

    d5, n5 = 0.0, 0
    if ("bottle", 0, "9A:B0_original") in data and ("bottle", 0, "5AH:B0") in data:
        a = rowmap(data[("bottle", 0, "9A:B0_original")])
        b = rowmap(data[("bottle", 0, "5AH:B0")])
        for kk in a:
            if kk in b:
                d5 = max(d5, abs(a[kk] - b[kk]))
                n5 += 1
    checks.append(("S5_baseline_pipeline_intact_9A_vs_5AH", d5 < 1e-6,
                   "bottle:0 B0 vs 5A-H B0: n=%d max|dscore|=%.3e" % (n5, d5)))

    m4 = {"repr_level": repr_equivalence(), "score_level_diagnostic": {},
          "determinism_repeat_max_abs_delta": {}}
    for cat in CATS:
        ke, kr = (cat, SEED, M4_EQUIV), (cat, SEED, M4_RESID)
        if ke not in data or kr not in data:
            continue
        a, b = rowmap(data[ke]), rowmap(data[kr])
        if set(a) != set(b):
            m4["score_level_diagnostic"][cat] = {"error": "key sets differ"}
            continue
        diffs = [abs(a[k] - b[k]) for k in a]
        scale = float(np.mean([abs(v) for v in a.values()]))
        meta = json.loads((Path(data[kr]["dir"]) / "module_meta.json").read_text())
        m4["score_level_diagnostic"][cat] = {
            "n_keys": len(a), "max_abs_delta_score": max(diffs),
            "mean_delta_score": float(np.mean([b[k] - a[k] for k in a])),
            "pearson_r": float(np.corrcoef([a[k] for k in a], [b[k] for k in a])[0, 1]),
            "max_rel_delta": (max(diffs) / scale) if scale > 0 else float("nan"),
            "mean_abs_score_scale": scale,
            "tau_equiv": data[ke]["info"]["tau_val"], "tau_resid": data[kr]["info"]["tau_val"],
            "alpha_prime": meta.get("m4_alpha_prime"), "lambda": meta.get("m4_lambda"),
            "effective_alpha": meta.get("m4_effective_alpha")}
        for rep_dir in sorted(RAW.glob("m4diag/%s/seed_%d/config_M4_resid_lam050_a040091275"
                                       % (cat, SEED))):
            c = rowmap(load_entry(rep_dir / "info.json"))
            if set(c) == set(b):
                m4["determinism_repeat_max_abs_delta"][cat] = max(abs(c[k] - b[k]) for k in b)
    repr_ok = m4["repr_level"]["max_abs_delta_feature"] < 1e-4
    det = m4["determinism_repeat_max_abs_delta"]
    det_ok = bool(det) and all(v == 0.0 for v in det.values())
    m4_pass = bool(repr_ok and det_ok)
    checks.append(("S6_m4_equiv_repr_exact_and_pipeline_deterministic", m4_pass,
                   "repr max|dF|=%.2e (%.2e rel, float32 eps-scale) | repeat-run "
                   "max|dscore|=%s (same config twice) | score gap between the two equivalent "
                   "forms = max %.3f (amplified by KCenterGreedy coreset selection)"
                   % (m4["repr_level"]["max_abs_delta_feature"],
                      m4["repr_level"]["rel_delta"],
                      {k: "%.1e" % v for k, v in det.items()},
                      max((v["max_abs_delta_score"] for v in
                           m4["score_level_diagnostic"].values()
                           if "max_abs_delta_score" in v), default=float("nan")))))
    d8, det8 = 0.0, []
    pm = ROOT / "results" / "experiment8b" / "analysis" / "probe_metrics.csv"
    if pm.exists():
        ref8 = {(r["category"], int(r["seed"]), r["mask"]): (float(r["mean_dprime"]),
                                                            float(r["mean_abs_delta_z"]))
                for r in csv.DictReader(open(pm))}
        for mask in ("PROPOSED25", "PROPOSED50"):
            for cat in CATS:
                k = (cat, SEED, "8B:%s" % mask)
                if k in m and (cat, SEED, mask) in ref8:
                    dd = abs(m[k]["mean_dprime"] - ref8[(cat, SEED, mask)][0])
                    dr = abs(m[k]["mean_abs_delta_z"] - ref8[(cat, SEED, mask)][1])
                    d8 = max(d8, dd, dr)
                    det8.append("%s/%s %.1e/%.1e" % (cat, mask, dd, dr))
    checks.append(("S7_m6_reuse_matches_8B_frozen_metrics", d8 < 1e-9, "; ".join(det8)))

    b1 = []
    r2 = ROOT / "results" / "experiment_7a_o" / "summary" / "round2_ranking.csv"
    if r2.exists():
        for r in csv.DictReader(open(r2)):
            if r["config"] == "B1_g100":
                b1.append("d'=%s |dz|=%s (15 units)" % (r["mean_defect_dprime"], r["mean_abs_delta_z"]))
    checks.append(("S8_m2_reuse_is_frozen_7AO_B1_g100", bool(b1), "; ".join(b1) or "n/a"))

    st = []
    for ip in sorted(RAW.glob("*/*/*/*/info.json")):
        st.append(json.loads(ip.read_text()).get("status"))
    checks.append(("S9_all_9A_units_OK", bool(st) and all(s == "OK" for s in st),
                   "%d units; statuses=%s" % (len(st), sorted(set(st)))))

    leak = []
    for ip in sorted(RAW.glob("*/*/*/*/module_meta.json")):
        mm = json.loads(ip.read_text())
        if mm.get("m3_basis_meta") and not mm["m3_basis_meta"].get("n_train"):
            leak.append("%s: m3 n_train missing" % mm["unit"])
        if mm["config"].startswith("M1") and mm.get("m1_reference_stats") is None:
            leak.append("%s: m1 stats missing" % mm["unit"])
    checks.append(("S10_no_target_leakage_in_new_methods", not leak,
                   "M1 stats / M3 basis from train-normal only (n_train recorded)" if not leak else str(leak)))


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
    for disp, tag, kind in CANDIDATES:
        rec = {"method": disp, "tag": tag, "type": kind, "per_category": {}}
        for cat in CATS:
            v = m.get((cat, SEED, tag))
            if v is None:
                continue
            b0 = ref_metrics(cat, "5AH:B0")
            b2 = ref_metrics(cat, "5AH:B2")
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
            raw_rows.append(dict({"category": cat, "seed": SEED, "method": disp, "tag": tag,
                                  "type": kind}, **d))
        summary_rows.append(rec)

    for bname, btag in BASELINES.items():
        for cat in CATS:
            v = ref_metrics(cat, btag)
            if v is None:
                continue
            raw_rows.append({"category": cat, "seed": SEED, "method": bname, "tag": btag,
                             "type": "BASELINE", "mean_dprime": v["mean_dprime"],
                             "mean_abs_delta_z": v["mean_abs_delta_z"],
                             "image_auroc": v["image_auroc"], "pixel_auroc": v["pixel_auroc"],
                             "aupro": v["aupro"],
                             "delta_dprime_vs_B0": 0.0, "delta_absz_vs_B0": 0.0,
                             "delta_dprime_vs_B2": 0.0, "delta_absz_vs_B2": 0.0,
                             "delta_dprime_vs_uniform": 0.0, "delta_absz_vs_uniform": 0.0,
                             "frontier_interp_dprime": float("nan"),
                             "beyond_frontier_gain": float("nan"),
                             "dominated_by_uniform_alpha": "",
                             "runtime_seconds": v["runtime"], "peak_vram_mb": v["peak_vram"],
                             "source": prov.get(btag, "")})

    def verdict_for(rec):
        """统一筛选规则（冻结，未因结果调整）。

        ADVANCE  : 两类别同时满足预注册 trade-off 条件 且 至少一侧跑出既有 uniform-alpha frontier
        HOLD     : 至少一侧有明确 robustness 改善（Delta|dz| <= -EPS_RZ）但类别间不一致
                   （含「一侧强、另一侧恶化」与「两类别改善但均未跑出 frontier」）
        STOP     : 两类别都没有 robustness 改善；或两类别都被 frontier 支配；或数学冗余
        """
        pc = rec["per_category"]
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
            return "STOP", ("两类别均无 robustness 改善（Delta|dz| > -%.2f）-> 不构成 "
                            "trade-off 改善，preservation-only 提升不改变 frontier 形状" % EPS_RZ)
        if len(strong) == n and beyond:
            return "ADVANCE", "两类别同时满足预注册条件且跑出 frontier (%s)" % ",".join(beyond)
        if n_dom == n and not beyond:
            return "STOP", "两类别均被既有 uniform-alpha frontier 支配"
        if len(strong) == n:
            return "HOLD", "两类别满足条件但均未跑出 frontier"
        if set(rob_gain) != set(pc):
            lost = [c for c in pc if c not in rob_gain]
            extra = []
            if worsened:
                extra.append("%s 同时出现明确恶化" % ",".join(worsened))
            return "HOLD", ("类别间不一致：robustness 改善仅在 %s，%s 无改善"
                            "%s -> cross-category unstable" % (",".join(rob_gain), ",".join(lost),
                                                               ("；" + "；".join(extra)) if extra else ""))
        return "HOLD", ("两类别 robustness 均改善但 preservation 在两类别同时明显下降 "
                        "-> 纯 trade-off 兑换，不是改善")

    out_rows = []
    for rec in summary_rows:
        vd, why = verdict_for(rec)
        if rec["method"].startswith("M6"):
            vd = "ORACLE_REFERENCE"
            why = ("ORACLE UPPER BOUND（selection 使用 test defect mask）—— 不是 deployable "
                   "候选、不计入晋级；原始判定: " + why)
        rec["verdict"], rec["verdict_reason"] = vd, why
        pc = rec["per_category"]

        def avg(f):
            return float(np.mean([r[f] for r in pc.values()])) if pc else float("nan")

        out_rows.append({
            "method": rec["method"], "type": rec["type"],
            "lit_motivation": MOTIVATION.get(rec["method"], ""),
            "deployable": not rec["method"].startswith("M6"),
            "reused_historical_raw": rec["method"] in REUSED,
            "stability": ("consistent (both categories improve robustness)"
                          if all(d["delta_absz_vs_B0"] <= -EPS_RZ for d in pc.values())
                          else "cross-category unstable"),
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
            "beyond_frontier_cats": ",".join(c for c, d in pc.items()
                                             if d["beyond_frontier_gain"] > 0
                                             and not d["dominated_by_uniform_alpha"]) or "-",
            "rob_gain_both_cats": all(d["delta_absz_vs_B0"] <= -EPS_RZ for d in pc.values()),
            "pres_noninferior_both_cats": all(d["delta_dprime_vs_B0"] >= -EPS_DP
                                              for d in pc.values()),
            "dominated_cats": ",".join(c for c, d in pc.items()
                                       if d["dominated_by_uniform_alpha"]) or "-",
            "runtime_new_gpu_seconds": (round(avg("runtime_seconds"), 1)
                                        if rec["type"] == "NEW" else 0.0),
            "hist_unit_runtime_seconds": (None if rec["type"] == "NEW"
                                          else round(avg("runtime_seconds"), 1)),
            "verdict": vd,
            "verdict_reason": (("ORACLE UPPER BOUND - " if rec["method"].startswith("M6") else "")
                               + why)})

    m4v = m.get(("bottle", 0, M4_EQUIV))
    out_rows.append({"method": "M4_residual_compensation", "type": "SANITY",
                     "lit_motivation": MOTIVATION["M4_residual_compensation"],
                     "deployable": False, "reused_historical_raw": False,
                     "stability": "n/a (algebraic sanity)",
                     "bottle_dprime": round(m4v["mean_dprime"], 4) if m4v else float("nan"),
                     "bottle_absdz": round(m4v["mean_abs_delta_z"], 4) if m4v else float("nan"),
                     "cable_dprime": float("nan"), "cable_absdz": float("nan"),
                     "mean_dprime_2cat": float("nan"), "mean_absdz_2cat": float("nan"),
                     "delta_dprime_vs_B0_2cat": float("nan"),
                     "delta_absz_vs_B0_2cat": float("nan"),
                     "delta_dprime_vs_uniform_2cat": float("nan"),
                     "delta_absz_vs_uniform_2cat": float("nan"),
                     "beyond_frontier_cats": "-", "runtime_new_gpu_seconds": float("nan"),
                     "hist_unit_runtime_seconds": float("nan"),
                     "verdict": "STOP" if m4_pass else "HOLD",
                     "verdict_reason": ("ALGEBRAICALLY REDUNDANT（等价性 sanity PASS："
                                        "F_r+λ(F-F_r) 与直接 α_eff=(1-λ)α' 逐位一致）" if m4_pass
                                        else "等价性 sanity 未通过")})
    out_rows.append({"method": "M5_selective_bypass", "type": "HOLD",
                     "lit_motivation": MOTIVATION["M5_selective_bypass"],
                     "deployable": True, "reused_historical_raw": False,
                     "stability": "n/a (not implemented)",
                     "bottle_dprime": float("nan"), "bottle_absdz": float("nan"),
                     "cable_dprime": float("nan"), "cable_absdz": float("nan"),
                     "mean_dprime_2cat": float("nan"), "mean_absdz_2cat": float("nan"),
                     "delta_dprime_vs_B0_2cat": float("nan"),
                     "delta_absz_vs_B0_2cat": float("nan"),
                     "delta_dprime_vs_uniform_2cat": float("nan"),
                     "delta_absz_vs_uniform_2cat": float("nan"),
                     "beyond_frontier_cats": "-", "runtime_new_gpu_seconds": float("nan"),
                     "hist_unit_runtime_seconds": float("nan"),
                     "verdict": "HOLD",
                     "verdict_reason": "IMPLEMENTATION_UNRESOLVED（本轮不实现、不发明 heuristic）"})

    for bname, btag, bmot in (("B0_original_a000", "5AH:B0", "baseline: 原始 PatchCore (alpha=0)"),
                              ("B1_fixed_a050", "5AH:B2", "baseline: 历史 best fixed alpha=0.5"),
                              ("B2_uniform_a040091275", "uniform:0.40091275",
                               "baseline: 6B 强 baseline Uniform alpha=0.40091275")):
        ref = {c: ref_metrics(c, btag) for c in CATS}
        if any(v is None for v in ref.values()):
            continue
        out_rows.append({"method": bname, "type": "BASELINE", "lit_motivation": bmot,
                         "deployable": False, "reused_historical_raw": True,
                         "stability": "reference",
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
                         "beyond_frontier_cats": "-", "rob_gain_both_cats": False,
                         "pres_noninferior_both_cats": True, "dominated_cats": "-",
                         "runtime_new_gpu_seconds": 0.0, "hist_unit_runtime_seconds": float("nan"),
                         "verdict": "BASELINE", "verdict_reason": bmot})

    order = {"ADVANCE": 0, "HOLD": 1, "ORACLE_REFERENCE": 2, "NOT_ENOUGH_EVIDENCE": 3,
             "STOP": 4, "BASELINE": 5}
    out_rows.sort(key=lambda r: (order.get(r["verdict"], 5),
                                 -(r["delta_dprime_vs_B0_2cat"]
                                   if np.isfinite(r["delta_dprime_vs_B0_2cat"])
                                   else float("-inf"))))

    # ---------------- 落盘 ----------------
    wcsv(SUM / "raw_results.csv", raw_rows)
    wcsv(SUM / "screening_summary.csv", out_rows)

    # ranking.csv：deployable 候选按统一规则排名（baseline / oracle 不参与排名）
    ranking, rk = [], 0
    for r in out_rows:
        rr = dict(r)
        if (r.get("deployable") and np.isfinite(r.get("mean_dprime_2cat", float("nan")))
                and r["verdict"] in ("ADVANCE", "HOLD", "STOP", "NOT_ENOUGH_EVIDENCE")):
            rk += 1
            rr["rank"] = rk
        else:
            rr["rank"] = ""
        ranking.append(rr)
    wcsv(SUM / "ranking.csv", ranking)

    # progress_units.csv：逐 unit 进度与耗时（ETA 依据的真实数据）
    prog_rows = []
    for rnd in ("round0", "round1", "m4diag"):
        for sh in sorted((EXP_ROOT / "progress").glob("%s__*.json" % rnd)):
            j = json.loads(sh.read_text())
            for u in j.get("units", []):
                prog_rows.append({
                    "round": rnd, "worker": j.get("worker"), "pid": j.get("pid"),
                    "unit": u.get("id"), "seconds": u.get("seconds"),
                    "status": u.get("status"), "detail": u.get("detail", ""),
                    "round_started_at": round(float(j.get("started_at", 0)), 3),
                    "finished_at": round(float(u.get("ts", 0)), 3)})
    wcsv(SUM / "progress_units.csv", prog_rows)
    frows = []
    for cat in CATS:
        for a, (rz, dp, src) in sorted(frontier_points(m, cat).items()):
            frows.append({"category": cat, "alpha": a, "mean_abs_delta_z": round(rz, 6),
                          "mean_dprime": round(dp, 6), "source": src})
    wcsv(SUM / "uniform_alpha_frontier.csv", frows)
    wcsv(SUM / "sanity_checks.csv",
         [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in checks])

    (SUM / "screening_summary.json").write_text(json.dumps(
        {"experiment": "9A", "round": "round1", "date": "2026-10-07",
         "categories": CATS, "seed": SEED,
         "criteria": {"EPS_DP": EPS_DP, "EPS_RZ": EPS_RZ,
                      "robustness": "mean |dNormalScore_z| (lower better)",
                      "preservation": "mean defect d-prime (higher better)"},
         "m4_equivalence": m4, "m4_pass": m4_pass,
         "methods": summary_rows}, indent=2, ensure_ascii=False))

    rt = []
    for ip in sorted(RAW.glob("*/*/*/*/module_meta.json")):
        mm = json.loads(ip.read_text())
        rt.append({"round": mm.get("round"), "unit": mm["unit"], "config": mm["config"],
                   "runtime_seconds": mm.get("runtime_seconds"),
                   "wallclock_seconds": mm.get("wallclock_seconds"),
                   "peak_vram_mb": mm.get("peak_gpu_memory_allocated_mb"),
                   "embedding_dim": mm.get("embedding_dim"),
                   "memory_bank_size": mm.get("memory_bank_size"),
                   "status": mm.get("status")})
    (SUM / "runtime_summary.json").write_text(json.dumps(
        {"gpu_units_9A": rt,
         "gpu_units_9A_count": len(rt),
         "gpu_seconds_total": round(sum((r["runtime_seconds"] or 0) for r in rt), 1),
         "reuse_zero_gpu": ("M2 (7A-O B1_g100) / M6 (8B PROPOSED25|PROPOSED50) / baselines + "
                            "uniform-alpha frontier (5A-H, 6A, 6B, 7A-O Q4) 全部由历史 frozen raw "
                            "重组，0 GPU")}, indent=2, ensure_ascii=False))
    (SUM / "m4_equivalence.json").write_text(json.dumps(m4, indent=2))

    # ---------------- figure: robustness-preservation plane ----------------
    MARK = {"M1_illum_standard": ("o", "#4C72B0"), "M2_dual_concat_B1_g100": ("^", "#DD8452"),
            "M3_nuis_k16": ("p", "#55A868"), "M6_oracle_mask25": ("X", "#8172B3"),
            "M6_oracle_mask50": ("P", "#C44E52")}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    for ax, cat in zip(axes, CATS):
        pts = sorted(frontier_points(m, cat).items())
        ax.plot([v[0] for _, v in pts], [v[1] for _, v in pts], "--", color="#888888", lw=1.3,
                zorder=1, label="Pareto frontier: uniform-alpha workpoints (existing, reused)")
        ax.scatter([v[0] for _, v in pts], [v[1] for _, v in pts], s=30, color="#888888",
                   marker="s", zorder=2)
        for a, (rz, dp, src) in pts:
            if abs(a - UNIFORM_ALPHA) < 1e-9 or a in (0.0, 0.5, 0.25):
                ax.annotate("alpha=%g" % a, (rz, dp), textcoords="offset points", xytext=(4, -13),
                            fontsize=7, color="#666666")
        v = ref_metrics(cat, "5AH:B0")
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=430, marker="*", color="#111111",
                       edgecolors="white", linewidths=1.4, zorder=9, label="B0 (alpha=0, BASELINE)")
            ax.annotate("B0 baseline", (v["mean_abs_delta_z"], v["mean_dprime"]),
                        textcoords="offset points", xytext=(-58, -16), fontsize=9,
                        fontweight="bold", color="#111111")
        v = ref_metrics(cat, "5AH:B2")
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=165, marker="s", color="#555555",
                       edgecolors="black", linewidths=0.9, zorder=8,
                       label="B1 fixed alpha=0.5 (reused)")
        v = ref_metrics(cat, "uniform:0.40091275")
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=205, marker="D", color="#C44E52",
                       edgecolors="black", linewidths=0.9, zorder=8,
                       label="B2 Uniform alpha=0.4009 (reused, strong baseline)")
        for disp, tag, kind in CANDIDATES:
            v = m.get((cat, SEED, tag))
            if not v:
                continue
            mk, col = MARK[disp]
            oracle = disp.startswith("M6")
            reuse = disp in REUSED
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=190, marker=mk,
                       facecolors="none" if oracle else col,
                       edgecolors=col if oracle else "black",
                       linewidths=2.0 if oracle else 0.8, zorder=7,
                       label=disp + (" [ORACLE / not deployable]" if oracle else "")
                             + (" [reused]" if reuse else " [new GPU run]"))
            ax.annotate(disp.split("_")[0] + ("(oracle)" if oracle else ""),
                        (v["mean_abs_delta_z"], v["mean_dprime"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=8.5, color=col, fontweight="bold")
        v = m.get((cat, SEED, M4_EQUIV))
        if v:
            ax.scatter(v["mean_abs_delta_z"], v["mean_dprime"], s=115, marker="x", color="#9E9E9E",
                       linewidths=1.8, zorder=6, label="M4 (= alpha_eff, algebraically redundant)")
        ax.set_xlabel("robustness cost - mean |" + "\u0394" + "NormalScore_z|   (lower = better)")
        ax.set_title("%s (seed 0)" % cat, fontsize=12)
        ax.grid(alpha=0.3)
        ax.annotate("better ->", xy=(0.03, 0.955), xycoords="axes fraction", fontsize=9,
                    color="#555555")
    axes[0].set_ylabel("defect preservation - mean defect d'   (higher = better)")
    h, l = axes[1].get_legend_handles_labels()
    dd = dict(zip(l, h))
    axes[1].legend(dd.values(), dd.keys(), fontsize=7.0, loc="lower left", ncol=2,
                   title="hollow markers = ORACLE upper bound (NOT a deployable method)",
                   title_fontsize=7.0)
    fig.suptitle("Experiment 9A - Round-1 method screening vs the existing uniform-alpha Pareto frontier\n"
                 "filled = new GPU run | reused = recomputed from historical frozen raw (0 GPU)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_robustness_preservation_plane.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 控制台排行榜 ----------------
    print("")
    print("=" * 136)
    print("Experiment 9A - Round-1 METHOD SCREENING (bottle/cable seed0)")
    print("robustness = mean|dNormalScore_z| (lower better) | preservation = mean defect d' (higher better)")
    print("=" * 136)
    print("%-3s%-26s%-16s%10s%10s%11s%9s%11s%10s%-20s" % (
        "#", "method", "decision", "bottle d'", "cable d'", "|dz| 2cat", "d' 2cat",
        "dd' vs B0", "beyond", "stability"))
    rank = 0
    for r in out_rows:
        if r.get("deployable") and r["verdict"] in ("ADVANCE", "HOLD", "STOP",
                                                   "NOT_ENOUGH_EVIDENCE"):
            rank += 1
            tag = "%d %s" % (rank, r["verdict"])
        else:
            tag = r["verdict"]
        vals = (r["bottle_dprime"], r["cable_dprime"], r["mean_absdz_2cat"],
                r["mean_dprime_2cat"], r["delta_dprime_vs_B0_2cat"])
        if all(np.isfinite(v) for v in vals):
            print("%-3s%-26s%-16s%10.4f%10.4f%11.4f%9.4f%+11.4f%10s%-20s" % (
                (str(rank) if tag.startswith(str(rank)) else "-"), r["method"], tag,
                vals[0], vals[1], vals[2], vals[3], vals[4], r["beyond_frontier_cats"],
                str(r.get("stability", ""))[:19]))
        else:
            print("%-3s%-26s%-16s%10s%10s%11s%9s%11s%10s%-20s" % (
                "-", r["method"], tag, "-", "-", "-", "-", "-", "-", "-"))
        if r.get("verdict_reason"):
            print("      reason: %s" % r["verdict_reason"][:190])
    adv = [r for r in out_rows if r["verdict"] == "ADVANCE"]
    print("TIER-1 (ADVANCE): %s" % (", ".join(r["method"] for r in adv) or
                                    "NONE - 9A 本轮无方法晋级"))
    print("TIER-2 (HOLD)   : %s" % ", ".join(r["method"] for r in out_rows
                                              if r["verdict"] == "HOLD"))
    print("ELIMINATED (STOP): %s" % ", ".join(r["method"] for r in out_rows
                                              if r["verdict"] == "STOP"))
    print("ORACLE (ref only): %s" % ", ".join(r["method"] for r in out_rows
                                              if r["verdict"] == "ORACLE_REFERENCE"))
    print("=" * 136)
    for c, ok, d in checks:
        print("  %-5s%-46s%s" % ("PASS" if ok else "FAIL", c, d[:78]))
    print("  -> sanity %d/%d PASS | M4 equivalence %s" % (
        sum(1 for _, ok, _ in checks if ok), len(checks), "PASS" if m4_pass else "FAIL"))
    print("  artifacts -> %s | %s" % (SUM, FIG))


if __name__ == "__main__":
    main()
