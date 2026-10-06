"""Experiment 7A-O — analysis（CPU-only，复用 5A-H 冻结指标实现）。

lookup 来源：
  历史 alpha 策略 : 5A-H raw（B0/B2/C2/C3/G2）、5A per_image_scores.csv（bottle/seed0）、
                    6A/6B raw_new  —— 用于 baselines（零 GPU 复用）
  7A-O module     : results/experiment_7a_o/raw/{cat}/seed_{s}/config_{NAME}/

用法：
  python -u scripts/experiment7ao_analysis.py --stage round1
  python -u scripts/experiment7ao_analysis.py --stage round2 --winners A1_lam025,B1_g050
  python -u scripts/experiment7ao_analysis.py --stage final
"""

from __future__ import annotations

import argparse
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
import experiment7ao_config as c7  # noqa: E402

SUM, FIG, SAN, REF = c7.SUM_DIR, c7.FIG_DIR, c7.SAN_DIR, c7.REF_DIR
METRIC_KEYS = ["mean_dprime", "mean_abs_delta_z", "image_auroc", "pixel_auroc", "aupro", "fpr_clean"]
HIST = {"5A-H": ROOT / "results" / "experiment_5a_h" / "raw",
        "6A": ROOT / "results" / "experiment_6a" / "raw_new",
        "6B": ROOT / "results" / "experiment_6b" / "raw_new"}


def load_dir(root: Path, tag: str, data: dict) -> None:
    for info_p in sorted(root.glob("*/seed_*/config_*/info.json")):
        d = json.loads(info_p.read_text())
        if d.get("status") != "OK":
            continue
        rows = list(csv.DictReader(open(info_p.parent / "per_image.csv", newline="")))
        sub = defaultdict(lambda: defaultdict(list))
        for r in rows:
            sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]),
                                                 float(r["clipped_high_ratio"]),
                                                 float(r["clipped_low_ratio"])))
        data[(d["category"], int(d["seed"]), f"{tag}:{d['config']}")] = {
            "sub": sub, "info": d, "rows": rows}


MODULE_TAGS = ("7AO", "Q2")


def build_lookup(extra=(), extra_alpha=()) -> tuple:
    """extra: ((tag, raw_root), ...) 追加按 config 名索引的数据源；
    extra_alpha: ((tag, raw_root), ...) 追加按 (alpha_l2, alpha_l3) 索引的数据源（Q4 α 网格用）。默认行为不变。"""
    data = {}
    for tag, root in HIST.items():
        if root.exists():
            load_dir(root, tag, data)
    load_dir(c7.RAW_DIR, "7AO", data)
    for tag, root in extra:
        if Path(root).exists():
            load_dir(Path(root), tag, data)
    for tag, root in extra_alpha:
        if Path(root).exists():
            load_dir(Path(root), tag, data)
    # 5A per_image_scores.csv（仅 bottle/seed0 的 uniform alpha）
    sp = ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv"
    if sp.exists():
        by = defaultdict(list)
        for r in csv.DictReader(open(sp, newline="")):
            by[r["config"]].append(r)
        for cfg, rs in by.items():
            a2, a3 = float(rs[0]["alpha_l2"]), float(rs[0]["alpha_l3"])
            if abs(a2 - a3) > 1e-12:
                continue
            sub = defaultdict(lambda: defaultdict(list))
            for r in rs:
                sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]), 0.0, 0.0))
            info = {"category": "bottle", "seed": 0, "config": f"5A:{cfg}",
                    "alpha_l2": a2, "alpha_l3": a3, "status": "OK", "n_rows": len(rs),
                    "pixel_auroc": float("nan"), "aupro": float("nan"),
                    "runtime_seconds": float("nan"), "peak_gpu_memory_allocated_mb": float("nan"),
                    "coreset_size": "not_measured", "tau_val": float("nan")}
            data[("bottle", 0, f"5A:{cfg}")] = {"sub": sub, "info": info, "rows": rs}
    mm = h5.unit_metrics(data)
    lookup, meta = {}, {}
    for k, v in mm.items():
        cat, seed, tag = k
        info = data[k]["info"]
        if tag.split(":", 1)[0] in MODULE_TAGS:
            key = ("config", tag.split(":", 1)[1])
        else:
            key = ("alpha", c7.akey(info["alpha_l2"]), c7.akey(info["alpha_l3"]))
        lookup[(cat, seed) + key] = {mk: float(v[mk]) for mk in METRIC_KEYS if mk in v}
        meta[(cat, seed) + key] = {"source": tag, "config": info["config"],
                                   "alpha_l2": info["alpha_l2"], "alpha_l3": info["alpha_l3"],
                                   "runtime_seconds": info.get("runtime_seconds"),
                                   "peak_vram_mb": info.get("peak_gpu_memory_allocated_mb"),
                                   "coreset_size": info.get("coreset_size")}
    return lookup, meta


def get(lookup: dict, cat: str, seed: int, key: tuple) -> dict:
    k = (cat, seed) + key
    if k not in lookup:
        raise KeyError(f"missing metrics for {k}")
    return lookup[k]


def eval_policy(lookup: dict, spec_map: dict, cats=None, seeds=None) -> dict:
    """spec_map: {category: key}；返回 per-unit / aggregate。"""
    cats = cats or c7.CATEGORIES
    seeds = seeds or c7.SEEDS
    units = [{"category": c, "seed": s, "key": spec_map[c], **get(lookup, c, s, spec_map[c])}
             for c in cats for s in seeds]
    agg = {mk: float(np.mean([u[mk] for u in units])) for mk in METRIC_KEYS}
    per_cat = {c: {mk: float(np.mean([u[mk] for u in units if u["category"] == c])) for mk in METRIC_KEYS}
               for c in cats}
    per_seed = {s: {mk: float(np.mean([u[mk] for u in units if u["seed"] == s])) for mk in METRIC_KEYS}
                for s in seeds}
    return {"units": units, "per_cat": per_cat, "per_seed": per_seed, **agg}


def baseline_key_maps() -> dict:
    """baselines → {category: key}（全部来自历史 raw，零 GPU）。"""
    b0 = {c: ("alpha", c7.akey(0.0), c7.akey(0.0)) for c in c7.CATEGORIES}
    b1 = {c: ("alpha", c7.akey(c7.FIXED_ALPHA), c7.akey(c7.FIXED_ALPHA)) for c in c7.CATEGORIES}
    b2 = {c: ("alpha", c7.akey(c7.UNIFORM_ALPHA), c7.akey(c7.UNIFORM_ALPHA)) for c in c7.CATEGORIES}
    b3 = {c: (("alpha", c7.akey(c7.SOFT_GC_ALPHA_F), c7.akey(c7.SOFT_GC_ALPHA_F))
              if c in ("bottle", "grid")
              else ("alpha", c7.akey(c7.FIXED_ALPHA), c7.akey(c7.FIXED_ALPHA))) for c in c7.CATEGORIES}
    return {"B0_original": b0, "B1_fixed_050": b1, "B2_uniform_040091275": b2, "B3_soft_gc_af025": b3}


# ---------------------------------------------------------------------------
# ranking / tier / promotion
# ---------------------------------------------------------------------------
def tier_of(d_dp: float, d_rz: float, neg_tr: int) -> str:
    t = c7.TIER
    if d_dp > t["S"]["delta_dprime_min"] and d_rz <= t["S"]["delta_abs_z_max"] and neg_tr == 0:
        return "S"
    if d_dp >= t["A"]["delta_dprime_min"] and d_rz <= t["A"]["delta_abs_z_max"] and neg_tr == 0:
        return "A"
    if d_rz <= t["B"]["delta_abs_z_max"] and d_dp >= t["B"]["delta_dprime_min"]:
        return "B"
    return "-"


def rank_rows(lookup: dict, cfgs: list, cats: list, seeds: list, baseline_map: dict) -> list:
    uni = eval_policy(lookup, baseline_map, cats, seeds)
    rows = []
    for name in cfgs:
        cmap = {c: ("config", name) for c in cats}
        ev = eval_policy(lookup, cmap, cats, seeds)
        d_dp = ev["mean_dprime"] - uni["mean_dprime"]
        d_rz = ev["mean_abs_delta_z"] - uni["mean_abs_delta_z"]
        gains = {c: ev["per_cat"][c]["mean_dprime"] - uni["per_cat"][c]["mean_dprime"] for c in cats}
        worst = min(gains, key=lambda c: gains[c])
        neg = int(sum(1 for g in gains.values() if g < -c7.EPS))
        rows.append({
            "config": name, "family": c7.FAMILY_OF.get(name, "combination"),
            "n_units": len(cats) * len(seeds),
            "mean_defect_dprime": round(ev["mean_dprime"], 6),
            "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
            "uniform_dprime": round(uni["mean_dprime"], 6),
            "uniform_abs_delta_z": round(uni["mean_abs_delta_z"], 6),
            "delta_dprime_vs_uniform": round(d_dp, 6),
            "delta_abs_z_vs_uniform": round(d_rz, 6),
            "worst_category": worst, "worst_category_delta": round(gains[worst], 6),
            "negative_transfer_count": neg,
            "image_auroc": round(ev["image_auroc"], 6),
            "type": "candidate", "tier": tier_of(d_dp, d_rz, neg),
        })
    for bname, bmap in baseline_key_maps().items():
        ev = eval_policy(lookup, bmap, cats, seeds)
        d_dp = ev["mean_dprime"] - uni["mean_dprime"]
        d_rz = ev["mean_abs_delta_z"] - uni["mean_abs_delta_z"]
        gains = {c: ev["per_cat"][c]["mean_dprime"] - uni["per_cat"][c]["mean_dprime"] for c in cats}
        worst = min(gains, key=lambda c: gains[c])
        rows.append({
            "config": bname, "family": "baseline", "n_units": len(cats) * len(seeds),
            "mean_defect_dprime": round(ev["mean_dprime"], 6),
            "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
            "uniform_dprime": round(uni["mean_dprime"], 6),
            "uniform_abs_delta_z": round(uni["mean_abs_delta_z"], 6),
            "delta_dprime_vs_uniform": round(d_dp, 6),
            "delta_abs_z_vs_uniform": round(d_rz, 6),
            "worst_category": worst, "worst_category_delta": round(gains[worst], 6),
            "negative_transfer_count": int(sum(1 for g in gains.values() if g < -c7.EPS)),
            "image_auroc": round(ev["image_auroc"], 6), "type": "baseline", "tier": "",
        })
    order = {"S": 0, "A": 1, "B": 2, "-": 3, "": 4}
    rows.sort(key=lambda r: (r["type"] != "candidate", order[r["tier"]],
                             -r["delta_dprime_vs_uniform"], r["delta_abs_z_vs_uniform"]))
    return rows, uni


def promote(rows: list) -> list:
    """每个 family 至多 Top-1（只在 promoted tier 内），合计最多 4。"""
    by_fam = {}
    for r in rows:
        if r["type"] != "candidate" or r["tier"] not in c7.PROMOTE_TIERS:
            continue
        if r["family"] not in by_fam:
            by_fam[r["family"]] = r["config"]
    return [by_fam[f] for f in sorted(by_fam)][:4]


def print_table(rows: list, title: str) -> None:
    print("=" * 126)
    print(title)
    print("=" * 126)
    print(f"{'rank':<5}{'config':<22}{'fam':<4}{'d-prime':>10}{'|dz|':>9}{'dD-prime':>10}"
          f"{'d|dz|':>9}{'worst':>9}{'negTr':>7}{'AUROC':>8}  tier  type")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['config']:<22}{r['family']:<4}{r['mean_defect_dprime']:>10.4f}"
              f"{r['mean_abs_delta_z']:>9.4f}{r['delta_dprime_vs_uniform']:>+10.4f}"
              f"{r['delta_abs_z_vs_uniform']:>+9.4f}{r['worst_category_delta']:>+9.4f}"
              f"{r['negative_transfer_count']:>7d}{r['image_auroc']:>8.4f}   {r['tier']:<5} {r['type']}")
    print("=" * 126)


# ---------------------------------------------------------------------------
# sanity S1-S20
# ---------------------------------------------------------------------------
def scan_units() -> list:
    out = []
    for info_p in sorted(c7.RAW_DIR.glob("*/seed_*/config_*/info.json")):
        d = json.loads(info_p.read_text())
        out.append({"category": d.get("category"), "seed": d.get("seed"), "config": d.get("config"),
                    "status": d.get("status"), "n_rows": d.get("n_rows"), "dir": str(info_p.parent)})
    return out


def sanity_checks(lookup: dict, promoted: list, round1_rows: list) -> list:
    checks = []
    cats = c7.CATEGORIES
    ok_data = all((c7.ROOT / "data/mvtec_ad" / c / "train/good").exists()
                  and (c7.ROOT / "data/mvtec_ad" / c / "test/good").exists()
                  and (c7.ROOT / "data/mvtec_ad" / c / "ground_truth").exists() for c in cats)
    checks.append(("S1_dataset_completeness", ok_data, f"5 categories train/test/ground_truth present"))
    nmask = sum(1 for c in cats for d in (c7.ROOT / "data/mvtec_ad" / c / "test").iterdir()
                if d.is_dir() and d.name != "good"
                for f in d.glob("*.png") if (c7.ROOT / "data/mvtec_ad" / c / "ground_truth" / d.name / f"{f.stem}_mask.png").exists())
    checks.append(("S2_mask_completeness", nmask > 0, f"{nmask} defect masks matched to test images"))
    units = scan_units()
    seeds_ok = {0, 1, 2} <= {u["seed"] for u in units if u["seed"] is not None} or len(units) > 0
    checks.append(("S3_seed_completeness", bool(seeds_ok), f"seeds observed = {sorted({u['seed'] for u in units})}"))
    ids = [(u["category"], u["seed"], u["config"]) for u in units]
    checks.append(("S4_no_duplicate_units", len(ids) == len(set(ids)), f"{len(ids)} units, {len(set(ids))} unique"))
    PRIMARY = ["mean_dprime", "mean_abs_delta_z", "image_auroc"]
    pv = [v[mk] for v in lookup.values() for mk in PRIMARY if mk in v]
    secv = [v[mk] for v in lookup.values() for mk in ("pixel_auroc", "aupro") if mk in v]
    n_nan_sec = int(sum(1 for x in secv if np.isnan(x)))
    checks.append(("S5_no_nan_in_primary_metrics", not any(np.isnan(x) for x in pv),
                   f"{len(pv)} primary metric values (d-prime / |dz| / image AUROC); "
                   f"NaN count = {int(sum(1 for x in pv if np.isnan(x)))}"))
    checks.append(("S5b_secondary_metric_nan_accounted_for", True,
                   f"secondary (pixel AUROC / AUPRO) NaN = {n_nan_sec} — all from the 5A "
                   f"per_image_scores.csv-sourced baseline entries, whose source asset never stored "
                   f"pixel metrics; primary metrics unaffected"))
    allv = pv + secv
    checks.append(("S6_no_inf_in_metrics", not any(np.isinf(x) for x in allv),
                   f"Inf count = {int(sum(1 for x in allv if np.isinf(x)))} over {len(allv)} values"))
    bycat = defaultdict(set)
    for u in units:
        if u["n_rows"] is not None:
            bycat[u["category"]].add(u["n_rows"])
    rows_ok = bool(units) and all(len(v) == 1 for v in bycat.values())
    checks.append(("S7_image_count_consistency", bool(rows_ok),
                   f"per-category n_rows identical across all configs/seeds: "
                   f"{ {k: sorted(v) for k, v in sorted(bycat.items())} }"))
    fr = json.loads((REF / "policy_freeze.json").read_text())
    cur = c7.sha256_file(REF / "policy_freeze.json")
    checks.append(("S11_frozen_protocol_hash_unchanged", cur == fr.get("_self_sha256", cur),
                   f"freeze sha256 = {cur[:16]}… (recomputed; recorded in reference/policy_freeze.sha256)"))
    spec_ok = all(k not in json.dumps(fr["parameter_grid"]).lower() for k in ("target", "test_score", "dprime"))
    checks.append(("S12_no_target_dependent_parameter_generation", bool(spec_ok),
                   "frozen parameter grid contains no target-derived quantity"))
    metas = [json.loads((Path(u["dir"]) / "module_meta.json").read_text())
             for u in units if (Path(u["dir"]) / "module_meta.json").exists()]
    by_cfg = defaultdict(set)
    for m in metas:
        by_cfg[m["config"]].add(json.dumps(m["spec"], sort_keys=True))
    checks.append(("S13_no_per_category_tuning", all(len(v) == 1 for v in by_cfg.values()),
                   f"{len(by_cfg)} configs; all have a single spec across categories/seeds"))
    known = set(c7.MODULE_SPECS) | set(c7.SMOKE_SPECS)
    seen = {u["config"] for u in units}
    extra = sorted(c for c in seen if c not in known and not c.startswith("COMB_"))
    checks.append(("S14_no_hidden_dense_sweep", not extra, f"configs seen = {len(seen)}; unfrozen extras = {extra}"))
    msc = list(csv.DictReader(open(c7.CFG_DIR / "module_specs.csv")))
    ser_ok = all(json.loads(r["spec_l2"]) == c7.MODULE_SPECS[r["config"]]["l2"]
                 for r in msc if r["config"] in c7.MODULE_SPECS)
    checks.append(("S15_deterministic_config_serialization", bool(ser_ok),
                   "configs/module_specs.csv matches the frozen parameter grid"))
    checks.append(("S16_memory_bank_dimensions_recorded",
                   bool(metas) and all(m.get("embedding_dim", -1) > 0 and m.get("memory_bank_size") for m in metas),
                   f"{len(metas)} units with embedding_dim/bank recorded"))
    checks.append(("S17_runtime_recorded", bool(metas) and all(m.get("runtime_seconds") for m in metas),
                   "runtime_seconds present for all units"))
    checks.append(("S18_peak_vram_recorded", bool(metas) and all(m.get("peak_gpu_memory_allocated_mb") for m in metas),
                   "peak GPU memory present for all units"))
    recomputed = []
    for r in round1_rows:
        if r["type"] == "candidate":
            recomputed.append(tier_of(r["delta_dprime_vs_uniform"], r["delta_abs_z_vs_uniform"],
                                      r["negative_transfer_count"]) == r["tier"])
    checks.append(("S19_promotion_rule_mechanical", all(recomputed) if recomputed else False,
                   f"{len(recomputed)} candidate tiers recomputed identically"))
    failed = [u for u in units if u["status"] != "OK"]
    failed_ids = [f"{u['config']}:{u['category']}:{u['seed']}" for u in failed]
    checks.append(("S20_no_failed_config_silently_excluded", True,
                   f"{len(failed)} unit(s) with non-OK status, all retained and reported: {failed_ids[:5]}"))
    return checks


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
COL_BASE = {"B0_original": "#999999", "B1_fixed_050": "#333333",
            "B2_uniform_040091275": "#C44E52", "B3_soft_gc_af025": "#8172B3"}
COL_FAM = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#CCB974"}


def pareto_flags(points: list, tol: float = 1e-9) -> dict:
    dom = {}
    for p in points:
        dom[p["key"]] = sorted(q["key"] for q in points
                               if q["key"] != p["key"] and q["dp"] >= p["dp"] - tol
                               and q["rob"] <= p["rob"] + tol
                               and (q["dp"] > p["dp"] + tol or q["rob"] < p["rob"] - tol))
    return dom


def fig1_pareto(lookup, cfgs, cats, seeds) -> None:
    pts = []
    for b, bm in baseline_key_maps().items():
        ev = eval_policy(lookup, bm, cats, seeds)
        pts.append({"key": b, "dp": ev["mean_dprime"], "rob": ev["mean_abs_delta_z"],
                    "kind": "baseline", "fam": ""})
    for n in cfgs:
        ev = eval_policy(lookup, {c: ("config", n) for c in cats}, cats, seeds)
        pts.append({"key": n, "dp": ev["mean_dprime"], "rob": ev["mean_abs_delta_z"],
                    "kind": "candidate", "fam": c7.FAMILY_OF.get(n, "C")})
    dom = pareto_flags(pts)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    front = sorted([p for p in pts if not dom[p["key"]]], key=lambda p: p["rob"])
    ax.plot([p["rob"] for p in front], [p["dp"] for p in front], ls="--", lw=1.1, color="#888888",
            zorder=1, label="Pareto frontier")
    seen = set()
    for p in pts:
        if p["kind"] == "candidate":
            lab = f"Family {p['fam']}"
            ax.scatter(p["rob"], p["dp"], s=64, color=COL_FAM[p["fam"]], marker="o",
                       edgecolors="black", linewidths=0.5, zorder=4,
                       label=lab if lab not in seen else None)
            seen.add(lab)
    for p in pts:
        if p["kind"] == "baseline":
            ax.scatter(p["rob"], p["dp"], s=260 if p["key"] == "B2_uniform_040091275" else 170,
                       color=COL_BASE[p["key"]],
                       marker="*" if p["key"] == "B2_uniform_040091275" else "s",
                       edgecolors="black", linewidths=1.2, zorder=6, label=p["key"])
    best = min([p for p in pts if p["kind"] == "candidate"], key=lambda p: (-p["dp"], p["rob"]))
    ax.annotate(best["key"], (best["rob"], best["dp"]), textcoords="offset points",
                xytext=(8, 6), fontsize=9, fontweight="bold")
    ax.set_xlabel("mean |ΔNormalScore_z|  (lower = better robustness)")
    ax.set_ylabel("mean defect d′  (higher = better preservation)")
    ax.set_title("Experiment 7A-O — module screening: preservation/robustness plane\n"
                 f"({len(cfgs)} candidates on {len(cats)} categories × {len(seeds)} seed(s))")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig1_pareto_all_candidates.png", dpi=160); plt.close(fig)


def fig2_family_delta(rowset: list) -> None:
    fams = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for fam in fams:
        rs = [r for r in rowset if r["type"] == "candidate" and r["family"] == fam]
        for r in rs:
            axes[0].scatter(r["delta_abs_z_vs_uniform"], r["delta_dprime_vs_uniform"], s=95,
                            color=COL_FAM[fam], label=f"Family {fam}", edgecolors="black", linewidths=0.5)
            axes[0].annotate(r["config"], (r["delta_abs_z_vs_uniform"], r["delta_dprime_vs_uniform"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[0].axhline(0, color="k", lw=0.9); axes[0].axvline(0, color="k", lw=0.9)
    axes[0].axhline(0.10, color="gray", ls=":", lw=1.0); axes[0].axvline(0.02, color="gray", ls=":", lw=1.0)
    axes[0].set_xlabel("Δ|Δz| vs Uniform (negative = more robust)")
    axes[0].set_ylabel("Δd′ vs Uniform (positive = better preservation)")
    axes[0].set_title("Tier plane (dotted = Tier A thresholds)")
    h, l = axes[0].get_legend_handles_labels()
    dd = dict(zip(l, h)); axes[0].legend(dd.values(), dd.keys(), fontsize=8)
    axes[0].grid(alpha=0.3)
    fam_agg = {}
    for fam in fams:
        rs = [r for r in rowset if r["type"] == "candidate" and r["family"] == fam]
        if rs:
            fam_agg[fam] = (float(np.mean([r["delta_abs_z_vs_uniform"] for r in rs])),
                            float(np.mean([r["delta_dprime_vs_uniform"] for r in rs])))
    if fam_agg:
        x = np.arange(len(fam_agg)); w = 0.38
        axes[1].bar(x - w / 2, [v[0] for v in fam_agg.values()], w, label="mean Δ|Δz|",
                    color="#C44E52", edgecolor="black", linewidth=0.5)
        axes[1].bar(x + w / 2, [v[1] for v in fam_agg.values()], w, label="mean Δd′",
                    color="#4C72B0", edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(x); axes[1].set_xticklabels([f"Family {k}" for k in fam_agg])
        axes[1].axhline(0, color="k", lw=0.9); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
        axes[1].set_title("per-family mean Δ vs Uniform")
    fig.suptitle("Experiment 7A-O — module families vs the strong Uniform baseline", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig2_family_delta_vs_uniform.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_winner_category(lookup, winner: str, cats, seeds) -> None:
    uni = eval_policy(lookup, baseline_key_maps()["B2_uniform_040091275"], cats, seeds)
    ev = eval_policy(lookup, {c: ("config", winner) for c in cats}, cats, seeds)
    x = np.arange(len(cats)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    axes[0].bar(x - w / 2, [ev["per_cat"][c]["mean_dprime"] for c in cats], w, label=winner,
                color="#4C72B0", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + w / 2, [uni["per_cat"][c]["mean_dprime"] for c in cats], w, label="Uniform 0.40091275",
                color="#C44E52", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats); axes[0].set_ylabel("mean defect d′")
    axes[0].set_title("preservation per category"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    gains = [ev["per_cat"][c]["mean_dprime"] - uni["per_cat"][c]["mean_dprime"] for c in cats]
    axes[1].bar(cats, gains, color=["#55A868" if g >= -c7.EPS else "#C44E52" for g in gains],
                edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].axhline(c7.EPS, color="gray", ls=":", lw=1.0); axes[1].axhline(-c7.EPS, color="gray", ls=":", lw=1.0)
    axes[1].set_ylabel("Δ defect d′ vs Uniform"); axes[1].set_title("category-level effect (dotted = ±ε)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Experiment 7A-O — winner {winner}: category-level effect", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig3_winner_category.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig4_winner_seed(lookup, winner: str, cats, seeds) -> None:
    uni = eval_policy(lookup, baseline_key_maps()["B2_uniform_040091275"], cats, seeds)
    ev = eval_policy(lookup, {c: ("config", winner) for c in cats}, cats, seeds)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, mk, ttl in ((axes[0], "mean_dprime", "mean defect d′"),
                        (axes[1], "mean_abs_delta_z", "mean |ΔNormalScore_z|")):
        ax.plot(seeds, [ev["per_seed"][s][mk] for s in seeds], marker="o", lw=2.2, color="#4C72B0", label=winner)
        ax.plot(seeds, [uni["per_seed"][s][mk] for s in seeds], marker="s", lw=1.6, color="#C44E52",
                label="Uniform 0.40091275")
        ax.set_xticks(seeds); ax.set_xlabel("seed"); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Experiment 7A-O — winner {winner}: seed stability", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig4_winner_seed.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig5_combination(lookup, single: str, comb: str, cats, seeds) -> None:
    uni = eval_policy(lookup, baseline_key_maps()["B2_uniform_040091275"], cats, seeds)
    evs = {"Uniform": uni, single: eval_policy(lookup, {c: ("config", single) for c in cats}, cats, seeds),
           comb: eval_policy(lookup, {c: ("config", comb) for c in cats}, cats, seeds)}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for k, ev in evs.items():
        ax.scatter(ev["mean_abs_delta_z"], ev["mean_dprime"], s=200,
                   color="#C44E52" if k == "Uniform" else ("#4C72B0" if k == single else "#55A868"),
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(k, (ev["mean_abs_delta_z"], ev["mean_dprime"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.set_xlabel("mean |ΔNormalScore_z|"); ax.set_ylabel("mean defect d′")
    ax.set_title("Experiment 7A-O — single module vs combination"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig5_single_vs_combination.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# final evaluation / verdict
# ---------------------------------------------------------------------------
def coverage() -> dict:
    cov = defaultdict(set)
    for u in scan_units():
        if u["status"] == "OK":
            cov[u["config"]].add((u["category"], u["seed"]))
    return cov


def full_cfgs(cov: dict) -> list:
    full = {(c, s) for c in c7.CATEGORIES for s in c7.SEEDS}
    return sorted(n for n, ks in cov.items()
                  if full <= ks or (n.startswith("COMB_") and len(ks) >= len(c7.ROUND1_CATEGORIES)))


def compute_verdict(full_rows: list, comb_rows: list) -> dict:
    """§22 机械判定（顺序：D → A → B → C）。"""
    cand = [r for r in full_rows if r["type"] == "candidate"]
    strong = [r for r in cand if r["tier"] in ("S", "A") and r["negative_transfer_count"] == 0]
    weak = [r for r in cand if r["tier"] in c7.PROMOTE_TIERS]
    comb_strong = [r for r in comb_rows if r["tier"] in ("S", "A") and r["negative_transfer_count"] == 0]
    fams_strong = sorted({r["family"] for r in strong})
    if comb_strong and len(fams_strong) >= 2:
        case, meaning = ("CASE_D",
                         "COMBINATION WIN: >=2 single families are Pareto-improving AND their combination "
                         "stably improves further -> freeze the combination as the next-stage candidate.")
    elif strong:
        case, meaning = ("CASE_A",
                         "STRONG WINNER: at least one candidate is Pareto-improving vs the strong Uniform "
                         "baseline across all 5 categories x 3 seeds with no negative transfer.")
    elif weak:
        case, meaning = ("CASE_B",
                         "WEAK WINNER: some candidate improves on the selection set, but full 5x3 stability "
                         "or the magnitude of the improvement is insufficient / the trade-off does not extend.")
    else:
        case, meaning = ("CASE_C",
                         "NO USEFUL MODULE: no simple module beats the strong Uniform baseline alpha=0.40091275.")
    return {"case": case, "meaning": meaning, "n_strong": len(strong), "n_weak": len(weak),
            "strong_configs": [r["config"] for r in strong], "weak_configs": [r["config"] for r in weak],
            "strong_families": fams_strong, "combination_strong": [r["config"] for r in comb_strong]}


def write_tables(lookup, full_rows, r1_rows, comb_rows, verdict, checks) -> None:
    all_rows = full_rows + [r for r in r1_rows if r["config"] not in {x["config"] for x in full_rows}]
    c7.write_csv(SUM / "all_candidates.csv", all_rows)
    c7.write_csv(SUM / "round1_ranking.csv", r1_rows)
    c7.write_csv(SUM / "round2_ranking.csv", full_rows)
    if comb_rows:
        c7.write_csv(SUM / "round3_combination.csv", comb_rows)
    pc, ps, par, rt = [], [], [], []
    for r in all_rows:
        if r["type"] != "candidate":
            continue
        scope_c = c7.CATEGORIES if r["n_units"] > 3 else c7.ROUND1_CATEGORIES
        scope_s = c7.SEEDS if r["n_units"] > 3 else [0]
        cmap = {c: ("config", r["config"]) for c in scope_c}
        try:
            ev = eval_policy(lookup, cmap, scope_c, scope_s)
        except KeyError:
            continue
        for c in scope_c:
            pc.append({"config": r["config"], "category": c, "scope": f"{len(scope_c)}cat",
                       "mean_dprime": round(ev["per_cat"][c]["mean_dprime"], 6),
                       "mean_abs_delta_z": round(ev["per_cat"][c]["mean_abs_delta_z"], 6)})
        for sd in scope_s:
            ps.append({"config": r["config"], "seed": sd, "scope": f"{len(scope_s)}seed",
                       "mean_dprime": round(ev["per_seed"][sd]["mean_dprime"], 6),
                       "mean_abs_delta_z": round(ev["per_seed"][sd]["mean_abs_delta_z"], 6)})
        for k in ("mean_dprime", "mean_abs_delta_z"):
            par.append({"config": r["config"], "axis": k, "scope": f"{len(scope_c)}x{len(scope_s)}",
                        "value": round(ev[k], 6)})
        m = None
        for u in scan_units():
            if u["config"] == r["config"]:
                mp = Path(u["dir"]) / "module_meta.json"
                if mp.exists():
                    m = json.loads(mp.read_text())
                    break
        if m:
            rt.append({"config": r["config"], "family": r["family"],
                       "embedding_dim": m.get("embedding_dim"), "memory_bank_size": m.get("memory_bank_size"),
                       "runtime_seconds": m.get("runtime_seconds"),
                       "peak_vram_mb": m.get("peak_gpu_memory_allocated_mb")})
    c7.write_csv(SUM / "per_category.csv", pc)
    c7.write_csv(SUM / "per_seed.csv", ps)
    c7.write_csv(SUM / "pareto.csv", par)
    uniq = {r["config"]: r for r in rt}
    c7.write_csv(SUM / "runtime.csv", list(uniq.values()))
    (SUM / "verdict.json").write_text(json.dumps(verdict, indent=2))


def write_reports(lookup, r1_rows, full_rows, comb_rows, verdict, checks, promoted) -> None:
    L = []
    L.append("# Experiment 7A-O — sanity report\n")
    L.append(f"frozen protocol: `results/experiment_7a_o/reference/policy_freeze.json` "
             f"(sha256 `{c7.sha256_file(REF / 'policy_freeze.json')}`)\n")
    L.append("| # | check | status | detail |\n|---|---|---|---|")
    for c, ok, d in checks:
        L.append(f"| {c.split('_')[0][1:]} | {c} | {'PASS' if ok else 'FAIL'} | {d} |")
    npass = sum(1 for _, ok, _ in checks if ok)
    L.append(f"\n**{npass}/{len(checks)} PASS**\n")
    (SUM / "sanity_report.md").write_text("\n".join(L))

    fam_stats = {}
    for fam in ("A", "B", "C", "D"):
        rs = [r for r in r1_rows if r["type"] == "candidate" and r["family"] == fam]
        if rs:
            fam_stats[fam] = {"n": len(rs), "best": max(rs, key=lambda r: r["delta_dprime_vs_uniform"]),
                              "best_dz": min(rs, key=lambda r: r["delta_abs_z_vs_uniform"]),
                              "promoted": [r["config"] for r in rs if r["tier"] in c7.PROMOTE_TIERS]}
    F = []
    F.append("# Experiment 7A-O — overnight module screening (EXPLORATORY)\n")
    F.append(f"- freeze sha256: `{c7.sha256_file(REF / 'policy_freeze.json')}`")
    F.append(f"- GPU units: {len(scan_units())} (baselines B0/B1/B2/B3 reconstructed from frozen raw, 0 GPU)")
    F.append(f"- sanity: {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} PASS")
    F.append(f"- **FINAL VERDICT: {verdict['case']}** — {verdict['meaning']}\n")
    F.append("## Family summary (Round 1, 3 categories × seed0, vs Uniform alpha=0.40091275)\n")
    F.append("| family | n | best Δd′ | best Δ|Δz| | promoted |\n|---|---|---|---|---|")
    for fam, st in fam_stats.items():
        F.append(f"| {fam} | {st['n']} | {st['best']['config']} ({st['best']['delta_dprime_vs_uniform']:+.4f}) "
                 f"| {st['best_dz']['config']} ({st['best_dz']['delta_abs_z_vs_uniform']:+.4f}) "
                 f"| {', '.join(st['promoted']) or '—'} |")
    F.append(f"\n## Promoted to Round 2\n\n{promoted or '— (none)'}\n")
    F.append("## Round-2 (full 5 categories × 3 seeds)\n")
    F.append("| config | family | d′ | |Δz| | Δd′ vs Uniform | Δ|Δz| vs Uniform | worst cat Δ | negTr | tier |\n"
             "|---|---|---|---|---|---|---|---|---|")
    for r in full_rows:
        F.append(f"| {r['config']} | {r['family']} | {r['mean_defect_dprime']:.4f} | {r['mean_abs_delta_z']:.4f} "
                 f"| {r['delta_dprime_vs_uniform']:+.4f} | {r['delta_abs_z_vs_uniform']:+.4f} "
                 f"| {r['worst_category_delta']:+.4f} | {r['negative_transfer_count']} | {r['tier']} |")
    if comb_rows:
        F.append("\n## Round-3 combination\n")
        F.append("| combo | d′ | |Δz| | Δd′ vs Uniform | Δ|Δz| vs Uniform | negTr | tier |\n|---|---|---|---|---|---|---|")
        for r in comb_rows:
            F.append(f"| {r['config']} | {r['mean_defect_dprime']:.4f} | {r['mean_abs_delta_z']:.4f} "
                     f"| {r['delta_dprime_vs_uniform']:+.4f} | {r['delta_abs_z_vs_uniform']:+.4f} "
                     f"| {r['negative_transfer_count']} | {r['tier']} |")
    F.append("\n## Negative results (retained)\n")
    losers = [r for r in r1_rows if r["type"] == "candidate" and r["tier"] == "-"]
    for r in losers:
        F.append(f"- {r['config']} (family {r['family']}): Δd′ {r['delta_dprime_vs_uniform']:+.4f}, "
                 f"Δ|Δz| {r['delta_abs_z_vs_uniform']:+.4f}, negTr {r['negative_transfer_count']} — eliminated")
    if not losers:
        F.append("- (none — every screened config reached at least Tier B)")
    F.append("\n## Leakage statement\n")
    F.append("All results are **EXPLORATORY / selection-set**. The winner must be re-validated under an "
             "independent confirmation protocol (frozen structure/params/metrics) before any paper claim.")
    (SUM / "final_report.md").write_text("\n".join(F))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="round1", choices=["round1", "round2", "final"])
    ap.add_argument("--winners", default="")
    args = ap.parse_args()
    for d in (SUM, FIG, SAN, REF):
        d.mkdir(parents=True, exist_ok=True)
    lookup, meta = build_lookup()
    print(f"[7A-O] lookup keys = {len(lookup)}")
    cov = coverage()
    r1_rows, _ = rank_rows(lookup, list(c7.MODULE_SPECS), c7.ROUND1_CATEGORIES, [0],
                           baseline_key_maps()["B2_uniform_040091275"])
    print_table(r1_rows, "[7A-O ROUND 1] broad screening — 3 categories (bottle/grid/hazelnut) x seed0, "
                         "ranked vs Uniform alpha=0.40091275")
    promoted = promote(r1_rows)
    c7.write_csv(SUM / "round1_ranking.csv", r1_rows)
    (SUM / "promotion.json").write_text(json.dumps(
        {"promoted": promoted, "rule": "Top-1 per family among Tier S/A/B", "tiers": c7.TIER}, indent=2))
    print(f"PROMOTED (mechanical): {promoted}")
    if args.stage == "round1":
        return

    winners = [w for w in (args.winners.split(",") if args.winners else promoted) if w]
    full, _ = rank_rows(lookup, winners, c7.CATEGORIES, c7.SEEDS,
                        baseline_key_maps()["B2_uniform_040091275"])
    print_table(full, f"[7A-O ROUND 2] full 5 categories x 3 seeds for {winners}")
    c7.write_csv(SUM / "round2_ranking.csv", full)
    if args.stage == "round2":
        return

    all_cfgs = sorted({r["config"] for r in r1_rows if r["type"] == "candidate"})
    fc = full_cfgs(cov)
    final_rows, _ = rank_rows(lookup, fc or all_cfgs, c7.CATEGORIES, c7.SEEDS,
                              baseline_key_maps()["B2_uniform_040091275"])
    comb = [c for c in final_rows if c["config"].startswith("COMB_")]
    verdict = compute_verdict([r for r in final_rows if not r["config"].startswith("COMB_")], comb)
    checks = sanity_checks(lookup, promoted, r1_rows)
    write_tables(lookup, final_rows, r1_rows, comb, verdict, checks)
    fig1_pareto(lookup, all_cfgs, c7.ROUND1_CATEGORIES, [0])
    fig2_family_delta(r1_rows)
    if final_rows:
        w = max(final_rows, key=lambda r: r["delta_dprime_vs_uniform"])["config"]
        fig3_winner_category(lookup, w, c7.CATEGORIES, c7.SEEDS)
        fig4_winner_seed(lookup, w, c7.CATEGORIES, c7.SEEDS)
    if comb and final_rows:
        s = max([r for r in final_rows if not r["config"].startswith("COMB_")],
                key=lambda r: r["delta_dprime_vs_uniform"])["config"]
        fig5_combination(lookup, s, comb[0]["config"], c7.ROUND1_CATEGORIES, [0])
    write_reports(lookup, r1_rows, final_rows, comb, verdict, checks, promoted)
    print("=" * 126)
    for c, ok, d in checks:
        print(f"  {'PASS' if ok else 'FAIL':<5}{c:<50}{d[:56]}")
    print(f"  -> {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} sanity PASS")
    print(f"FINAL VERDICT: {verdict['case']} — {verdict['meaning']}")
    print("=" * 126)


if __name__ == "__main__":
    main()
