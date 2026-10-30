#!/usr/bin/env python
"""E27: 真实光照下的同工件配对得分变化（仅用已有 F0/R1 逐图得分，无训练/推理）。

口径（预注册，见同目录 protocol.md）：
  - 数据：M²AD 10 类 × view120 × seed{0,1,2}，来自 results/e22/cells/*.json
  - 主指标：正常测试工件在 10 个真实光照下的**样本标准差**（ddof=1）的跨工件均值
  - 聚合：类别内工件等权 → 3 seed 平均 → 10 类等权
  - 区间：工件级配对 bootstrap，3 seed 共用同一组工件抽样，3000 次

输出：
  illum_sd_per_specimen.csv     逐 (cat, seed, specimen) 的 F0/R1 跨光照 SD
  illum_sd_category.csv         逐 (cat, seed) 汇总 + 3-seed 平均 + 排除计数
  illum_sd_macro.csv            10 类等权宏平均 + bootstrap 区间
  illum_level_stats.csv         正常/异常得分的均值与标准差变化（主评价同口径）
"""
from __future__ import annotations

import csv
import json
import glob
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]      # results/e27_illum_paired/
CELLS = ROOT / "results" / "e22" / "cells"
N_BOOT = 3000
SEED = 20261030
CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
DEV = {"Bird", "Car", "Cube"}


def load(cat, seed):
    c = json.loads((CELLS / f"{cat}_seed{seed}.json").read_text())
    y = np.array(c["labels"], int)
    det = np.array([x is not False for x in c["detectable_raw"]], bool)
    sp = np.array(c["specimen"])
    ill = np.array(c["illumination"])
    s0 = np.array(c["scores_f0"], float)
    s1 = np.array(c["scores_r1"], float)
    return y, det, sp, ill, s0, s1


def per_specimen_sd(sp, ill, s, mask, need=10):
    """在 mask 内按工件分组，返回 {specimen: sd} 以及排除计数。"""
    out, excl = {}, Counter()
    for sid in sorted(set(sp[mask].tolist())):
        m = mask & (sp == sid)
        n_ill = len(set(ill[m].tolist()))
        if n_ill < need:
            excl[f"only_{n_ill}_illum"] += 1
            continue
        v = s[m]
        if len(v) < 2:
            excl["n_lt2"] += 1
            continue
        out[sid] = float(np.std(v, ddof=1))
    return out, excl


def main() -> int:
    per_spec, cat_rows = [], []
    level_rows = []
    # {cat: {seed: {specimen: (sd0, sd1)}}}  正常工件
    normal = {}
    anomaly_cov = []

    for cat in CATS:
        normal[cat] = {}
        for seed in SEEDS:
            y, det, sp, ill, s0, s1 = load(cat, seed)
            nmask = det & (y == 0)
            amask = det & (y == 1)
            sd0, ex0 = per_specimen_sd(sp, ill, s0, nmask)
            sd1, ex1 = per_specimen_sd(sp, ill, s1, nmask)
            common = sorted(set(sd0) & set(sd1))
            normal[cat][seed] = {k: (sd0[k], sd1[k]) for k in common}
            for k in common:
                per_spec.append({"category": cat, "seed": seed, "specimen": k,
                                 "sd_f0": sd0[k], "sd_r1": sd1[k],
                                 "delta": sd1[k] - sd0[k],
                                 "rel_delta": (sd1[k] - sd0[k]) / sd0[k] if sd0[k] > 0 else float("nan"),
                                 "n_illum": 10})
            d = np.array([sd1[k] - sd0[k] for k in common], float)
            cat_rows.append({
                "category": cat, "seed": seed,
                "n_specimens_used": len(common),
                "n_specimens_excluded": sum(ex0.values()),
                "exclude_reasons": dict(ex0),
                "mean_sd_f0": float(np.mean([sd0[k] for k in common])),
                "mean_sd_r1": float(np.mean([sd1[k] for k in common])),
                "mean_delta_sd": float(d.mean()),
                "median_delta_sd": float(np.median(d)),
                "frac_specimens_pos": float((d > 0).mean()),
                "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            })
            # 异常工件的描述性覆盖统计
            a_spec = {}
            for sid in sorted(set(sp[amask].tolist())):
                m = amask & (sp == sid)
                a_spec[sid] = len(set(ill[m].tolist()))
            anomaly_cov.append({"category": cat, "seed": seed,
                                "n_anom_specimens": len(a_spec),
                                "n_with_10_illum": sum(1 for v in a_spec.values() if v == 10),
                                "min_illum": min(a_spec.values()) if a_spec else 0,
                                "max_illum": max(a_spec.values()) if a_spec else 0})

            # 主评价同口径的均值/标准差
            for tag, msk in (("normal", nmask), ("anomaly", amask)):
                for meth, s in (("F0", s0), ("R1", s1)):
                    level_rows.append({"category": cat, "seed": seed, "subset": tag,
                                       "method": meth, "n_images": int(msk.sum()),
                                       "n_specimens": len(set(sp[msk].tolist())),
                                       "mean": float(s[msk].mean()),
                                       "sd": float(s[msk].std(ddof=1)),
                                       "median": float(np.median(s[msk]))})

    # ---------- 3 seed 平均（类别层）
    cat_avg = []
    for cat in CATS:
        rows = [r for r in cat_rows if r["category"] == cat]
        cat_avg.append({
            "category": cat,
            "n_specimens_used": rows[0]["n_specimens_used"],
            "mean_sd_f0": float(np.mean([r["mean_sd_f0"] for r in rows])),
            "mean_sd_r1": float(np.mean([r["mean_sd_r1"] for r in rows])),
            "mean_delta_sd": float(np.mean([r["mean_delta_sd"] for r in rows])),
            "sd_over_seeds_of_delta": float(np.std([r["mean_delta_sd"] for r in rows], ddof=1)),
            "mean_frac_pos": float(np.mean([r["frac_specimens_pos"] for r in rows])),
            "seed_deltas": [r["mean_delta_sd"] for r in rows],
            "group": "dev" if cat in DEV else "ext",
        })

    # ---------- bootstrap（工件重采样，3 seed 共用）
    rng = np.random.default_rng(SEED)
    per_cat_boot = {c: [] for c in CATS}
    macro_boot, dev_boot, ext_boot = [], [], []
    spec_ids = {c: sorted(next(iter(normal[c].values())).keys()) for c in CATS}
    for _ in range(N_BOOT):
        draws = {}
        for c in CATS:
            ids = spec_ids[c]
            pick = rng.integers(0, len(ids), len(ids))
            draws[c] = [ids[i] for i in pick]
        for c in CATS:
            vals = []
            for seed in SEEDS:
                d = np.array([normal[c][seed][k][1] - normal[c][seed][k][0] for k in draws[c]], float)
                vals.append(d.mean())
            per_cat_boot[c].append(float(np.mean(vals)))
        macro_boot.append(float(np.mean([per_cat_boot[c][-1] for c in CATS])))
        dev_boot.append(float(np.mean([per_cat_boot[c][-1] for c in CATS if c in DEV])))
        ext_boot.append(float(np.mean([per_cat_boot[c][-1] for c in CATS if c not in DEV])))

    def q(a):
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}

    point_macro = float(np.mean([r["mean_delta_sd"] for r in cat_avg]))
    macro = {"level": "macro_10cat", "point_delta_sd": point_macro, **q(macro_boot),
             "n_boot": N_BOOT, "rng_seed": SEED, "resample_unit": "specimen (3 seeds share draw)"}
    dev = {"level": "dev_3cat", "point_delta_sd": float(np.mean(
        [r["mean_delta_sd"] for r in cat_avg if r["group"] == "dev"])), **q(dev_boot),
        "n_boot": N_BOOT, "rng_seed": SEED, "resample_unit": "specimen (3 seeds share draw)"}
    ext = {"level": "ext_7cat", "point_delta_sd": float(np.mean(
        [r["mean_delta_sd"] for r in cat_avg if r["group"] == "ext"])), **q(ext_boot),
        "n_boot": N_BOOT, "rng_seed": SEED, "resample_unit": "specimen (3 seeds share draw)"}

    def dump(path, rows):
        cols = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in cols})
        print(f"[e27] {path.name}: {len(rows)} rows")

    dump(OUT / "illum_sd_per_specimen.csv", per_spec)
    dump(OUT / "illum_sd_category.csv", cat_rows)
    dump(OUT / "illum_sd_category_seedavg.csv", cat_avg)
    dump(OUT / "illum_level_stats.csv", level_rows)
    dump(OUT / "anomaly_illum_coverage.csv", anomaly_cov)
    dump(OUT / "illum_sd_bootstrap.csv", [macro, dev, ext])

    print("\n=== 正常工件跨光照 SD（3 seed 平均）===")
    for r in cat_avg:
        print(f"  {r['category']:7s} n_spec={r['n_specimens_used']:2d} "
              f"F0={r['mean_sd_f0']:.5f} R1={r['mean_sd_r1']:.5f} "
              f"Δ={r['mean_delta_sd']:+.5f} (seed sd {r['sd_over_seeds_of_delta']:.5f}) "
              f"pos={r['mean_frac_pos']:.2f}")
    print(f"\n  macro Δ = {point_macro:+.6f}  CI [{macro['lo']:+.6f}, {macro['hi']:+.6f}]")
    print(f"  dev3   Δ = {dev['point_delta_sd']:+.6f}  CI [{dev['lo']:+.6f}, {dev['hi']:+.6f}]")
    print(f"  ext7   Δ = {ext['point_delta_sd']:+.6f}  CI [{ext['lo']:+.6f}, {ext['hi']:+.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
