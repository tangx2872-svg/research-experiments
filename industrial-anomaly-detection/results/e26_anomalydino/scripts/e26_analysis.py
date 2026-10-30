#!/usr/bin/env python
"""E26 分析：AnomalyDINO（官方实现·本协议配置）vs F0 / R1。

- 逐 cell（10 类 × 3 seed）：AUROC / AP（official image_anomaly + detectable != False）
- 类别层：3 seed 均值 ± 样本标准差（ddof=1）
- 宏平均：10 类等权（另给开发 3 类 / 扩展 7 类描述性汇总）
- R1 − AnomalyDINO：工件级配对 bootstrap；同工件全部有效光照共同抽样；
  两方法共用同一次工件抽样；3 seed 共用同一组抽样；3000 次。
- 附带 cache vs orig 输入（仅 Bird 有原图）的偏差量化。

只读：results/e26_anomalydino/scores/*.json、scores_originput/*.json、results/e22/cells/*.json
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SCORES = OUT / "scores"
SCORES_ORIG = OUT / "scores_originput"
CELLS = ROOT / "results" / "e22" / "cells"

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
DEV = {"Bird", "Car", "Cube"}
N_BOOT = 3000
BOOT_SEED = 20261030


def bank_hash(paths) -> str:
    h = hashlib.sha256()
    for s in paths:
        h.update(str(s).encode())
        h.update(b"\n")
    return h.hexdigest()


def metrics(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def load_cell(cat, seed):
    ad = json.loads((SCORES / f"{cat}_seed{seed}.json").read_text())
    e22 = json.loads((CELLS / f"{cat}_seed{seed}.json").read_text())
    assert ad["identity_ok"], f"{cat} seed{seed} identity check failed"
    bh = bank_hash([r["orig"] for r in ad["refs"]])
    assert bh == e22["bank_hash"], f"{cat} seed{seed} bank mismatch"
    keep = np.array([t["detectable"] is not False for t in ad["tests"]], bool)
    y = np.array([t["label"] for t in ad["tests"]], int)[keep]
    s_ad = np.array([t["score"] for t in ad["tests"]], float)[keep]
    sp = np.array([t["specimen"] for t in ad["tests"]])[keep]
    s0 = np.array(e22["scores_f0"], float)[keep]
    s1 = np.array(e22["scores_r1"], float)[keep]
    return {"y": y, "s_ad": s_ad, "s0": s0, "s1": s1, "sp": sp,
            "n_test": ad["n_test"], "n_keep": int(keep.sum())}


def main() -> int:
    rows, cache = [], {}
    for cat in CATS:
        for seed in SEEDS:
            d = load_cell(cat, seed)
            cache[(cat, seed)] = d
            a_ad, p_ad = metrics(d["y"], d["s_ad"])
            a0, p0 = metrics(d["y"], d["s0"])
            a1, p1 = metrics(d["y"], d["s1"])
            rows.append({
                "category": cat, "seed": seed,
                "n_ok": int((d["y"] == 0).sum()), "n_anomaly": int((d["y"] == 1).sum()),
                "auroc_ad": a_ad, "ap_ad": p_ad,
                "auroc_f0": a0, "ap_f0": p0,
                "auroc_r1": a1, "ap_r1": p1,
                "d_auroc_r1_minus_ad": a1 - a_ad, "d_ap_r1_minus_ad": p1 - p_ad,
                "d_auroc_ad_minus_f0": a_ad - a0, "d_ap_ad_minus_f0": p_ad - p0,
                "group": "dev" if cat in DEV else "ext",
            })

    cat_rows = []
    for cat in CATS:
        sub = [r for r in rows if r["category"] == cat]
        row = {"category": cat, "group": sub[0]["group"], "n_seeds": len(sub),
               "n_ok": sub[0]["n_ok"], "n_anomaly": sub[0]["n_anomaly"]}
        for k in ("auroc_ad", "ap_ad", "auroc_f0", "ap_f0", "auroc_r1", "ap_r1",
                  "d_auroc_r1_minus_ad", "d_ap_r1_minus_ad",
                  "d_auroc_ad_minus_f0", "d_ap_ad_minus_f0"):
            v = np.array([r[k] for r in sub], float)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_sd"] = float(v.std(ddof=1))
        cat_rows.append(row)

    def macro(key, subset=CATS):
        return float(np.mean([r[key] for r in cat_rows if r["category"] in subset]))

    macro_rows = []
    for tag, subset in (("macro_10cat", CATS), ("dev_3cat", sorted(DEV)),
                        ("ext_7cat", [c for c in CATS if c not in DEV])):
        r = {"level": tag, "n_categories": len(subset)}
        for k in ("auroc_ad_mean", "ap_ad_mean", "auroc_f0_mean", "ap_f0_mean",
                  "auroc_r1_mean", "ap_r1_mean", "d_auroc_r1_minus_ad_mean",
                  "d_ap_r1_minus_ad_mean", "d_auroc_ad_minus_f0_mean",
                  "d_ap_ad_minus_f0_mean"):
            r[k.replace("_mean", "") + "_macro"] = macro(k, subset)
        macro_rows.append(r)

    # ---------- 工件级配对 bootstrap
    spec_ids = {c: sorted(set(cache[(c, 0)]["sp"].tolist())) for c in CATS}
    rng = np.random.default_rng(BOOT_SEED)
    per_cat = {c: {"d_auroc": [], "d_ap": []} for c in CATS}
    macro = {"d_auroc": [], "d_ap": []}
    dev_a, ext_a = [], []
    for _ in range(N_BOOT):
        draws = {c: rng.integers(0, len(spec_ids[c]), len(spec_ids[c])) for c in CATS}
        cell_boot = {}
        for c in CATS:
            ids = spec_ids[c]
            sel = [ids[i] for i in draws[c]]
            au, ap = [], []
            for seed in SEEDS:
                d = cache[(c, seed)]
                idx = np.concatenate([np.flatnonzero(d["sp"] == s) for s in sel])
                yy = d["y"][idx]
                if len(set(yy.tolist())) < 2:
                    au.append(np.nan); ap.append(np.nan); continue
                au.append(roc_auc_score(yy, d["s1"][idx]) - roc_auc_score(yy, d["s_ad"][idx]))
                ap.append(average_precision_score(yy, d["s1"][idx])
                          - average_precision_score(yy, d["s_ad"][idx]))
            cell_boot[c] = (float(np.nanmean(au)), float(np.nanmean(ap)))
            per_cat[c]["d_auroc"].append(cell_boot[c][0])
            per_cat[c]["d_ap"].append(cell_boot[c][1])
        macro["d_auroc"].append(float(np.mean([cell_boot[c][0] for c in CATS])))
        macro["d_ap"].append(float(np.mean([cell_boot[c][1] for c in CATS])))
        dev_a.append(float(np.mean([cell_boot[c][0] for c in DEV])))
        ext_a.append(float(np.mean([cell_boot[c][0] for c in CATS if c not in DEV])))

    def q(a):
        a = np.asarray(a, float)
        a = a[~np.isnan(a)]
        return {"boot_mean": float(a.mean()), "boot_sd": float(a.std(ddof=1)),
                "ci_lo": float(np.percentile(a, 2.5)), "ci_hi": float(np.percentile(a, 97.5)),
                "n_eff": int(len(a))}

    boot_rows = []
    for c in CATS:
        for m in ("d_auroc", "d_ap"):
            pt = [r for r in cat_rows if r["category"] == c][0][f"{m}_r1_minus_ad_mean"]
            boot_rows.append({"level": c, "metric": m, "point": pt, **q(per_cat[c][m]),
                              "n_boot": N_BOOT, "rng_seed": BOOT_SEED,
                              "resample_unit": "specimen (3 seeds share draw)"})
    for tag, arr, key in (("macro_10cat", macro["d_auroc"], "d_auroc_r1_minus_ad_macro"),
                          ("macro_10cat", macro["d_ap"], "d_ap_r1_minus_ad_macro")):
        boot_rows.append({"level": tag, "metric": "d_auroc" if key.startswith("d_auroc") else "d_ap",
                          "point": [r for r in macro_rows if r["level"] == tag][0][key],
                          **q(arr), "n_boot": N_BOOT, "rng_seed": BOOT_SEED,
                          "resample_unit": "specimen (3 seeds share draw)"})
    for tag, arr in (("dev_3cat", dev_a), ("ext_7cat", ext_a)):
        boot_rows.append({"level": tag, "metric": "d_auroc",
                          "point": [r for r in macro_rows if r["level"] == tag][0][
                              "d_auroc_r1_minus_ad_macro"],
                          **q(arr), "n_boot": N_BOOT, "rng_seed": BOOT_SEED,
                          "resample_unit": "specimen (3 seeds share draw)"})

    # ---------- cache vs orig（仅 Bird）
    dev_rows = []
    for seed in SEEDS:
        po = SCORES_ORIG / f"Bird_seed{seed}.json"
        if not po.exists():
            continue
        a = json.loads(po.read_text())
        b = json.loads((SCORES / f"Bird_seed{seed}.json").read_text())
        keep = np.array([t["detectable"] for t in a["tests"]], bool)
        y = np.array([t["label"] for t in a["tests"]], int)[keep]
        sa = np.array([t["score"] for t in a["tests"]], float)[keep]
        sb = np.array([t["score"] for t in b["tests"]], float)[keep]
        dev_rows.append({"category": "Bird", "seed": seed,
                         "max_abs_dscore": float(np.max(np.abs(sa - sb))),
                         "auroc_orig_input": float(roc_auc_score(y, sa)),
                         "auroc_cache_input": float(roc_auc_score(y, sb)),
                         "d_auroc_cache_minus_orig": float(roc_auc_score(y, sb)
                                                           - roc_auc_score(y, sa)),
                         "ap_orig_input": float(average_precision_score(y, sa)),
                         "ap_cache_input": float(average_precision_score(y, sb)),
                         "d_ap_cache_minus_orig": float(average_precision_score(y, sb)
                                                        - average_precision_score(y, sa))})

    cond = [
        ("backbone", "dinov2_vits14 (official default)", "DINOv2 ViT-B/14", "不同（容量）"),
        ("resolution", "448 短边 BICUBIC + 裁 14 倍数", "448 l0 缓存", "几何等价"),
        ("input file", "写盘 l0_448 (PNG uint8) — 8bit 量化", "内存 float", "轻微数值偏差（已量化）"),
        ("reference set", "300 张 = 该 seed 的 bank（bank_hash 逐 cell 核对）", "同", "完全一致"),
        ("reference augmentation", "8 角度旋转（官方 agnostic 默认）", "无", "未对齐（官方组件）"),
        ("PCA masking", "关闭（官方对自定义数据集 fallback）", "无", "一致"),
        ("retrieval", "L2_normalized, k=1, 精确", "cosine 1-NN 精确", "等价"),
        ("image score", "mean of top 1% patch distances", "max patch distance", "不同（官方组件）"),
        ("adapter/training", "无（training-free）", "线性残差 W，normal-only 训练", "R1 有训练"),
        ("test set", "view120, 700 图, 逐图身份一致", "同", "完全一致"),
        ("label / filter", "image_anomaly + detectable != False", "同", "完全一致"),
    ]

    def dump(path, data):
        if not data:
            return
        cols = list(data[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in data:
                w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                            for k in cols})
        print(f"[e26] {path.name}: {len(data)} rows")

    dump(OUT / "baseline_30cells.csv", rows)
    dump(OUT / "baseline_category_summary.csv", cat_rows)
    dump(OUT / "baseline_macro.csv", macro_rows)
    dump(OUT / "baseline_bootstrap.csv", boot_rows)
    dump(OUT / "input_mode_cache_vs_orig.csv", dev_rows)
    with open(OUT / "comparison_conditions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "anomalydino_official_this_protocol", "this_paper_f0_r1",
                    "aligned"])
        w.writerows(cond)

    m = [r for r in macro_rows if r["level"] == "macro_10cat"][0]
    b_au = [r for r in boot_rows if r["level"] == "macro_10cat" and r["metric"] == "d_auroc"][0]
    b_ap = [r for r in boot_rows if r["level"] == "macro_10cat" and r["metric"] == "d_ap"][0]
    print("\n=== 10 类宏平均（3 seed 先平均，再类等权）===")
    print(f"  AnomalyDINO  AUROC={m['auroc_ad_macro']:.5f}  AP={m['ap_ad_macro']:.5f}")
    print(f"  F0           AUROC={m['auroc_f0_macro']:.5f}  AP={m['ap_f0_macro']:.5f}")
    print(f"  R1           AUROC={m['auroc_r1_macro']:.5f}  AP={m['ap_r1_macro']:.5f}")
    print(f"  R1 - AD      ΔAUROC={m['d_auroc_r1_minus_ad_macro']:+.5f} "
          f"CI [{b_au['ci_lo']:+.5f}, {b_au['ci_hi']:+.5f}]")
    print(f"               ΔAP   ={m['d_ap_r1_minus_ad_macro']:+.5f} "
          f"CI [{b_ap['ci_lo']:+.5f}, {b_ap['ci_hi']:+.5f}]")
    print("\n=== 逐类别（3 seed 均值）===")
    for r in cat_rows:
        print(f"  {r['category']:7s} AD={r['auroc_ad_mean']:.5f} F0={r['auroc_f0_mean']:.5f} "
              f"R1={r['auroc_r1_mean']:.5f} | Δ(R1-AD)={r['d_auroc_r1_minus_ad_mean']:+.5f} "
              f"Δ(AD-F0)={r['d_auroc_ad_minus_f0_mean']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
