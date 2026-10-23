#!/usr/bin/env python
"""投稿前收尾：修正版「工件级配对 cluster bootstrap」（3 seed 先算差再平均）。

为什么需要修正版（与 results/paper_supplement_20261023/audit_a3_specimen_cluster_bootstrap.json 的区别）：
  旧脚本对每个 seed 单独做 bootstrap，然后取「各 seed CI 端点的平均」作为区间，
  并用 bootstrap 均值替代点估计。那不是一个统计量（三种子平均增益）的 95% 区间。

本脚本的估计量（estimand）：
  固定 3 个冻结 seed，测试工件（specimen）抽样不确定性下，
  dAUROC(or dAP) = mean_seed[ metric(R1) - metric(F0) ] 的分布，类别内 3 个 seed 与 F0/R1
  共用同一次工件重采样；类别等权宏平均在每次 replicate 内先算再取分位数。

规定：
  - 三个 seed 的图像身份、标签、工件 ID 按身份（文件名）对齐，不假定数组顺序相同；
  - 只使用冻结 cell 的 scores_f0 / scores_r1 / labels / specimen / detectable_raw；
  - 只用 detectable 子集（与主表 official 口径一致）；
  - 同一 specimen 的全部图像一起进入（cluster）；
  - 每次每个 seed 分别算 AUROC/AP 差，再对 3 个 seed 取平均；
  - 点估计来自原始样本（weights=1），不来自 bootstrap 均值；
  - rng 固定，n_boot=3000，退化重采样记数不静默丢弃；
  - P(d>0) 是 bootstrap 中差值 > 0 的重采样比例，不是 p 值。

实现说明：以工件计数作为 sample_weight 传给 sklearn（整数权重 == 把该工件复制 k 次），
并在启动时用「显式复制」抽样做等价性自检（见 check_weight_equivalence）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e22" / "cells"
OUTDIR = ROOT / "results" / "paper_final_pack_20261023"

CATS_DEV = ["Bird", "Car", "Cube"]
CATS_EXT = ["Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
CATS = CATS_DEV + CATS_EXT
SEEDS = [0, 1, 2]

N_BOOT = 3000
RNG_SEED = 20261023
ALPHA = 0.05


# --------------------------------------------------------------------------- loading
def load_category(cat: str) -> dict:
    """按图像身份对齐 3 个 seed，返回统一顺序下的阵列。"""
    per_seed = {}
    for seed in SEEDS:
        c = json.loads((CELLDIR / f"{cat}_seed{seed}.json").read_text())
        det = np.asarray(c["detectable_raw"], dtype=bool)
        # E22 cells 未存文件名；身份用 (specimen, illumination, kind)，已逐 cell 校验唯一。
        rows = {}
        for i in range(len(c["labels"])):
            if not det[i]:
                continue
            key = (str(c["specimen"][i]), str(c["illumination"][i]), str(c["kind"][i]))
            rows[key] = (int(c["labels"][i]), str(c["specimen"][i]),
                         float(c["scores_f0"][i]), float(c["scores_r1"][i]))
        per_seed[seed] = rows

    keys = sorted(per_seed[SEEDS[0]])
    same_order = all(list(per_seed[s]) == list(per_seed[SEEDS[0]]) for s in SEEDS)
    for s in SEEDS:
        assert sorted(per_seed[s]) == keys, f"{cat}: seed{s} identity set differs"
        for f in keys:
            assert per_seed[s][f][0] == per_seed[0][f][0], f"{cat}: label differs {f}"
            assert per_seed[s][f][1] == per_seed[0][f][1], f"{cat}: specimen differs {f}"

    labels = np.array([per_seed[0][f][0] for f in keys], dtype=int)
    specimen = [per_seed[0][f][1] for f in keys]
    s0 = {s: np.array([per_seed[s][f][2] for f in keys]) for s in SEEDS}
    s1 = {s: np.array([per_seed[s][f][3] for f in keys]) for s in SEEDS}

    specs = sorted(set(specimen))
    spec_arr = np.array(specimen)
    spec_of_row = np.searchsorted(specs, spec_arr)
    assert len(specs) == len(set(specs))
    # 每个工件只能有一种标签（否则 AUROC 的簇定义不成立）
    for k in specs:
        labs = set(labels[spec_arr == k].tolist())
        assert len(labs) == 1, f"{cat}: specimen {k} has mixed labels"
    return {"keys": keys, "labels": labels, "specimen": specs,
            "spec_of_row": spec_of_row, "s0": s0, "s1": s1,
            "same_array_order": same_order, "n_rows": len(keys)}


# --------------------------------------------------------------------------- metrics
def metrics(y, s0, s1, w):
    a0 = roc_auc_score(y, s0, sample_weight=w)
    a1 = roc_auc_score(y, s1, sample_weight=w)
    p0 = average_precision_score(y, s0, sample_weight=w)
    p1 = average_precision_score(y, s1, sample_weight=w)
    return a1 - a0, p1 - p0


def check_weight_equivalence(d: dict, rng, n_check: int = 5) -> float:
    """整数权重 == 显式复制：对随机重采样比较两种实现的最大绝对差。"""
    y, idx = d["labels"], d["spec_of_row"]
    n_spec = len(d["specimen"])
    worst = 0.0
    for _ in range(n_check):
        cnt = np.bincount(rng.integers(0, n_spec, size=n_spec), minlength=n_spec)
        w = cnt[idx].astype(float)
        rows = np.repeat(np.arange(len(y)), cnt[idx])
        for s in SEEDS:
            dw, pw = metrics(y, d["s0"][s], d["s1"][s], w)
            de, pe = metrics(y[rows], d["s0"][s][rows], d["s1"][s][rows], None)
            worst = max(worst, abs(dw - de), abs(pw - pe))
    return worst


def point_estimate(d: dict) -> dict:
    y, w = d["labels"], np.ones(d["n_rows"])
    du, dp = [], []
    for s in SEEDS:
        a, p = metrics(y, d["s0"][s], d["s1"][s], w)
        du.append(a)
        dp.append(p)
    return {"d_auroc": float(np.mean(du)), "d_ap": float(np.mean(dp)),
            "per_seed_d_auroc": [float(x) for x in du],
            "per_seed_d_ap": [float(x) for x in dp]}


# --------------------------------------------------------------------------- main
def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = {c: load_category(c) for c in CATS}

    rng_check = np.random.default_rng(RNG_SEED - 1)
    eq = {c: check_weight_equivalence(data[c], rng_check) for c in CATS}
    worst_eq = max(eq.values())
    print(f"[check] 权重-vs-显式复制 最大绝对差 = {worst_eq:.3e}")

    rng = np.random.default_rng(RNG_SEED)
    du = {c: np.full(N_BOOT, np.nan) for c in CATS}
    dp = {c: np.full(N_BOOT, np.nan) for c in CATS}
    degen = {c: 0 for c in CATS}

    for b in range(N_BOOT):
        for c in CATS:
            d = data[c]
            n_spec = len(d["specimen"])
            cnt = np.bincount(rng.integers(0, n_spec, size=n_spec), minlength=n_spec)
            w = cnt[d["spec_of_row"]].astype(float)
            y = d["labels"]
            wpos = w[y == 1].sum()
            wneg = w[y == 0].sum()
            if wpos == 0 or wneg == 0:
                degen[c] += 1
                continue
            us, ps = [], []
            for s in SEEDS:
                a, p = metrics(y, d["s0"][s], d["s1"][s], w)
                us.append(a)
                ps.append(p)
            du[c][b] = float(np.mean(us))
            dp[c][b] = float(np.mean(ps))

    def summarise(arr_by_cat, cats, point_by_cat):
        """类别等权宏平均：每次 replicate 内先平均类别，再取分位数。

        point 来自原始样本（未重采样），不是 bootstrap 均值。
        """
        mat = np.vstack([arr_by_cat[c] for c in cats])
        macro = np.nanmean(mat, axis=0)
        valid = macro[~np.isnan(macro)]
        return {"point": float(np.mean([point_by_cat[c] for c in cats])),
                "n_boot_effective": int(valid.size),
                "n_boot_missing": int(N_BOOT - valid.size),
                "ci_2p5": float(np.percentile(valid, 100 * ALPHA / 2)),
                "ci_97p5": float(np.percentile(valid, 100 * (1 - ALPHA / 2))),
                "p_gt0": float((valid > 0).mean())}

    out = {
        "analysis": "specimen-level paired cluster bootstrap, corrected (seeds differenced first)",
        "estimand": "mean over 3 frozen seeds of [metric(R1)-metric(F0)], uncertainty from "
                    "resampling test specimens (clusters); category-equal-weight macro is "
                    "computed inside each replicate",
        "not_estimand": ["training randomness / cross-seed variability",
                         "population generalisation across categories",
                         "a hypothesis-test p-value"],
        "n_boot": N_BOOT, "rng_seed": RNG_SEED, "alpha": ALPHA,
        "weight_equivalence_max_absdiff": worst_eq,
        "per_category": {}, "macro": {},
    }
    for c in CATS:
        pe = point_estimate(data[c])
        v = du[c][~np.isnan(du[c])]
        vp = dp[c][~np.isnan(dp[c])]
        out["per_category"][c] = {
            "n_specimen": len(data[c]["specimen"]),
            "n_rows_detectable": data[c]["n_rows"],
            "same_array_order_across_seeds": data[c]["same_array_order"],
            "degenerate_resamples": degen[c],
            "point_d_auroc": pe["d_auroc"],
            "point_d_ap": pe["d_ap"],
            "per_seed_point_d_auroc": pe["per_seed_d_auroc"],
            "per_seed_point_d_ap": pe["per_seed_d_ap"],
            "ci_auroc": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_d_auroc_gt0": float((v > 0).mean()),
            "ci_ap": [float(np.percentile(vp, 2.5)), float(np.percentile(vp, 97.5))],
            "p_d_ap_gt0": float((vp > 0).mean()),
        }
    point_auroc = {c: out["per_category"][c]["point_d_auroc"] for c in CATS}
    point_ap = {c: out["per_category"][c]["point_d_ap"] for c in CATS}
    for name, cats in (("all10", CATS), ("dev3", CATS_DEV), ("extension7", CATS_EXT)):
        out["macro"][name] = {"categories": cats,
                              "auroc": summarise(du, cats, point_auroc),
                              "ap": summarise(dp, cats, point_ap)}

    with open(OUTDIR / "specimen_bootstrap_v2.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    with open(OUTDIR / "specimen_bootstrap_v2_per_category.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["category", "n_specimen", "n_rows", "point_d_auroc", "ci_lo", "ci_hi",
                    "p_gt0_auroc", "point_d_ap", "ci_lo_ap", "ci_hi_ap", "p_gt0_ap",
                    "degenerate"])
        for c in CATS:
            r = out["per_category"][c]
            w.writerow([c, r["n_specimen"], r["n_rows_detectable"],
                        f"{r['point_d_auroc']:.6f}", f"{r['ci_auroc'][0]:.6f}",
                        f"{r['ci_auroc'][1]:.6f}", f"{r['p_d_auroc_gt0']:.4f}",
                        f"{r['point_d_ap']:.6f}", f"{r['ci_ap'][0]:.6f}",
                        f"{r['ci_ap'][1]:.6f}", f"{r['p_d_ap_gt0']:.4f}",
                        r["degenerate_resamples"]])

    print(f"[done] n_boot={N_BOOT} rng={RNG_SEED} -> {OUTDIR/'specimen_bootstrap_v2.json'}")
    for c in CATS:
        r = out["per_category"][c]
        print(f"  {c:8s} spec={r['n_specimen']:3d} dAUROC={r['point_d_auroc']:+.5f} "
              f"CI=[{r['ci_auroc'][0]:+.5f},{r['ci_auroc'][1]:+.5f}] "
              f"P>0={r['p_d_auroc_gt0']:.3f} degen={r['degenerate_resamples']}")
    for name in ("all10", "dev3", "extension7"):
        a = out["macro"][name]["auroc"]
        p = out["macro"][name]["ap"]
        print(f"  macro[{name:10s}] dAUROC={a['point']:+.5f} "
              f"CI=[{a['ci_2p5']:+.5f},{a['ci_97p5']:+.5f}] P>0={a['p_gt0']:.3f} | "
              f"dAP={p['point']:+.5f} CI=[{p['ci_2p5']:+.5f},{p['ci_97p5']:+.5f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
