"""Experiment 5B — per-seed consistency decomposition (descriptive).

预注册分析计划 §4.3（seed stability）的描述性分解：对每个 seed 单独计算
category-level（n=5）Spearman rho，核对方向是否跨 seed 一致。
仅读取已生成的 summary CSV，不引入新 predictor / 新 target / 新口径。

输出：results/experiment_5b/summary/seed_consistency_descriptive.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "experiment_5b" / "summary"
CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]

preds = {}
with open(SUM / "normal_only_predictors.csv", newline="") as f:
    for r in csv.DictReader(f):
        preds[(r["category"], int(r["seed"]))] = r
tgt = {}
with open(SUM / "target_damage.csv", newline="") as f:
    for r in csv.DictReader(f):
        tgt[(r["category"], int(r["seed"]))] = float(r["damage_C2"])

rows = []
for n in preds[("bottle", 0)].keys():
    if n in ("category", "seed"):
        continue
    r = {"predictor": n}
    rhos = []
    for s in SEEDS:
        x = [float(preds[(c, s)][n]) for c in CATEGORIES]
        y = [tgt[(c, s)] for c in CATEGORIES]
        rho = float(spearmanr(x, y)[0])
        r[f"rho_seed{s}"] = round(rho, 3)
        rhos.append(rho)
    r["direction_consistent"] = bool(all(np.sign(v) == np.sign(rhos[0]) for v in rhos)
                                     and rhos[0] != 0)
    rows.append(r)

with open(SUM / "seed_consistency_descriptive.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
