"""Experiment 13-P — Primary A/B/C + Diagnostic D（CPU-only；eigh）。

定义（冻结）：
  PRIMARY_centered  : cov_c = sum_ddT - sum_d sum_d^T / n   （任务书 §8: 先 center 再 PCA）
  SECONDARY_uncentered: cov_u = sum_ddT                     （= 9A nuisance_basis 的定义，含平均响应方向）
  EV@K = 累积解释方差比；K80 = 达到 0.80 的最小 K（>32 记 33）
  subspace similarity = ||U^T V||_F^2 / K （mean cos^2 principal angles）
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_13p"
ANA, FIG = EXP / "analysis", EXP / "figures"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
LAYERS = ["layer2", "layer3"]
PERT = [("T1", "gamma"), ("T2", "gamma"), ("T3", "brightness"), ("T4", "brightness")]
KGRID = [1, 2, 4, 8, 16, 32]
K_MAIN = 8
K80_CAP = 33


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
        w.writeheader(); w.writerows(rows)


def stats(cat: str, seed: int) -> dict:
    return dict(np.load(EXP / "raw" / cat / f"seed_{seed}" / "stats.npz"))


def counts(cat: str, seed: int) -> dict:
    return json.loads((EXP / "raw" / cat / f"seed_{seed}" / "meta.json").read_text())["counts"]


def cov(st: dict, layer: str, perts: tuple, centered: bool, cnt: dict) -> np.ndarray:
    C = st[f"sum_d_{layer}_{perts[0]}"].shape[0]
    out = np.zeros((C, C), dtype=np.float64)
    for pid in perts:
        s = st[f"sum_d_{layer}_{pid}"].astype(np.float64)
        m = st[f"sum_ddT_{layer}_{pid}"].astype(np.float64)
        n = float(max(cnt[f"{layer}_{pid}"], 1))
        out += (m - np.outer(s, s) / n) if centered else m
    return (out + out.T) / 2.0


def basis(covm: np.ndarray, k: int):
    ev, evec = np.linalg.eigh(covm)
    ev = ev[::-1]
    U = evec[:, ::-1][:, :k]
    tot = float(ev.clip(min=0).sum())
    cum = np.cumsum(ev.clip(min=0)) / max(tot, 1e-30)
    return U, ev, cum


def ev_at(cum: np.ndarray, ks=KGRID) -> dict:
    out = {}
    for k in ks:
        out[k] = float(cum[k - 1]) if k <= len(cum) else float(cum[-1])
    return out


def k80(cum: np.ndarray) -> int:
    idx = np.argmax(cum >= 0.80) + 1
    return int(idx) if cum.max() >= 0.80 else K80_CAP


def sim(U: np.ndarray, V: np.ndarray) -> float:
    k = U.shape[1]
    return float(np.linalg.norm(U.T @ V, ord="fro") ** 2 / k)


def build(cats=CATS, seeds=(0, 1, 2), centered=True) -> dict:
    res = {}
    for cat in cats:
        for seed in seeds:
            st, cnt = stats(cat, seed), counts(cat, seed)
            for layer in LAYERS:
                key = (cat, seed, layer)
                res[key] = {}
                for tag, perts in (("all", tuple(p[0] for p in PERT)),
                                   ("gamma", ("T1", "T2")), ("brightness", ("T3", "T4"))):
                    cm = cov(st, layer, perts, centered, cnt)
                    U, ev, cum = basis(cm, max(KGRID))
                    res[key][tag] = {"U": U, "ev": ev, "cum": cum, **ev_at(cum), "K80": k80(cum)}
    return res


def main() -> None:
    ANA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    prim = build(centered=True)
    sec = build(centered=False)

    # ---------- Primary A: 低维性 ----------
    rows_a = []
    for (cat, seed, layer), v in sorted(prim.items()):
        for defname, res in (("PRIMARY_centered", prim), ("SECONDARY_uncentered", sec)):
            d = res[(cat, seed, layer)]["all"]
            rows_a.append({"definition": defname, "category": cat, "seed": seed, "layer": layer,
                           **{f"EV@{k}": round(d[k], 4) for k in KGRID}, "K80": d["K80"],
                           "trace": round(float(d["ev"].clip(min=0).sum()), 3)})
    wcsv(ANA / "A_low_dimensionality.csv", rows_a)

    # ---------- Primary B: gamma vs brightness ----------
    rows_b = []
    for (cat, seed, layer), v in sorted(prim.items()):
        for defname, res in (("PRIMARY_centered", prim), ("SECONDARY_uncentered", sec)):
            d = res[(cat, seed, layer)]
            rows_b.append({"definition": defname, "category": cat, "seed": seed, "layer": layer,
                           "gamma_vs_brightness_sim_K8": round(sim(d["gamma"]["U"][:, :8],
                                                                   d["brightness"]["U"][:, :8]), 4)})
    wcsv(ANA / "B_augmentation_stability.csv", rows_b)

    # ---------- Primary C: cross-category ----------
    rows_c = []
    for defname, res in (("PRIMARY_centered", prim), ("SECONDARY_uncentered", sec)):
        for layer in LAYERS:
            for i, c1 in enumerate(CATS):
                for c2 in CATS[i + 1:]:
                    rows_c.append({"definition": defname, "layer": layer, "cat_a": c1, "cat_b": c2,
                                   "sim_K8_seed0": round(sim(res[(c1, 0, layer)]["all"]["U"][:, :8],
                                                             res[(c2, 0, layer)]["all"]["U"][:, :8]), 4)})
    wcsv(ANA / "C_cross_category.csv", rows_c)

    # ---------- Diagnostic D: seed stability ----------
    rows_d = []
    for defname, res in (("PRIMARY_centered", prim), ("SECONDARY_uncentered", sec)):
        for cat in CATS:
            for layer in LAYERS:
                su = [sim(res[(cat, s, layer)]["all"]["U"][:, :8], res[(cat, t, layer)]["all"]["U"][:, :8])
                      for s in (0, 1, 2) for t in (0, 1, 2) if s < t]
                ev8 = [res[(cat, s, layer)]["all"][8] for s in (0, 1, 2)]
                kk = [res[(cat, s, layer)]["all"]["K80"] for s in (0, 1, 2)]
                rows_d.append({"definition": defname, "category": cat, "layer": layer,
                               "EV@8_seed0/1/2": "/".join("%.3f" % x for x in ev8),
                               "K80_seed0/1/2": "/".join(str(x) for x in kk),
                               "mean_pairwise_subspace_sim": round(float(np.mean(su)), 4),
                               "min_pairwise_subspace_sim": round(float(np.min(su)), 4)})
    wcsv(ANA / "D_seed_stability.csv", rows_d)

    # ---------- 冻结 illumination subspace (K=8, all-pert, seed0, PRIMARY) ----------
    freeze = {"K": K_MAIN, "definition": "PRIMARY_centered", "seed": 0, "layers": {},
              "sha256": {}}
    for cat in CATS:
        for layer in LAYERS:
            U = prim[(cat, 0, layer)]["all"]["U"][:, :K_MAIN].astype(np.float64)
            np.savez_compressed(ANA / f"illum_subspace_{cat}_{layer}_k8.npz", U=U)
            h = hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest()
            freeze["layers"].setdefault(cat, {})[layer] = {"shape": list(U.shape), "sha256": h}
    (ANA / "illumination_subspace_freeze.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False))

    # ---------- console ----------
    print("=" * 118)
    print("Experiment 13-P — PRIMARY A/B/C + D  (K grid %s, K_MAIN=%d)" % (KGRID, K_MAIN))
    print("=" * 118)
    print("%-20s%-12s%6s%7s%7s%7s%7s%7s%7s%6s" % ("definition", "cat/layer", "seed", "EV@1", "EV@2", "EV@4",
                                                  "EV@8", "EV@16", "EV@32", "K80"))
    for r in rows_a:
        if r["seed"] != 0:
            continue
        print("%-20s%-12s%6d%7.3f%7.3f%7.3f%7.3f%7.3f%7.3f%6d" % (
            r["definition"], "%s/%s" % (r["category"], r["layer"]), r["seed"], r["EV@1"], r["EV@2"],
            r["EV@4"], r["EV@8"], r["EV@16"], r["EV@32"], r["K80"]))
    print("-" * 118)
    print("Primary B (gamma vs brightness, K=8):")
    for r in rows_b:
        if r["seed"] == 0:
            print("   %-20s %-12s sim=%.4f" % (r["definition"], "%s/%s" % (r["category"], r["layer"]),
                                               r["gamma_vs_brightness_sim_K8"]))
    print("-" * 118)
    print("Primary C (cross-category, K=8, seed0):")
    for r in rows_c:
        print("   %-20s %-7s %-9s%-9s sim=%.4f" % (r["definition"], r["layer"], r["cat_a"], r["cat_b"],
                                                    r["sim_K8_seed0"]))
    print("-" * 118)
    print("Diagnostic D (seed stability):")
    for r in rows_d:
        print("   %-20s %-12s EV@8=%s K80=%s mean_pair_sim=%.4f (min %.4f)" % (
            r["definition"], "%s/%s" % (r["category"], r["layer"]), r["EV@8_seed0/1/2"],
            r["K80_seed0/1/2"], r["mean_pairwise_subspace_sim"], r["min_pairwise_subspace_sim"]))
    print("=" * 118)
    print("frozen subspace ->", ANA / "illumination_subspace_freeze.json")


if __name__ == "__main__":
    main()
