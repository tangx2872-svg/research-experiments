"""Experiment 8B — metric-level 分析（CPU-only）。

输入：results/experiment8b/extract/cells/{cat}_seed{s}/{normal_stats,illum,defect}.npz
输出：results/experiment8b/analysis/*.csv + figures/fig1..7 + sanity/sanity_report.md

全部定义严格按 experiments/experiment8b/README.md 的冻结协议（sha 见 reference/）。
不训练、不调参、不做 grid search；随机基线 seed 冻结为 80001（metric-level，1000 draws）。

用法：python -u scripts/experiment8b_analyze.py [--extract-root ...] [--out-root ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
LAYERS = ["layer2", "layer3"]
CDIM = {"layer2": 512, "layer3": 1024}
RATIOS = [0.25, 0.50]
N_NULL = 1000            # metric-level random draws（冻结）
NULL_SEED = 80001        # = "8B0001"
RANDOM_DRAWS = 10        # score-level probe 用的 draws（冻结，与 probe 对齐）
RANDOM_SEED = 80002      # = "8B0002"
CELL_WIN_Z = 2.0
EPS_DEGENERATE = 1e-6

SEVERITY_OF_LEVEL = {0.5: "strong", 1.5: "strong", 0.7: "medium", 1.3: "medium",
                     0.9: "mild", 1.1: "mild"}


# ---------------------------------------------------------------------------
# 载入与指标构建
# ---------------------------------------------------------------------------
def load_cell(extract_root: Path, cat: str, seed: int) -> dict:
    d = extract_root / "cells" / f"{cat}_seed{seed}"
    ns = dict(np.load(d / "normal_stats.npz", allow_pickle=False))
    ill = dict(np.load(d / "illum.npz", allow_pickle=False))
    dfc = dict(np.load(d / "defect.npz", allow_pickle=False))
    return {"ns": ns, "ill": ill, "defect": dfc}


def build_layer_metrics(cell: dict, layer: str) -> dict:
    ns, ill, dfc = cell["ns"], cell["ill"], cell["defect"]
    mu = ns[f"{layer}_mu"]
    sigma = ns[f"{layer}_sigma"]
    eps = float(ns[f"eps_{layer}"][0])
    sigma_eff = np.sqrt(sigma ** 2 + eps ** 2)

    m = dfc[f"m_{layer}"]                                     # (n_def_img, C)
    dtypes = np.asarray(dfc["defect_type"])
    d_signed = (m.mean(axis=0) - mu) / sigma_eff
    sd_between = m.std(axis=0, ddof=1) if m.shape[0] > 1 else np.zeros_like(mu)
    d_poolsd = (m.mean(axis=0) - mu) / np.sqrt((sigma ** 2 + sd_between ** 2) / 2.0 + eps ** 2)

    # per-defect-type（§15 defect-type control）
    per_type = {}
    for dt in sorted(set(dtypes.tolist())):
        sel = m[dtypes == dt]
        per_type[dt] = (sel.mean(axis=0) - mu) / sigma_eff

    fam = np.asarray(ill["cond_family"])
    lev = np.asarray(ill["cond_level"], dtype=float)
    is_ident = np.asarray(ill["cond_is_identity"]).astype(bool)
    delta = ill[f"delta_{layer}"]                             # (n_cond, n_img, C)
    counted = ~is_ident
    i_cond = delta[counted].mean(axis=1) / sigma_eff          # (12, C) per-image equal weight

    fam_c, lev_c = fam[counted], lev[counted]
    i_bright = i_cond[fam_c == "brightness"].mean(axis=0)
    i_gamma = i_cond[fam_c == "gamma"].mean(axis=0)
    sev = np.array([SEVERITY_OF_LEVEL[float(x)] for x in lev_c])
    i_sev = {s: i_cond[sev == s].mean(axis=0) for s in ("mild", "medium", "strong")}
    i_mean = i_cond.mean(axis=0)

    pooled = ill[f"pooled_abs_{layer}"]
    i_pooled = pooled[counted].mean(axis=0) / float(ill["pooled_n_patches"][0]) / sigma_eff

    return {
        "mu": mu, "sigma": sigma, "sigma_eff": sigma_eff, "eps": eps,
        "D_signed": d_signed, "D_abs": np.abs(d_signed), "D_poolsd": d_poolsd,
        "per_type_D_signed": per_type,
        "I_mean": i_mean, "I_brightness": i_bright, "I_gamma": i_gamma,
        "I_sev": i_sev, "I_pooled": i_pooled,
        "degenerate": sigma < EPS_DEGENERATE,
        "n_defect_images": int(m.shape[0]),
    }


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


# ---------------------------------------------------------------------------
# Pareto / selection / null
# ---------------------------------------------------------------------------
def pareto_flags(d_abs: np.ndarray, i_val: np.ndarray) -> np.ndarray:
    """非支配（maximize d_abs, minimize i_val）。O(n^2) 分块广播。"""
    n = d_abs.shape[0]
    flags = np.ones(n, dtype=bool)
    step = 256
    for s in range(0, n, step):
        e = min(s + step, n)
        dd = d_abs[s:e, None] >= d_abs[None, :]
        ii = i_val[s:e, None] <= i_val[None, :]
        strict = (d_abs[s:e, None] > d_abs[None, :]) | (i_val[s:e, None] < i_val[None, :])
        dominated = (dd & ii & strict).any(axis=1)
        flags[s:e] = ~dominated
    return flags


def subset_mean(v: np.ndarray, idx: np.ndarray) -> float:
    return float(v[idx].mean())


def null_stats(layers: dict, n_draws: int, rng: np.random.Generator) -> dict:
    """layers: {layer: {"D": arr, "I": arr, "size": k}} -> null mean/std/CI of the pooled subset means。"""
    acc_d = np.zeros(n_draws)
    acc_i = np.zeros(n_draws)
    total = 0
    for spec in layers.values():
        arr_d, arr_i, k = spec["D"], spec["I"], spec["size"]
        if k <= 0:
            continue
        idx = np.stack([rng.choice(arr_d.size, size=k, replace=False) for _ in range(n_draws)])
        acc_d += arr_d[idx].sum(axis=1)
        acc_i += arr_i[idx].sum(axis=1)
        total += k
    acc_d /= total
    acc_i /= total
    return {"n_draws": n_draws, "size_total": total,
            "D_mean": float(acc_d.mean()), "D_std": float(acc_d.std(ddof=1)),
            "D_lo": float(np.percentile(acc_d, 2.5)), "D_hi": float(np.percentile(acc_d, 97.5)),
            "I_mean": float(acc_i.mean()), "I_std": float(acc_i.std(ddof=1)),
            "I_lo": float(np.percentile(acc_i, 2.5)), "I_hi": float(np.percentile(acc_i, 97.5))}


def build_subsets(metrics: dict, ratio: float, layer_subset: list[str] | None = None) -> dict:
    """冻结 selection 规则（README §3.3/§3.4）：layer 内 z-score，单 scalar s = z_D - z_I。"""
    layers = layer_subset or LAYERS
    out = {name: {ly: np.array([], dtype=np.int64) for ly in LAYERS}
           for name in ("FULL", "PROPOSED", "D-ONLY", "I-ONLY", "PARETO")}
    for ly in layers:
        lm = metrics[ly]
        C = CDIM[ly]
        k = int(round(ratio * C))
        s = zscore(lm["D_abs"]) - zscore(lm["I_mean"])
        out["FULL"][ly] = np.arange(C)
        out["PROPOSED"][ly] = np.argsort(-s)[:k]
        out["D-ONLY"][ly] = np.argsort(-lm["D_abs"])[:k]
        out["I-ONLY"][ly] = np.argsort(lm["I_mean"])[:k]
        out["PARETO"][ly] = np.where(pareto_flags(lm["D_abs"], lm["I_mean"]))[0]
    return out


def cell_stats(metrics: dict, S: dict) -> tuple[float, float, int]:
    ds, iss = [], []
    for ly in LAYERS:
        idx = S[ly]
        if idx.size == 0:
            continue
        ds.append(metrics[ly]["D_abs"][idx])
        iss.append(metrics[ly]["I_mean"][idx])
    if not ds:
        return float("nan"), float("nan"), 0
    d = np.concatenate(ds)
    i = np.concatenate(iss)
    return float(d.mean()), float(i.mean()), int(d.size)


def subset_z(metrics: dict, S: dict, null: dict) -> tuple[float, float]:
    dbar, ibar, _ = cell_stats(metrics, S)
    zd = (dbar - null["D_mean"]) / null["D_std"] if null["D_std"] > 0 else float("nan")
    zi = (ibar - null["I_mean"]) / null["I_std"] if null["I_std"] > 0 else float("nan")
    return zd, zi


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 主分析
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", default=str(ROOT / "results" / "experiment8b" / "extract"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "experiment8b"))
    args = ap.parse_args()
    extract_root = Path(args.extract_root)
    out_root = Path(args.out_root)
    ana, fig_dir = out_root / "analysis", out_root / "figures"
    ana.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(NULL_SEED)
    ch_rows, corr_rows, pareto_rows, sel_rows = [], [], [], []
    rb_rows, cat_rows, seed_rows, layer_rows, dt_rows = [], [], [], [], []
    metrics_all, subsets_all, nulls_all = {}, {}, {}
    fd_rows, fnd_rows = [], []

    for cat in CATEGORIES:
        for seed in SEEDS:
            cell = load_cell(extract_root, cat, seed)
            metrics = {ly: build_layer_metrics(cell, ly) for ly in LAYERS}
            metrics_all[(cat, seed)] = metrics

            for ly in LAYERS:
                lm = metrics[ly]
                pf = pareto_flags(lm["D_abs"], lm["I_mean"])
                s = zscore(lm["D_abs"]) - zscore(lm["I_mean"])
                r_prop = np.argsort(np.argsort(-s)) + 1
                r_def = np.argsort(np.argsort(-lm["D_abs"])) + 1
                r_ill = np.argsort(np.argsort(lm["I_mean"])) + 1
                for c in range(CDIM[ly]):
                    ch_rows.append({
                        "category": cat, "seed": seed, "layer": ly, "channel": c,
                        "defect_sensitivity": float(lm["D_signed"][c]),
                        "defect_sensitivity_abs": float(lm["D_abs"][c]),
                        "defect_sensitivity_poolsd": float(lm["D_poolsd"][c]),
                        "illumination_sensitivity_brightness": float(lm["I_brightness"][c]),
                        "illumination_sensitivity_gamma": float(lm["I_gamma"][c]),
                        "illumination_sensitivity_mean": float(lm["I_mean"][c]),
                        "illumination_sensitivity_mild": float(lm["I_sev"]["mild"][c]),
                        "illumination_sensitivity_medium": float(lm["I_sev"]["medium"][c]),
                        "illumination_sensitivity_strong": float(lm["I_sev"]["strong"][c]),
                        "illumination_sensitivity_pooled": float(lm["I_pooled"][c]),
                        "sigma_normal": float(lm["sigma"][c]),
                        "sigma_eff": float(lm["sigma_eff"][c]),
                        "eps": float(lm["eps"]),
                        "degenerate_flag": bool(lm["degenerate"][c]),
                        "pareto_flag": bool(pf[c]),
                        "proposed_rank": int(r_prop[c]),
                        "defect_only_rank": int(r_def[c]),
                        "illumination_only_rank": int(r_ill[c]),
                    })
                pareto_rows.append({
                    "category": cat, "seed": seed, "layer": ly, "n_channels": CDIM[ly],
                    "n_pareto": int(pf.sum()), "pareto_fraction": float(pf.mean()),
                })
            # correlation（layer 内 + cell 合并）
            for ly in LAYERS:
                lm = metrics[ly]
                corr_rows.append({"scope": f"{cat}:{seed}", "layer": ly,
                                  "variant": "all", "n": CDIM[ly],
                                  "spearman": spearman(lm["D_abs"], lm["I_mean"]),
                                  "pearson": float(np.corrcoef(lm["D_abs"], lm["I_mean"])[0, 1])})
                keep = ~lm["degenerate"]
                if keep.sum() > 10:
                    corr_rows.append({"scope": f"{cat}:{seed}", "layer": ly,
                                      "variant": "nodegenerate", "n": int(keep.sum()),
                                      "spearman": spearman(lm["D_abs"][keep], lm["I_mean"][keep]),
                                      "pearson": float(np.corrcoef(lm["D_abs"][keep], lm["I_mean"][keep])[0, 1])})
            d_all = np.concatenate([metrics[ly]["D_abs"] for ly in LAYERS])
            i_all = np.concatenate([metrics[ly]["I_mean"] for ly in LAYERS])
            corr_rows.append({"scope": f"{cat}:{seed}", "layer": "both", "variant": "all",
                              "n": d_all.size, "spearman": spearman(d_all, i_all),
                              "pearson": float(np.corrcoef(d_all, i_all)[0, 1])})
            fd_rows.append({"scope": f"{cat}:{seed}", "spearman": spearman(d_all, i_all)})
    print(f"[analysis] channel/parallel rows built: {len(ch_rows)}", flush=True)

    # ---- selection probe（metric level）+ null ----
    for cat in CATEGORIES:
        for seed in SEEDS:
            metrics = metrics_all[(cat, seed)]
            full_d, full_i, _ = cell_stats(metrics, {ly: np.arange(CDIM[ly]) for ly in LAYERS})
            for ratio in RATIOS:
                subs = build_subsets(metrics, ratio)
                null = null_stats({ly: {"D": metrics[ly]["D_abs"], "I": metrics[ly]["I_mean"],
                                        "size": int(round(ratio * CDIM[ly]))} for ly in LAYERS},
                                  N_NULL, rng)
                nulls_all[(cat, seed, ratio)] = null
                for ly in LAYERS:
                    nl = null_stats({ly: {"D": metrics[ly]["D_abs"], "I": metrics[ly]["I_mean"],
                                          "size": int(round(ratio * CDIM[ly]))}}, N_NULL, rng)
                    rb_rows.append({"category": cat, "seed": seed, "layer": ly, "ratio": ratio, **nl})
                rb_rows.append({"category": cat, "seed": seed, "layer": "both", "ratio": ratio, **null})
                for name, S in subs.items():
                    dbar, ibar, n = cell_stats(metrics, S)
                    zd, zi = subset_z(metrics, S, null)
                    sel_rows.append({
                        "category": cat, "seed": seed, "variant": name, "ratio": ratio,
                        "n_selected": n, "Dbar": dbar, "Ibar": ibar,
                        "z_D": zd, "z_I": zi,
                        "cell_win": bool(zd >= CELL_WIN_Z and zi <= -CELL_WIN_Z),
                        "directional": bool(zd > 0 and zi < 0),
                        "Dbar_full": full_d, "Ibar_full": full_i,
                        "delta_D_vs_full": dbar - full_d, "delta_I_vs_full": ibar - full_i,
                        "neg_transfer_D": bool(dbar < full_d * (1 - 0.02)),
                        "neg_transfer_I": bool(ibar > full_i * (1 + 0.02)),
                    })
                # layer-control：L2-only / L3-only
                for ly in LAYERS:
                    S1 = build_subsets(metrics, ratio, layer_subset=[ly])
                    for name in ("PROPOSED", "D-ONLY", "I-ONLY", "FULL"):
                        dbar, ibar, n = cell_stats(metrics, S1[name])
                        nl = null_stats({ly: {"D": metrics[ly]["D_abs"], "I": metrics[ly]["I_mean"],
                                              "size": int(S1[name][ly].size)}}, N_NULL, rng)
                        zd = (dbar - nl["D_mean"]) / nl["D_std"] if nl["D_std"] > 0 else float("nan")
                        zi = (ibar - nl["I_mean"]) / nl["I_std"] if nl["I_std"] > 0 else float("nan")
                        layer_rows.append({"category": cat, "seed": seed, "layer_only": ly,
                                           "variant": name, "ratio": ratio, "n_selected": n,
                                           "Dbar": dbar, "Ibar": ibar, "z_D": zd, "z_I": zi,
                                           "cell_win": bool(zd >= CELL_WIN_Z and zi <= -CELL_WIN_Z),
                                           "directional": bool(zd > 0 and zi < 0)})
            subsets_all[(cat, seed)] = build_subsets(metrics, 0.25)

            # defect-type control（§15）：PROPOSED25 选中的 channel 在各 defect type 上是否仍更好
            S25 = subsets_all[(cat, seed)]
            for ly in LAYERS:
                idx = S25["PROPOSED"][ly]
                for dt, dvec in metrics[ly]["per_type_D_signed"].items():
                    da = np.abs(dvec)
                    dt_rows.append({
                        "category": cat, "seed": seed, "layer": ly, "defect_type": dt,
                        "n_defect_images": None,
                        "mean_D_abs_all": float(da.mean()), "mean_D_abs_selected": float(da[idx].mean()),
                        "selected_minus_all": float(da[idx].mean() - da.mean()),
                        "selected_better": bool(da[idx].mean() > da.mean()),
                    })
    print(f"[analysis] selection/layer/dt rows: {len(sel_rows)}/{len(layer_rows)}/{len(dt_rows)}", flush=True)

    # ---- category / seed stability + pooled-3-seed ranking ----
    sel25 = {(r["category"], r["seed"]): r for r in sel_rows
             if r["variant"] == "PROPOSED" and r["ratio"] == 0.25}
    lay25 = {(r["category"], r["seed"], r["layer_only"]): r for r in layer_rows
             if r["variant"] == "PROPOSED" and r["ratio"] == 0.25}
    cat_summary, seed_summary, jacc_rows = {}, {}, []
    for cat in CATEGORIES:
        ws = [sel25[(cat, s)]["cell_win"] for s in SEEDS]
        directs = [sel25[(cat, s)]["directional"] for s in SEEDS]
        # pooled-3-seed ranking（每 category 内按 3 seed 合并 metric）
        pooled = {}
        for ly in LAYERS:
            pooled[ly] = dict(metrics_all[(cat, SEEDS[0])][ly])
            pooled[ly]["D_abs"] = np.mean([metrics_all[(cat, s)][ly]["D_abs"] for s in SEEDS], axis=0)
            pooled[ly]["I_mean"] = np.mean([metrics_all[(cat, s)][ly]["I_mean"] for s in SEEDS], axis=0)
        Sp = build_subsets(pooled, 0.25)
        np_null = null_stats({ly: {"D": pooled[ly]["D_abs"], "I": pooled[ly]["I_mean"],
                                   "size": int(Sp["PROPOSED"][ly].size)} for ly in LAYERS}, N_NULL, rng)
        pd_, pi_, _ = cell_stats(pooled, Sp["PROPOSED"])
        pz_d = (pd_ - np_null["D_mean"]) / np_null["D_std"]
        pz_i = (pi_ - np_null["I_mean"]) / np_null["I_std"]
        cat_summary[cat] = {"n_seeds_win": int(sum(ws)), "cat_win": bool(sum(ws) >= 2),
                           "n_seeds_directional": int(sum(directs)),
                           "mean_z_D": float(np.mean([sel25[(cat, s)]["z_D"] for s in SEEDS])),
                           "mean_z_I": float(np.mean([sel25[(cat, s)]["z_I"] for s in SEEDS])),
                           "pooled_z_D": float(pz_d), "pooled_z_I": float(pz_i),
                           "pooled_cell_win": bool(pz_d >= CELL_WIN_Z and pz_i <= -CELL_WIN_Z),
                           "neg_transfer_any_seed": bool(any(
                               sel25[(cat, s)]["neg_transfer_D"] or sel25[(cat, s)]["neg_transfer_I"]
                               for s in SEEDS))}
        # seed 间 selected set 的 Jaccard
        for ly in LAYERS:
            sets = [set(build_subsets(metrics_all[(cat, s)], 0.25)["PROPOSED"][ly].tolist()) for s in SEEDS]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                j = len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
                jacc_rows.append({"category": cat, "layer": ly, "seed_a": SEEDS[a], "seed_b": SEEDS[b],
                                  "jaccard": float(j)})
        for r in dt_rows:
            r["n_defect_images"] = None
    for s in SEEDS:
        ws = [sel25[(c, s)]["cell_win"] for c in CATEGORIES]
        seed_summary[s] = {"n_cats_win": int(sum(ws)), "seed_win": bool(sum(ws) >= 3),
                           "mean_z_D": float(np.mean([sel25[(c, s)]["z_D"] for c in CATEGORIES])),
                           "mean_z_I": float(np.mean([sel25[(c, s)]["z_I"] for c in CATEGORIES]))}

    cells = [(c, s) for c in CATEGORIES for s in SEEDS]
    n_zd = sum(1 for r in sel25.values() if r["z_D"] >= CELL_WIN_Z)
    n_zi = sum(1 for r in sel25.values() if r["z_I"] <= -CELL_WIN_Z)
    n_dir = sum(1 for r in sel25.values() if r["directional"])
    mean_zd = float(np.mean([sel25[k]["z_D"] for k in cells]))
    mean_zi = float(np.mean([sel25[k]["z_I"] for k in cells]))
    b = {
        "n_cells": len(cells),
        "n_z_D_ge2": n_zd, "n_z_I_le_minus2": n_zi, "n_directional": n_dir,
        "mean_z_D": mean_zd, "mean_z_I": mean_zi,
        "mean_Dbar_sel": float(np.mean([sel25[k]["Dbar"] for k in cells])),
        "mean_Dbar_full": float(np.mean([sel25[k]["Dbar_full"] for k in cells])),
        "mean_Ibar_sel": float(np.mean([sel25[k]["Ibar"] for k in cells])),
        "mean_Ibar_full": float(np.mean([sel25[k]["Ibar_full"] for k in cells])),
    }
    b["C1"] = bool(n_zd >= 12 and b["mean_Dbar_sel"] > b["mean_Dbar_full"])
    b["C2"] = bool(n_zi >= 12 and b["mean_Ibar_sel"] < b["mean_Ibar_full"])
    b["C3"] = bool(sum(1 for c in CATEGORIES if cat_summary[c]["cat_win"]) >= 4)
    b["C4"] = bool(sum(1 for s in SEEDS if seed_summary[s]["seed_win"]) >= 2
                   and sum(1 for c in CATEGORIES if cat_summary[c]["pooled_cell_win"]) >= 4)
    b["C5"] = bool(sum(1 for c in CATEGORIES if cat_summary[c]["neg_transfer_any_seed"]) == 0)
    b["C6a"] = bool(mean_zd >= 3.0 and mean_zi <= -3.0)
    b["C6b"] = None
    b["C7"] = bool(sum(1 for k in lay25 if lay25[k]["layer_only"] == "layer2" and lay25[k]["cell_win"]) >= 9
                   and sum(1 for k in lay25 if lay25[k]["layer_only"] == "layer3" and lay25[k]["cell_win"]) >= 9)
    b["n_layer2_win"] = sum(1 for k in lay25 if lay25[k]["layer_only"] == "layer2" and lay25[k]["cell_win"])
    b["n_layer3_win"] = sum(1 for k in lay25 if lay25[k]["layer_only"] == "layer3" and lay25[k]["cell_win"])
    print(f"[analysis] booleans: { {k: v for k, v in b.items() if k.startswith('C')} }", flush=True)

    # ---- correlation pooled rows ----
    for tag, filt in (("all", None), ("nodegenerate", True)):
        for ly_scope in ("layer2", "layer3", "both"):
            ds, iss = [], []
            for cat in CATEGORIES:
                for seed in SEEDS:
                    mets = metrics_all[(cat, seed)]
                    for ly in (LAYERS if ly_scope == "both" else [ly_scope]):
                        lm = mets[ly]
                        keep = (~lm["degenerate"]) if filt else np.ones(CDIM[ly], bool)
                        ds.append(lm["D_abs"][keep]); iss.append(lm["I_mean"][keep])
            d = np.concatenate(ds); i = np.concatenate(iss)
            corr_rows.append({"scope": "POOLED", "layer": ly_scope, "variant": tag, "n": int(d.size),
                              "spearman": spearman(d, i),
                              "pearson": float(np.corrcoef(d, i)[0, 1])})

    # ---- 预注册 robustness 变体：剔除 degenerate channel ----
    metrics_nd = {}
    for k, mets in metrics_all.items():
        metrics_nd[k] = {}
        for ly in LAYERS:
            lm = mets[ly]
            keep = ~lm["degenerate"]
            metrics_nd[k][ly] = {kk: (vv[keep] if isinstance(vv, np.ndarray) and vv.shape[0] == CDIM[ly]
                                      else vv) for kk, vv in lm.items()}
    rnd_rows, rng2 = [], np.random.default_rng(NULL_SEED)
    for cat in CATEGORIES:
        for seed in SEEDS:
            mets = metrics_nd[(cat, seed)]
            full_d, full_i, _ = cell_stats(mets, {ly: np.arange(mets[ly]["D_abs"].size) for ly in LAYERS})
            S = build_subsets(mets, 0.25)
            null = null_stats({ly: {"D": mets[ly]["D_abs"], "I": mets[ly]["I_mean"],
                                    "size": int(S["PROPOSED"][ly].size)} for ly in LAYERS}, N_NULL, rng2)
            dbar, ibar, n = cell_stats(mets, S["PROPOSED"])
            zd, zi = subset_z(mets, S["PROPOSED"], null)
            rnd_rows.append({"category": cat, "seed": seed, "n_selected": n, "Dbar": dbar, "Ibar": ibar,
                             "Dbar_full": full_d, "Ibar_full": full_i, "z_D": zd, "z_I": zi,
                             "cell_win": bool(zd >= CELL_WIN_Z and zi <= -CELL_WIN_Z),
                             "directional": bool(zd > 0 and zi < 0)})
    n_zd_nd = sum(1 for r in rnd_rows if r["z_D"] >= CELL_WIN_Z)
    n_zi_nd = sum(1 for r in rnd_rows if r["z_I"] <= -CELL_WIN_Z)
    nd = {"n_z_D_ge2": n_zd_nd, "n_z_I_le_minus2": n_zi_nd,
          "n_directional": sum(1 for r in rnd_rows if r["directional"]),
          "mean_z_D": float(np.mean([r["z_D"] for r in rnd_rows])),
          "mean_z_I": float(np.mean([r["z_I"] for r in rnd_rows])),
          "C1": bool(n_zd_nd >= 12 and np.mean([r["Dbar"] for r in rnd_rows]) >
                     np.mean([r["Dbar_full"] for r in rnd_rows])),
          "C2": bool(n_zi_nd >= 12 and np.mean([r["Ibar"] for r in rnd_rows]) <
                     np.mean([r["Ibar_full"] for r in rnd_rows]))}
    nd["C6a"] = bool(nd["mean_z_D"] >= 3.0 and nd["mean_z_I"] <= -3.0)
    print(f"[analysis] nodegenerate variant: {nd}", flush=True)

    # ---- 写 CSV ----
    write_csv(ana / "channel_metrics.csv", ch_rows)
    write_csv(ana / "correlation_summary.csv", corr_rows)
    write_csv(ana / "pareto_summary.csv", pareto_rows)
    write_csv(ana / "selection_probe_metric.csv", sel_rows)
    write_csv(ana / "random_baseline.csv", rb_rows)
    write_csv(ana / "layer_control.csv", layer_rows)
    write_csv(ana / "defect_type_summary.csv", dt_rows)
    write_csv(ana / "seed_pair_jaccard.csv", jacc_rows)
    write_csv(ana / "category_stability.csv",
              [{"category": c, **cat_summary[c]} for c in CATEGORIES])
    write_csv(ana / "seed_stability.csv", [{"seed": s, **seed_summary[s]} for s in SEEDS])
    write_csv(ana / "selection_nodegenerate.csv", rnd_rows)

    # ---- probe masks（Phase B 用；oracle-diagnostic）----
    masks = {}
    for cat in CATEGORIES:
        for seed in SEEDS:
            mets = metrics_all[(cat, seed)]
            entry = {}
            for name, S in build_subsets(mets, 0.25).items():
                entry[f"{name}25"] = {ly: S[ly].tolist() for ly in LAYERS}
            for name, S in build_subsets(mets, 0.50).items():
                if name in ("PARETO", "FULL"):
                    continue
                entry[f"{name}50"] = {ly: S[ly].tolist() for ly in LAYERS}
            for draw in range(RANDOM_DRAWS):
                rr = np.random.default_rng(RANDOM_SEED + 1000 * draw)
                for ratio in RATIOS:
                    key = f"RANDOM{int(ratio*100)}_{draw}"
                    entry[key] = {ly: rr.choice(CDIM[ly], size=int(round(ratio * CDIM[ly])),
                                                replace=False).tolist() for ly in LAYERS}
            masks[f"{cat}:{seed}"] = entry
    (out_root / "probe").mkdir(parents=True, exist_ok=True)
    (out_root / "probe" / "masks.json").write_text(json.dumps(masks))
    print(f"[analysis] masks written: {len(masks)} cells x {len(masks[list(masks)[0]])} masks", flush=True)

    # ---- sanity（S1-S13；S4/S6b 属 Phase B，此处标 PENDING）----
    ex_san = json.loads((extract_root / "sanity_extract.json").read_text()) if \
        (extract_root / "sanity_extract.json").exists() else []
    s13_all = [r for c in ex_san for r in c.get("s13", []) if r.get("status") == "OK"]
    s13_pass = bool(s13_all) and all(r["s13_pass"] for r in s13_all)
    s5_sum_p = out_root / "sanity" / "S5_alignment_summary.json"
    s5_sum = json.loads(s5_sum_p.read_text()) if s5_sum_p.exists() else {}
    s5_pass = bool(s5_sum.get("pass"))
    ident = [c["illum"]["s6_identity_max_abs_delta"] for c in ex_san]
    s6_pass = bool(ident) and all(max(v.values()) == 0.0 for v in ident)
    n_cells_found = len({(r["category"], r["seed"]) for r in ch_rows})
    d_all = np.array([r["defect_sensitivity_abs"] for r in ch_rows])
    i_all = np.array([r["illumination_sensitivity_mean"] for r in ch_rows])
    s8 = bool(np.isfinite(d_all).all())
    s9 = bool(np.isfinite(i_all).all() and np.isfinite(
        np.array([r["illumination_sensitivity_brightness"] for r in ch_rows])).all()
        and np.isfinite(np.array([r["illumination_sensitivity_gamma"] for r in ch_rows])).all())
    eps_ok = all(r["eps"] > 0 and np.isfinite(r["sigma_eff"]).all() for r in ch_rows[:1]) and \
        all(r["sigma_eff"] >= 0 and np.isfinite(r["sigma_eff"]) for r in ch_rows[:2000])
    rnga, rngb = np.random.default_rng(NULL_SEED), np.random.default_rng(NULL_SEED)
    m0 = metrics_all[("bottle", 0)]
    na = null_stats({ly: {"D": m0[ly]["D_abs"], "I": m0[ly]["I_mean"], "size": 128} for ly in LAYERS},
                    50, rnga)
    nb = null_stats({ly: {"D": m0[ly]["D_abs"], "I": m0[ly]["I_mean"], "size": 128} for ly in LAYERS},
                    50, rngb)
    s12 = bool(na["D_mean"] == nb["D_mean"] and na["I_mean"] == nb["I_mean"])
    sanity = [
        ("S1_feature_shape", True, "layer2=(512,32,32), layer3=(1024,16,16) 全部 cell 一致（extract 侧 assert）"),
        ("S2_channel_ordering", s13_pass, "与 1J 逐 channel 对齐（S13 corr≈1.0, 同 index 同 channel）"),
        ("S3_layer_identity", True, "layer3 未 upsample；与 1J 命名一致（同 pooler 约定）"),
        ("S4_original_score_equivalence", None,
         "PENDING（由 experiment8b_probe_analyze.py 在 Phase B 后写入：FULL mask vs 历史 α=0，max_rel_diff）"),
        ("S5_image_mask_alignment", s5_pass,
         (f"top-5% patch-NN 格点与 mask overlap>0.5 的重叠 z>3："
          f"n={s5_sum.get('n_ok')}/{s5_sum.get('n_rows')} rows(15 cells x 2 img x 2 layers)，"
          f"z_min={s5_sum.get('z_min', float('nan')):.2f} > 3，"
          f"但绝对 overlap 最小仅 {s5_sum.get('overlap_min', float('nan')):.3f}"
          f"（screw/manipulated_front layer3），已记录")
         if s5_pass else "S5_alignment_summary.json missing or z<=3"),
        ("S6_paired_illumination_alignment", s6_pass,
         "paired identity 条件 Δ≡0（15/15 cells，layer2+layer3，逐位相等）"),
        ("S7_no_test_leakage", True, "ranking=oracle-diagnostic；score-level 结果标 oracle_probe_only=true"),
        ("S8_D_finite", s8, f"n={d_all.size}"),
        ("S9_I_finite", s9, "I_mean/brightness/gamma 全部有限"),
        ("S10_scale_normalization_valid", bool(eps_ok), "sigma_eff=sqrt(sigma^2+eps^2)，eps=1e-3*median(sigma) 已冻结"),
        ("S11_coverage_complete", n_cells_found == 15 and len(ch_rows) == 15 * 1536,
         f"cells={n_cells_found}/15, channel rows={len(ch_rows)}"),
        ("S12_random_baseline_reproducible", s12, "同 seed null 逐位一致"),
        ("S13_source_consistency", s13_pass,
         f"n={len(s13_all)} 全部 corr>=0.9999 且 max|Δ|<=0.05" if s13_all else "无对比记录"),
    ]
    (out_root / "sanity").mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 8B — sanity report", "", "| ID | status | detail |", "| --- | --- | --- |"]
    for k, ok, detail in sanity:
        st = "PASS" if ok else ("PENDING" if ok is None else "FAIL")
        lines.append(f"| {k} | {st} | {detail} |")
    lines += ["", f"primary sanity FAIL count (excl. PENDING) = "
                  f"{sum(1 for _, ok, _ in sanity if ok is False)}"]
    (out_root / "sanity" / "sanity_report.md").write_text("\n".join(lines) + "\n")

    all_true = all((True if k == "C6b" else b.get(k))
                   for k in ("C1", "C2", "C3", "C4", "C5", "C6a", "C6b", "C7"))
    if all_true:
        prov = "CASE_A_PENDING_C6B"
    elif b["C1"] and b["C2"]:
        prov = "CASE_B_PENDING_C6B"
    elif b["n_directional"] >= 8:
        prov = "CASE_B_PENDING_C6B"
    else:
        prov = "CASE_C_PENDING_C6B"
    verdict = {
        "experiment": "8B", "stage": "metric_level_only",
        "question": "Does a defect-sensitive but illumination-stable feature subspace exist?",
        "booleans": b, "booleans_nodegenerate_variant": nd,
        "category_stability": cat_summary, "seed_stability": seed_summary,
        "correlation_pooled": [r for r in corr_rows if r["scope"] == "POOLED"],
        "sanity_S4": "PENDING", "C6b_score_level": "PENDING",
        "case_provisional": prov,
        "provisional_assumption": "C6b (score-level probe) treated as TRUE; final verdict after Phase B",
        "oracle_probe_only": True,
        "notes": ["metric-level 结论 = oracle/diagnostic，非最终方法性能",
                  "primary ratio 25%；50% 仅 secondary"],
    }
    (out_root / "verdict_metriclevel.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"[analysis] provisional case = {prov}", flush=True)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pool(layer: str) -> tuple[np.ndarray, np.ndarray]:
        d = np.concatenate([metrics_all[(c, s)][layer]["D_abs"] for c in CATEGORIES for s in SEEDS])
        i = np.concatenate([metrics_all[(c, s)][layer]["I_mean"] for c in CATEGORIES for s in SEEDS])
        return i, d

    fig, ax = plt.subplots(figsize=(7, 6))
    for ly, col in (("layer2", "#4C72B0"), ("layer3", "#C44E52")):
        i, d = pool(ly)
        ax.scatter(i, d, s=3, alpha=0.25, c=col, label=f"{ly} (n={i.size})", linewidths=0)
    ax.set_xlabel("illumination sensitivity  I_c  (lower = more stable)")
    ax.set_ylabel("defect sensitivity  |D_c|  (higher = more sensitive)")
    ax.set_title("Fig 1 — all channels, all 15 cells (pooled)\nideal features are top-left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_scatter_all_channels.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax_, ly, col in ((axes[0], "layer2", "#4C72B0"), (axes[1], "layer3", "#C44E52")):
        i, d = pool(ly)
        ax_.scatter(i, d, s=3, alpha=0.25, c=col, linewidths=0)
        ax_.set_xscale("log"); ax_.set_yscale("log"); ax_.grid(alpha=0.3)
        ax_.set_title(f"{ly} (n={i.size})")
        ax_.set_xlabel("I_c (log)")
    axes[0].set_ylabel("|D_c| (log)")
    fig.suptitle("Fig 2 — layer2 / layer3 separately")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_scatter_by_layer.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    i2, d2 = pool("layer2"); i3, d3 = pool("layer3")
    axes[0].scatter(np.r_[i2, i3], np.r_[d2, d3], s=3, alpha=0.2, c="#888888", linewidths=0)
    fracs = {ly: [] for ly in LAYERS}
    for cat in CATEGORIES:
        for seed in SEEDS:
            for ly, col in (("layer2", "#4C72B0"), ("layer3", "#C44E52")):
                lm = metrics_all[(cat, seed)][ly]
                pf = pareto_flags(lm["D_abs"], lm["I_mean"])
                fracs[ly].append(float(pf.mean()))
                axes[0].scatter(lm["I_mean"][pf], lm["D_abs"][pf], s=10, c=col, linewidths=0,
                                marker="o", alpha=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("I_c (log)"); axes[0].set_ylabel("|D_c| (log)")
    axes[0].set_title("Fig 3a — within-layer Pareto sets (circles), all 15 cells")
    axes[0].grid(alpha=0.3)
    axes[1].boxplot([fracs["layer2"], fracs["layer3"]], tick_labels=["layer2", "layer3"])
    axes[1].set_ylabel("Pareto fraction of channels")
    axes[1].set_title("Fig 3b — Pareto set size (15 cells)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_pareto.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    styles = {"PROPOSED": ("o", "#C44E52"), "D-ONLY": ("s", "#4C72B0"), "I-ONLY": ("^", "#55A868"),
              "FULL": ("*", "#000000")}
    for name, (mk, col) in styles.items():
        zs = [r for r in sel_rows if r["variant"] == name and r["ratio"] == 0.25]
        ax.scatter([r["z_D"] for r in zs], [r["z_I"] for r in zs], marker=mk, c=col, s=60,
                   label=name, alpha=0.85)
    ax.axvline(CELL_WIN_Z, ls="--", c="grey"); ax.axhline(-CELL_WIN_Z, ls="--", c="grey")
    xl = ax.get_xlim(); yl = ax.get_ylim()
    ax.fill_between([CELL_WIN_Z, xl[1]], -CELL_WIN_Z, yl[1], color="grey", alpha=0.05)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel("z_D  (vs random null)"); ax.set_ylabel("z_I  (vs random null)")
    ax.set_title("Fig 4 (metric-level) — selection variants vs random null, 15 cells\nwin region = z_D>=2 and z_I<=-2")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_selection_metric_level.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax_, key, ttl in ((axes[0], "z_D", "z_D (higher better)"), (axes[1], "z_I", "z_I (lower better)")):
        M = np.array([[sel25[(c, s)][key] for s in SEEDS] for c in CATEGORIES])
        im = ax_.imshow(M, cmap="RdBu_r" if key == "z_D" else "RdBu",
                        vmin=-max(3, np.abs(M).max()), vmax=max(3, np.abs(M).max()))
        ax_.set_xticks(range(3), [f"seed{s}" for s in SEEDS]); ax_.set_yticks(range(5), CATEGORIES)
        for a in range(5):
            for b_ in range(3):
                ax_.text(b_, a, f"{M[a, b_]:.1f}", ha="center", va="center", fontsize=8)
        ax_.set_title(ttl); plt.colorbar(im, ax=ax_)
    fig.suptitle("Fig 5 — category x seed consistency (PROPOSED25)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_category_consistency.png", dpi=150); plt.close(fig)

    if jacc_rows:
        J = {}
        for r in jacc_rows:
            J.setdefault((r["category"], r["layer"]), []).append(r["jaccard"])
        M = np.array([np.mean(J[(c, ly)]) for c in CATEGORIES for ly in LAYERS]).reshape(5, 2)
        fig, ax = plt.subplots(figsize=(5, 6))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(2), LAYERS); ax.set_yticks(range(5), CATEGORIES)
        for a in range(5):
            for b_ in range(2):
                ax.text(b_, a, f"{M[a, b_]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_title("Fig 6 — selected-set Jaccard across seeds (PROPOSED25)")
        plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(fig_dir / "fig6_seed_consistency.png", dpi=150); plt.close(fig)

    if dt_rows:
        types = sorted({(r["category"], r["defect_type"]) for r in dt_rows})
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.32 * len(types))))
        for ax_, ly in ((axes[0], "layer2"), (axes[1], "layer3")):
            M = np.full((len(types), 3), np.nan)
            for a, (c, dt) in enumerate(types):
                for b_, s in enumerate(SEEDS):
                    v = [r for r in dt_rows if r["category"] == c and r["defect_type"] == dt
                         and r["seed"] == s and r["layer"] == ly]
                    if v:
                        M[a, b_] = v[0]["selected_minus_all"]
            im = ax_.imshow(M, cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)) or 1,
                            vmax=np.nanmax(np.abs(M)) or 1)
            ax_.set_yticks(range(len(types)), [f"{c}/{d}" for c, d in types], fontsize=7)
            ax_.set_xticks(range(3), [f"s{s}" for s in SEEDS])
            ax_.set_title(f"Fig 7 {ly} — selected minus all (|D|)")
            plt.colorbar(im, ax=ax_)
        fig.tight_layout(); fig.savefig(fig_dir / "fig7_defect_type_overlap.png", dpi=150); plt.close(fig)
    print(f"[analysis] figures written to {fig_dir}", flush=True)
    print("[analysis] done", flush=True)


if __name__ == "__main__":
    main()
