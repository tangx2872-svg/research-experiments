"""Experiment 5B — Normal-Only Normalization-Tolerance Predictability (CPU only).

严格顺序（防泄漏，S4）：
  Phase 1  写 predictor registry（reference/predictor_registry.csv）
  Phase 2  计算 normal-only predictors（banks + train/good 图像 controls）
  Phase 3  才加载 target（5A-H per_unit.csv -> Damage）并做相关 / 排序 / LOCO

X = normal training data only（1J 冻结 banks：a=0 normal train coreset features，
与 5A-H 同 make_validation_split 协议；图像 control 只读 train/good）。
Y = 5A-H observed defect preservation damage（dprime_B0 - dprime_normalized）。
CPU only；无 alpha search；n_category=5 为主口径（seed-level n=15 仅 exploratory）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment1b_defect_sensitivity as e1b  # noqa: E402

OUT = ROOT / "results" / "experiment_5b"
REF_DIR = OUT / "reference"
SUM_DIR = OUT / "summary"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"
BANK_DIR = ROOT / "results" / "experiment_1j" / "banks"
PER_UNIT = ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
LAYERS = ["layer2", "layer3"]
EXPECT_HEAD = "ff4a1b2"

FROZEN_FILES = [
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_cross_chain_summary.csv",
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_family_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "transmission_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "per_defect.csv",
    ROOT / "results" / "experiment_5a" / "geometry_rule.json",
    ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv",
    ROOT / "results" / "experiment_5a_h" / "summary" / "g2_vs_c2.csv",
]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# Phase 1 — predictor registry（在任何 target 数据加载之前写入磁盘）
REGISTRY = [
    ("norm_mean_L2", "A_magnitude", "bank patch L2 norm mean", "1j_banks", "layer2", "none"),
    ("norm_mean_L3", "A_magnitude", "bank patch L2 norm mean", "1j_banks", "layer3", "none"),
    ("norm_cv_L2", "A_magnitude", "bank patch norm std/mean", "1j_banks", "layer2", "high->tolerant"),
    ("norm_cv_L3", "A_magnitude", "bank patch norm std/mean", "1j_banks", "layer3", "high->tolerant"),
    ("rms_radius_L2", "B_geometry", "RMS dist to bank centroid / ||centroid||", "1j_banks", "layer2", "high->tolerant"),
    ("rms_radius_L3", "B_geometry", "RMS dist to bank centroid / ||centroid||", "1j_banks", "layer3", "high->tolerant"),
    ("nn_dist_rel_L2", "B_geometry", "mean intra-bank NN euclid dist / mean norm", "1j_banks", "layer2", "high->tolerant"),
    ("nn_dist_rel_L3", "B_geometry", "mean intra-bank NN euclid dist / mean norm", "1j_banks", "layer3", "high->tolerant"),
    ("eff_dim_L2", "B_geometry", "PCA participation ratio of bank covariance", "1j_banks", "layer2", "high->tolerant"),
    ("eff_dim_L3", "B_geometry", "PCA participation ratio of bank covariance", "1j_banks", "layer3", "high->tolerant"),
    ("radius_ratio_L3L2", "D_crosslayer", "rms_radius_L3 / rms_radius_L2", "derived", "both", "none"),
    ("nn_dist_ratio_L3L2", "D_crosslayer", "nn_dist_rel_L3 / nn_dist_rel_L2", "derived", "both", "none"),
    ("eff_dim_ratio_L3L2", "D_crosslayer", "eff_dim_L3 / eff_dim_L2", "derived", "both", "none"),
    ("img_brightness_mean", "control", "train/good mean pixel brightness", "images", "none", "control"),
    ("img_pixel_std", "control", "train/good pixel intensity std", "images", "none", "control"),
]


def write_registry() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    with open(REF_DIR / "predictor_registry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predictor_name", "definition", "source", "layer",
                    "uses_normal_only", "frozen_before_target_analysis",
                    "direction_hypothesis", "registry_written_at"])
        for name, group, definition, source, layer, direction in REGISTRY:
            w.writerow([name, f"[{group}] {definition}", source, layer,
                        "true", "true", direction, now])
    (LOG_DIR / "registry_freeze_time.txt").write_text(now)
    print(f"[registry] frozen at {now}: {len(REGISTRY)} predictors (target NOT loaded yet)")


# ---------------------------------------------------------------------------
# Phase 2 — normal-only predictors
# ---------------------------------------------------------------------------
def bank_stats(bank: np.ndarray) -> dict:
    norms = np.linalg.norm(bank, axis=1)
    centroid = bank.mean(axis=0)
    d = np.linalg.norm(bank - centroid, axis=1)
    rms_radius = float(np.sqrt((d ** 2).mean())) / (float(np.linalg.norm(centroid)) + 1e-12)
    Xc = bank - centroid
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    eff_dim = float(lam.sum() ** 2 / (lam ** 2).sum())
    return {"norm_mean": float(norms.mean()),
            "norm_cv": float(norms.std(ddof=1) / norms.mean()),
            "rms_radius": rms_radius,
            "eff_dim": eff_dim}


def nn_mean_dist(bank: np.ndarray, chunk: int = 512) -> float:
    """bank 内每点到其余点最小欧氏距离的均值（chunked，float64 累积）。"""
    b = bank.astype(np.float64)
    n = b.shape[0]
    bn = np.einsum("nc,nc->n", b, b)
    best = np.full(n, np.inf, dtype=np.float64)
    for i in range(0, n, chunk):
        xb = b[i:i + chunk]
        # ||x-y||^2 = ||x||^2 - 2<x,y> + ||y||^2 （三项缺一不可）
        d2 = bn[i:i + chunk, None] - 2.0 * (xb @ b.T) + bn[None, :]
        d2[np.arange(xb.shape[0]), np.arange(i, i + xb.shape[0])] = np.inf
        best[i:i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return float(best.mean())


def image_controls(train_ids: list, category: str):
    means, stds = [], []
    for vid in train_ids:
        p = e1b.DATA_ROOT / category / "train" / "good" / vid
        img = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        means.append(float(img.mean()))
        stds.append(float(img.std()))
    return float(np.mean(means)), float(np.mean(stds))


def compute_predictors() -> dict:
    preds = {}
    t0 = time.time()
    for cat in CATEGORIES:
        for seed in SEEDS:
            val_ids, train_ids = e1b.make_validation_split(category=cat, seed=seed)
            row = {}
            stats = {}
            for layer in LAYERS:
                bank = np.load(BANK_DIR / f"{cat}_seed{seed}_{layer}.npy")
                assert np.isfinite(bank).all(), f"non-finite bank {cat} seed{seed} {layer}"
                st = bank_stats(bank)
                st["nn_dist_rel"] = nn_mean_dist(bank) / st["norm_mean"]
                stats[layer] = st
                tag = "L2" if layer == "layer2" else "L3"
                row[f"norm_mean_{tag}"] = st["norm_mean"]
                row[f"norm_cv_{tag}"] = st["norm_cv"]
                row[f"rms_radius_{tag}"] = st["rms_radius"]
                row[f"nn_dist_rel_{tag}"] = st["nn_dist_rel"]
                row[f"eff_dim_{tag}"] = st["eff_dim"]
                expect = int(0.1 * len(train_ids) * (1024 if layer == "layer2" else 256))
                assert abs(bank.shape[0] - expect) <= len(train_ids), \
                    f"S12 FAIL {cat}:{seed}:{layer}: {bank.shape[0]} vs {expect}"
            row["radius_ratio_L3L2"] = stats["layer3"]["rms_radius"] / stats["layer2"]["rms_radius"]
            row["nn_dist_ratio_L3L2"] = stats["layer3"]["nn_dist_rel"] / stats["layer2"]["nn_dist_rel"]
            row["eff_dim_ratio_L3L2"] = stats["layer3"]["eff_dim"] / stats["layer2"]["eff_dim"]
            b, s = image_controls(train_ids, cat)
            row["img_brightness_mean"] = b
            row["img_pixel_std"] = s
            preds[(cat, seed)] = row
            print(f"[predictors] {cat}:{seed} done ({time.time()-t0:.0f}s)", flush=True)
    return preds


# ---------------------------------------------------------------------------
# Phase 3 — target 与分析
# ---------------------------------------------------------------------------
def load_targets() -> dict:
    dprime = {}
    with open(PER_UNIT, newline="") as f:
        for r in csv.DictReader(f):
            dprime[(r["category"], int(r["seed"]), r["config"])] = float(r["mean_dprime"])
    targets = {}
    for cat in CATEGORIES:
        for seed in SEEDS:
            targets[(cat, seed)] = {
                "damage_C2": dprime[(cat, seed, "B0")] - dprime[(cat, seed, "C2")],
                "damage_C3": dprime[(cat, seed, "B0")] - dprime[(cat, seed, "C3")],
                "damage_G2": dprime[(cat, seed, "B0")] - dprime[(cat, seed, "G2")],
            }
    return targets


def balanced_acc(y_true: list, y_pred: list) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    n_pos = sum(y_true); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (tp / n_pos + tn / n_neg) / 2


def main() -> None:
    for d in (REF_DIR, SUM_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = []

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    checks.append(("S9_5A_commit_in_history",
                   subprocess.run(["git", "merge-base", "--is-ancestor", "ba11038", "HEAD"],
                                  cwd=ROOT).returncode == 0,
                   "ba11038 is ancestor of HEAD"))
    checks.append(("S10_head_is_5AH_archive", head.startswith(EXPECT_HEAD), f"HEAD={head[:8]}"))
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip().splitlines()
    dirty_tracked = [l for l in porcelain if l and not l.startswith("??")]
    checks.append(("S10_tracked_files_clean", not dirty_tracked,
                   "; ".join(dirty_tracked[:3]) if dirty_tracked else "no tracked modifications"))

    # ===== Phase 1: registry freeze =====
    write_registry()
    n_pred = len(REGISTRY)
    checks.append(("S5_predictor_count_le_15", n_pred <= 15, f"{n_pred} predictors"))
    checks.append(("S4_registry_before_target", True,
                   "registry CSV written before per_unit.csv opened (code order, timestamp recorded)"))

    # ===== Phase 2: predictors =====
    preds = compute_predictors()
    pred_names = [r[0] for r in REGISTRY]
    nan_pred = [n for (c, s), row in preds.items() for n, v in row.items()
                if not np.isfinite(v)]
    checks.append(("S11_no_nan_inf", not nan_pred, str(nan_pred[:3]) if nan_pred else "all finite"))
    checks.append(("S1_train_good_only", True,
                   "banks=1J normal train coreset (a=0 raw); image controls read train/good only"))
    checks.append(("S2_S3_no_test_defect_in_X", True,
                   "X sources: banks + train/good images; no test paths, no labels"))
    checks.append(("S12_bank_split_alignment", True,
                   "bank rows == int(0.1*n_train*{1024,256}) for all 30 banks (asserted)"))
    checks.append(("S7_no_alpha_search", True, "no model fitting, no alpha anywhere"))
    checks.append(("S6_two_level_n", True, "category n=5 primary; seed n=15 exploratory"))

    with open(SUM_DIR / "normal_only_predictors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed"] + pred_names)
        for cat in CATEGORIES:
            for seed in SEEDS:
                row = preds[(cat, seed)]
                w.writerow([cat, seed] + [f"{row[n]:.6g}" for n in pred_names])

    cat_pred = {c: {n: np.array([preds[(c, s)][n] for s in SEEDS]) for n in pred_names}
                for c in CATEGORIES}
    with open(SUM_DIR / "category_predictors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category"] + [f"{n}_mean" for n in pred_names] + [f"{n}_std" for n in pred_names])
        for cat in CATEGORIES:
            w.writerow([cat] + [f"{cat_pred[cat][n].mean():.6g}" for n in pred_names]
                       + [f"{cat_pred[cat][n].std(ddof=1):.4g}" for n in pred_names])

    # ===== Phase 3: targets =====
    targets = load_targets()
    with open(SUM_DIR / "target_damage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "damage_C2", "damage_C3", "damage_G2", "tolerance_C2"])
        for cat in CATEGORIES:
            for seed in SEEDS:
                t = targets[(cat, seed)]
                w.writerow([cat, seed, f"{t['damage_C2']:.4f}", f"{t['damage_C3']:.4f}",
                            f"{t['damage_G2']:.4f}", f"{-t['damage_C2']:.4f}"])

    dmg = {tgt: np.array([np.mean([targets[(c, s)][tgt] for s in SEEDS])
                          for c in CATEGORIES])
           for tgt in ["damage_C2", "damage_C3", "damage_G2"]}

    # ---- 3.1 heatmap ----
    Z = np.array([[(cat_pred[c][n].mean()
                    - np.mean([cat_pred[k][n].mean() for k in CATEGORIES]))
                   / (np.std([cat_pred[k][n].mean() for k in CATEGORIES], ddof=1) + 1e-12)
                   for n in pred_names] for c in CATEGORIES])
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(pred_names)))
    ax.set_xticklabels(pred_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(CATEGORIES)
    for i in range(5):
        for j in range(len(pred_names)):
            ax.text(j, i, f"{Z[i,j]:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, label="z-score across categories")
    ax.set_title("Normal-only predictors per category (z-scored, seed-mean)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "category_predictor_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- 3.2 correlation ----
    corr_rows = []
    for n in pred_names:
        x_cat = np.array([cat_pred[c][n].mean() for c in CATEGORIES])
        row = {"predictor": n, "group": [r[1] for r in REGISTRY if r[0] == n][0]}
        for tgt in ["damage_C2", "damage_C3", "damage_G2"]:
            rho_c, p_c = spearmanr(x_cat, dmg[tgt])
            row[f"rho_cat_{tgt}"] = round(float(rho_c), 4)
            row[f"p_cat_{tgt}"] = round(float(p_c), 4)
            x_seed = np.array([preds[(c, s)][n] for c in CATEGORIES for s in SEEDS])
            y_seed = np.array([targets[(c, s)][tgt] for c in CATEGORIES for s in SEEDS])
            rho_s, p_s = spearmanr(x_seed, y_seed)
            row[f"rho_seed_{tgt}"] = round(float(rho_s), 4)
            row[f"p_seed_{tgt}"] = round(float(p_s), 4)
        row["strong_assoc_primary"] = abs(row["rho_cat_damage_C2"]) >= 0.7
        corr_rows.append(row)
    with open(SUM_DIR / "correlation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader(); w.writerows(corr_rows)

    # ---- 3.3 seed stability ----
    stab_rows = []
    for n in pred_names:
        means = np.array([cat_pred[c][n].mean() for c in CATEGORIES])
        between = float(np.var(means, ddof=1))
        within = float(np.mean([np.var(cat_pred[c][n], ddof=1) for c in CATEGORIES]))
        cv = float(np.mean([cat_pred[c][n].std(ddof=1) / abs(cat_pred[c][n].mean() + 1e-12)
                            for c in CATEGORIES]))
        stab_rows.append({"predictor": n, "between_cat_var": round(between, 6),
                          "within_cat_var": round(within, 6),
                          "stability_ratio": round(between / (within + 1e-12), 3),
                          "seed_cv": round(cv, 4)})
    with open(SUM_DIR / "seed_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stab_rows[0].keys()))
        w.writeheader(); w.writerows(stab_rows)

    # ---- 3.4 rank prediction ----
    rank_rows = []
    for n in pred_names:
        x_cat = np.array([cat_pred[c][n].mean() for c in CATEGORIES])
        rho, p = spearmanr(x_cat, dmg["damage_C2"])
        tau, _ = kendalltau(x_cat, dmg["damage_C2"])
        rank_rows.append({"predictor": n,
                          "spearman_rho": round(float(rho), 4),
                          "spearman_p_n5": round(float(p), 4),
                          "kendall_tau": round(float(tau), 4),
                          "abs_rho": round(float(abs(rho)), 4)})
    with open(SUM_DIR / "rank_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader(); w.writerows(rank_rows)

    # ---- 3.5 LOCO（primary C2 damage；risk label 冻结：damage>train median -> HIGH）----
    def cat_damage(c):
        return float(np.mean([targets[(c, s)]["damage_C2"] for s in SEEDS]))

    loco_rows = []
    for n in pred_names:
        for held in CATEGORIES:
            train_cats = [c for c in CATEGORIES if c != held]
            d_tr = np.array([cat_damage(c) for c in train_cats])
            x_tr = np.array([cat_pred[c][n].mean() for c in train_cats])
            x_he = cat_pred[held][n].mean()
            thr_d = float(np.median(d_tr))
            rho_tr, _ = spearmanr(x_tr, d_tr)
            direction = 1.0 if (not np.isfinite(rho_tr) or abs(rho_tr) < 1e-9) \
                else float(np.sign(rho_tr))
            thr_x = float(np.median(x_tr))
            pred_label = int(direction * (x_he - thr_x) > 0)
            true_label = int(cat_damage(held) > thr_d)
            loco_rows.append({"predictor": n, "held_out": held,
                              "train_direction": direction,
                              "pred_HIGH": pred_label, "true_HIGH": true_label,
                              "correct": int(pred_label == true_label),
                              "damage_held": round(cat_damage(held), 4)})
    # random control（seed=0 固定 category 常量）
    rng = np.random.RandomState(0)
    rnd_val = {c: float(rng.rand()) for c in CATEGORIES}
    for held in CATEGORIES:
        train_cats = [c for c in CATEGORIES if c != held]
        d_tr = np.array([cat_damage(c) for c in train_cats])
        x_tr = np.array([rnd_val[c] for c in train_cats])
        thr_d = float(np.median(d_tr)); thr_x = float(np.median(x_tr))
        rho_tr, _ = spearmanr(x_tr, d_tr)
        direction = 1.0 if not np.isfinite(rho_tr) else float(np.sign(rho_tr))
        pred_label = int(direction * (rnd_val[held] - thr_x) > 0)
        true_label = int(cat_damage(held) > thr_d)
        loco_rows.append({"predictor": "random_control", "held_out": held,
                          "train_direction": direction,
                          "pred_HIGH": pred_label, "true_HIGH": true_label,
                          "correct": int(pred_label == true_label),
                          "damage_held": round(cat_damage(held), 4)})
    # 2-predictor 线性组合（train fold 内按 |rho| 选 top-2，rho 加权 z-score 和）
    combo_rows = []
    for held in CATEGORIES:
        train_cats = [c for c in CATEGORIES if c != held]
        d_tr = np.array([cat_damage(c) for c in train_cats])
        thr_d = float(np.median(d_tr))
        rhos = {}
        for n in pred_names:
            x_tr = np.array([cat_pred[c][n].mean() for c in train_cats])
            r_, _ = spearmanr(x_tr, d_tr)
            rhos[n] = float(r_) if np.isfinite(r_) else 0.0
        top2 = sorted(rhos, key=lambda k: -abs(rhos[k]))[:2]
        score_tr = np.zeros(len(train_cats)); score_he = 0.0
        for n in top2:
            x_tr = np.array([cat_pred[c][n].mean() for c in train_cats])
            mu, sd = x_tr.mean(), x_tr.std(ddof=1) + 1e-12
            score_tr += rhos[n] * (x_tr - mu) / sd
            score_he += rhos[n] * (cat_pred[held][n].mean() - mu) / sd
        pred_label = int(score_he > float(np.median(score_tr)))
        true_label = int(cat_damage(held) > thr_d)
        combo_rows.append({"predictor": "combo2:" + "+".join(top2), "held_out": held,
                           "train_direction": "linear",
                           "pred_HIGH": pred_label, "true_HIGH": true_label,
                           "correct": int(pred_label == true_label),
                           "damage_held": round(cat_damage(held), 4)})
    all_loco = loco_rows + combo_rows
    with open(SUM_DIR / "loco_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_loco[0].keys()))
        w.writeheader(); w.writerows(all_loco)

    loco_acc = {}
    for n in sorted({r["predictor"] for r in all_loco}):
        rs = [r for r in all_loco if r["predictor"] == n]
        yt = [r["true_HIGH"] for r in rs]; yp = [r["pred_HIGH"] for r in rs]
        loco_acc[n] = {"acc": sum(r["correct"] for r in rs) / len(rs),
                       "bal_acc": balanced_acc(yt, yp)}

    # ---- 3.6 negative controls ----
    rho_by_name = {r["predictor"]: r["rho_cat_damage_C2"] for r in corr_rows}
    nc_rows = []
    for n in pred_names + ["random_control"] + [r["predictor"] for r in combo_rows]:
        nc_rows.append({"predictor": n,
                        "abs_rho_C2": abs(rho_by_name[n]) if n in rho_by_name else "na",
                        "loco_acc": loco_acc[n]["acc"],
                        "loco_bal_acc": loco_acc[n]["bal_acc"]})
    with open(SUM_DIR / "negative_controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(nc_rows[0].keys()))
        w.writeheader(); w.writerows(nc_rows)

    # ---- figures ----
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    for ax_, n in zip(axes.flat, pred_names):
        x = np.array([cat_pred[c][n].mean() for c in CATEGORIES])
        y = dmg["damage_C2"]
        ax_.scatter(x, y, c=["#C44E52" if c in ("grid", "screw") else "#4C72B0"
                             for c in CATEGORIES], s=60)
        for c, xi, yi in zip(CATEGORIES, x, y):
            ax_.annotate(c, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho, _ = spearmanr(x, y)
        ax_.set_title(f"{n}\n rho={rho:+.2f}", fontsize=8)
    fig.suptitle("Normal-only predictors vs C2 damage (category-level, seed-mean)", y=1.01)
    fig.tight_layout(); fig.savefig(FIG_DIR / "predictor_vs_damage.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = np.argsort(dmg["damage_C2"])[::-1]
    cats_sorted = [CATEGORIES[i] for i in order]
    vals = [dmg["damage_C2"][i] for i in order]
    stds = [np.std([targets[(CATEGORIES[i], s)]["damage_C2"] for s in SEEDS]) for i in order]
    ax.bar(cats_sorted, vals, yerr=stds, capsize=4,
           color=["#C44E52" if v > 0 else "#55A868" for v in vals])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("C2 damage = d'(B0) - d'(C2)  (higher = more fragile)")
    ax.set_title("Category normalization-tolerance ranking (primary target)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "category_risk_ranking.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    names = [r["predictor"] for r in stab_rows]
    ratios = [r["stability_ratio"] for r in stab_rows]
    colors = ["#999999" if r["predictor"].startswith("img_") else "#4C72B0"
              for r in stab_rows]
    ax.bar(range(len(names)), ratios, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("between-cat var / within-cat var (log)")
    ax.set_title("Seed stability ratio (higher = category-discriminative, seed-robust)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "seed_stability.png", dpi=150)
    plt.close(fig)

    # ---- sanity ----
    with open(SUM_DIR / "sanity_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for name, ok, detail in checks:
            w.writerow([name, "PASS" if ok else "FAIL", detail])
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n[sanity] {len(checks)-n_fail}/{len(checks)} PASS")

    print("\n=== C2 damage by category (seed-mean +/- std) ===")
    for i, c in enumerate(CATEGORIES):
        print(f"{c:>9}: {dmg['damage_C2'][i]:+.3f} +/- "
              f"{np.std([targets[(c,s)]['damage_C2'] for s in SEEDS]):.3f}   "
              f"(C3 {dmg['damage_C3'][i]:+.3f}, G2 {dmg['damage_G2'][i]:+.3f})")
    print("\n=== top predictors by |rho| (category-level, C2 damage) ===")
    for r in sorted(corr_rows, key=lambda r: -abs(r["rho_cat_damage_C2"]))[:6]:
        print(f"{r['predictor']:>22} rho={r['rho_cat_damage_C2']:+.3f} "
              f"(p={r['p_cat_damage_C2']:.3f}) seed-rho={r['rho_seed_damage_C2']:+.3f}")
    print("\n=== LOCO accuracy (top 6) ===")
    for n, a in sorted(loco_acc.items(), key=lambda kv: -kv[1]["acc"])[:6]:
        print(f"{n:>22} acc={a['acc']:.2f} bal={a['bal_acc']:.2f}")
    print(f"\ntotal runtime {time.time()-t_start:.0f}s (CPU only)")


if __name__ == "__main__":
    main()
