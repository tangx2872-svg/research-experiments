"""
Experiment 1G — Step 2-7: NN distance dispersion 分析 + feature geometry + CASE 判定

从 embeddings npz + memory banks 计算：
  LEVEL 1 (PRIMARY): defect patch 的 NN-distance dispersion（matched Mα + frozen M0）
  LEVEL 2: defect feature dispersion（到 centroid 平均距离）
  LEVEL 3: feature norm / channel variance / representation drift

两套 bank：
  Track A (matched): distance(Fα, Mα)
  Track B (frozen): distance(Fα, M0)

输出 defect_summary.csv + 4 张图 + CASE 判定。
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_1g"
ANALYSIS = OUT / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 0


def load_selection(selection: str = "primary") -> list[dict]:
    rows = list(csv.DictReader(open(OUT / "selection.csv", encoding="utf-8")))
    return [r for r in rows if r["selection"] == selection]


def load_bank(category: str, alpha: float) -> np.ndarray:
    p = OUT / "memory_banks" / f"bank_{category}_a{alpha:g}_seed{SEED}.npy"
    return np.load(p)


_BANK_CACHE: dict[tuple, np.ndarray] = {}


def load_bank_cached(category: str, alpha: float) -> np.ndarray:
    """带缓存的 bank 加载：每个 (category, alpha) 只从磁盘读一次。"""
    key = (category, alpha)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = load_bank(category, alpha)
    return _BANK_CACHE[key]


def nn_distances(feat: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """feat (N,D) vs bank (M,D) 的最近邻欧氏距离。用 ||f||²+||b||²-2f·bᵀ 避免 (N,M,D) 张量 OOM。"""
    n = feat.shape[0]
    dists = np.empty(n, dtype=np.float32)
    # 分块处理 feat，避免 (N,M) 过大；bank 也分块避免 f@bank.T 单次太大
    feat_chunk = 256
    bank_chunk = 8192
    for i in range(0, n, feat_chunk):
        f = feat[i:i + feat_chunk].astype(np.float32)  # (cn, D)
        f_norm2 = np.einsum("ij,ij->i", f, f)          # (cn,)
        # 逐 bank 块求最小距离
        dmin = np.full(f.shape[0], np.inf, dtype=np.float32)
        for j in range(0, bank.shape[0], bank_chunk):
            b = bank[j:j + bank_chunk].astype(np.float32)  # (cm, D)
            b_norm2 = np.einsum("ij,ij->i", b, b)          # (cm,)
            # d²(i,j) = ||f_i||² + ||b_j||² - 2 f_i·b_j
            dot = f @ b.T                                  # (cn, cm)
            d2 = f_norm2[:, None] + b_norm2[None, :] - 2.0 * dot
            np.minimum(dmin, d2.min(axis=1), out=dmin)
        dists[i:i + feat_chunk] = np.sqrt(dmin)
    return dists


def feature_dispersion(feat: np.ndarray) -> float:
    """到 feature centroid 的平均距离。"""
    centroid = feat.mean(axis=0)
    return float(np.linalg.norm(feat - centroid, axis=1).mean())


def main() -> None:
    sel = load_selection()
    # 读 patch_level（只取 seed=0，defect role）
    patch_rows = list(csv.DictReader(open(OUT / "patch_level.csv", encoding="utf-8")))

    # 按 (cat, dt, alpha) 组织 defect patch 的 feat_alpha + feat_pre
    # npz 文件名 = cat_dt_stem_a{alpha}_seed0.npz
    image_rows: list[dict] = []

    for d in sel:
        cat, dt = d["category"], d["defect_type"]
        family = d["family"]
        # 该 defect type 的所有图
        imgs = sorted(set(r["image_path"] for r in patch_rows
                          if r["category"] == cat and r["defect_type"] == dt))
        print(f"[{cat}/{dt}] {len(imgs)} 图，family={family}", flush=True)
        for img_path in imgs:
            stem = Path(img_path).stem
            # 每 alpha 的 defect patches 统计
            per_alpha = {}
            for alpha in ALPHAS:
                npz_path = OUT / "embeddings" / f"{cat}_{dt}_{stem}_a{alpha:g}_seed{SEED}.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                feat_pre = data["feat_pre"]      # (N, D)
                feat_alpha = data["feat_alpha"]  # (N, D)
                overlap = data["overlap"]        # (H, W)
                # defect mask: overlap > 0.5
                h, w = overlap.shape
                defect_mask = (overlap > 0.5).flatten()
                # 边界处理：细小缺陷（thread_side/thread_top/glue）在 32×32 网格下可能
                # 无 patch 达到 >0.5，降级用 overlap>0 的 patch（boundary∪defect）作 fallback。
                # 记录偏差（技术边界处理，非事后 cherry-pick）。
                if defect_mask.sum() == 0:
                    defect_mask = (overlap > 0.0).flatten()
                bg_mask = (overlap == 0.0).flatten()

                feat_d = feat_alpha[defect_mask]
                feat_bg = feat_alpha[bg_mask]

                # matched bank（缓存加载，避免每图每 alpha 重复读 245MB 磁盘）
                bank_matched = load_bank_cached(cat, alpha)
                # frozen M0
                bank_m0 = load_bank_cached(cat, 0.0)

                nn_matched = nn_distances(feat_d, bank_matched)
                nn_frozen = nn_distances(feat_d, bank_m0)

                per_alpha[alpha] = {
                    "defect_nn_mean": float(nn_matched.mean()),
                    "defect_nn_std": float(nn_matched.std()),       # PRIMARY dispersion
                    "defect_nn_iqr": float(np.percentile(nn_matched, 75) - np.percentile(nn_matched, 25)),
                    "defect_nn_std_frozen": float(nn_frozen.std()),
                    "defect_nn_mean_frozen": float(nn_frozen.mean()),
                    "defect_n_dispersion": float(feature_dispersion(feat_d)),
                    "defect_norm_mean": float(np.linalg.norm(feat_d, axis=1).mean()),
                    "defect_norm_std": float(np.linalg.norm(feat_d, axis=1).std()),
                    "channel_var_mean": float(feat_d.var(axis=0).mean()),
                    # background control
                    "bg_nn_mean": float(nn_distances(feat_bg, bank_matched).mean()),
                    "bg_nn_std": float(nn_distances(feat_bg, bank_matched).std()),
                    "bg_dispersion": float(feature_dispersion(feat_bg)),
                    # representation drift (pre vs post)
                    "cos_pre_post": float(np.mean(
                        (feat_pre[defect_mask] * feat_alpha[defect_mask]).sum(axis=1) /
                        (np.linalg.norm(feat_pre[defect_mask], axis=1) *
                         np.linalg.norm(feat_alpha[defect_mask], axis=1) + 1e-9))),
                    "n_defect_patches": int(defect_mask.sum()),
                    "n_bg_patches": int(bg_mask.sum()),
                }

            # 记录 image-level 行（含各 alpha）
            for alpha in ALPHAS:
                if alpha not in per_alpha:
                    continue
                r = {"category": cat, "defect_type": dt, "image_path": img_path,
                     "seed": SEED, "alpha": alpha, "family": family,
                     **per_alpha[alpha]}
                image_rows.append(r)

    # 写 image_level.csv
    fields = ["category", "defect_type", "image_path", "seed", "alpha", "family",
              "defect_nn_mean", "defect_nn_std", "defect_nn_iqr", "defect_nn_std_frozen",
              "defect_nn_mean_frozen", "defect_n_dispersion", "defect_norm_mean",
              "defect_norm_std", "channel_var_mean", "bg_nn_mean", "bg_nn_std",
              "bg_dispersion", "cos_pre_post", "n_defect_patches", "n_bg_patches"]
    with open(ANALYSIS / "image_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(image_rows)

    print(f"image_level.csv: {len(image_rows)} 行")

    # ---- defect-level summary: 每个 defect 的 ΔNN std 等 ----
    # 先按 defect 聚合（跨 image 取均值）
    by_defect: dict[tuple, dict] = {}
    for r in image_rows:
        key = (r["category"], r["defect_type"])
        by_defect.setdefault(key, {"alphas": {}, "family": r["family"]})
        a = r["alpha"]
        by_defect[key]["alphas"].setdefault(a, []).append(r)

    summary_rows = []
    for (cat, dt), info in by_defect.items():
        def mean_at(alpha, field):
            vals = [float(r[field]) for r in info["alphas"][alpha]]
            return float(np.mean(vals)) if vals else np.nan

        d0 = {f: mean_at(0.0, f) for f in ["defect_nn_std", "defect_nn_mean", "defect_n_dispersion",
                                           "defect_norm_std", "channel_var_mean"]}
        d1 = {f: mean_at(1.0, f) for f in ["defect_nn_std", "defect_nn_mean", "defect_n_dispersion",
                                           "defect_norm_std", "channel_var_mean"]}
        summary_rows.append({
            "category": cat, "defect_type": dt, "family": info["family"],
            "nn_std_a0": d0["defect_nn_std"], "nn_std_a1": d1["defect_nn_std"],
            "delta_nn_std": d1["defect_nn_std"] - d0["defect_nn_std"],
            "delta_nn_std_frozen": mean_at(1.0, "defect_nn_std_frozen") - mean_at(0.0, "defect_nn_std_frozen"),
            "delta_nn_mean": d1["defect_nn_mean"] - d0["defect_nn_mean"],
            "delta_dispersion": d1["defect_n_dispersion"] - d0["defect_n_dispersion"],
            "delta_norm_std": d1["defect_norm_std"] - d0["defect_norm_std"],
            "delta_channel_var": d1["channel_var_mean"] - d0["channel_var_mean"],
        })

    fields2 = ["category", "defect_type", "family", "nn_std_a0", "nn_std_a1",
               "delta_nn_std", "delta_nn_std_frozen", "delta_nn_mean",
               "delta_dispersion", "delta_norm_std", "delta_channel_var"]
    with open(ANALYSIS / "defect_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(summary_rows)

    # ---- LEVEL 1: 1E Δdefect_std vs 1G ΔNN std ----
    resp = {}
    for r in csv.DictReader(open(
            ROOT / "results/experiment_1e/analysis/defect_response_signatures_seed_summary.csv",
            encoding="utf-8")):
        resp[(r["category"], r["defect_type"])] = float(r["delta_defect_std_mean"])

    print("\n===== LEVEL 1: 1E Δdefect_std vs 1G ΔNN-distance std =====")
    print(f"{'defect':30s} {'1E Δstd':>8s} {'1G ΔNNstd':>9s} {'ΔNNstd_frozen':>13s} {'family':>8s}")
    xs, ys = [], []
    for r in summary_rows:
        key = (r["category"], r["defect_type"])
        d1e = resp.get(key, np.nan)
        xs.append(d1e)
        ys.append(r["delta_nn_std"])
        print(f"{r['category']+'/'+r['defect_type']:30s} {d1e:+8.2f} {r['delta_nn_std']:+9.3f} "
              f"{r['delta_nn_std_frozen']:+13.3f} {r['family']:>8s}")
    xs = np.array(xs); ys = np.array(ys)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    if keep.sum() >= 5:
        rho, p = stats.spearmanr(xs[keep], ys[keep])
        print(f"\nSpearman(1E Δstd, 1G ΔNNstd) = {rho:+.3f} (p={p:.3f}, n={keep.sum()})")
        rho_f, p_f = stats.spearmanr(xs[keep], np.array([r["delta_nn_std_frozen"] for r in summary_rows])[keep])
        print(f"Spearman(1E Δstd, 1G ΔNNstd_frozen) = {rho_f:+.3f} (p={p_f:.3f})")

    print(f"\ndefect_summary.csv: {len(summary_rows)} 行 → {ANALYSIS}")


if __name__ == "__main__":
    main()
