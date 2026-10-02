"""
Experiment 1F — Step 3-5: 聚合、Merge、Primary 分析、Exploratory 分析、CASE 判定

Step 3: sample attributes → defect-type level (median + IQR)，与 1E response signature merge
Step 4: Primary 分析（target = delta_defect_std）
  A1: Spearman rho + 95% bootstrap CI（attribute vs delta_defect_std）
  A2: partial correlation 控制 log(area)
  A3: leave-one-category-out
  A4: shrink vs expansion group 对比（排除 MIXED）
Step 5: Exploratory — 4D response 标准化 PCA + CASE A/B/C/D 判定
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "experiment_1f"
ANALYSIS_DIR = OUT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# 预注册属性（与 config.json 一致）
ATTRS = {
    "size": ["area_ratio", "log_area_ratio"],
    "contrast": ["intensity_contrast", "lab_contrast", "local_variance_ratio"],
    "frequency": ["gradient_mean_defect", "gradient_contrast", "laplacian_energy", "hf_energy_ratio"],
    "morphology": ["compactness", "elongation"],  # area_pixels/perimeter 与 size 冗余，取紧凑形态
}
ALL_ATTRS = [a for v in ATTRS.values() for a in v]
PRIMARY = "delta_defect_std"
SECONDARY = ["delta_mean_gap", "delta_z", "delta_dprime"]

N_BOOT = 10000
rng = np.random.default_rng(20261002)  # 固定种子，可复现


# ---------------------------------------------------------------- Step 3
def aggregate_attributes() -> dict[tuple[str, str], dict]:
    """sample-level attributes → defect-type median + IQR。"""
    rows = list(csv.DictReader(open(OUT_ROOT / "sample_attributes_d15.csv", encoding="utf-8")))
    by_type: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        key = (r["category"], r["defect_type"])
        by_type.setdefault(key, {a: [] for a in ALL_ATTRS})
        for a in ALL_ATTRS:
            try:
                v = float(r[a])
            except (ValueError, TypeError):
                continue
            if v == v:  # not NaN
                by_type[key][a].append(v)

    agg: dict[tuple[str, str], dict] = {}
    for key, vals in by_type.items():
        d = {"category": key[0], "defect_type": key[1], "n_samples": len(next(iter(vals.values())))}
        for a in ALL_ATTRS:
            arr = np.array(vals[a])
            d[f"{a}_median"] = float(np.median(arr)) if arr.size else np.nan
            q1, q3 = np.percentile(arr, [25, 75]) if arr.size else (np.nan, np.nan)
            d[f"{a}_iqr"] = float(q3 - q1) if arr.size else np.nan
        agg[key] = d
    return agg


def load_responses() -> dict[tuple[str, str], dict]:
    """1E seed_summary + seed_stability。"""
    sig = list(csv.DictReader(open(
        ROOT / "results/experiment_1e/analysis/defect_response_signatures_seed_summary.csv",
        encoding="utf-8")))
    stab = {(r["category"], r["defect_type"]): r for r in csv.DictReader(open(
        ROOT / "results/experiment_1e/analysis/seed_stability.csv", encoding="utf-8"))}
    out: dict[tuple[str, str], dict] = {}
    for r in sig:
        key = (r["category"], r["defect_type"])
        s = stab.get(key, {})
        d = {
            "category": r["category"], "defect_type": r["defect_type"],
            "delta_mean_gap": float(r["delta_mean_gap_mean"]),
            "delta_defect_std": float(r["delta_defect_std_mean"]),
            "delta_z": float(r["delta_z_mean"]),
            "delta_dprime": float(r["delta_dprime_mean"]),
            "median_area_ratio_1e": float(r["median_area_ratio"]),
            "sign_mean_gap": s.get("delta_mean_gap_sign", ""),
            "sign_defect_std": s.get("delta_defect_std_sign", ""),
            "sign_z": s.get("delta_z_sign", ""),
            "sign_dprime": s.get("delta_dprime_sign", ""),
        }
        out[key] = d
    return out


def build_merged_table(agg, resp) -> list[dict]:
    """核心分析表：25 defect types × (attributes + responses)。"""
    merged = []
    for key in sorted(resp.keys()):
        if key not in agg:
            raise KeyError(f"1E response 有但 attribute 缺失: {key}")
        row = dict(agg[key])
        row.update(resp[key])
        merged.append(row)
    return merged


# ---------------------------------------------------------------- Step 4 stats
def spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Spearman rho + 95% bootstrap CI（percentile）。"""
    rho, pval = stats.spearmanr(x, y)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            boots[i] = np.nan
            continue
        boots[i] = stats.spearmanr(x[idx], y[idx]).statistic
    boots = boots[~np.isnan(boots)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": float(rho), "p": float(pval), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": n}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """控制 z（log area）后的 partial Spearman：对 rank 残差求相关。

    x,y,z → rank 变换 → 线性回归取残差 → Pearson（= partial Spearman 的标准做法）。
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    ex = resid(rx, rz)
    ey = resid(ry, rz)
    r, p = stats.pearsonr(ex, ey)
    return {"partial_rho": float(r), "p": float(p), "n": len(x)}


def loco_spearman(rows: list[dict], attr: str, target: str) -> list[dict]:
    """leave-one-category-out：每次去掉一个 category 重算 Spearman。"""
    cats = sorted(set(r["category"] for r in rows))
    out = []
    for drop in cats:
        sub = [r for r in rows if r["category"] != drop]
        x = np.array([r[f"{attr}_median"] for r in sub])
        y = np.array([r[target] for r in sub])
        keep = ~(np.isnan(x) | np.isnan(y))
        if keep.sum() < 5:
            out.append({"drop": drop, "rho": np.nan, "n": int(keep.sum())})
            continue
        res = stats.spearmanr(x[keep], y[keep])
        out.append({"drop": drop, "rho": float(res.statistic), "p": float(res.pvalue),
                    "n": int(keep.sum())})
    return out


def shrink_expand(rows: list[dict], attrs: list[str]) -> list[dict]:
    """Shrink (Δstd<0) vs Expansion (Δstd>0) group 对比，排除 MIXED。"""
    shrink = [r for r in rows if r["sign_defect_std"] == "NEGATIVE_3_OF_3"]
    expand = [r for r in rows if r["sign_defect_std"] == "POSITIVE_3_OF_3"]
    out = []
    for a in attrs:
        xs = np.array([r[f"{a}_median"] for r in shrink])
        xe = np.array([r[f"{a}_median"] for r in expand])
        xs = xs[~np.isnan(xs)]
        xe = xe[~np.isnan(xe)]
        if xs.size < 3 or xe.size < 3:
            continue
        # Mann-Whitney U（组间比较，n 小）
        u, p = stats.mannwhitneyu(xs, xe, alternative="two-sided")
        # rank-biserial effect size
        n1, n2 = xs.size, xe.size
        rb = 2 * u / (n1 * n2) - 1
        # bootstrap CI for median difference
        diffs = []
        for _ in range(N_BOOT):
            bs = rng.choice(xs, n1, replace=True)
            be = rng.choice(xe, n2, replace=True)
            diffs.append(np.median(be) - np.median(bs))
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        out.append({
            "attr": a,
            "median_shrink": float(np.median(xs)), "median_expand": float(np.median(xe)),
            "median_diff_expand_minus_shrink": float(np.median(xe) - np.median(xs)),
            "diff_ci_lo": float(ci_lo), "diff_ci_hi": float(ci_hi),
            "rank_biserial": float(rb), "p_mw": float(p),
            "n_shrink": int(n1), "n_expand": int(n2),
        })
    return out


def main() -> None:
    # ---- Step 3 ----
    agg = aggregate_attributes()
    resp = load_responses()
    merged = build_merged_table(agg, resp)
    print(f"merged: {len(merged)} defect types")

    # 写核心分析表
    fields = list(merged[0].keys())
    with open(ANALYSIS_DIR / "merged_attributes_responses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)

    # ---- Step 4: Primary analysis (delta_defect_std) ----
    y_primary = np.array([r[PRIMARY] for r in merged])
    log_area = np.array([r["log_area_ratio_median"] for r in merged])

    a1_rows, a2_rows, a3_rows = [], [], []
    for fam, attrs in ATTRS.items():
        for a in attrs:
            x = np.array([r[f"{a}_median"] for r in merged])
            keep = ~(np.isnan(x) | np.isnan(y_primary))
            if keep.sum() < 10:
                continue
            r1 = spearman_boot(x[keep], y_primary[keep])
            a1_rows.append({"attr": a, "family": fam, **r1})
            # A2 partial（控制 log area）
            r2 = partial_spearman(x[keep], y_primary[keep], log_area[keep])
            a2_rows.append({"attr": a, "family": fam, **r2})
            # A3 LOCO
            for lo in loco_spearman([m for m, k in zip(merged, keep) if k], a, PRIMARY):
                a3_rows.append({"attr": a, "family": fam, **lo})

    # 写 A1/A2/A3
    for name, rows_, fields_ in [
        ("a1_spearman_primary.csv", a1_rows,
         ["attr", "family", "rho", "p", "ci_lo", "ci_hi", "n"]),
        ("a2_partial_control_area.csv", a2_rows,
         ["attr", "family", "partial_rho", "p", "n"]),
        ("a3_loco_primary.csv", a3_rows,
         ["attr", "family", "drop", "rho", "p", "n"]),
    ]:
        with open(ANALYSIS_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields_)
            w.writeheader()
            w.writerows(rows_)

    # A4 shrink vs expand
    a4_rows = shrink_expand(merged, ALL_ATTRS)
    with open(ANALYSIS_DIR / "a4_shrink_vs_expand.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(a4_rows[0].keys()))
        w.writeheader()
        w.writerows(a4_rows)

    # ---- 控制台摘要 ----
    print("\n===== A1: Spearman (attribute vs delta_defect_std) =====")
    for r in sorted(a1_rows, key=lambda r: -abs(r["rho"])):
        star = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
        print(f"  [{r['family']:10s}] {r['attr']:22s} rho={r['rho']:+.3f} "
              f"CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]{star}")

    print("\n===== A2: Partial Spearman (control log area) =====")
    for r in sorted(a2_rows, key=lambda r: -abs(r["partial_rho"])):
        print(f"  [{r['family']:10s}] {r['attr']:22s} partial_rho={r['partial_rho']:+.3f} p={r['p']:.4f}")

    print("\n===== A3: LOCO（去掉某 category 后的 rho） =====")
    by_attr: dict[str, list] = {}
    for r in a3_rows:
        by_attr.setdefault(r["attr"], []).append(r)
    for r1 in a1_rows:
        a = r1["attr"]
        locos = by_attr.get(a, [])
        if not locos:
            continue
        s = "  ".join(f"-{l['drop']}={l['rho']:+.2f}" for l in locos)
        print(f"  {a:22s} full={r1['rho']:+.3f} | {s}")

    print("\n===== A4: Shrink vs Expansion (Δstd sign groups) =====")
    for r in a4_rows:
        print(f"  {r['attr']:22s} shrink={r['median_shrink']:8.3f} expand={r['median_expand']:8.3f} "
              f"diff={r['median_diff_expand_minus_shrink']:+8.3f} CI[{r['diff_ci_lo']:+.3f},{r['diff_ci_hi']:+.3f}] "
              f"rb={r['rank_biserial']:+.2f} p={r['p_mw']:.3f}")

    # ---- Step 5: Exploratory PCA (4D response) ----
    R = np.array([[r["delta_mean_gap"], r["delta_defect_std"], r["delta_z"], r["delta_dprime"]]
                  for r in merged])
    # 标准化（z-score 每列）
    Rz = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-12)
    C = np.cov(Rz.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    proj = Rz @ eigvecs
    explained = eigvals / eigvals.sum()

    pca_rows = []
    for i, r in enumerate(merged):
        pca_rows.append({
            "category": r["category"], "defect_type": r["defect_type"],
            "pc1": float(proj[i, 0]), "pc2": float(proj[i, 1]),
            "delta_mean_gap": r["delta_mean_gap"], "delta_defect_std": r["delta_defect_std"],
            "delta_z": r["delta_z"], "delta_dprime": r["delta_dprime"],
            "sign_defect_std": r["sign_defect_std"],
        })
    with open(ANALYSIS_DIR / "a5_pca_response.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pca_rows[0].keys()))
        w.writeheader()
        w.writerows(pca_rows)

    print("\n===== A5: PCA of 4D response (exploratory) =====")
    print(f"  explained variance: PC1={explained[0]:.3f} PC2={explained[1]:.3f} "
          f"PC3={explained[2]:.3f} PC4={explained[3]:.3f}")
    print(f"  PC1 loadings (mean_gap,std,z,dprime): {np.round(eigvecs[:,0],3)}")
    print(f"  PC2 loadings: {np.round(eigvecs[:,1],3)}")

    print("\nStep 3-5 分析完成，全部写入 results/experiment_1f/analysis/")


if __name__ == "__main__":
    main()
