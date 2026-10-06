"""Experiment 5B — NN-distance implementation sanity check (CPU only).

背景：experiment5b_analysis.py 的 `nn_mean_dist` 首次运行曾因 d² 展开式漏掉
||x||² 项导致 nn_dist_rel_L2 = 0。修复后（d² = ||x||² - 2<x,y> + ||y||²），
在重新运行完整 5B 之前，按协议对本实现做以下检查：

  A. 非负性      raw d²（未 clamp）最小值 >= -1e-6 * mean(||y||²)（相对容差）
  B. Self-distance d²(x,x) ≈ 0（相对容差）
  C. Brute-force  小 subset 与 torch.cdist 对照（NN 距离逐点一致）
  D. NN 合理性    每 bank 随机子采样（<=4000 点）排除 self 后 mean/median NN > 0，
                  零距离比例检查；另对 bottle:0 两层 + grid:0:L2 跑完整
                  nn_mean_dist 端到端确认 > 0
  E. 30 banks     全部可加载 / shape 合法 / 无 NaN / 无 Inf

输出：results/experiment_5b/logs/nn_sanity_report.{json,txt}
不修改主分析脚本、不写任何 summary/figures、不加载 target。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment5b_analysis import nn_mean_dist, BANK_DIR, CATEGORIES, SEEDS, LAYERS  # noqa: E402

OUT = ROOT / "results" / "experiment_5b" / "logs"
REPORT = {}

rng = np.random.RandomState(123)


def rel_tol_ref(b: np.ndarray) -> float:
    """相对容差基准：平均平方范数。"""
    return float((b.astype(np.float64) ** 2).sum(axis=1).mean())


def raw_d2_rows(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """与主脚本完全相同的展开式，返回未 clamp 的 d²（x 行 × b 列）。"""
    x = x.astype(np.float64)
    b64 = b.astype(np.float64)
    xn = np.einsum("nc,nc->n", x, x)
    bn = np.einsum("nc,nc->n", b64, b64)
    return xn[:, None] - 2.0 * (x @ b64.T) + bn[None, :]


def nn_subsample(b: np.ndarray, k: int = 4000) -> np.ndarray:
    """子采样内（排除 self）每点 NN 距离，走与主实现相同的展开式路径。"""
    idx = rng.permutation(b.shape[0])[: min(k, b.shape[0])]
    sub = b[idx]
    s = sub.astype(np.float64)
    n = s.shape[0]
    bn = np.einsum("nc,nc->n", s, s)
    d2 = bn[:, None] - 2.0 * (s @ s.T) + bn[None, :]
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def check_bank(b: np.ndarray, name: str, checks: dict) -> None:
    ref = rel_tol_ref(b)

    # A. 非负性（raw d²，子采样 800 行 × 全 bank）
    rows = rng.choice(b.shape[0], size=min(800, b.shape[0]), replace=False)
    d2 = raw_d2_rows(b[rows], b)
    # 排除自身列（行 i 对应 bank 行 rows[i]）
    d2[np.arange(len(rows)), rows] = np.inf
    min_d2 = float(d2.min())
    checks["A_min_raw_d2"][name] = min_d2
    checks["A_pass"] &= min_d2 >= -1e-6 * ref

    # B. self-distance
    sample = rng.choice(b.shape[0], size=min(50, b.shape[0]), replace=False)
    sd = raw_d2_rows(b[sample], b[sample])
    self_d2 = np.diag(sd)
    checks["B_max_self_d2"][name] = float(np.abs(self_d2).max())
    checks["B_pass"] &= float(np.abs(self_d2).max()) <= 1e-6 * ref

    # D. NN 合理性（子采样）
    nn = nn_subsample(b)
    zero_frac = float((nn <= 1e-9).mean())
    checks["D_mean_nn"][name] = float(nn.mean())
    checks["D_median_nn"][name] = float(np.median(nn))
    checks["D_zero_frac"][name] = zero_frac
    checks["D_pass"] &= nn.mean() > 0 and np.median(nn) > 0 and zero_frac < 0.01


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks = {
        "A_pass": True, "B_pass": True, "D_pass": True,
        "E_pass": True,
        "A_min_raw_d2": {}, "B_max_self_d2": {},
        "D_mean_nn": {}, "D_median_nn": {}, "D_zero_frac": {},
        "E_shapes": {},
    }

    # E + A + B + D：全部 30 banks
    for cat in CATEGORIES:
        for seed in SEEDS:
            for layer in LAYERS:
                name = f"{cat}:s{seed}:{layer}"
                p = BANK_DIR / f"{cat}_seed{seed}_{layer}.npy"
                try:
                    b = np.load(p)
                except Exception as e:  # noqa: BLE001
                    checks["E_pass"] = False
                    REPORT.setdefault("load_errors", {})[name] = str(e)
                    continue
                ok_shape = b.ndim == 2 and b.shape[0] >= 2 and b.shape[1] >= 2
                ok_finite = bool(np.isfinite(b).all())
                checks["E_shapes"][name] = list(b.shape)
                checks["E_pass"] &= ok_shape and ok_finite
                if not (ok_shape and ok_finite):
                    continue
                check_bank(b, name, checks)
                print(f"[bank] {name} shape={b.shape} done ({time.time()-t0:.0f}s)",
                      flush=True)

    # C. brute-force 对照（3 个 bank，各抽 N=150，subset 内 NN vs torch.cdist）
    checks["C_pass"] = True
    checks["C_max_abs_diff"] = {}
    for cat, seed, layer in [("bottle", 0, "layer2"), ("grid", 0, "layer2"),
                             ("screw", 0, "layer3")]:
        name = f"{cat}:s{seed}:{layer}"
        b = np.load(BANK_DIR / f"{cat}_seed{seed}_{layer}.npy")
        idx = rng.choice(b.shape[0], size=150, replace=False)
        sub = b[idx]
        # optimized 路径（同展开式，subset 内排除 self）
        s = sub.astype(np.float64)
        bn = np.einsum("nc,nc->n", s, s)
        d2 = bn[:, None] - 2.0 * (s @ s.T) + bn[None, :]
        np.fill_diagonal(d2, np.inf)
        nn_opt = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        # brute-force：torch.cdist（CPU）
        with torch.no_grad():
            ref_d = torch.cdist(torch.from_numpy(sub.astype(np.float64)),
                                torch.from_numpy(sub.astype(np.float64)))
        ref_np = ref_d.numpy().copy()
        np.fill_diagonal(ref_np, np.inf)
        nn_ref = ref_np.min(axis=1)
        diff = float(np.abs(nn_opt - nn_ref).max())
        checks["C_max_abs_diff"][name] = diff
        checks["C_pass"] &= diff <= 1e-6
        print(f"[brute] {name} max|ΔNN|={diff:.3e}", flush=True)

    # D-full：完整 nn_mean_dist 端到端（原 bug 首发位置 bottle:0 两层 + grid:0:L2）
    checks["D_full"] = {}
    for cat, seed, layer in [("bottle", 0, "layer2"), ("bottle", 0, "layer3"),
                             ("grid", 0, "layer2")]:
        name = f"{cat}:s{seed}:{layer}"
        b = np.load(BANK_DIR / f"{cat}_seed{seed}_{layer}.npy")
        v = nn_mean_dist(b)
        norm_mean = float(np.linalg.norm(b.astype(np.float64), axis=1).mean())
        checks["D_full"][name] = {"nn_mean": v,
                                  "norm_mean": norm_mean,
                                  "nn_dist_rel": v / norm_mean}
        checks["D_pass"] &= v > 0
        print(f"[full-nn] {name} nn_mean={v:.6f} rel={v/norm_mean:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    overall = all(checks[k] for k in ["A_pass", "B_pass", "C_pass", "D_pass", "E_pass"])
    REPORT["checks"] = checks
    REPORT["n_banks"] = len(checks["E_shapes"])
    REPORT["overall"] = "PASS" if overall else "FAIL"
    REPORT["elapsed_s"] = round(time.time() - t0, 1)

    with open(OUT / "nn_sanity_report.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=float)
    lines = [
        f"NN sanity overall: {REPORT['overall']}  ({REPORT['n_banks']} banks, "
        f"{REPORT['elapsed_s']}s)",
        f"A non-negativity : {'PASS' if checks['A_pass'] else 'FAIL'}  "
        f"min raw d2 = {min(checks['A_min_raw_d2'].values()):.3e}",
        f"B self-distance  : {'PASS' if checks['B_pass'] else 'FAIL'}  "
        f"max |d2(x,x)| = {max(checks['B_max_self_d2'].values()):.3e}",
        f"C brute-force    : {'PASS' if checks['C_pass'] else 'FAIL'}  "
        f"max |dNN| = {max(checks['C_max_abs_diff'].values()):.3e}",
        f"D nn>0 (subsample): {'PASS' if checks['D_pass'] else 'FAIL'}  "
        f"min mean NN = {min(checks['D_mean_nn'].values()):.4e}, "
        f"max zero-frac = {max(checks['D_zero_frac'].values()):.4f}",
        f"E 30 banks finite: {'PASS' if checks['E_pass'] else 'FAIL'}",
    ]
    (OUT / "nn_sanity_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
