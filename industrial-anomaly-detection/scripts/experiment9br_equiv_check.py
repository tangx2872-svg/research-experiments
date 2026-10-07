"""Experiment 9B-R — Control 2 逐元素 representation 等价性检查（GPU，1 次，约 1-2 min）。

9C 只报告了聚合量（sum_abs / mean）。本脚本补 **真正的逐元素 max|dF|**：
在**同一进程**内，让两个数学等价实现（uniform alpha=0.25 vs constant-gate 0.25）
遍历**与 fit 相同的训练集**（bottle train/good 全部 209 张，batch=16），
逐元素累加 max 差异并统计不同元素个数。

输出：analysis/equivalence_evidence.json（逐元素证据）
     并与 9C-v2b 的跨进程证据（coreset / bank / score）合并。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment9b_model as m9b  # noqa: E402
from experiment5a_model import FAlphaDualLayerPatchcoreModel  # noqa: E402

EXP = ROOT / "results" / "experiment_9b_r_strict_replay"
SUM = EXP / "analysis"
ALPHA = 0.25


def main() -> None:
    SUM.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
              num_neighbors=9)
    g2, _ = m9b.make_gate(np.zeros(512, np.float32), "const", value=ALPHA)
    g3, _ = m9b.make_gate(np.zeros(1024, np.float32), "const", value=ALPHA)
    m_alpha = FAlphaDualLayerPatchcoreModel(**kw, alpha_l2=ALPHA, alpha_l3=ALPHA).eval().to(dev)
    m_gate = m9b.ChannelGatePatchcoreModel(**kw, gate_l2=g2, gate_l3=g3).eval().to(dev)

    imgs = sorted((ROOT / "data" / "mvtec_ad" / "bottle" / "train" / "good").glob("*.png"))
    bs = 16
    max_abs, n_diff, n_tot = 0.0, 0, 0
    sum_abs_a, sum_abs_b = 0.0, 0.0
    per_batch = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.stack([e1b.preprocess_for_model(e1b.load_image_as_tensor(p), dev)
                             for p in imgs[i:i + bs]])
            f = m_alpha.feature_extractor(x)
            f = {k: m_alpha.feature_pooler(v) for k, v in f.items()}
            ea = m_alpha.generate_embedding({k: v.clone() for k, v in f.items()})
            eb = m_gate.generate_embedding({k: v.clone() for k, v in f.items()})
            d = (ea - eb).abs()
            per_batch.append({"batch": i // bs, "n_images": int(x.shape[0]),
                              "max_abs_dF": float(d.max()),
                              "n_diff_elements": int((ea != eb).sum()), "n_elements": int(ea.numel())})
            max_abs = max(max_abs, float(d.max()))
            n_diff += int((ea != eb).sum())
            n_tot += int(ea.numel())
            sum_abs_a += float(ea.abs().sum())
            sum_abs_b += float(eb.abs().sum())

    ev = {
        "control": "Control 2 — theoretically equivalent pair (uniform alpha=0.25 vs constant-gate 0.25)",
        "scope": "bottle train/good 全部 %d 张（与 fit 实际使用的训练集一致），batch=%d，同进程逐元素" % (len(imgs), bs),
        "elementwise_max_abs_dF": max_abs,
        "elementwise_identical": bool(n_diff == 0),
        "n_diff_elements": n_diff, "n_elements_total": n_tot,
        "sum_abs_uniform": sum_abs_a, "sum_abs_gate": sum_abs_b,
        "per_batch": per_batch,
        "note": ("9C 只报告聚合量；此处为真正的逐元素检查。跨进程的 bit-level 证据见 9C-v2b "
                 "coreset / memory bank sha256 与 max|dscore|（analysis/cross_process_equivalence.json）"),
    }
    (SUM / "equivalence_evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    print("elementwise max|dF| = %.3e | n_diff=%d / %d | sum_abs %.1f vs %.1f"
          % (max_abs, n_diff, n_tot, sum_abs_a, sum_abs_b))
    print("identical:", n_diff == 0, "| per-batch (first3):", per_batch[:3])


if __name__ == "__main__":
    main()
