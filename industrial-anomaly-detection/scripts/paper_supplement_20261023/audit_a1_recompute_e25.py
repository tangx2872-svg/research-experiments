#!/usr/bin/env python
"""A1: 从冻结 E25 cell 的逐图分数独立复算 AUROC/AP，并对账冻结 cell 值。

- 仅读 results/e25/cells/*.json 的 scores_f0/scores_r1/labels/files，不依赖原图。
- 独立用 sklearn roc_auc_score / average_precision_score 复算，与 cell 内 f0/r1 对账。
- 检查样本对齐：files 唯一性、长度一致、labels 二进制、score NaN/Inf、排序错位。
- 检查"分数越大越异常"的实现方向（score = 1 - max cosine，越大越异常）。
- 不修改任何冻结产物；结果写独立目录。

产物: metric_reconciliation.csv / sample_alignment_audit.json
"""
from __future__ import annotations
import json
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e25" / "cells"
OUTDIR = ROOT / "results" / "paper_supplement_20261023"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOL = 1e-9  # 复算对账容差（全精度 float 应几乎位级一致）


def main():
    cells = sorted(CELLDIR.glob("*.json"))
    if not cells:
        raise SystemExit("no cells found in results/e25/cells")

    rec_rows = []       # metric_reconciliation
    align_rows = []     # sample_alignment_audit

    for cp in cells:
        cell = json.loads(cp.read_text())
        domain = cell["domain"]
        split = cell["split"]
        seed = cell["seed"]
        valid = cell["valid"]
        status = cell.get("status")
        s0 = np.asarray(cell["scores_f0"], dtype=float)
        s1 = np.asarray(cell["scores_r1"], dtype=float)
        labels = np.asarray(cell["labels"], dtype=int)
        files = cell["files"]

        # --- 样本对齐检查 ---
        align = {
            "cell": cp.name, "domain": domain, "split": split, "seed": seed,
            "valid": valid, "status": status,
            "n_total": cell.get("n_total"),
            "len_scores_f0": len(s0), "len_scores_r1": len(s1),
            "len_labels": len(labels), "len_files": len(files),
            "length_mismatch": not (len(s0) == len(s1) == len(labels) == len(files)),
            "n_unique_files": len(set(files)),
            "has_duplicate_files": len(set(files)) != len(files),
            "nan_inf_f0": bool(np.isnan(s0).any() or np.isinf(s0).any()),
            "nan_inf_r1": bool(np.isnan(s1).any() or np.isinf(s1).any()),
            "labels_binary": bool(set(labels.tolist()) <= {0, 1}),
            "n_label_0": int((labels == 0).sum()),
            "n_label_1": int((labels == 1).sum()),
            "two_classes": bool(len(set(labels.tolist())) == 2),
            "score_range_f0": [float(np.min(s0)), float(np.max(s0))],
            "score_range_r1": [float(np.min(s1)), float(np.max(s1))],
            # 分数越大越异常：异常类(label=1)的平均分数应 >= 正常类(label=0)平均分数（若非，方向可疑）
            "mean_score_label0_f0": float(s0[labels == 0].mean()),
            "mean_score_label1_f0": float(s0[labels == 1].mean()),
            "direction_ok_f0": bool(s0[labels == 1].mean() >= s0[labels == 0].mean()),
        }
        align_rows.append(align)

        # --- 独立复算 AUROC/AP ---
        if align["two_classes"] and not align["nan_inf_f0"] and not align["nan_inf_r1"]:
            rec_f0 = roc_auc_score(labels, s0)
            rec_f1 = roc_auc_score(labels, s1)
            ap_f0 = average_precision_score(labels, s0)
            ap_f1 = average_precision_score(labels, s1)
        else:
            rec_f0 = rec_f1 = ap_f0 = ap_f1 = np.nan

        d_auroc = rec_f1 - rec_f0
        d_ap = ap_f1 - ap_f0
        rec_rows.append({
            "cell": cp.name, "domain": domain, "split": split, "seed": seed,
            "valid": valid, "status": status,
            "n_total": cell.get("n_total"),
            "auroc_f0_cell": cell["f0"]["auroc"],
            "auroc_f0_recomp": rec_f0,
            "auroc_f0_absdiff": None if np.isnan(rec_f0) else abs(cell["f0"]["auroc"] - rec_f0),
            "ap_f0_cell": cell["f0"]["ap"],
            "ap_f0_recomp": ap_f0,
            "ap_f0_absdiff": None if np.isnan(ap_f0) else abs(cell["f0"]["ap"] - ap_f0),
            "auroc_r1_cell": cell["r1"]["auroc"],
            "auroc_r1_recomp": rec_f1,
            "auroc_r1_absdiff": None if np.isnan(rec_f1) else abs(cell["r1"]["auroc"] - rec_f1),
            "ap_r1_cell": cell["r1"]["ap"],
            "ap_r1_recomp": ap_f1,
            "ap_r1_absdiff": None if np.isnan(ap_f1) else abs(cell["r1"]["ap"] - ap_f1),
            "d_auroc_cell": cell["d_auroc"],
            "d_auroc_recomp": d_auroc,
            "d_auroc_absdiff": None if np.isnan(d_auroc) else abs(cell["d_auroc"] - d_auroc),
            "d_ap_cell": cell["d_ap"],
            "d_ap_recomp": d_ap,
            "d_ap_absdiff": None if np.isnan(d_ap) else abs(cell["d_ap"] - d_ap),
            "recompute_status": "OK" if (not np.isnan(rec_f0) and
                abs(cell["f0"]["auroc"] - rec_f0) <= TOL and
                abs(cell["r1"]["auroc"] - rec_f1) <= TOL and
                abs(cell["f0"]["ap"] - ap_f0) <= TOL and
                abs(cell["r1"]["ap"] - ap_f1) <= TOL) else "MISMATCH_OR_SKIP",
        })

    # --- 写出 metric_reconciliation.csv ---
    csv_path = OUTDIR / "metric_reconciliation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec_rows[0].keys()))
        w.writeheader()
        w.writerows(rec_rows)

    # --- 汇总 ---
    n_ok = sum(1 for r in rec_rows if r["recompute_status"] == "OK")
    n_mismatch = sum(1 for r in rec_rows if r["recompute_status"] == "MISMATCH_OR_SKIP")
    max_absdiff = max((r["auroc_f0_absdiff"] for r in rec_rows
                       if r["auroc_f0_absdiff"] is not None), default=0.0)
    any_align_bad = [a for a in align_rows if (a["length_mismatch"] or a["has_duplicate_files"]
                     or a["nan_inf_f0"] or a["nan_inf_r1"] or not a["labels_binary"])]

    summary = {
        "n_cells": len(cells),
        "n_recompute_ok": n_ok,
        "n_mismatch_or_skip": n_mismatch,
        "max_absdiff_auroc_f0": max_absdiff,
        "tol": TOL,
        "cells_with_alignment_issue": [a["cell"] for a in any_align_bad],
        "n_alignment_issue": len(any_align_bad),
        "direction_check_all_ok": all(a["direction_ok_f0"] for a in align_rows),
    }

    # --- 写出 sample_alignment_audit.json ---
    with open(OUTDIR / "sample_alignment_audit.json", "w") as f:
        json.dump({"summary": summary, "per_cell": align_rows}, f, indent=1)

    # --- 打印结论 ---
    print(f"cells={len(cells)}  recompute_ok={n_ok}  mismatch/skip={n_mismatch}  "
          f"max_absdiff={max_absdiff:.3e}  align_issue={len(any_align_bad)}  "
          f"direction_ok={summary['direction_check_all_ok']}")
    if n_mismatch:
        for r in rec_rows:
            if r["recompute_status"] != "OK":
                print("  MISMATCH:", r["cell"], "f0", r["auroc_f0_cell"], "vs",
                      r["auroc_f0_recomp"], "| r1", r["auroc_r1_cell"], "vs", r["auroc_r1_recomp"])
    print(f"wrote {csv_path}")
    print(f"wrote {OUTDIR / 'sample_alignment_audit.json'}")


if __name__ == "__main__":
    main()
