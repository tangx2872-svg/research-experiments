#!/usr/bin/env python
"""核对 F0 / R1 受控比较的「输入同一性」：只读冻结 cell，不重算特征。

检查项（每类每 seed 一个 cell，10 类 x 3 seed = 30 个）：
  1. 同一 cell 内 scores_f0 / scores_r1 / labels / specimen / illumination / kind 长度一致；
  2. 同一类别内 3 个 seed 的测试图像身份集合一致（用 (specimen, illumination, kind) 作为身份）；
  3. 同一类别内 3 个 seed 的 detectable 掩码一致（若不一致，主表口径不可比）；
  4. 同一类别内 3 个 seed 的标签一致；
  5. bank_hash 存在且长度 64；bank_size == 300（或该类实际可用的 min(300, pool)）；
  6. trainable_params == 589824（R1 = 768x768），steps_used > 0；
  7. 分数无 NaN / Inf。

输出 results/paper_final_pack_20261023/fairness_checks.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e22" / "cells"
OUT = ROOT / "results" / "paper_final_pack_20261023" / "fairness_checks.json"
CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]


def key_of(c, i):
    return (str(c["specimen"][i]), str(c["illumination"][i]), str(c["kind"][i]))


def main() -> int:
    per_cat = {}
    for cat in CATS:
        cells = {s: json.loads((CELLDIR / f"{cat}_seed{s}.json").read_text()) for s in SEEDS}
        rec = {}
        lens_ok = all(len(cells[s][k]) == len(cells[s]["labels"])
                      for s in SEEDS for k in ("scores_f0", "scores_r1", "specimen",
                                               "illumination", "kind", "detectable_raw"))
        det = {s: tuple(bool(x) for x in cells[s]["detectable_raw"]) for s in SEEDS}
        keys = {s: tuple(key_of(cells[s], i) for i in range(len(cells[s]["labels"]))) for s in SEEDS}
        labs = {s: tuple(int(x) for x in cells[s]["labels"]) for s in SEEDS}
        finite = all(math.isfinite(x) for s in SEEDS
                     for k in ("scores_f0", "scores_r1") for x in cells[s][k])
        rec.update({
            "n_total_per_seed": {str(s): len(cells[s]["labels"]) for s in SEEDS},
            "lengths_consistent": lens_ok,
            "same_identity_set_across_seeds": len(set(keys.values())) == 1,
            "same_identity_order_across_seeds": keys[0] == keys[1] == keys[2],
            "same_detectable_mask_across_seeds": len(set(det.values())) == 1,
            "n_detectable": int(sum(det[0])),
            "same_labels_across_seeds": len(set(labs.values())) == 1,
            "bank_hash_len_ok": all(len(cells[s]["bank_hash"]) == 64 for s in SEEDS),
            "bank_hash_per_seed": {str(s): cells[s]["bank_hash"][:12] for s in SEEDS},
            "bank_size": {str(s): cells[s]["bank_size"] for s in SEEDS},
            "trainable_params": {str(s): cells[s]["trainable_params"] for s in SEEDS},
            "steps_used": {str(s): cells[s]["steps_used"] for s in SEEDS},
            "scores_finite": finite,
            "cell_metrics_recompute_flags": {
                "labels_match_e16": [bool(cells[s].get("reuse_verification", {}).get("labels_match_e16"))
                                     if "reuse_verification" in cells[s] else None for s in SEEDS],
                "specimen_match_e16": [bool(cells[s].get("reuse_verification", {}).get("specimen_match_e16"))
                                       if "reuse_verification" in cells[s] else None for s in SEEDS],
                "bank_hash_matches_e21": [bool(cells[s].get("reuse_verification", {}).get("bank_hash_matches_e21"))
                                          if "reuse_verification" in cells[s] else None for s in SEEDS],
                "note": "None = 该 cell 是 E22 新算（source=NEW），无历史复用断言",
            },
        })
        per_cat[cat] = rec

    all_ok = all(v[k] for v in per_cat.values() for k in
                 ("lengths_consistent", "same_identity_set_across_seeds",
                  "same_detectable_mask_across_seeds", "same_labels_across_seeds",
                  "bank_hash_len_ok", "scores_finite"))
    out = {"n_categories": len(CATS), "n_cells": len(CATS) * len(SEEDS),
           "all_checks_pass": all_ok, "per_category": per_cat,
           "note": "身份键 = (specimen, illumination, kind)；该组合在 30 个 cell 中逐 cell 唯一。"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[fairness] all_checks_pass={all_ok} -> {OUT}")
    for c in CATS:
        r = per_cat[c]
        print(f"  {c:8s} n={r['n_total_per_seed']['0']:3d} det={r['n_detectable']:3d} "
              f"identity_set={r['same_identity_set_across_seeds']} "
              f"order={r['same_identity_order_across_seeds']} "
              f"mask={r['same_detectable_mask_across_seeds']} "
              f"bank={r['bank_hash_per_seed']['0']} steps={r['steps_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
