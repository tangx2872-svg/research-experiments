#!/usr/bin/env python
"""E26-B: read-only early-stopping / checkpoint audit.

Static scan (no execution of the trainer) answers three questions and writes a
machine-readable step statistics table:

  Q1 does the final trainer track a best validation value?   -> yes (scalar only)
  Q2 does it snapshot the best weights?                      -> no
  Q3 which weights are used for scoring after early stop?    -> the current ones

Usage
-----
    python scripts/e26_b_checkpoint_audit.py
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e26" / "checkpoint_audit"

FINAL_TRAINER = "scripts/e13b_screen.py"
FINAL_IMPORTERS = {
    "scripts/e16_qualify.py": "E16 qualification",
    "scripts/e17_refine.py": "E17 BETA refine / freeze",
    "scripts/e21_multiview_validation.py": "E21 multi-view",
    "scripts/e22_full_category_validation.py": "E22 10-category x 3 seeds (main table)",
    "scripts/e24_runner.py": "E24 photometric sensitivity",
    "scripts/e25_robustad_eval.py": "E25 RobustAD external validation",
}
DEPRECATED = {"scripts/e12b_screen.py": "E12-B gate method (not the final method)"}

PATTERNS = {
    "snapshot_keywords": r"best_state|best_model|best_checkpoint|deepcopy|state_dict|"
                         r"torch\.save|load_state_dict",
    "tracking_keywords": r"best_val|no_improve|patience|EVAL_EVERY",
}

CELL_GLOBS = {
    "E22": "results/e22/cells/*.json",
    "E25": "results/e25/cells/*.json",
    "E24": "results/e24/cells/*.json",
}


def scan(path: Path) -> dict:
    text = path.read_text()
    hits = {}
    for name, pat in PATTERNS.items():
        found = []
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(pat, line):
                found.append({"line": i, "code": line.strip()[:120]})
        hits[name] = found
    return hits


def step_stats() -> dict:
    out = {}
    for exp, pattern in CELL_GLOBS.items():
        files = sorted(glob.glob(str(ROOT / pattern)))
        steps, best = [], []
        for f in files:
            rec = json.loads(Path(f).read_text())
            if isinstance(rec.get("steps_used"), int):
                steps.append(rec["steps_used"])
            for key in ("best_val", "best_val_cons"):
                if isinstance(rec.get(key), (int, float)):
                    best.append(float(rec[key]))
        out[exp] = {
            "n_cells": len(files),
            "n_with_steps_used": len(steps),
            "steps_min": min(steps) if steps else None,
            "steps_median": st.median(steps) if steps else None,
            "steps_max": max(steps) if steps else None,
            "n_hit_max_steps_300": sum(1 for s in steps if s >= 300),
            "n_early_stopped": sum(1 for s in steps if s < 300),
            "n_with_best_val_recorded": len(best),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    trainer = scan(ROOT / FINAL_TRAINER)
    result = {
        "date": "2026-11-01",
        "experiment": "E26-B",
        "final_trainer": FINAL_TRAINER,
        "final_config": "results/e17/final_method.json",
        "final_importers": FINAL_IMPORTERS,
        "deprecated_excluded": DEPRECATED,
        "findings": {
            "Q1_tracks_best_val": True,
            "Q2_saves_best_weights": False,
            "Q3_weights_used_for_scoring": "current weights at the stopping iteration "
                                           "(no best-checkpoint restore)",
            "Q3_code_evidence": [
                {"file": FINAL_TRAINER, "lines": "52-82",
                 "what": "best_val / no_improve updated; break when no_improve*EVAL_EVERY >= PATIENCE_STEPS"},
                {"file": FINAL_TRAINER, "lines": "85-90",
                 "what": "report uses the live adapter; no snapshot, no restore"},
                {"file": FINAL_TRAINER, "lines": "137-141",
                 "what": "run_cell() scores bank/test with the same adapter returned from train_r1"},
            ],
            "snapshot_keywords_in_final_trainer": trainer["snapshot_keywords"],
            "tracking_keywords_in_final_trainer": trainer["tracking_keywords"],
            "deprecated_contrast": {
                "file": "scripts/e12b_screen.py",
                "lines": "326, 333-334",
                "what": "this older E12-B path DOES snapshot/restore best_state; it is NOT on the "
                        "final import chain and must not be used to infer the final behaviour",
            },
            "minor_observations": [
                "steps_used is written after the eval branch, so on an early-stop break it records "
                "one step less than the number of executed steps (reporting only)",
                "best_val is a dead variable: it decides when to stop, never which weights to score",
            ],
        },
        "protocol_check": {
            "frozen_protocol_states_checkpoint_policy": False,
            "paper_facing_material_states_checkpoint_policy": False,
            "evidence": [
                "results/e16/protocol.md:45",
                "results/e16/audit.md:64",
                "results/e17/e17_decision_memo.md:31",
                "results/e17/audit.md:25",
                "results/e17/final_method.json (PATIENCE_STEPS/EVAL_EVERY/MAX_STEPS only)",
                "results/paper_ready/EVIDENCE_MAP.md:36",
            ],
            "verdict": "PASS_A",
            "verdict_reason": "implementation = current weights; neither the frozen protocol nor the "
                              "paper-facing material requires best-checkpoint restore, so there is "
                              "no contradiction. If the intended behaviour was best-restore, this "
                              "becomes ESCALATE and requires human adjudication.",
            "required_manuscript_sentence": "Early stopping monitors a validation consistency loss on "
                                            "60 held-out normal images (every 5 steps, patience 30); "
                                            "the parameters at the stopping iteration are used for "
                                            "evaluation - no best-validation checkpoint is restored.",
        },
        "step_stats": step_stats(),
    }
    p = OUT / "step_stats.json"
    p.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["step_stats"], indent=1))
    print(f"[E26-B] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
