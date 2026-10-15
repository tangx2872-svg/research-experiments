#!/usr/bin/env python
"""E22 independent verification: >= 20 checks -> results/e22/verification.csv (target 0 FAIL).

Verification is INDEPENDENT of the runner: it re-derives scopes, labels, metrics and the
verdict from first principles (metadata + stored per-image scores) and compares against the
artifacts written by scripts/e22_full_category_validation.py and scripts/e22_analysis.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e11_p2_runner as P2  # noqa: E402
import e13a_screen as E13A  # noqa: E402
import e16_qualify as E16  # noqa: E402

OUT = Path(__import__("os").environ.get("E22_OUT", str(ROOT / "results" / "e22")))
CELLD = Path(__import__("os").environ.get("E22_CELLS", str(OUT / "cells")))
HIST = ["Bird", "Car", "Cube"]
NEW = ["Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
CATS = HIST + NEW
SEEDS = [0, 1, 2]
FROZEN_FILES = ["scripts/e16_qualify.py", "scripts/e13a_screen.py", "scripts/e13b_screen.py",
                "scripts/e11_p2_runner.py", "scripts/e8_common.py", "scripts/e4d1_m2ad_loader.py"]
LEGACY = ["monitor/exp_monitor.py", "results/e8/status.json", "monitor/events.jsonl",
          "scripts/_lockprobe.txt", "scripts/_probe2.py", "scripts/e11_p4_runner.py"]

results = []


def add(cid: str, check: str, expected, observed, ok: bool) -> None:
    results.append({"id": cid, "check": check, "expected": str(expected),
                    "observed": str(observed), "pass": "PASS" if ok else "FAIL"})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {check} | expected={expected} "
          f"observed={observed}", flush=True)


def rel_of(rec) -> str:
    return str(Path(rec["image_path"]).relative_to(LD.DATA_ROOT))


def load_cells():
    out = {}
    for c in CATS:
        for s in SEEDS:
            p = CELLD / f"{c}_seed{s}.json"
            if p.exists():
                out[(c, s)] = json.loads(p.read_text())
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True).stdout


def main() -> int:
    cells = load_cells()
    meta = json.loads((ROOT / "data/m2ad/jsons_extract/meta_unsupervised.json").read_text())
    main_rows = {r["category"]: r for r in csv.DictReader(open(OUT / "main_table_10cat_3seed.csv"))}
    win = json.loads((OUT / "win_rates.json").read_text())
    verd = json.loads((OUT / "verdict.json").read_text())
    boot = json.loads((OUT / "bootstrap_e22.json").read_text())
    gate = json.loads((OUT / "repro_gate.json").read_text()) if (OUT / "repro_gate.json").exists() else {}

    # ---- 1-3. completeness
    cats_present = sorted({c for (c, _) in cells})
    add("V01", "10/10 categories present in results/e22/cells", sorted(CATS), cats_present,
        cats_present == sorted(CATS))
    add("V02", "30/30 cells present", 30, len(cells), len(cells) == 30)
    per_cat = {c: sorted(s for (cc, s) in cells if cc == c) for c in CATS}
    add("V03", "exactly 3 seeds per category", {c: SEEDS for c in CATS}, per_cat,
        all(per_cat[c] == SEEDS for c in CATS))
    add("V04", "no duplicate cells (unique category x seed keys)", 30, len(set(cells)),
        len(set(cells)) == 30)
    add("V05", "view120 only", {"120"}, sorted({c["view"] for c in cells.values()}),
        {c["view"] for c in cells.values()} == {"120"})

    # ---- 6-9. independent re-derivation of scope, bank, labels, metrics
    lbl_ok = det_ok = met_ok = bank_ok = spec_ok = True
    trap_hits = 0
    details = []
    for (cat, seed), cell in cells.items():
        recs = LD.load_records(category=cat, require_exists=False)
        est = LD.estimate_scope(recs)
        pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
        bank = E8.bank_for(pool, seed)
        test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
        rawmap = P2.raw_test_map(cat)
        labels = [int(rawmap[rel_of(r)]["image_anomaly"]) for r in test]
        det = [rawmap[rel_of(r)]["detectable"] is not False for r in test]
        specs = [r["specimen"] for r in test]
        # bank hash = sha256 over "<image_path>\n" lines (frozen runner definition)
        h = hashlib.sha256()
        for r in bank:
            h.update(r["image_path"].encode()); h.update(b"\n")
        bank_ok &= (len(bank) == 300 and h.hexdigest() == cell["bank_hash"])
        lbl_ok &= (labels == cell["labels"])
        det_ok &= (det == cell["detectable_raw"])
        spec_ok &= (specs == cell["specimen"])
        trap_hits += sum(1 for r, lab in zip(test, labels) if "/NG/" in rel_of(r) and lab == 0)
        keep = np.array(det, bool)
        y = np.array(labels, int)[keep]
        a = np.array(cell["scores_f0"], float)[keep]
        b = np.array(cell["scores_r1"], float)[keep]
        ra, rb = roc_auc_score(y, a), roc_auc_score(y, b)
        pa, pb = average_precision_score(y, a), average_precision_score(y, b)
        met_ok &= (abs(ra - cell["f0"]["auroc"]) <= 1e-12 and abs(rb - cell["r1"]["auroc"]) <= 1e-12
                   and abs(pa - cell["f0"]["ap"]) <= 1e-12 and abs(pb - cell["r1"]["ap"]) <= 1e-12)
        if abs(ra - cell["f0"]["auroc"]) > 1e-12:
            details.append(f"{cat}|{seed} f0 {ra} vs {cell['f0']['auroc']}")
    add("V06", "labels identical to official metadata image_anomaly (all 30 cells)", True, lbl_ok,
        lbl_ok)
    add("V07", "detectable filter identical to metadata (all 30 cells)", True, det_ok, det_ok)
    add("V08", "specimen order identical to metadata scope (all 30 cells)", True, spec_ok, spec_ok)
    add("V09", "bank hash reproduced from frozen bank_for rule (all 30 cells)", True, bank_ok,
        bank_ok)
    add("V10", "AUROC/AP recomputed from stored per-image scores (tol 1e-12)", True,
        met_ok if not details else details[:2], met_ok)
    add("V11", "label trap handled: NG-directory files with image_anomaly=0 exist and are "
               "labelled 0", ">=1", trap_hits, trap_hits >= 1)

    # ---- 12-15. frozen hyperparameters / code
    fm = json.loads((ROOT / "results" / "e17" / "final_method.json").read_text())["parameters"]
    hp = {"BETA": E13A.BETA, "LR": E13A.LR_ADAPTER, "MAX_STEPS": E13A.MAX_STEPS,
          "PATIENCE": E13A.PATIENCE_STEPS, "TRAIN_N": E13A.TRAIN_N, "VAL_N": E13A.VAL_N,
          "BATCH": E13A.BATCH_IMGS}
    exp_hp = {"BETA": fm["BETA"], "LR": fm["LR_ADAPTER"], "MAX_STEPS": fm["MAX_STEPS"],
              "PATIENCE": fm["PATIENCE_STEPS"], "TRAIN_N": fm["TRAIN_N"], "VAL_N": fm["VAL_N"],
              "BATCH": fm["BATCH_IMGS"]}
    add("V12", "frozen hyperparameters (BETA/LR/steps/patience/train/val/batch)", exp_hp, hp,
        hp == exp_hp)
    aug_expected = [("brightness", 0.85), ("brightness", 1.15), ("contrast", 0.85),
                    ("contrast", 1.15), ("gamma", 0.85), ("gamma", 1.15)]
    add("V13", "augmentation list unchanged (6 photometric variants)", aug_expected,
        [tuple(x) for x in E13A.AUG_LIST], [tuple(x) for x in E13A.AUG_LIST] == aug_expected)
    add("V14", "bank rule unchanged (seed 20261009, size 300)",
        {"seed": 20261009, "size": 300},
        {"seed": E8.BANK_SUBSET_SEED, "size": E8.BANK_SIZE},
        E8.BANK_SUBSET_SEED == 20261009 and E8.BANK_SIZE == 300)
    steps_max = max(c["steps_used"] for c in cells.values())
    add("V15", "R1-ILA training never exceeded MAX_STEPS=300 (all cells)",
        f"<=300", steps_max, steps_max <= 300)
    add("V16", "trainable params = 589824 in every cell", {589824},
        sorted({c["trainable_params"] for c in cells.values()}),
        {c["trainable_params"] for c in cells.values()} == {589824})

    # ---- 17-19. reproduction gate
    ga = gate.get("results", {}).get("GateA_Bird0_two_code_paths", {})
    add("V17", "Gate A: Bird|0 frozen path vs E22 copy bit-identical", True,
        ga.get("bit_identical"), bool(ga.get("bit_identical")))
    gb = {k: v for k, v in gate.get("results", {}).items() if k.startswith("GateB")}
    add("V18", "Gate B: Bird|0, Car|0, Cube|0, Car|1 reproduce history (<=1e-6)",
        {k: True for k in gb}, {k: v.get("repro_pass") for k, v in gb.items()},
        len(gb) == 4 and all(v.get("repro_pass") for v in gb.values()))
    add("V19", "repro gate overall all_pass=True", True, gate.get("all_pass"),
        bool(gate.get("all_pass")))

    # ---- 20-21. historical reuse
    reuse_ok = all(all(c.get("reuse_verification", {}).values())
                   for (cat, seed), c in cells.items() if c["source"] == "HISTORICAL_REUSED")
    n_hist = sum(1 for c in cells.values() if c["source"] == "HISTORICAL_REUSED")
    add("V20", "all 9 historical cells passed reuse verification (labels/bank/metrics)",
        "9 cells, all checks true", f"{n_hist} cells, reuse_ok={reuse_ok}",
        n_hist == 9 and reuse_ok)
    n_new = sum(1 for c in cells.values() if c["source"] == "NEW")
    add("V21", "21 new cells computed by E22 (7 categories x 3 seeds)", 21, n_new, n_new == 21)

    # ---- 22-24. analysis artifacts consistency
    csv_rows = list(csv.DictReader(open(OUT / "all10_3seed_cells.csv")))
    add("V22", "all10_3seed_cells.csv has 30 rows", 30, len(csv_rows), len(csv_rows) == 30)
    complete = len(cells) == len(CATS) * len(SEEDS)
    avail = [c for c in CATS if c in main_rows]
    mt_ok = True
    for cat in avail:
        v = [cells[(cat, s)]["r1"]["auroc"] - cells[(cat, s)]["f0"]["auroc"]
             for s in SEEDS if (cat, s) in cells]
        if v:
            mt_ok &= abs(float(main_rows[cat]["d_auroc_mean"]) - float(np.mean(v))) < 5e-7
    add("V23", "main_table ΔAUROC mean matches recomputation from cells (<5e-7)",
        True, f"{mt_ok} (complete={complete})", mt_ok and complete)
    n_pos = sum(1 for c in avail if float(main_rows[c]["d_auroc_mean"]) > 0)
    add("V24", "win_rates.json positive-category count recomputed", n_pos,
        win["n_positive_dauroc"], n_pos == win["n_positive_dauroc"] and complete)

    # ---- 25. verdict independently recomputed
    macro = float(np.mean([float(main_rows[c]["d_auroc_mean"]) for c in avail]))
    med = float(np.median([float(main_rows[c]["d_auroc_mean"]) for c in avail]))
    n_sev = sum(1 for c in avail if float(main_rows[c]["d_auroc_mean"]) < -0.03)
    strong = n_pos >= 7 and macro > 0 and med > 0 and n_sev == 0
    stop = n_pos <= 4 or (macro <= 0 and med <= 0) or n_sev >= 2
    v_exp = "STRONG GO" if strong else ("STOP BROAD-GENERALIZATION CLAIM" if stop else "HOLD")
    add("V25", "verdict.json matches independent recomputation of the pre-registered rule",
        v_exp, verd["verdict"], v_exp == verd["verdict"] and complete)
    add("V26", "verdict.json conditions consistent with observed counts",
        {"n_pos": n_pos, "macro>0": macro > 0, "median>0": med > 0, "n_severe": n_sev},
        verd.get("conditions"), bool(verd.get("conditions")) and (
            verd["conditions"]["c1_pos_categories_ge_7"] == (n_pos >= 7)
            and verd["conditions"]["c2_macro_mean_gt_0"] == (macro > 0)
            and verd["conditions"]["c3_median_gt_0"] == (med > 0)
            and verd["conditions"]["c4_no_category_below_-0.03"] == (n_sev == 0)))

    # ---- 27. bootstrap determinism
    if not complete:
        print("[E22 verification] matrix incomplete -> remaining checks evaluate on partial data",
              flush=True)
    c = cells.get(("Bird", 0), next(iter(cells.values())))
    keep = np.array(c["detectable_raw"], bool)
    y = np.array(c["labels"], int)[keep]
    a = np.array(c["scores_f0"], float)[keep]
    b = np.array(c["scores_r1"], float)[keep]
    def rerun():
        rng = np.random.default_rng(20261015)
        n = len(y)
        das = []
        for _ in range(3000):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            das.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
        return [float(np.percentile(das, 2.5)), float(np.percentile(das, 97.5))]
    ci_a = rerun()
    ci_b = rerun()
    stored = boot["Bird"]["per_seed"]["0"]["d_auroc_ci"]
    add("V27", "bootstrap deterministic (same RNG -> identical CI) and matches stored CI",
        True, {"identical": ci_a == ci_b, "matches_stored": ci_a == stored},
        ci_a == ci_b and ci_a == stored)

    # ---- 28-29. figures + README/artifact presence
    figs = sorted(p.name for p in (OUT / "figures").glob("*.png")) if (OUT / "figures").is_dir() else []
    pdfs = sorted(p.name for p in (OUT / "figures").glob("*.pdf")) if (OUT / "figures").is_dir() else []
    add("V28", "8 figures present as PNG and PDF", 8, f"{len(figs)} png / {len(pdfs)} pdf",
        len(figs) == 8 and len(pdfs) == 8)
    req = ["README.md", "protocol.md", "main_table_10cat_3seed.csv", "category_summary.csv",
           "illumination_breakdown.csv", "statistical_summary.md", "failure_cases.md",
           "verification.csv", "live_results.csv", "all10_3seed_cells.csv"]
    missing = [f for f in req if not (OUT / f).exists()]
    add("V29", "all required artifacts exist", "none missing", missing or "none", not missing)

    # ---- 30. negative results retained
    neg_cells = [(k, v) for k, v in cells.items() if v["delta_auroc"] < 0]
    fc = (OUT / "failure_cases.md").read_text()
    listed = sum(1 for (cat, seed), _ in neg_cells if f"| {cat} | {seed} |" in fc)
    add("V30", "all negative cells listed in failure_cases.md", len(neg_cells), listed,
        listed == len(neg_cells))

    # ---- 31-34. hygiene: frozen files untouched, E21 untouched, legacy files untouched
    changed = set(git("diff", "--name-only", "HEAD").split())
    add("V31", "no frozen core file modified (e16/e13a/e13b/e11p2/e8_common/e4d1)",
        "none", sorted(changed & set(FROZEN_FILES)) or "none",
        not (changed & set(FROZEN_FILES)))
    e21_changed = sorted(x for x in changed if x.startswith("results/e21") or x.startswith("scripts/e21"))
    add("V32", "E21 artifacts and scripts untouched", "none", e21_changed or "none",
        not e21_changed)
    e22_base = "8957237"                       # last commit before E22 started
    since = {ln.strip() for ln in git("log", "--name-only", "--pretty=format:", f"{e22_base}..HEAD").splitlines()
             if ln.strip()}
    since |= {ln[3:].strip() for ln in git("status", "--porcelain").splitlines()
              if ln[:2].strip() and not ln.startswith("??")}
    legacy_in_e22 = sorted(x for x in LEGACY if x in since)
    add("V33", "six legacy files never included in any E22 commit/stage", "none",
        legacy_in_e22 or "none", not legacy_in_e22)
    porcelain = git("status", "--porcelain")
    present = all((ROOT / x).exists() for x in LEGACY)
    still_dirty = all(x in porcelain for x in LEGACY)
    add("V34", "six legacy files still present and still uncommitted (state preserved)", True,
        {"present": present, "still_dirty": still_dirty}, present and still_dirty)

    # ---- 35. cache integrity (data provenance)
    agg_p = ROOT / "results" / "e22" / "cache_l0_448_aggregate_sha256.txt"
    agg = agg_p.read_text().strip()
    add("V35", "E22 cache aggregate sha256 unchanged (data provenance)",
        "1569cc57f8c316056b3f445c13779e36aee6e861433c020c3d0a683a16e47812", agg,
        agg == "1569cc57f8c316056b3f445c13779e36aee6e861433c020c3d0a683a16e47812")

    # ---- 36. illumination breakdown completeness
    illum_rows = list(csv.DictReader(open(OUT / "illumination_breakdown.csv")))
    exp_rows = len(CATS) * len(SEEDS) * 2 * 11
    add("V36", "illumination_breakdown.csv complete (cats x seeds x methods x (ALL+10))",
        exp_rows, len(illum_rows), len(illum_rows) == exp_rows)

    with open(OUT / "verification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "check", "expected", "observed", "pass"])
        w.writeheader()
        w.writerows(results)
    n_fail = sum(1 for r in results if r["pass"] == "FAIL")
    print(f"\n[E22 verification] {len(results)} checks, {n_fail} FAIL -> {OUT / 'verification.csv'}",
          flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
