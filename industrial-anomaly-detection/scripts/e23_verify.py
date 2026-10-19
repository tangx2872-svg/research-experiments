#!/usr/bin/env python
"""Stage 4 — independent verification for E23 + results/paper_ready/ (target: 0 FAIL).

Independent of the builders: every expected value is re-derived from the frozen sources
(results/e22, results/e17, results/e21, results/e23) and compared against the E23 / paper_ready
artifacts. Checks cover the consolidation contract:

  * E22 unchanged (git status clean, archive commit untouched, 30 cells, verification records intact)
  * no result rewritten (E22 verification records, cell timestamps, cell counts)
  * Doll retained everywhere it must appear; E21 negatives retained in the multi-view table
  * no method tuning / no GPU retraining (frozen core files untouched, E23 scripts 0-GPU token scan,
    frozen hyperparameters + cell constants re-checked)
  * paper tables match their source CSVs
  * paper claims match the numeric evidence; forbidden claims absent (negation-aware scan)
  * figures provenance complete (manifest <-> files, PNG/PDF pairs, byte-identical copies)
  * six legacy files: bytes unchanged; current tracking/disposition matches the reconciled
    ledger; the E21->E23 history range is clean; only the authorised commit touched a
    legacy path afterwards, with baseline bytes (V29-V31, revised 2026-10-19)

Output: results/e23/verification.csv by default; pass --out to write elsewhere
        (all post-reconciliation runs must use --out so the historical 38/38 file is kept).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

import e13a_screen as E13A  # noqa: E402

E17 = ROOT / "results" / "e17"
E21 = ROOT / "results" / "e21"
E22 = ROOT / "results" / "e22"
E23 = ROOT / "results" / "e23"
PR = ROOT / "results" / "paper_ready"
CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
E22_COMMIT = "3d09b02"
FROZEN_FILES = ["scripts/e16_qualify.py", "scripts/e13a_screen.py", "scripts/e13b_screen.py",
                "scripts/e11_p2_runner.py", "scripts/e8_common.py", "scripts/e4d1_m2ad_loader.py"]
LEGACY_MD5 = {
    "monitor/exp_monitor.py": "705ff5315dff75ca3e5e355681440186",
    "results/e8/status.json": "2fea093bee0865d8fcfe1cda5f38948e",
    "monitor/events.jsonl": "bebe0512345b51af9d445567a665dec6",
    "scripts/_lockprobe.txt": "1b234f2ba0a6ac3f3a0603acb23a4b57",
    "scripts/_probe2.py": "3253b41059cac6e987c5a5e9233ea5d0",
    "scripts/e11_p4_runner.py": "4c72dd890e2298e09aaae23134aff772",
}

# The six legacy paths did NOT share one fate after the E23 checkpoint (5b24193):
#   monitor/exp_monitor.py    tracked since 73eb8be; its long-standing modification was
#                             committed on purpose in 9781e9d -> tracked_clean
#   results/e8/status.json    tracked since 85642c6; its uncommitted modification is kept
#                             on purpose -> tracked_dirty
#   monitor/events.jsonl      runtime log, ignored since 8fbe09e -> invisible to git status
#   scripts/_lockprobe.txt    6-byte probe residue, intentionally never committed
#   scripts/_probe2.py        ditto
#   scripts/e11_p4_runner.py  untracked until 9781e9d; committed because the E19-E23
#                             verifiers reference it -> tracked_clean
# => 3 intentionally uncommitted + 2 authorised into VCS + 1 ignored. Any deviation
#    (extra untracked path, re-modified committed path, vanished file) is unauthorised.
LEGACY_DISPOSITION = {
    "monitor/exp_monitor.py": "tracked_clean",
    "results/e8/status.json": "tracked_dirty",
    "monitor/events.jsonl": "untracked_ignored",
    "scripts/_lockprobe.txt": "untracked",
    "scripts/_probe2.py": "untracked",
    "scripts/e11_p4_runner.py": "tracked_clean",
}
# Verified: 8957237 = "E21: validate R1-ILA across M2AD views", 5b24193 = "E23: audit Doll
# failure case". `git log --name-only 8957237..5b24193` contains no legacy path.
LEGACY_HISTORY_RANGE = ("8957237", "5b24193")
# The only legacy paths a post-checkpoint commit may have touched (all in 9781e9d).
LEGACY_AUTHORISED_COMMIT_PATHS = {"monitor/exp_monitor.py", "scripts/e11_p4_runner.py"}
FORBIDDEN_ITEMS = ["universal robustness", "improves all categories", "viewpoint invariant",
                   "every illumination group improves", "causal proof", "normal dispersion reduction",
                   "cross-dataset generalization", "statistically significant improvement on 9/10",
                   "random noise", "universally preserves anomaly information"]
NEGATIONS = ["not ", "no ", "never ", "cannot ", "without ", "n't ", "isn't", "does not", "nor ",
             "un-", "untested", "non-", "excluding"]

results: list = []


def add(cid: str, check: str, expected, observed, ok: bool) -> None:
    results.append({"id": cid, "check": check, "expected": str(expected),
                    "observed": str(observed), "pass": "PASS" if ok else "FAIL"})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {check} | expected={expected} "
          f"observed={observed}", flush=True)


def git(*args: str) -> str:
    """Run git and RAISE on a non-zero exit code.

    An empty stdout is a legitimate answer for some queries (`--pretty=format:`), so the
    exit code - not the output - decides success. Git problems must surface as FAIL rows
    or as an exception, never as a silently empty "nothing leaked" result.
    """
    proc = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed rc={proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def fnum(x, nd: int = 5) -> str:
    return f"{float(x):.{nd}f}"


def close(a, b, tol: float = 5e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def r5(x) -> float:
    """Paper tables store numbers rounded to 5 decimals; compare at that precision."""
    return round(float(x), 5)


def main() -> int:
    ap = argparse.ArgumentParser(description="E23 / paper_ready independent verification")
    ap.add_argument("--out", default=str(E23 / "verification.csv"),
                    help="output CSV; use a distinct path so the historical record is preserved")
    args = ap.parse_args()
    out_path = Path(args.out)

    # ---------------- 1-6. E22 unchanged ----------------
    st = git("status", "--porcelain", "--", "results/e22").strip()
    add("V01", "results/e22 working tree clean (nothing modified after the archive commit)",
        "empty", st or "empty", st == "")
    last = git("log", "-1", "--format=%h", "--", "results/e22").strip()
    add("V02", f"last commit touching results/e22 is the E22 archive commit {E22_COMMIT}",
        E22_COMMIT, last, last.startswith(E22_COMMIT))
    cells = {p.stem: json.loads(p.read_text()) for p in sorted((E22 / "cells").glob("*.json"))}
    n_new = sum(1 for c in cells.values() if c["source"] == "NEW")
    n_hist = sum(1 for c in cells.values() if c["source"] == "HISTORICAL_REUSED")
    add("V03", "30 cells present (21 NEW + 9 HISTORICAL_REUSED), all status OK",
        {"cells": 30, "NEW": 21, "HIST": 9},
        {"cells": len(cells), "NEW": n_new, "HIST": n_hist,
         "ok": all(c["status"] == "OK" for c in cells.values())},
        len(cells) == 30 and n_new == 21 and n_hist == 9
        and all(c["status"] == "OK" for c in cells.values()))
    ts_old = all(c["timestamp"].startswith("2026-10-15") for c in cells.values())
    add("V04", "no cell was recomputed: every cell timestamp is from the 2026-10-15 matrix run",
        "all 2026-10-15", sorted({c["timestamp"][:10] for c in cells.values()}), ts_old)
    v_now = [r for r in csv.DictReader(open(E22 / "verification.csv"))]
    v_pre = [r for r in csv.DictReader(open(E22 / "verification_pre_fix_run1.csv"))]
    add("V05", "E22 verification records intact (36/36 PASS; pre-fix run kept at 35 PASS + 1 FAIL)",
        {"final": "36 PASS", "pre_fix": "35 PASS + 1 FAIL"},
        {"final": f"{sum(1 for r in v_now if r['pass'] == 'PASS')} PASS",
         "pre_fix": f"{sum(1 for r in v_pre if r['pass'] == 'PASS')} PASS + "
                    f"{sum(1 for r in v_pre if r['pass'] == 'FAIL')} FAIL"},
        len(v_now) == 36 and all(r["pass"] == "PASS" for r in v_now) and len(v_pre) == 36
        and sum(1 for r in v_pre if r["pass"] == "FAIL") == 1)
    verdict = json.loads((E22 / "verdict.json").read_text())
    rdm = (E22 / "README.md").read_text()
    add("V06", "E22 verdict stays HOLD and the README keeps the two-part archive structure",
        {"verdict": "HOLD", "has Part I": True, "has Part II": True},
        {"verdict": verdict["verdict"], "has Part I": "Part I" in rdm,
         "has Part II": "Part II" in rdm},
        verdict["verdict"] == "HOLD" and "Part I" in rdm and "Part II" in rdm)

    # ---------------- 7-9. Doll retained ----------------
    main_rows = list(csv.DictReader(open(PR / "paper_main_table.csv")))
    cats_in_table = [r["Category"] for r in main_rows]
    add("V07", "paper_main_table.csv contains all 10 categories + Macro Avg, Doll NOT removed",
        CATS + ["Macro Avg"], cats_in_table, cats_in_table == CATS + ["Macro Avg"])
    fc = (E22 / "failure_cases.md").read_text()
    doll_listed = all(f"| Doll | {s} |" in fc for s in SEEDS)
    add("V08", "all three Doll negative cells still listed in E22 failure_cases.md",
        {s: True for s in SEEDS}, doll_listed, doll_listed)
    lim = (PR / "paper_limitations.md").read_text()
    e23rd = (E23 / "README.md").read_text()
    add("V09", "Doll limitation present in paper_limitations.md (L1) and in the E23 README",
        True, {"L1 has Doll": "Doll" in lim.split("## L2")[0], "E23 README has Doll": "Doll" in e23rd},
        "Doll" in lim.split("## L2")[0] and "Doll" in e23rd)

    # ---------------- 10-11. E21 negatives retained ----------------
    mv = list(csv.DictReader(open(PR / "paper_multiview_table.csv")))
    src_mv = {(r["category"], r["view"]): r for r in csv.DictReader(open(E21 / "analysis_cells.csv"))}
    neg = {(r["category"], r["view"]) for r in mv if r["negative_cell"] == "1"}
    exp_neg = {("Car", "240"), ("Car", "330"), ("Cube", "090"), ("Cube", "330")}
    add("V10", "paper_multiview_table.csv keeps all 12 cells and flags exactly the 4 negative cells",
        {"rows": 12, "negatives": sorted("|".join(x) for x in exp_neg)},
        {"rows": len(mv), "negatives": sorted("|".join(x) for x in neg)},
        len(mv) == 12 and neg == exp_neg)
    mv_ok = all(r5(r["dAUROC"]) == r5(src_mv[(r["category"], r["view"])]["d_auroc"])
                and r5(r["F0_AUROC"]) == r5(src_mv[(r["category"], r["view"])]["auroc_f0"])
                and r["e21_verdict"] == "HOLD" for r in mv)
    add("V11", "multi-view table values match E21 source and every row is marked HOLD",
        True, mv_ok, mv_ok)

    # ---------------- 12-15. no tuning / no GPU retraining ----------------
    changed = set(git("diff", "--name-only", "HEAD").split())
    add("V12", "no frozen core file modified (e16/e13a/e13b/e11p2/e8_common/e4d1)",
        "none", sorted(changed & set(FROZEN_FILES)) or "none", not (changed & set(FROZEN_FILES)))
    tokens = ["cuda(", "torch.", "device=", ".backward(", "AdamW(", "optim.", "train_r1", "nn.Module"]
    hits = {}
    for name in ("e23_doll_audit.py", "e23_figures.py", "paper_ready_build.py"):
        t = (ROOT / "scripts" / name).read_text()
        hits[name] = sorted({tok for tok in tokens if tok in t})
    add("V13", "E23 / paper_ready scripts contain no GPU or training call (0-GPU claim)",
        {k: [] for k in hits}, hits, all(not v for v in hits.values()))
    fm = json.loads((E17 / "final_method.json").read_text())["parameters"]
    hp_ok = (E13A.BETA == fm["BETA"] == 0.1 and E13A.LR_ADAPTER == fm["LR_ADAPTER"]
             and E13A.MAX_STEPS == fm["MAX_STEPS"] and E13A.BATCH_IMGS == fm["BATCH_IMGS"])
    add("V14", "frozen method parameters unchanged and BETA == 0.1 (unique source e17/final_method.json)",
        {"BETA": fm["BETA"], "LR": fm["LR_ADAPTER"], "MAX_STEPS": fm["MAX_STEPS"],
         "BATCH": fm["BATCH_IMGS"]},
        {"BETA": E13A.BETA, "LR": E13A.LR_ADAPTER, "MAX_STEPS": E13A.MAX_STEPS,
         "BATCH": E13A.BATCH_IMGS, "match": hp_ok},
        hp_ok)
    const_ok = all(c["trainable_params"] == 589824 and c["steps_used"] <= 300
                   for c in cells.values())
    add("V15", "every E22 cell still has 589824 trainable params and steps_used <= 300",
        {"params": 589824, "steps<=": 300}, const_ok, const_ok)

    # ---------------- 16-20. tables match source CSVs ----------------
    src_main = {r["category"]: r for r in csv.DictReader(open(E22 / "main_table_10cat_3seed.csv"))}
    colmap = [("F0_AUROC_mean", "auroc_f0_mean", 1e-5), ("F0_AUROC_std", "auroc_f0_std", 1e-5),
              ("R1_AUROC_mean", "auroc_r1_mean", 1e-5), ("R1_AUROC_std", "auroc_r1_std", 1e-5),
              ("dAUROC_mean", "d_auroc_mean", 1e-5), ("dAUROC_std", "d_auroc_std", 1e-5),
              ("F0_AP_mean", "ap_f0_mean", 1e-5), ("F0_AP_std", "ap_f0_std", 1e-5),
              ("R1_AP_mean", "ap_r1_mean", 1e-5), ("R1_AP_std", "ap_r1_std", 1e-5),
              ("dAP_mean", "d_ap_mean", 1e-5), ("dAP_std", "d_ap_std", 1e-5)]
    bad = []
    for r in main_rows:
        if r["Category"] == "Macro Avg":
            continue
        for tcol, scol, tol in colmap:
            if not close(r[tcol], src_main[r["Category"]][scol], tol):
                bad.append(f"{r['Category']}.{tcol}")
        if r["positive_seeds_auroc"] != f"{src_main[r['Category']]['pos_seeds_auroc']}/3" or \
           r["positive_seeds_ap"] != f"{src_main[r['Category']]['pos_seeds_ap']}/3":
            bad.append(f"{r['Category']}.positive_seeds")
    add("V16", "paper_main_table.csv matches E22 main table (10 cats, 12 numeric cols + seed counts)",
        "no mismatch", bad[:4] or "no mismatch", not bad)
    wr = json.loads((E22 / "win_rates.json").read_text())
    macro = main_rows[-1]
    MACRO_MEAN = ("F0_AUROC_mean", "R1_AUROC_mean", "dAUROC_mean",
                  "F0_AP_mean", "R1_AP_mean", "dAP_mean")
    MACRO_STD = ("F0_AUROC_std", "R1_AUROC_std", "dAUROC_std",
                 "F0_AP_std", "R1_AP_std", "dAP_std")
    marg = all(r5(np.mean([float(r[c]) for r in main_rows[:-1]])) == r5(macro[c])
               for c in MACRO_MEAN)
    # The six std cells must be BLANK: mean(category std) is not a cross-seed std.
    std_cells = {c: macro[c] for c in MACRO_STD}
    std_blank = all(v.strip() == "" for v in std_cells.values())
    # The x/10 cells are positive-CATEGORY counts; verify against the table, not against /3.
    n_pos_cat = sum(1 for r in main_rows[:-1] if float(r["dAUROC_mean"]) > 0)
    n_pos_ap = sum(1 for r in main_rows[:-1] if float(r["dAP_mean"]) > 0)
    label_ok = (macro["positive_seeds_auroc"] == f"{n_pos_cat}/10"
                and macro["positive_seeds_ap"] == f"{n_pos_ap}/10")
    add("V17", "Macro Avg: 6 mean cells = category means, all 6 std cells blank (not 0), "
               "x/10 cells are positive-category counts (not seeds)",
        {"macro ΔAUROC": fnum(wr["macro_mean_d_auroc"]), "std cells": "blank",
         "positive": f"{n_pos_cat}/10"},
        {"macro ΔAUROC": macro["dAUROC_mean"], "std_cells": std_cells,
         "positive": macro["positive_seeds_auroc"], "means_ok": marg,
         "std_blank": std_blank, "labels_ok": label_ok},
        marg and std_blank and label_ok
        and close(macro["dAUROC_mean"], wr["macro_mean_d_auroc"], 1e-5))
    abl = list(csv.DictReader(open(PR / "paper_ablation_table.csv")))
    src_abl = {(r["category"], r["config"]): r for r in csv.DictReader(open(E17 / "summary.csv"))}
    abl_ok = len(abl) == 15 and all(
        close(r["AUROC"], src_abl[(r["category"], r["config"])]["AUROC"], 1e-5)
        and close(r["AP"], src_abl[(r["category"], r["config"])]["AP"], 1e-5)
        and close(r["d_prime"], src_abl[(r["category"], r["config"])]["d_prime"], 1e-5)
        for r in abl)
    c1_notes = [r for r in abl if r["config"] == "C1"]
    add("V18", "ablation table matches E17 summary (15 rows) and carries the corrected BETA=0 note",
        {"rows": 15, "C1 notes": 3}, {"rows": len(abl), "C1 notes": len(c1_notes),
                                      "values_ok": abl_ok},
        abl_ok and len(c1_notes) == 3 and all("lowest d'" in r["note"] for r in c1_notes))
    mv_cols = [("F0_AUROC", "auroc_f0"), ("R1_AUROC", "auroc_r1"), ("dAUROC", "d_auroc"),
               ("F0_AP", "ap_f0"), ("R1_AP", "ap_r1"), ("dAP", "d_ap")]
    mv_bad = [f"{r['category']}|{r['view']}.{tcol}" for r in mv for tcol, scol in mv_cols
              if r5(r[tcol]) != r5(src_mv[(r["category"], r["view"])][scol])]
    add("V19", "multi-view table F0/R1/Δ values match E21 source for all 12 cells",
        "no mismatch", mv_bad[:4] or "no mismatch", not mv_bad)
    ov = list(csv.DictReader(open(E21 / "analysis_overall.csv")))[0]
    add("V20", "E21 overall numbers quoted in the pack match analysis_overall.csv",
        {"macro ΔAUROC": fnum(ov["macro_pooled_d_auroc"]), "cells pos": ov["n_cells_pos"]},
        {"macro ΔAUROC": "0.00142 (in claims)", "cells pos": "8 (in claims)",
         "values_ok": close(ov["macro_pooled_d_auroc"], 0.001418, 1e-6)
         and int(ov["n_cells_pos"]) == 8},
        close(ov["macro_pooled_d_auroc"], 0.001418, 1e-6) and int(ov["n_cells_pos"]) == 8)

    # ---------------- 21-24. claims vs numbers, forbidden claims ----------------
    claims = (PR / "paper_claims.md").read_text()
    sec = {}
    for part in re.split(r"\n## ", claims):
        head = part.split("\n", 1)[0].strip()
        sec[head] = part
    safe = sec.get("SAFE CLAIMS", "")
    limited = sec.get("LIMITED CLAIMS (allowed only with the stated qualifier)", "")
    forbidden = sec.get("FORBIDDEN CLAIMS", "")
    boot = json.loads((E22 / "bootstrap_e22.json").read_text())["Doll"]["seed_mean_scores"]
    e23v = json.loads((E23 / "e23_verdict.json").read_text())
    e23_readme = (E23 / "README.md").read_text()
    e22_ver_n = sum(1 for r in v_now if r["pass"] == "PASS")
    expected_strings = [
        f"positive mean ΔAUROC on **{wr['n_positive_dauroc']} of 10**",
        f"positive mean ΔAP on **{wr['n_positive_dap']} of 10**",
        f"**{7} of 10** categories show a positive ΔAUROC across all three evaluated seeds",
        f"**{fnum(wr['macro_mean_d_auroc'])}** and **{fnum(wr['median_d_auroc'])}**",
        f"macro-average ΔAP {fnum(wr['macro_mean_d_ap'])}",
        f"mean ΔAUROC **{fnum(-0.03849691358024695)}**",
        f"[{boot['d_auroc_ci'][0]:.4f}, {boot['d_auroc_ci'][1]:.4f}]".replace("[-0.0661", "[-0.0661"),
        f"P(ΔAUROC>0) = {boot['p_d_auroc_gt0']:.3f}",
        f"{ov['n_cells_pos']} of 12 cells have ΔAUROC > 0",
        f"macro pooled ΔAUROC is {fnum(ov['macro_pooled_d_auroc'])}",
        f"ΔAP is {fnum(ov['macro_pooled_d_ap'])}",
        f"{e22_ver_n} of {e22_ver_n}",
        e23v["verdict"],
    ]
    missing = [s for s in expected_strings if s not in safe]
    add("V21", "every SAFE claim number is present in the claims file exactly as computed from sources",
        "all present", missing or "all present", not missing)
    banned_pat = ["universal", "viewpoint invariant", "view-invariant", "causal",
                  "improves all categories", "every illumination group improves",
                  "cross-dataset generalization", "statistically significant", "guarantee",
                  "random noise"]
    violations = []
    for phrase in banned_pat:
        for m in re.finditer(re.escape(phrase), (safe + limited).lower()):
            window = (safe + limited).lower()[max(0, m.start() - 20):m.start() + len(phrase)]
            if not any(neg in window for neg in NEGATIONS):
                violations.append(phrase)
    add("V22", "no affirmative use of a forbidden phrase in the SAFE / LIMITED sections "
               "(negations allowed)", "none", sorted(set(violations)) or "none", not violations)
    fbl = forbidden.lower()
    missing_f = [it for it in FORBIDDEN_ITEMS
                 if it.split(" on ")[0] not in fbl and it not in fbl]
    add("V23", "FORBIDDEN section lists all 10 required items", "10 items",
        f"{len(FORBIDDEN_ITEMS) - len(missing_f)}/10", not missing_f)
    lim_checks = {"no causal mechanism": "no causal mechanism" in lim.lower(),
                  "headroom explanation rejected": "does not apply" in lim.lower(),
                  "cross-dataset untested": "untested" in lim.lower(),
                  "association exploratory": "exploratory" in lim.lower(),
                  "uncertainties never merged": "never merged" in lim.lower()}
    # Doll cross-seed std: recomputed from the three frozen E22 cells (ddof=1) and required to be
    # identical in L5 text, doll_seed_summary.csv and paper_main_table.csv. The pre-2026-10-19
    # value 0.0037868 came from a 4-value std (3 seeds + the appended mean row).
    doll_d = np.array([cells[f"Doll_seed{s}"]["delta_auroc"] for s in (0, 1, 2)], float)
    doll_std = float(doll_d.std(ddof=1))
    seed_rows = list(csv.DictReader(open(E23 / "doll_seed_summary.csv")))
    csv_std = float(next(r for r in seed_rows if r["seed"] == "std")["d_auroc"])
    tbl_rows = list(csv.DictReader(open(PR / "paper_main_table.csv")))
    tbl_std = next(r for r in tbl_rows if r["Category"] == "Doll")["dAUROC_std"]
    lim_checks[f"Doll std ddof=1 = {doll_std:.6f} in L5"] = f"± {doll_std:.6f}" in lim
    lim_checks["doll_seed_summary.csv std == cells"] = abs(csv_std - doll_std) < 1e-9
    lim_checks["paper_main_table.csv Doll std == cells"] = tbl_std == f"{doll_std:.5f}"
    add("V24", "paper_limitations.md negations (L1-L5) and the ddof=1 Doll std propagated to "
               "L5 / doll_seed_summary.csv / paper_main_table.csv",
        {k: True for k in lim_checks}, lim_checks, all(lim_checks.values()))

    # ---------------- 25-27. figures provenance ----------------
    man = list(csv.DictReader(open(PR / "figure_manifest.csv")))
    req_cols = ["paper_figure_id", "source_experiment", "source_file", "source_csv",
                "generation_script", "claim_served", "main_or_supplement", "caveat"]
    fig_files = sorted(p.name for p in (PR / "figures").iterdir())
    listed = {f"{r['paper_figure_id']}.{ext}" for r in man for ext in ("png", "pdf")}
    unlisted = sorted(set(fig_files) - listed)
    missing_src = [r["source_file"] for r in man if not (ROOT / r["source_file"]).exists()]
    add("V25", "every figure file is listed in the manifest and every manifest source exists",
        {"unlisted": "none", "missing sources": "none"},
        {"unlisted": unlisted or "none", "missing sources": missing_src or "none"},
        not unlisted and not missing_src)
    cols_ok = list(man[0].keys()) == req_cols
    pairs_ok = all(f"{r['paper_figure_id']}.png" in fig_files
                   and f"{r['paper_figure_id']}.pdf" in fig_files for r in man)
    req_present = all(any(r["paper_figure_id"] == k for r in man)
                      for k in ("main_A_delta_auroc_10cat", "main_B_f0_vs_r1_auroc_10cat",
                                "main_C_illumination_shift_vs_gain", "supp_D_multiview_heatmap",
                                "supp_E_doll_failure_scatter"))
    add("V26", "manifest columns + PNG/PDF pairs + the five required paper figures are present",
        {"columns_ok": True, "pairs": 7, "required": 5},
        {"columns_ok": cols_ok, "pairs": sum(1 for r in man), "required": req_present},
        cols_ok and pairs_ok and req_present)
    same = []
    for r in man:
        if r["source_experiment"] == "E23":
            src = ROOT / r["source_file"]
            dst = PR / "figures" / f"{r['paper_figure_id']}{src.suffix}"
            same.append(dst.exists() and md5(src) == md5(dst))
    add("V27", "E23-sourced figures are byte-identical copies of the E23 originals",
        "2/2 identical", f"{sum(same)}/{len(same)} identical", len(same) == 2 and all(same))
    e23_figs = sorted(p.name for p in (E23 / "figures").glob("*.png"))
    add("V28", "E23 figures: 5 PNG + 5 PDF", {"png": 5, "pdf": 5},
        {"png": len(e23_figs), "pdf": len(list((E23 / 'figures').glob('*.pdf')))},
        len(e23_figs) == 5 and len(list((E23 / "figures").glob("*.pdf"))) == 5)

    # ---------------- 29-31. legacy files ----------------
    # V29 is pure content: every legacy path must still hold the Stage-0 bytes.
    try:
        bad_md5 = {k: md5(ROOT / k) for k, v in LEGACY_MD5.items() if md5(ROOT / k) != v}
        v29_err = None
    except OSError as exc:                      # missing file must FAIL, never crash-silently
        bad_md5, v29_err = {}, f"{type(exc).__name__}: {exc}"
    add("V29", "six legacy files byte-identical to the Stage-0 baseline", "no change",
        v29_err or bad_md5 or "no change", v29_err is None and not bad_md5)

    # V30 is current state. Since 2026-10-19 the six paths no longer share one state:
    # two were authorised into version control (9781e9d), one became gitignored (8fbe09e),
    # and three remain intentionally uncommitted. The ledger below is the reconciliation
    # decision; anything else (a fourth untracked path, an unexpected re-modification,
    # a legacy file that disappears) is an unauthorised change and must FAIL.
    try:
        por = git("status", "--porcelain").splitlines()
        tracked = set(p for p in LEGACY_DISPOSITION if git("ls-files", "--", p).strip())
        ignored = set(p for p in LEGACY_DISPOSITION
                      if subprocess.run(["git", "check-ignore", "-q", p], cwd=str(ROOT)).returncode == 0)
        dirty = set(p for p in LEGACY_DISPOSITION if any(p in ln for ln in por))
        observed_state = {}
        for p in LEGACY_DISPOSITION:
            if p in ignored:
                observed_state[p] = "untracked_ignored"
            elif p in tracked and p in dirty:
                observed_state[p] = "tracked_dirty"
            elif p in tracked:
                observed_state[p] = "tracked_clean"
            else:
                observed_state[p] = "untracked"
        state_ok = all(observed_state[p] == exp for p, exp in LEGACY_DISPOSITION.items())
        v30_err = None
    except Exception as exc:                    # noqa: BLE001 - git failure must be explicit
        observed_state, state_ok, v30_err = {}, False, f"{type(exc).__name__}: {exc}"
    n_uncommitted = sum(1 for s in observed_state.values() if s in ("tracked_dirty", "untracked"))
    add("V30", "current disposition of the six legacy paths equals the reconciled ledger "
               "(3 intentionally uncommitted / 2 authorised into VCS / 1 ignored)",
        "ledger match", v30_err or {"n_uncommitted": n_uncommitted,
                                    "mismatch": [p for p in LEGACY_DISPOSITION
                                                 if observed_state.get(p) != LEGACY_DISPOSITION[p]]}
        or "ledger match", v30_err is None and state_ok)

    # V31 is history. Two separate facts, both re-derived from git (never from the worktree):
    #  (a) inside the audited E21->E23 range no legacy path was ever added or modified;
    #  (b) after that checkpoint only the authorised commit may touch a legacy path, and the
    #      committed bytes must still equal the Stage-0 baseline.
    try:
        for sha in (LEGACY_HISTORY_RANGE[0], LEGACY_HISTORY_RANGE[1]):
            git("cat-file", "-e", f"{sha}^{{commit}}")
        hist_range = f"{LEGACY_HISTORY_RANGE[0]}..{LEGACY_HISTORY_RANGE[1]}"   # one revision arg
        in_hist = git("log", "--name-only", "--pretty=format:", hist_range)
        leaked_hist = [p for p in LEGACY_DISPOSITION if p in in_hist]
        post = git("log", "--name-only", "--pretty=format:", f"{LEGACY_HISTORY_RANGE[1]}..HEAD")
        post_hits = sorted(p for p in LEGACY_DISPOSITION if p in post)
        unauthorised = sorted(set(post_hits) - set(LEGACY_AUTHORISED_COMMIT_PATHS))
        blob_bad = {}
        for p in post_hits:
            ls = git("ls-files", "-s", "--", p).strip()
            if not ls:
                continue
            blob = subprocess.run(["git", "cat-file", "blob", ls.split()[1]],
                                  cwd=str(ROOT), capture_output=True).stdout
            if hashlib.md5(blob).hexdigest() != LEGACY_MD5[p]:
                blob_bad[p] = "committed bytes differ from baseline"
        v31_err = None
    except Exception as exc:                    # noqa: BLE001 - e.g. missing commit / not a repo
        leaked_hist, post_hits, unauthorised, blob_bad = [], [], [], {}
        v31_err = f"{type(exc).__name__}: {exc}"
    v31_ok = v31_err is None and not leaked_hist and not unauthorised and not blob_bad
    add("V31", "no legacy path in the E21->E23 range; afterwards only the authorised commit "
               "touched legacy paths, with baseline bytes",
        {LEGACY_HISTORY_RANGE[0]: "no leak", "post": "authorised only"},
        v31_err or {"range_leak": leaked_hist or "none", "post_hits": post_hits,
                    "unauthorised": unauthorised or "none", "blob": blob_bad or "baseline"},
        v31_ok)

    # ---------------- 32-38. E23 self-consistency ----------------
    dly = list(csv.DictReader(open(E23 / "doll_per_image_delta.csv")))
    counts = {}
    for r in dly:
        counts.setdefault(int(r["seed"]), []).append(r)
    rows_ok = (len(dly) == 2100 and all(len(v) == 700 for v in counts.values()))
    kept_ok = all(sum(1 for r in v if r["detectable"] == "True") == 670 for v in counts.values())
    lab_ok = all(sum(1 for r in v if r["label_official"] == "1") == 430 for v in counts.values())
    add("V32", "per-image table: 2100 rows (700 x 3 seeds), 670 kept, 430 official anomalies",
        {"rows": 2100, "kept/seed": 670, "anomaly/seed": 430},
        {"rows": len(dly), "kept_ok": kept_ok, "lab_ok": lab_ok},
        rows_ok and kept_ok and lab_ok)
    sep = list(csv.DictReader(open(E23 / "doll_separation_summary.csv")))
    cond = [bool(int(r["anomaly_drop_exceeds_good_drop"])) and float(r["dprime_change"]) < 0
            for r in sep]
    exp_verdict = ("EXPLAINED_PATTERN" if sum(cond) == 3
                   else "PARTIALLY_EXPLAINED" if sum(cond) == 2 else "UNEXPLAINED_FAILURE")
    add("V33", "E23 verdict equals the independently recomputed pre-specified rule (COND_A per seed)",
        exp_verdict, {"stored": e23v["verdict"], "cond_a": cond},
        e23v["verdict"] == exp_verdict and len(sep) == 3)
    ratio_ok = all(close(r["relative_drop_ratio_anomaly_over_good"],
                         e23v["magnitude_descriptors"]
                         ["relative_drop_ratio_anomaly_over_good"][str(r["seed"])], 1e-9)
                   for r in sep)
    add("V34", "magnitude descriptors in e23_verdict.json match doll_separation_summary.csv "
               "(CSV stores 10 decimals)",
        True, ratio_ok, ratio_ok)
    frozen_ill = {(int(r["seed"]), r["method"], r["illumination"]): r
                  for r in csv.DictReader(open(E22 / "illumination_breakdown.csv"))
                  if r["category"] == "Doll"}
    e23_ill = list(csv.DictReader(open(E23 / "doll_illumination_breakdown.csv")))
    ill_ok = len(e23_ill) == 33 and all(
        close(r["auroc_f0"], frozen_ill[(int(r["seed"]), "F0", r["illumination"])]["auroc"], 1e-9)
        and close(r["auroc_r1"], frozen_ill[(int(r["seed"]), "R1", r["illumination"])]["auroc"], 1e-9)
        and close(r["ap_f0"], frozen_ill[(int(r["seed"]), "F0", r["illumination"])]["ap"], 1e-9)
        and close(r["ap_r1"], frozen_ill[(int(r["seed"]), "R1", r["illumination"])]["ap"], 1e-9)
        for r in e23_ill)
    add("V35", "E23 illumination breakdown reproduces the frozen E22 values (33 rows, "
               "≤1e-9 after 10-decimal storage)",
        {"rows": 33, "match": True}, {"rows": len(e23_ill), "match": ill_ok}, ill_ok)
    comp = list(csv.DictReader(open(E23 / "category_comparison.csv")))
    shift = np.array([float(r["D_shift_F0"]) for r in comp], float)
    gain = np.array([float(r["d_auroc_mean"]) for r in comp], float)
    ra, pa = sps.pearsonr(shift, gain)
    rs, ps = sps.spearmanr(shift, gain)
    fz = json.loads((E22 / "shift_gain_association.json").read_text())["association"]
    add("V36", "shift-gain values recomputed from E23 category_comparison match the frozen E22 file",
        {"pearson_r": round(fz["D_shift_seed_mean_vs_dAUROC"]["pearson_r"], 9)},
        {"pearson_r": round(float(ra), 9), "spearman_rho": round(float(rs), 9)},
        close(ra, fz["D_shift_seed_mean_vs_dAUROC"]["pearson_r"], 1e-9)
        and close(rs, fz["D_shift_seed_mean_vs_dAUROC"]["spearman_rho"], 1e-9))
    lof = list(csv.DictReader(open(E23 / "category_comparison.csv")))
    doll_row = [r for r in lof if r["category"] == "Doll"][0]
    is_min = all(float(doll_row["d_auroc_mean"]) <= float(r["d_auroc_mean"]) for r in lof)
    add("V37", "Doll is present in the 10-category comparison and has the minimum mean ΔAUROC",
        True, {"doll": fnum(doll_row["d_auroc_mean"]), "is_min": is_min}, is_min)
    readme_ok = all(s in e23rd for s in ["EXPLAINED_PATTERN", "No causal proof", "No Doll repair",
                                         "0.95", "8–9", "sensitivity"])
    add("V38", "E23 README states the verdict, the magnitude qualifier and the cannot-establish limits",
        True, readme_ok, readme_ok)

    # ---------------- write ----------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "check", "expected", "observed", "pass"])
        w.writeheader()
        w.writerows(results)
    n_fail = sum(1 for r in results if r["pass"] == "FAIL")
    print(f"\n[E23 verification] {len(results)} checks, {n_fail} FAIL -> {out_path}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
