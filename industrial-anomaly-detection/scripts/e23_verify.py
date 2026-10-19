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
  * six legacy files: bytes unchanged; per-path current disposition matches the reconciled
    ledger (V30, scoped to those six paths, never a whole-repo claim); every post-checkpoint
    commit touching a legacy path is inspected individually using the blob of THAT commit
    (V31: full hashes, endpoint subjects, A/M only, D/R/merge/restore rejected)
  * errata provenance: Doll summaries recomputed from the three frozen cells (V39), blank
    macro-std convention (V17/V40), E18 anchor identity (V41), bootstrap-proportion wording
    (V42), E25 invariants (V43)
  * V30/V31 reverse tests in isolated repositories must actually FAIL every broken scenario
    (V44); the generator must reproduce all six text artefacts byte-for-byte in a temp OUT (V45)

Output: results/e23/verification.csv by default; pass --out to write elsewhere
        (all post-reconciliation runs must use --out so historical records are kept).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
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
E22_ARCHIVE_COMMIT = "3d09b02d5b862c4f50d7122c9441c07e29207f05"
E22_ARCHIVE_SUBJECT = "E22: complete ten-category three-seed validation"
# The only two E22 documents a post-archive commit may change, and ONLY by appending this exact
# text (byte-for-byte). The appendix hash is the contract; the phrases make the intent readable
# and stop a future edit from satisfying the hash with unrelated prose.
E22_AUTHORISED_DOCS = {
    "results/e22/README.md": {
        "appendix_sha256":
            "36b75953bd78b2990d1d4a3800fb0158dbb3702e171be1aa0123c86a25401d77",
        "phrases": ["P_boot(ΔAUROC > 0)", "传统假设检验的单侧或双侧 p 值",
                    "scripts/e22_analysis.py:226", "不改变"]},
    "results/e22/statistical_summary.md": {
        "appendix_sha256":
            "ed1eb8d7d6e800b3d6dae552646dbdbd8d89ace91881bdeda0dce0d630e2a473",
        "phrases": ["重采样比例", "不是假设检验的单侧或双侧 p 值", "P_boot(ΔAUROC>0)"]},
}
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
# Full hashes (V31 resolves and re-checks them, so a short-hash collision cannot retarget the
# audit). Subjects are asserted, not assumed:
#   8957237... = "E21: validate R1-ILA across M2AD views"
#   5b24193... = "E23: audit Doll failure case and consolidate paper evidence"
LEGACY_HISTORY_RANGE = ("89572372e1edd580d173b59dc31afa3ac6581d1a",
                        "5b24193ff26998d109f23e789625e568e7957708")
LEGACY_HISTORY_SUBJECTS = {"start": "E21: validate R1-ILA across M2AD views",
                           "end": "E23: audit Doll failure case and consolidate paper evidence"}
# Commit full hash -> the ONLY legacy paths that commit may touch (9781e9d committed
# monitor/exp_monitor.py as M and scripts/e11_p4_runner.py as A, both with baseline bytes).
LEGACY_AUTHORISED = {"9781e9de09b902bcf22765f169524efd09bbbd83":
                     {"monitor/exp_monitor.py", "scripts/e11_p4_runner.py"}}
MACRO_STD_COLS = ("F0_AUROC_std", "R1_AUROC_std", "dAUROC_std",
                  "F0_AP_std", "R1_AP_std", "dAP_std")


def macro_std_blank(rows) -> bool:
    """True iff the Macro Avg row leaves every std cell blank (blank = not defined, never 0)."""
    macro = next(r for r in rows if r["Category"] == "Macro Avg")
    return all(macro[c].strip() == "" for c in MACRO_STD_COLS)


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


class GitFailure(RuntimeError):
    """A git subprocess exited non-zero. Never downgrade this to an empty/negative answer."""


def git_in(repo: Path, *args: str, binary: bool = False):
    """Run git in `repo`; raise GitFailure on a non-zero exit code.

    An empty stdout is a legitimate answer for some queries (`--pretty=format:`), so the
    exit code - not the output - decides success. `binary=True` returns raw bytes (needed
    for blob hashing); text mode decodes as UTF-8.
    """
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=not binary)
    if proc.returncode != 0:
        err = proc.stderr
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        raise GitFailure(f"git {' '.join(args)} failed rc={proc.returncode} in {repo}: "
                         f"{err.strip()[:300]}")
    return proc.stdout


def git(*args: str) -> str:
    """git in the project root (raises GitFailure, which is a RuntimeError)."""
    return git_in(ROOT, *args)


_PREFIX_CACHE: dict = {}


def repo_prefix(repo: Path) -> str:
    """Workspace-relative prefix of `repo` inside its git toplevel ('' when equal).

    `git status --porcelain -z`, `diff-tree` and `show <rev>:<path>` do not all speak the
    same path dialect (some are repo-relative, `ls-files` is cwd-relative), so every path is
    normalised through this prefix before comparison.
    """
    key = str(repo.resolve())
    if key not in _PREFIX_CACHE:
        _PREFIX_CACHE[key] = git_in(repo, "rev-parse", "--show-prefix").strip()
    return _PREFIX_CACHE[key]


def ws_path(path: str, prefix: str) -> str:
    return path[len(prefix):] if prefix and path.startswith(prefix) else path


def status_map(repo: Path) -> dict:
    """Exact workspace-relative path -> two-letter porcelain status (no substring matching)."""
    prefix = repo_prefix(repo)
    raw = git_in(repo, "status", "--porcelain", "-z", "--untracked-files=all")
    fields = raw.split("\0")
    out, i = {}, 0
    while i < len(fields):
        entry = fields[i]
        if not entry:
            i += 1
            continue
        xy, path = entry[:2], ws_path(entry[3:], prefix)
        out[path] = xy
        # porcelain -z packs "XY PATH" into ONE NUL-terminated field; only rename/copy adds a
        # second field (the old path). Advancing by 2 for every entry silently skips every other
        # file, which would hide a dirty legacy path.
        i += 2 if (xy[0] in "RC" or xy[1] in "RC") else 1
    return out


def is_ignored(repo: Path, path: str) -> bool:
    """True/False from `git check-ignore`; any other exit code is a hard error.

    rc=0 -> ignored, rc=1 -> not ignored, anything else (128: not a repo, bad pathspec, ...)
    must NOT be mistaken for "not ignored".
    """
    proc = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=str(repo),
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise GitFailure(f"git check-ignore {path} failed rc={proc.returncode}: "
                     f"{proc.stderr.strip()[:200]}")


def legacy_states(repo: Path, ledger: dict) -> tuple:
    """Observed disposition of the ledger paths only (never a whole-repo claim).

    A vanished file is reported as `missing`, never as `untracked`: `ls-files` still lists
    deleted-but-tracked paths, so existence has to be checked on disk first.
    """
    st = status_map(repo)
    observed = {}
    for p in ledger:
        tracked = p in [x for x in git_in(repo, "ls-files", "-z", "--", p).split("\0") if x]
        if not (repo / p).exists():
            observed[p] = "missing"
        elif tracked:
            observed[p] = "tracked_dirty" if p in st else "tracked_clean"
        elif is_ignored(repo, p):
            observed[p] = "untracked_ignored"
        else:
            observed[p] = "untracked"
    # Only untracked paths are reported as extras; ordinary modifications elsewhere in the
    # repository are not "extra legacy files" and must not be dressed up as such.
    extras = sorted(p for p in st if p not in ledger and st[p].startswith("??"))
    return observed, extras


def audit_v30(repo: Path, ledger: dict) -> dict:
    observed, extras = legacy_states(repo, ledger)
    mismatch = [p for p in ledger if observed.get(p) != ledger[p]]
    return {"ok": not mismatch, "observed": observed, "mismatch": mismatch,
            "extra_untracked": extras}


def changed_paths(repo: Path, commit: str) -> tuple:
    """[(status, workspace path, old path)] for one commit, plus is_merge.

    Merge commits are compared against their FIRST parent only (documented choice); rename
    detection is enabled so a rename surfaces as R instead of A+D, but both forms are caught
    because either the old or the new path is matched against the legacy set.
    """
    prefix = repo_prefix(repo)
    parents = git_in(repo, "rev-list", "--parents", "-n", "1", commit).split()
    is_merge = len(parents) > 2
    if is_merge:
        raw = git_in(repo, "diff-tree", "-r", "-M", "--no-commit-id", "--name-status", "-z",
                     f"{commit}^1", commit)
    else:
        raw = git_in(repo, "diff-tree", "-r", "-M", "--no-commit-id", "--name-status", "-z",
                     "--root", commit)
    fields = [x for x in raw.split("\0") if x != ""]
    out, i = [], 0
    while i < len(fields):
        status = fields[i]
        if status[0] in "RC":
            out.append((status, ws_path(fields[i + 2], prefix), ws_path(fields[i + 1], prefix)))
            i += 3
        else:
            out.append((status, ws_path(fields[i + 1], prefix), None))
            i += 2
    return out, is_merge


def audit_v31(repo: Path, legacy_paths, baseline_md5: dict, hist_range, authorised: dict,
              expected_subjects: dict) -> dict:
    """Per-commit provenance of the legacy paths (never a path-set diff).

    1. Both endpoints are resolved to full commit hashes and their subjects are checked, so
       the audited range cannot silently drift.
    2. The historical range must contain no legacy change at all.
    3. Every commit after the checkpoint is inspected individually: identity, path, change
       type, and the blob of THAT commit (never the index/HEAD blob). Only `authorised`
       commits may touch only their authorised paths, with status A/M and baseline bytes.
    4. Deletions, renames, merges (first parent) and unauthorised commits all fail; a
       modify-then-restore pair fails on the first commit even when HEAD bytes match again.
    """
    start = git_in(repo, "rev-parse", f"{hist_range[0]}^{{commit}}").strip()
    end = git_in(repo, "rev-parse", f"{hist_range[1]}^{{commit}}").strip()
    subjects = {"start": git_in(repo, "log", "-1", "--format=%s", start).strip(),
                "end": git_in(repo, "log", "-1", "--format=%s", end).strip()}
    subject_ok = (subjects["start"] == expected_subjects["start"]
                  and subjects["end"] == expected_subjects["end"])
    legacy = set(legacy_paths)
    prefix = repo_prefix(repo)

    def scan(rev_range):
        hits = []
        for c in [x for x in git_in(repo, "rev-list", rev_range).split() if x]:
            changes, is_merge = changed_paths(repo, c)
            for status, path, old in changes:
                if path in legacy or (old in legacy if old else False):
                    hits.append({"commit": c, "path": path if path in legacy else old,
                                 "new_path": path if old else None,
                                 "status": status, "merge": is_merge})
        return hits

    hist_hits = scan(f"{start}..{end}")
    post_hits = scan(f"{end}..HEAD")
    n_hist = len([x for x in git_in(repo, "rev-list", f"{start}..{end}").split() if x])
    # Count only post-checkpoint commits that TOUCH a legacy path. The raw commit count grows
    # with every (unrelated) commit, including the commit that records this very CSV, which
    # would make the record self-referential and force an endless sequence of record commits.
    n_post_legacy = len({h["commit"] for h in post_hits})
    unauthorised, blob_bad = [], []
    for h in post_hits:
        # Hash the blob of THAT commit for every A/M entry, including unauthorised ones: this is
        # what makes "modify then restore" visible even when HEAD bytes match again.
        h["blob_md5"] = None
        if h["status"][0] in "AM" and not h["merge"]:
            try:
                blob = git_in(repo, "show", f"{h['commit']}:{prefix}{h['path']}", binary=True)
                h["blob_md5"] = hashlib.md5(blob).hexdigest()
            except GitFailure:
                h["blob_md5"] = "missing"
        allowed = authorised.get(h["commit"], set())
        if h["path"] not in allowed or h["status"][0] not in "AM" or h["merge"]:
            unauthorised.append(h)
            continue
        if h["blob_md5"] != baseline_md5[h["path"]]:
            blob_bad.append(h)
    # Current HEAD bytes for the authorised paths must still be the baseline (the worktree is
    # covered by V29 separately).
    head_bad = {}
    for p in sorted(set().union(*authorised.values()) if authorised else set()):
        try:
            digest = hashlib.md5(git_in(repo, "show", f"HEAD:{prefix}{p}", binary=True)).hexdigest()
        except GitFailure:
            head_bad[p] = "missing at HEAD"
            continue
        if digest != baseline_md5[p]:
            head_bad[p] = "HEAD bytes differ from baseline"
    restored = sorted({h["path"] for h in post_hits
                       if isinstance(h["blob_md5"], str) and h["blob_md5"] not in (None, "missing")
                       and h["blob_md5"] != baseline_md5[h["path"]]})
    ok = subject_ok and not hist_hits and not unauthorised and not blob_bad and not head_bad
    return {"ok": ok, "range": {"start": start, "end": end, "subjects": subjects,
                                "subjects_ok": subject_ok, "hist_leaks": hist_hits,
                                "n_hist_commits": n_hist,
                                "n_legacy_post_commits": n_post_legacy},
            "post": post_hits, "unauthorised": unauthorised, "blob_bad": blob_bad,
            "head_bad": head_bad, "touched_with_wrong_bytes": restored}


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

    # ---------------- 1-6. E22 integrity ----------------
    # Historical fact and current completeness are split on purpose: the archive commit froze
    # every artefact, and a later commit may only APPEND an authorised erratum to two named
    # documents. Nothing here is a blanket "E22 may change" waiver.
    prefix = repo_prefix(ROOT)
    # sorted, not a set: the observed dict is written to CSV, and set iteration order
    # depends on PYTHONHASHSEED, so the record would not be byte-reproducible.
    doc_paths = sorted(E22_AUTHORISED_DOCS)
    doc_set = set(doc_paths)
    archived = git("rev-parse", f"{E22_ARCHIVE_COMMIT}^{{commit}}").strip()
    arc_subject = git("log", "-1", "--format=%s", archived).strip()
    hist_paths = {ws_path(p, prefix)
                  for p in git("log", "--name-only", "--pretty=format:",
                               f"{archived}..HEAD", "--", "results/e22").split("\n") if p}
    frozen_leaks = sorted(hist_paths - doc_set)
    dirty_now = sorted(p for p in status_map(ROOT)
                       if p.startswith("results/e22/") and p not in doc_set)
    v01_ok = arc_subject == E22_ARCHIVE_SUBJECT and not frozen_leaks and not dirty_now
    add("V01", "E22 frozen artefacts unchanged since the archive commit (history AND worktree); "
               "only the two authorised documents may differ",
        {"archive": E22_ARCHIVE_COMMIT[:12], "frozen leaks": "none", "uncommitted": "none"},
        {"archive_subject_ok": arc_subject == E22_ARCHIVE_SUBJECT,
         "frozen_leaks": frozen_leaks or "none", "uncommitted": dirty_now or "none"},
        v01_ok)

    baseline = {p: git_in(ROOT, "show", f"{archived}:{prefix}{p}", binary=True) for p in doc_paths}
    current = {p: (ROOT / p).read_bytes() for p in doc_paths}
    appendix = {p: current[p][len(baseline[p]):] for p in doc_paths}
    prefix_ok = all(current[p].startswith(baseline[p]) for p in doc_paths)
    app_hash_ok = all(hashlib.sha256(appendix[p]).hexdigest()
                      == E22_AUTHORISED_DOCS[p]["appendix_sha256"] for p in doc_paths)
    app_text_ok = all(all(s in appendix[p].decode("utf-8")
                          for s in E22_AUTHORISED_DOCS[p]["phrases"]) for p in doc_paths)
    bad_doc_commits = []
    for c in [x for x in git("rev-list", f"{archived}..HEAD").split() if x]:
        changes, _merge = changed_paths(ROOT, c)
        for status, path, old in changes:
            if path.startswith("results/e22/") or (old or "").startswith("results/e22/"):
                if path not in doc_paths or status[0] != "M" or (old is not None and old != path):
                    bad_doc_commits.append(f"{c[:7]}:{status}:{path}")
    v02_ok = prefix_ok and app_hash_ok and app_text_ok and not bad_doc_commits
    add("V02", "every post-archive change under results/e22 is a byte-exact APPEND to the two "
               "authorised documents (historical body preserved; no other path, no D/R)",
        {"baseline": "archive blob", "appendix": "fixed sha256", "other paths": "none"},
        {"baseline_sha256": {p: hashlib.sha256(baseline[p]).hexdigest()[:16] for p in doc_paths},
         "appendix_sha256": {p: hashlib.sha256(appendix[p]).hexdigest() for p in doc_paths},
         "prefix_preserved": prefix_ok, "hash_ok": app_hash_ok, "phrases_ok": app_text_ok,
         "bad_commits": bad_doc_commits or "none"},
        v02_ok)
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
    std_blank = macro_std_blank(main_rows)
    # Negative self-test: the same predicate must REJECT a single filled std cell, otherwise a
    # regression that fills the row would still be reported as PASS.
    poisoned = [dict(r) for r in main_rows]
    next(r for r in poisoned if r["Category"] == "Macro Avg")["dAUROC_std"] = "0.0"
    detects_filled = not macro_std_blank(poisoned)
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
         "std_blank": std_blank, "catches_filled": detects_filled, "labels_ok": label_ok},
        marg and std_blank and detects_filled and label_ok
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

    # V30 is current state, SCOPED to the six legacy paths. Missing files are their own state;
    # status parsing is exact (no substring matching); check-ignore exit codes are distinguished
    # (0 ignored, 1 not ignored, else error); extra untracked paths are reported but are NOT
    # counted as legacy violations.
    try:
        v30 = audit_v30(ROOT, LEGACY_DISPOSITION)
        v30_err = None
    except Exception as exc:                    # noqa: BLE001 - git failure must be explicit
        v30, v30_err = {"observed": {}, "mismatch": [], "extra_untracked": [], "ok": False}, \
            f"{type(exc).__name__}: {exc}"
    n_uncommitted = sum(1 for s in v30["observed"].values() if s in ("tracked_dirty", "untracked"))
    add("V30", "six legacy paths' disposition equals the reconciled ledger (NOT a whole-repo "
               "audit); missing/mismatched state FAILs, extra untracked paths are reported",
        "ledger match", v30_err or {"n_uncommitted": n_uncommitted,
                                    "mismatch": v30["mismatch"],
                                    "extra_untracked": v30["extra_untracked"]},
        v30_err is None and v30["ok"])

    # V31 is history, checked COMMIT BY COMMIT (not a path-set diff). audit_v31 resolves full
    # hashes, asserts endpoint subjects, inspects every post-checkpoint commit (identity, path,
    # change type, blob of THAT commit), rejects D/R/merge and detects modify-then-restore.
    # Any git failure becomes a FAIL row, never "no leak".
    try:
        v31 = audit_v31(ROOT, list(LEGACY_DISPOSITION), LEGACY_MD5, LEGACY_HISTORY_RANGE,
                        LEGACY_AUTHORISED, LEGACY_HISTORY_SUBJECTS)
        v31_err = None
    except Exception as exc:                    # noqa: BLE001 - e.g. missing commit / not a repo
        v31, v31_err = None, f"{type(exc).__name__}: {exc}"
    if v31 is None:
        v31_obs, v31_ok = v31_err, False
    else:
        v31_obs = {
            "range": f"{v31['range']['start'][:12]}..{v31['range']['end'][:12]}",
            "commits_scanned": f"{v31['range']['n_hist_commits']} hist / "
                               f"{v31['range']['n_legacy_post_commits']} legacy-touching post",
            "subjects_ok": v31["range"]["subjects_ok"],
            "hist_leaks": v31["range"]["hist_leaks"] or "none",
            "post": [f"{h['commit'][:7]}:{h['path']}:{h['status']}"
                     for h in v31["post"]] or "none",
            "unauthorised": v31["unauthorised"] or "none",
            "blob_bad": v31["blob_bad"] or "none",
            "head_bad": v31["head_bad"] or "none",
            "wrong_bytes_seen": v31["touched_with_wrong_bytes"] or "none",
        }
        v31_ok = v31["ok"]
    add("V31", "legacy paths: no change in the E21->E23 range; every later change is per-commit "
               "authorised (A/M only, baseline bytes in THAT commit); D/R/merge/restore/git "
               "failure FAIL",
        {"range": "fixed full hashes", "post": "authorised only"}, v31_obs, v31_ok)

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

    # ---------------- 39-45. errata provenance, generator consistency, reverse tests --------
    # V39: every Doll summary cell must be reproducible from the THREE frozen seed cells alone
    # (ddof=1). This re-derives the table independently of the buggy code path, instead of
    # trusting the regenerated CSV.
    doll_cells = [json.loads((E22 / "cells" / f"Doll_seed{s}.json").read_text()) for s in SEEDS]
    col_of = {
        "auroc_f0": lambda c: c["f0"]["auroc"], "auroc_r1": lambda c: c["r1"]["auroc"],
        "d_auroc": lambda c: c["delta_auroc"],
        "ap_f0": lambda c: c["f0"]["ap"], "ap_r1": lambda c: c["r1"]["ap"],
        "d_ap": lambda c: c["delta_ap"],
        "dprime_f0": lambda c: c["f0"]["d_prime"], "dprime_r1": lambda c: c["r1"]["d_prime"],
        "dprime_change": lambda c: c["r1"]["d_prime"] - c["f0"]["d_prime"],
        "n_good": lambda c: c["n_ok"], "n_anomaly": lambda c: c["n_ng"],
        "n_filtered": lambda c: c["n_filtered"],
    }
    summ = {r["seed"]: r for r in csv.DictReader(open(E23 / "doll_seed_summary.csv"))}
    summ_mismatch = []
    for k, fn in col_of.items():
        v = np.array([fn(c) for c in doll_cells], float)
        for stat, val in (("std", v.std(ddof=1)), ("min", v.min()), ("max", v.max())):
            if abs(float(summ[stat][k]) - float(val)) > 1e-9:
                summ_mismatch.append(f"{stat}.{k}")
    audit_src = (ROOT / "scripts" / "e23_doll_audit.py").read_text()
    raw_only = ("rows = [dict(r) for r in raw]" in audit_src
                and "for k in A_FLOAT_COLS + A_COUNT_COLS" in audit_src
                and "np.array([float(r[k]) for r in raw], float)" in audit_src)
    add("V39", "Doll std/min/max for ALL 12 numeric columns recomputed from the 3 frozen cells "
               "(ddof=1); analysis_a reads only the raw seed rows",
        {"mismatch": "none", "raw_only": True},
        {"mismatch": summ_mismatch or "none", "raw_only": raw_only},
        not summ_mismatch and raw_only)

    # V40: the "blank is not zero" convention must be documented, not just implemented.
    preadme = (PR / "README.md").read_text()
    note_ok2 = ("never 0" in preadme
                and "not defined at this aggregation level" in preadme)
    add("V40", "pack README states that a blank Macro Avg std cell means NOT DEFINED, never 0",
        {"never 0": True, "not defined": True},
        {"never 0": "never 0" in preadme,
         "not defined": "not defined at this aggregation level" in preadme}, note_ok2)

    # V41: E18 anchor identity — seed0 F0 -> seed0 Ours, with 3-seed means named separately.
    e18 = {r["method"]: r for r in csv.DictReader(open(ROOT / "results/e18/main_table.csv"))}
    e18_f0, e18_ours = e18["F0"], e18["Ours (R1-ILA)"]
    anchor = {"Bird": (0.7472503753753754, 0.757864114114114),
              "Car": (0.7855243445692883, 0.8051123595505618),
              "Cube": (0.8540945896209056, 0.8700956937799043)}
    anchor_ok = all(close(e18_f0[f"{c}_AUROC"], a, 1e-12)
                    and close(e18_ours[f"{c}_AUROC"], b, 1e-12)
                    for c, (a, b) in anchor.items())
    seed_cols_ok = all(f"{c}_AUROC_3seed_mean" in e18_ours
                       and f"{c}_AUROC_3seed_std" in e18_ours for c in anchor)
    emap_txt = (PR / "EVIDENCE_MAP.md").read_text()
    anchor_text_ok = all(s in emap_txt for s in ["0.74725→0.75786", "0.78552→0.80511",
                                                 "0.85409→0.87010"])
    e24_pair_ok = True
    for c, (a, b) in anchor.items():
        cell = json.loads((ROOT / "results/e24/cells"
                           / f"strength_S15_{c}_s0.json").read_text())
        e24_pair_ok = (e24_pair_ok and close(cell["f0"]["auroc"], a, 1e-12)
                       and close(cell["r1"]["auroc"], b, 1e-12))
    add("V41", "E18 anchor is seed0 F0->Ours (Bird/Car/Cube), 3-seed columns separately named, "
               "and E24 S15 seed0 reproduces the same 6 values",
        {"anchor": "seed0 F0->Ours", "text": True, "e24": True},
        {"anchor_ok": anchor_ok, "seed_mean_cols": seed_cols_ok,
         "EVIDENCE_MAP_text": anchor_text_ok, "e24_pair": e24_pair_ok},
        anchor_ok and seed_cols_ok and anchor_text_ok and e24_pair_ok)

    # V42: the bootstrap quantity is a resampling proportion, not a p-value — checked in the
    # implementation AND in every paper-facing document.
    src22 = (ROOT / "scripts" / "e22_analysis.py").read_text()
    defn_ok2 = '"p_d_auroc_gt0": float((das > 0).mean())' in src22
    facing = {"claims": (PR / "paper_claims.md").read_text(),
              "limitations": (PR / "paper_limitations.md").read_text(),
              "generated map": (PR / "paper_evidence_map.md").read_text(),
              "hand map": emap_txt,
              "e23 README": e23rd}
    label_ok2 = all(("P(ΔAUROC>0)" in t) or ("P_boot(ΔAUROC>0)" in t)
                    or ("positive-difference proportion" in t) for t in facing.values())
    pvalue_free = all(not re.search(r"\bp\s*=\s*0\.001", t) for t in facing.values())
    add("V42", "p_d_auroc_gt0 is implemented as (das > 0).mean(); paper-facing text calls it "
               "P(ΔAUROC>0) / positive-difference proportion and never p = 0.001",
        {"definition": "(das > 0).mean()", "label": True, "no p-value": True},
        {"definition": defn_ok2, "label": label_ok2, "no_p_value": pvalue_free},
        defn_ok2 and label_ok2 and pvalue_free)

    # V43: E25 invariants — aggregation levels, cell accounting, absolute PCB levels, and the
    # retraining (not transfer) wording.
    e25m = json.loads((ROOT / "results/e25/multiseed_analysis.json").read_text())
    per25, ov25 = e25m["per_domain"], e25m["overall"]
    sub_macro = (per25["PCB"]["d_auroc_mean"] + per25["MetalParts"]["d_auroc_mean"]) / 2
    n25 = len(list((ROOT / "results/e25/cells").glob("*.json")))
    e25_num_ok = (close(ov25["macro_d_auroc_mean_over_splits"], -0.0010679714, 1e-9)
                  and close(sub_macro, 0.000550, 1e-6) and n25 == 39
                  and close(per25["PCB"]["auroc_f0_mean"], 0.444722, 1e-6)
                  and close(per25["PCB"]["auroc_r1_mean"], 0.463074, 1e-6))
    e25_txt = (ROOT / "results/e25" / "README.md").read_text()
    e25_doc_ok = all(s in e25_txt for s in ["0.444722", "0.463074", "180", "120", "60"])
    retrain_ok = ("不是 direct" in e25_txt and "external dataset validation" in e25_txt
                  and "external dataset validation" in emap_txt
                  and "direct cross-dataset transfer" in emap_txt)
    add("V43", "E25: 39 = 33+3+3, split-weighted macro and sub-dataset-weighted macro at their "
               "own levels, PCB absolute 0.444722->0.463074 reported, retraining (not transfer) "
               "wording intact",
        {"cells": 39, "macro split": "-0.001068", "macro subdataset": "+0.000550",
         "PCB absolute": "0.444722->0.463074"},
        {"cells": n25, "macro_split": round(ov25["macro_d_auroc_mean_over_splits"], 6),
         "macro_subdataset": round(sub_macro, 6),
         "pcb_abs": [round(per25["PCB"]["auroc_f0_mean"], 6),
                     round(per25["PCB"]["auroc_r1_mean"], 6)],
         "docs": e25_doc_ok, "retraining": retrain_ok},
        e25_num_ok and e25_doc_ok and retrain_ok)

    # V44: V30/V31 reverse tests in isolated temporary repositories. The verifier does not
    # trust "it passed before": the script must demonstrably FAIL each broken scenario.
    with tempfile.TemporaryDirectory() as td_rev:
        rev_out = Path(td_rev) / "reverse.csv"
        rev = subprocess.run([sys.executable,
                              str(ROOT / "scripts" / "verification_reverse_tests.py"),
                              "--out", str(rev_out)],
                             cwd=str(ROOT), capture_output=True, text=True)
        rev_rows = list(csv.DictReader(open(rev_out))) if rev_out.exists() else []
    rev_bad = [r["scenario"] for r in rev_rows if r["match"] != "True"]
    rev_ok = rev.returncode == 0 and bool(rev_rows) and not rev_bad
    add("V44", "V30/V31 reverse tests: every broken scenario must actually FAIL and the "
               "authorised scenario must PASS (isolated temp repos)",
        {"scenarios": "all matched", "rc": 0},
        {"scenarios": len(rev_rows), "unmatched": rev_bad or "none", "rc": rev.returncode,
         "stderr": (rev.stderr.strip()[-200:] or "none") if rev.returncode else "none"},
        rev_ok)

    # V45: the generator must reproduce every TEXT artefact byte-for-byte in an isolated OUT
    # directory. Figures/PDFs are deliberately not regenerated.
    spec = importlib.util.spec_from_file_location("prb_check",
                                                  ROOT / "scripts" / "paper_ready_build.py")
    prb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prb)
    with tempfile.TemporaryDirectory() as td_gen:
        tmp_out = Path(td_gen)
        prb.OUT = tmp_out
        prb.FIG = tmp_out / "figures"
        prb.FIG.mkdir()
        s_gen = prb.build_main_table()
        mv_gen = prb.build_multiview_table()
        prb.build_claims(s_gen, mv_gen)
        prb.build_limitations()
        prb.build_evidence_map(s_gen)
        prb.build_readme()
        gen_names = ["paper_main_table.csv", "paper_multiview_table.csv", "paper_claims.md",
                     "paper_limitations.md", "paper_evidence_map.md", "README.md"]
        gen_drift = [n for n in gen_names
                     if (tmp_out / n).read_bytes() != (PR / n).read_bytes()]
    add("V45", "paper_ready generator reproduces all six text artefacts byte-for-byte in an "
               "isolated OUT dir (no figures/PDFs regenerated)",
        "no drift", gen_drift or "no drift", not gen_drift)

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
