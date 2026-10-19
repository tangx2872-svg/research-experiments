#!/usr/bin/env python
"""Isolated reverse tests for the V30 / V31 provenance checks.

Every scenario builds a throw-away git repository under a temporary directory and calls the
SAME functions the verifier uses (`audit_v30`, `audit_v31`). Nothing here touches the real
repository's history or files. The point is to prove that the checks can FAIL: a check that
only ever printed PASS would match none of the broken scenarios.

Scenarios
  R01 missing legacy file                -> V30 FAIL (state "missing", not "untracked")
  R02 ledger says clean, file is dirty   -> V30 FAIL
  R03 extra untracked path               -> V30 PASS, path reported separately
  R04 authorised commit (A/M, baseline)  -> V31 PASS
  R05 unauthorised later modification    -> V31 FAIL
  R06 modify then restore to baseline    -> V31 FAIL (restore is an unauthorised commit)
  R07 unauthorised deletion              -> V31 FAIL
  R08 unauthorised rename                -> V31 FAIL
  R09 git command failure (not a repo)   -> GitFailure, never "no leak"
  R10 merge commit touching legacy path  -> V31 FAIL (first-parent rule)

Usage: python scripts/verification_reverse_tests.py --out <csv>
Exit code 0 iff every scenario matched its expectation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from e23_verify import GitFailure, audit_v30, audit_v31  # noqa: E402

EXP = "monitor/exp_monitor.py"
RUNNER = "scripts/e11_p4_runner.py"
BASE = b"base\n"


def sh(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def init_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    repo.mkdir(parents=True)
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "config", "user.email", "reverse@test")
    sh(repo, "config", "user.name", "reverse test")
    return repo


def write(repo: Path, rel: str, data: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)


def commit(repo: Path, msg: str) -> str:
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", msg)
    return sh(repo, "rev-parse", "HEAD").strip()


def build_v31_base(tmp: Path):
    """A = initial commit, B = checkpoint with no legacy change, baseline bytes recorded."""
    repo = init_repo(tmp)
    write(repo, EXP, "old\n")
    a = commit(repo, "base")
    write(repo, "docs/note.txt", "n\n")
    b = commit(repo, "checkpoint")
    baseline = {EXP: hashlib.md5(BASE).hexdigest(), RUNNER: hashlib.md5(BASE).hexdigest()}
    subjects = {"start": sh(repo, "log", "-1", "--format=%s", a).strip(),
                "end": sh(repo, "log", "-1", "--format=%s", b).strip()}
    return repo, a, b, baseline, subjects


def v31(repo, a, b, baseline, subjects, authorised):
    return audit_v31(repo, [EXP, RUNNER], baseline, (a, b), authorised, subjects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reverse_tests.csv")
    args = ap.parse_args()
    rows = []

    def record(name, expect_fail, audit_failed, mechanism, detail=""):
        """expect_fail: the audit SHOULD report FAIL; mechanism: the specific defect was seen."""
        matched = (bool(audit_failed) == bool(expect_fail)) and bool(mechanism)
        rows.append({"scenario": name,
                     "expected": "FAIL" if expect_fail else "PASS",
                     "observed": "FAIL" if audit_failed else "PASS",
                     "match": str(matched), "detail": detail})

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # R01 missing file: V30 must report `missing`, never `untracked`.
        repo = init_repo(root / "r01")
        write(repo, EXP, "x\n")
        commit(repo, "tracked")
        (repo / EXP).unlink()
        res = audit_v30(repo, {EXP: "tracked_clean"})
        record("R01 missing legacy file", True, not res["ok"],
               res["observed"][EXP] == "missing",
               f"ok={res['ok']} state={res['observed'][EXP]}")

        # R02 ledger/state mismatch.
        repo = init_repo(root / "r02")
        write(repo, EXP, "x\n")
        commit(repo, "tracked")
        write(repo, EXP, "dirty\n")
        res = audit_v30(repo, {EXP: "tracked_clean"})
        record("R02 ledger clean but file dirty", True, not res["ok"],
               res["observed"][EXP] == "tracked_dirty",
               f"ok={res['ok']} state={res['observed'][EXP]}")

        # R03 extra untracked path is reported but is not a legacy violation.
        repo = init_repo(root / "r03")
        write(repo, EXP, "x\n")
        commit(repo, "tracked")
        write(repo, "draft/paper.md", "text\n")
        res = audit_v30(repo, {EXP: "tracked_clean"})
        record("R03 extra untracked path", False, not res["ok"],
               "draft/paper.md" in res["extra_untracked"],
               f"ok={res['ok']} extras={res['extra_untracked']}")

        # R04 authorised commit (M with baseline bytes + A with baseline bytes) passes.
        repo, a, b, baseline, subjects = build_v31_base(root / "r04")
        write(repo, EXP, BASE.decode())
        write(repo, RUNNER, BASE.decode())
        c = commit(repo, "authorised")
        res = v31(repo, a, b, baseline, subjects, {c: {EXP, RUNNER}})
        record("R04 authorised history", False, not res["ok"],
               [h["status"] for h in res["post"]] == ["M", "A"],
               f"post={[h['status'] for h in res['post']]}")

        # R05 unauthorised later modification.
        repo, a, b, baseline, subjects = build_v31_base(root / "r05")
        write(repo, EXP, BASE.decode())
        write(repo, RUNNER, BASE.decode())
        c = commit(repo, "authorised")
        write(repo, EXP, "evil\n")
        d = commit(repo, "unauthorised")
        res = v31(repo, a, b, baseline, subjects, {c: {EXP, RUNNER}})
        record("R05 unauthorised later modification", True, not res["ok"],
               any(h["commit"] == d for h in res["unauthorised"]),
               # Deterministic detail: a temp-repo SHA changes on every run, so it
               # must not be rendered here (the identity check lives in `mechanism`).
               f"unauthorised={len(res['unauthorised'])} "
               f"flagged_injected_commit="
               f"{any(h['commit'] == d for h in res['unauthorised'])}")

        # R06 modify then restore: HEAD bytes match baseline, history must still FAIL.
        repo, a, b, baseline, subjects = build_v31_base(root / "r06")
        write(repo, EXP, BASE.decode())
        write(repo, RUNNER, BASE.decode())
        c = commit(repo, "authorised")
        write(repo, EXP, "evil\n")
        d = commit(repo, "break")
        write(repo, EXP, BASE.decode())
        e = commit(repo, "restore")
        res = v31(repo, a, b, baseline, subjects, {c: {EXP, RUNNER}})
        record("R06 modify then restore", True, not res["ok"],
               {d, e} <= {h["commit"] for h in res["unauthorised"]}
               and EXP in res["touched_with_wrong_bytes"],
               f"unauthorised={len(res['unauthorised'])} "
               f"wrong_bytes={res['touched_with_wrong_bytes']}")

        # R07 unauthorised deletion.
        repo, a, b, baseline, subjects = build_v31_base(root / "r07")
        write(repo, EXP, BASE.decode())
        write(repo, RUNNER, BASE.decode())
        c = commit(repo, "authorised")
        sh(repo, "rm", "-q", EXP)
        commit(repo, "delete legacy")
        res = v31(repo, a, b, baseline, subjects, {c: {EXP, RUNNER}})
        record("R07 unauthorised deletion", True, not res["ok"],
               any(h["status"].startswith("D") for h in res["unauthorised"]),
               f"types={[h['status'] for h in res['post']]}")

        # R08 unauthorised rename (R detection, old path matched).
        repo, a, b, baseline, subjects = build_v31_base(root / "r08")
        write(repo, EXP, BASE.decode())
        write(repo, RUNNER, BASE.decode())
        c = commit(repo, "authorised")
        sh(repo, "mv", EXP, "monitor/renamed.py")
        commit(repo, "rename legacy")
        res = v31(repo, a, b, baseline, subjects, {c: {EXP, RUNNER}})
        record("R08 unauthorised rename", True, not res["ok"],
               any(h["status"].startswith("R") for h in res["unauthorised"]),
               f"types={[h['status'] for h in res['post']]}")

        # R09 git failure must raise, not be read as an empty history.
        not_repo = root / "not_a_repo"
        not_repo.mkdir()
        raised = False
        try:
            audit_v30(not_repo, {EXP: "tracked_clean"})
        except GitFailure:
            raised = True
        record("R09 git command failure", True, raised, raised,
               "GitFailure raised" if raised else "no exception")

        # R11 many status entries: the dirty legacy path must still be parsed exactly.
        # (An off-by-one in NUL-delimited porcelain parsing would skip it and report clean.)
        repo = init_repo(root / "r11")
        for name in ("aaa.txt", "zzz.txt", EXP):
            write(repo, name, "x\n")
        commit(repo, "tracked")
        write(repo, "aaa.txt", "dirty\n")
        write(repo, EXP, "dirty\n")
        write(repo, "zzz.txt", "dirty\n")
        res = audit_v30(repo, {EXP: "tracked_dirty"})
        record("R11 dirty legacy among many status entries", False, not res["ok"],
               res["observed"][EXP] == "tracked_dirty",
               f"state={res['observed'][EXP]}")

        # R10 merge commit: first-parent diff must surface the legacy change.
        repo = init_repo(root / "r10")
        write(repo, EXP, "old\n")
        a = commit(repo, "base")
        write(repo, "docs/note.txt", "n\n")
        b = commit(repo, "checkpoint")
        sh(repo, "checkout", "-q", "-b", "side")
        write(repo, EXP, "side\n")
        commit(repo, "side changes legacy")
        sh(repo, "checkout", "-q", "main")
        write(repo, "docs/other.txt", "o\n")
        commit(repo, "main moves on")
        sh(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
        baseline = {EXP: hashlib.md5(BASE).hexdigest(), RUNNER: hashlib.md5(BASE).hexdigest()}
        subjects = {"start": sh(repo, "log", "-1", "--format=%s", a).strip(),
                    "end": sh(repo, "log", "-1", "--format=%s", b).strip()}
        res = v31(repo, a, b, baseline, subjects, {})
        record("R10 merge commit touching legacy", True, not res["ok"],
               any(h["merge"] for h in res["unauthorised"]),
               f"post={[(h['status'], h['merge']) for h in res['post']]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "expected", "observed", "match", "detail"])
        w.writeheader()
        w.writerows(rows)
    bad = [r["scenario"] for r in rows if r["match"] != "True"]
    print(f"[reverse tests] {len(rows)} scenarios, {len(bad)} unmatched -> {out}")
    for r in rows:
        print(f"  [{'OK' if r['match'] == 'True' else 'MISMATCH'}] {r['scenario']}: "
              f"expected {r['expected']}, observed {r['observed']} ({r['detail']})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
