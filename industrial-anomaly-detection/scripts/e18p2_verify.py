#!/usr/bin/env python
"""E18-P2 quality gate: independently recompute every table value from per-image scores."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import e18p2_main_table as M

def met(auth_or_rows, scorekey="score"):
    pass

def recompute(rows, scorekey):
    kept = [r for r in rows if r["detectable"] != False]
    y = np.array([r["image_anomaly"] for r in kept], int)
    s = np.array([r[scorekey] for r in kept], float)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), len(kept)

def main():
    auth = M.authoritative()
    checks, ok_all = [], True
    def chk(name, got, exp):
        nonlocal ok_all
        a, p, n = got if isinstance(got, tuple) else (got, None, None)
        ea, ep = exp
        good = abs(a - ea) < 1e-12 and (p is None or abs(p - ep) < 1e-12)
        ok_all &= good
        checks.append((name, a, ea, p, ep, n, good))

    # main table entries with per-image evidence
    for c in M.CATS:
        chk(f"PatchCore-emb unified / {c}", recompute(M.load_e11p2(c), "score"),
            auth["PatchCore-emb + unified scoring"][c])
    chk("PatchCore official / Bird", recompute(M.load_patchcore_official_bird(), "score"),
        (auth["_pc_official"]["official_protocol"]["auroc"],
         auth["_pc_official"]["official_protocol"]["ap"]))
    chk("AnomalyDINO / Cube", recompute(M.load_e18p1("anomalydino"), "score"), auth["AnomalyDINO"]["Cube"])
    chk("SoftPatch / Cube", recompute(M.load_e18p1("softpatch"), "score"), auth["SoftPatch"]["Cube"])
    for c in M.CATS:
        for s in M.SEEDS:
            d = M.load_e16(c, s)
            e16 = auth["_e16"][f"{c}|{s}"]
            chk(f"F0 / {c} / seed{s}", recompute(d, "score_f0"), (e16["f0"]["auroc"], e16["f0"]["ap"]))
            chk(f"Ours / {c} / seed{s}", recompute(d, "score_ours"), (e16["r1"]["auroc"], e16["r1"]["ap"]))

    # main_table.csv cross-check (must equal the authoritative JSON at full precision)
    mt = {r["method"]: r for r in csv.DictReader(open(ROOT / "results/e18/main_table.csv"))}
    for m, row in mt.items():
        for c in M.CATS:
            v = row.get(f"{c}_AUROC", "")
            if v and c in auth[m]:
                good = abs(float(v) - auth[m][c][0]) < 1e-12
                ok_all &= good
                checks.append((f"main_table.csv {m}/{c} AUROC", float(v), auth[m][c][0], None, None, None, good))

    with open(ROOT / "results/e18/verification.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "recomputed", "authoritative", "recomputed_ap", "auth_ap", "n", "PASS"])
        for name, a, ea, p, ep, n, good in checks:
            w.writerow([name, f"{a:.16f}", f"{ea:.16f}",
                        f"{p:.16f}" if p is not None else "",
                        f"{ep:.16f}" if ep is not None else "", n, good])
    npass = sum(1 for c in checks if c[-1])
    print(f"[E18-P2 VERIFY] {npass}/{len(checks)} checks PASS")
    for c in checks:
        if not c[-1]:
            print("  FAIL:", c)
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1

if __name__ == "__main__":
    raise SystemExit(main())
