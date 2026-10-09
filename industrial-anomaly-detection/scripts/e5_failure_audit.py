#!/usr/bin/env python
"""E5-FAILURE-AUDIT — CPU-only diagnosis of the E5-1 `0 GO / 1 HOLD / 3 STOP` outcome.

Triggered mechanically by E5-2A protocol section 2 CASE C (0 GO): GPU method work is stopped and
the round is routed here. ZERO GPU, ZERO model fitting, ZERO new inference.

Reads only already-frozen artifacts:
  results/e4_x/per_image_original.csv          (Original reference, view 120)
  results/e5_1/raw/<C>_per_image.csv           (E5-1 candidate scores, view 120)
  results/e5_1/raw/<C>.json                    (E5-1 metrics / decisions)

Answers, with numbers rather than narrative:
  Q1 why each candidate failed
  Q2 whether failure is consistent across illuminations
  Q3 whether all three insertion points (input / feature / memory) failed
  Q4 whether that means the "module insertion" route is globally unsupported
  Q5 which family the next stage should move to

Writes results/e5_failure_audit/{diagnostics.csv,per_illumination_delta.csv,summary.json} and
figures. No protocol of any kind is modified.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import e4x_common as C  # noqa: E402

OUT = ROOT / "results" / "e5_failure_audit"
RAW = ROOT / "results" / "e5_1" / "raw"
ORIG_CSV = ROOT / "results" / "e4_x" / "per_image_original.csv"

CANDIDATES = ["C01", "C06", "C07", "C08"]
INSERTION = {"C01": "input (photometric)", "C06": "post-concat (feature)",
             "C07": "post-concat (feature)", "C08": "memory representation"}


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return [{"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                 "score": float(r["score"])} for r in csv.DictReader(f)]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (average ranks for ties)."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def structured_variance(rows: list[dict], kind: str) -> dict:
    """Decompose Good (or NG) score variance into ILLUMINATION-driven and SPECIMEN-driven parts.

    illumination component : per-illumination mean of the score, taken over all specimens of `kind`
    specimen component     : per-specimen mean of the score, taken over all 10 illuminations
    A module that has learned illumination rather than object state shows a LARGE illumination
    component relative to the specimen component.
    """
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        return {"illum_std": float("nan"), "spec_std": float("nan"), "illum_share": float("nan")}
    by_ill = defaultdict(list)
    by_spec = defaultdict(list)
    for r in sub:
        by_ill[r["illumination"]].append(r["score"])
        by_spec[r["specimen"]].append(r["score"])
    ill_means = np.array([np.mean(v) for _, v in sorted(by_ill.items())])
    spec_means = np.array([np.mean(v) for _, v in sorted(by_spec.items())])
    tot = float(np.std([r["score"] for r in sub], ddof=1))
    i_std = float(np.std(ill_means, ddof=1))
    s_std = float(np.std(spec_means, ddof=1))
    return {"illum_std": i_std, "spec_std": s_std,
            "illum_share": i_std / (i_std + s_std) if (i_std + s_std) > 0 else float("nan"),
            "overall_std": tot}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    orig_rows = read_csv(ORIG_CSV)
    orig = C.metric_block(orig_rows, "Original")
    key = lambda r: (r["specimen"], r["illumination"])
    orig_map = {key(r): r["score"] for r in orig_rows}
    o_std = float(np.std([r["score"] for r in orig_rows], ddof=1))

    diag, per_ill_rows = [], []

    # reference row: Original itself (rho vs itself = +1, sigma ratio = 1)
    sv_o_good = structured_variance(orig_rows, "Good")
    sv_o_ng = structured_variance(orig_rows, "NG")
    diag.append({
        "candidate": "Original", "insertion": "(frozen reference)", "decision": "REFERENCE",
        "auroc": orig["image_auroc"], "d_auroc": 0.0, "dprime": orig["d_prime"], "d_dprime": 0.0,
        "R_all": orig["R_all"], "R_ratio": 1.0,
        "good_mean": orig["good_mean"], "good_std": orig["good_std"],
        "ng_mean": orig["ng_mean"], "ng_std": orig["ng_std"],
        "score_std_ratio_vs_orig": 1.0, "spearman_vs_original": 1.0,
        "good_illum_share": sv_o_good["illum_share"], "good_illum_std": sv_o_good["illum_std"],
        "good_spec_std": sv_o_good["spec_std"],
        "ng_illum_share": sv_o_ng["illum_share"], "ng_illum_std": sv_o_ng["illum_std"],
        "ng_spec_std": sv_o_ng["spec_std"], "auroc_below_chance": False,
    })

    for cand in CANDIDATES:
        rows = read_csv(RAW / f"{cand}_per_image.csv")
        m = C.metric_block(rows, cand)
        j = json.loads((RAW / f"{cand}.json").read_text())
        dec = j["decision"]
        g = np.array([r["score"] for r in rows if r["kind"] == "Good"], dtype=float)
        d = np.array([r["score"] for r in rows if r["kind"] == "NG"], dtype=float)

        # signal retention vs Original (aligned by specimen x illumination)
        pairs = [(r["score"], orig_map[key(r)]) for r in rows if key(r) in orig_map]
        rho_s = spearman(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))

        sv_good = structured_variance(rows, "Good")
        sv_ng = structured_variance(rows, "NG")

        diag.append({
            "candidate": cand, "insertion": INSERTION[cand], "decision": dec["decision"],
            "auroc": m["image_auroc"], "d_auroc": dec["d_auroc"],
            "dprime": m["d_prime"], "d_dprime": dec["d_dprime"],
            "R_all": m["R_all"], "R_ratio": dec["R_ratio"],
            "good_mean": m["good_mean"], "good_std": m["good_std"],
            "ng_mean": m["ng_mean"], "ng_std": m["ng_std"],
            "score_std_ratio_vs_orig": float(np.std([r["score"] for r in rows], ddof=1) / o_std),
            "spearman_vs_original": rho_s,
            "good_illum_share": sv_good["illum_share"], "good_illum_std": sv_good["illum_std"],
            "good_spec_std": sv_good["spec_std"],
            "ng_illum_share": sv_ng["illum_share"],
            "ng_illum_std": sv_ng["illum_std"], "ng_spec_std": sv_ng["spec_std"],
            "auroc_below_chance": m["image_auroc"] < 0.5,
        })

        # per-illumination delta (Q2)
        for p_c, p_o in zip(m["per_illumination"], orig["per_illumination"]):
            assert p_c["illumination"] == p_o["illumination"]
            per_ill_rows.append({
                "candidate": cand, "illumination": p_c["illumination"],
                "auroc_cand": round(p_c["image_auroc"], 5), "auroc_orig": round(p_o["image_auroc"], 5),
                "d_auroc": round(p_c["image_auroc"] - p_o["image_auroc"], 5),
                "dprime_cand": round(p_c["d_prime"], 5), "dprime_orig": round(p_o["d_prime"], 5),
                "d_dprime": round(p_c["d_prime"] - p_o["d_prime"], 5),
            })

    # ---- writes
    with open(OUT / "diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0].keys())); w.writeheader(); w.writerows(diag)
    with open(OUT / "per_illumination_delta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ill_rows[0].keys())); w.writeheader()
        w.writerows(per_ill_rows)

    # ---- Q2: consistency of the sign of dAUROC across illuminations
    q2 = {}
    for cand in CANDIDATES:
        dd = np.array([r["d_auroc"] for r in per_ill_rows if r["candidate"] == cand])
        q2[cand] = {"n_illum": int(len(dd)), "n_positive": int((dd > 0).sum()),
                    "n_negative": int((dd < 0).sum()), "min": float(dd.min()),
                    "max": float(dd.max()), "mean": float(dd.mean()),
                    "sign_consistent_negative": bool((dd < 0).all()),
                    "sign_consistent_positive": bool((dd > 0).all())}

    summary = {
        "original": {k: orig[k] for k in ("image_auroc", "d_prime", "R_all", "good_std", "ng_std")},
        "original_overall_score_std": o_std,
        "diagnostics": diag,
        "per_illumination_consistency": q2,
        "branch": "CASE C (0 GO) -> E5-FAILURE-AUDIT",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- console report
    print("=" * 108)
    print("E5-FAILURE-AUDIT (CPU-only) — diagnosis of E5-1  0 GO / 1 HOLD / 3 STOP")
    print("=" * 108)
    print(f"Original: AUROC={orig['image_auroc']:.5f} d'={orig['d_prime']:.5f} R_all={orig['R_all']:.5f} "
          f"overall score std={o_std:.4f}")
    print()
    hdr = f"{'cand':5s} {'insertion':24s} {'dec':6s} {'AUROC':>7s} {'dAUROC':>8s} {'d(x)':>7s} " \
          f"{'R_ratio':>8s} {'std_ratio':>9s} {'rho_orig':>9s} {'illum_share_G':>13s} {'illum_share_NG':>14s}"
    print(hdr); print("-" * len(hdr))
    for r in diag:
        print(f"{r['candidate']:5s} {r['insertion']:24s} {r['decision']:6s} {r['auroc']:7.4f} "
              f"{r['d_auroc']:+8.5f} {r['dprime']:7.4f} {r['R_ratio']:8.4f} "
              f"{r['score_std_ratio_vs_orig']:9.4f} {r['spearman_vs_original']:+9.4f} "
              f"{r['good_illum_share']:13.4f} {r['ng_illum_share']:14.4f}")
    print()
    print("Q2 — sign consistency of per-illumination dAUROC (n=10 illuminations):")
    for cand, v in q2.items():
        print(f"  {cand}: mean={v['mean']:+.5f} min={v['min']:+.5f} max={v['max']:+.5f} "
              f"pos/neg={v['n_positive']}/{v['n_negative']} "
              f"all_negative={v['sign_consistent_negative']} all_positive={v['sign_consistent_positive']}")
    print()
    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ill = C.ILLUMINATIONS
    for cand in CANDIDATES:
        axes[0].plot(ill, [r["d_auroc"] for r in per_ill_rows if r["candidate"] == cand],
                     marker="o", lw=1.3, label=f"{cand} ({INSERTION[cand].split(' ')[0]})")
    axes[0].axhline(0, color="black", lw=0.9)
    axes[0].axhline(0.03, color="green", ls="--", lw=0.9, label="win band +0.03")
    axes[0].set_xlabel("illumination id")
    axes[0].set_ylabel("dAUROC vs same-illumination Original")
    axes[0].set_title("Audit Fig 1 — per-illumination dAUROC\n(all 10 illuminations negative for C06/C07/C08)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    labels = [d["candidate"] for d in diag]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, [d["spearman_vs_original"] for d in diag], width=0.4,
                label="Spearman rho vs Original ordering", edgecolor="black")
    axes[1].bar(x + 0.2, [d["good_illum_share"] for d in diag], width=0.4,
                label="illumination share of Good-score variance", edgecolor="black")
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_title("Audit Fig 2 — does the module keep the signal,\nor does it track illumination?")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "e5_failure_audit.png", dpi=150)
    plt.close(fig)

    print(f"[written] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
