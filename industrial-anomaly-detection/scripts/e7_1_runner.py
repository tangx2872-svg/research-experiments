#!/usr/bin/env python
"""E7-1 — broad screen of the frozen 14-pipeline registry.

Protocol: `docs/E7_1_FROZEN_PROTOCOL.md` (frozen at af56dd3 BEFORE any candidate metric).

Usage:
  python scripts/e7_1_runner.py --smoke            # Stage 0: 10 normal + 20 test, all 14 pipelines
  python scripts/e7_1_runner.py                    # full E7-1 broad screen
  python scripts/e7_1_runner.py --only P00,P01     # subset (resume-friendly)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402
import e7_pipelines as PL  # noqa: E402

OUT = ROOT / "results" / "e7_1"
RAW = OUT / "raw"
PROGRESS = OUT / "progress.json"
PROTOCOL = ROOT / "docs" / "E7_1_FROZEN_PROTOCOL.md"

SPEC = {
    "P00": {"family": "patchcore", "image": None},
    "P01": {"family": "patchcore", "image": "retinex"},
    "P02": {"family": "patchcore", "image": "clahe_mild"},
    "P03": {"family": "patchcore", "image": "clahe_medium"},
    "P04": {"family": "patchcore", "image": "retinex_clahe_mild"},
    "P05": {"family": "dinov2", "image": None, "model": "dinov2_vits14"},
    "P06": {"family": "dinov2", "image": None, "model": "dinov2_vitb14"},
    "P07": {"family": "dinov2", "image": "retinex", "model": "dinov2_vits14"},
    "P08": {"family": "dinov2", "image": "retinex", "model": "dinov2_vitb14"},
    "P09": {"family": "dinov2", "image": "clahe_mild", "model": "dinov2_vits14"},
    "P10": {"family": "softpatch", "image": None, "weight": "nearest"},
    "P11": {"family": "softpatch", "image": None, "weight": "lof"},
    "P12": {"family": "softpatch", "image": "retinex", "weight": "nearest"},
    "P13": {"family": "softpatch", "image": "retinex", "weight": "lof"},
}
ORDER = list(SPEC.keys())
SMOKE_ONLY = None


def reroute(records: list, kind) -> list:
    if kind is None:
        return [dict(r) for r in records]
    return [dict(r, image_path=E7.cached_image_path(kind, r)) for r in records]


def run_one(pid: str, bank: list, test: list, dev) -> dict:
    s = SPEC[pid]
    bank_r, test_r = reroute(bank, s["image"]), reroute(test, s["image"])
    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        return {"status": "FAILED", "reason": f"{len(missing)} input images missing (cache not built): "
                                              f"{missing[:2]}"}
    if s["family"] == "patchcore":
        res = PL.run_patchcore(bank_r, test_r, dev, pid)
    elif s["family"] == "dinov2":
        res = PL.run_dinov2(s["model"], bank_r, test_r, dev, pid)
    elif s["family"] == "softpatch":
        res = PL.run_softpatch(s["weight"], bank_r, test_r, dev, pid)
    else:
        raise ValueError(pid)
    res["status"] = "OK"
    res["image_kind"] = s["image"]
    res["family"] = s["family"]
    res["spec"] = {k: v for k, v in s.items()}
    return res


# ------------------------------------------------------------------ Stage 0 smoke
def smoke(dev) -> int:
    scope = E7.load_scope()

    def spread(recs, n):
        """One image per DISTINCT specimen (so the score-direction check spans n specimens)."""
        seen, out = set(), []
        for r in recs:
            if r["specimen"] in seen:
                continue
            seen.add(r["specimen"]); out.append(r)
            if len(out) == n:
                break
        return out

    bank = spread(scope["bank_subset"], 10)
    good = spread([r for r in scope["test120"] if r["kind"] == "Good"], 10)
    ng = spread([r for r in scope["test120"] if r["kind"] == "NG"], 10)
    test = good + ng
    print(f"\n[Stage 0 smoke] bank={len(bank)} test={len(test)} ({len(good)} Good + {len(ng)} NG)", flush=True)
    kinds_needed = sorted({SPEC[p]["image"] for p in ORDER} - {None})
    if kinds_needed:
        bar = E7.Bar("cache", "build", len(bank) + len(test), unit="img")
        for kind in kinds_needed:
            E7.ensure_cache(kind, bank + test, bar)
    rows = []
    only = [p.strip() for p in (SMOKE_ONLY or ",".join(ORDER)).split(",")]
    for pid in only:
        t0 = time.time()
        print(f">>> starting {pid} {E7.PIPELINE_DESC[pid]}", flush=True)
        try:
            r = run_one(pid, bank, test, dev)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            rows.append({"pipeline": pid, "PASS": False, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if r.get("status") != "OK":
            rows.append({"pipeline": pid, "PASS": False, "reason": r.get("reason", "")})
            continue
        s = r["rows"]
        g = np.array([x["score"] for x in s if x["kind"] == "Good"], float)
        d = np.array([x["score"] for x in s if x["kind"] == "NG"], float)
        checks = {
            "shape": len(s) == len(test),
            "finite": bool(np.isfinite(np.concatenate([g, d])).all()),
            "dependency": True,
            "score_direction": bool(d.mean() > g.mean()),
            "normal_only": all(x["kind"] == "Good" for x in bank),
            "no_leakage": not ({x["specimen"] for x in bank} & {x["specimen"] for x in test}),
        }
        ok = all(checks.values())
        rows.append({"pipeline": pid, "PASS": ok, "seconds": round(time.time() - t0, 1),
                     "good_mean": round(float(g.mean()), 6), "ng_mean": round(float(d.mean()), 6),
                     "family": r["family"], **checks,
                     "reason": "" if ok else "failed: " + ",".join(k for k, v in checks.items() if not v)})
        print(f"  SMOKE {pid:4s} {'PASS' if ok else 'FAIL'}  {time.time()-t0:6.1f}s  "
              f"good={g.mean():.5g} ng={d.mean():.5g}  "
              f"{'' if ok else [k for k,v in checks.items() if not v]}", flush=True)
    E7.write_csv(OUT / "smoke.csv", rows)
    npass = sum(1 for r in rows if r["PASS"])
    print(f"\n[Stage 0] {npass}/{len(ORDER)} PASS -> {OUT/'smoke.csv'}", flush=True)
    return 0 if npass == len(ORDER) else 3


# ------------------------------------------------------------------ full broad screen
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of pipeline ids")
    ap.add_argument("--smoke-only", default=None, help="subset for the Stage-0 smoke")
    ap.add_argument("--max-minutes", type=float, default=90.0, help="E7-1 hard budget (protocol s14)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    print("=" * 100)
    print("E7-1 - Lightweight Composition Broad Screening   protocol commit af56dd3")
    print("=" * 100, flush=True)
    if args.smoke:
        global SMOKE_ONLY
        SMOKE_ONLY = args.smoke_only
        return smoke(dev)

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    wanted = [p.strip() for p in args.only.split(",")] if args.only else list(ORDER)
    scope = E7.load_scope()
    san = E7.sanity_scope(scope)
    for k, v in san.items():
        print(f"  sanity {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}", flush=True)
    if not all(v["pass"] for v in san.values()):
        print("[ABORT] scope sanity FAIL")
        return 3
    print(f"  bank subset specimens = {len(scope['bank_specimens'])}  "
          f"views={scope['bank_views']}  illum={scope['bank_illuminations']}", flush=True)

    kinds = sorted({SPEC[p]["image"] for p in wanted} - {None})
    t0 = time.time()
    if kinds:
        print(f"\n[stage] photometric caches {kinds}", flush=True)
        bar = E7.Bar("cache", "build", len(scope["bank_subset"]) + len(scope["test120"]), unit="img")
        for kind in kinds:
            E7.ensure_cache(kind, scope["bank_subset"] + scope["test120"], bar)
        print(f"  caches ready in {time.time()-t0:.1f}s", flush=True)

    st = E7.progress_json(PROGRESS)
    st.update({"protocol_commit": "af56dd3", "protocol_sha256": E7.sha256(PROTOCOL),
               "seed": E7.SEED, "frozen_view": E7.FROZEN_VIEW,
               "BANK_SUBSET_SEED": E7.BANK_SUBSET_SEED, "bank_n": len(scope["bank_subset"]),
               "test_n": len(scope["test120"]), "sanity": {k: v["pass"] for k, v in san.items()}})
    E7.save_progress(PROGRESS, st)

    T0 = time.time()
    done_times = []
    for i, pid in enumerate(wanted, 1):
        prev = st["pipelines"].get(pid, {})
        if prev.get("status") == "DONE" and not args.force:
            print(f"E7-1 [{i}/{len(wanted)}] {pid}: already DONE - skipped", flush=True)
            continue
        elapsed_min = (time.time() - T0) / 60.0
        if elapsed_min > args.max_minutes and pid != "P00":
            st["pipelines"][pid] = {"status": "SLOW_CANDIDATE",
                                    "reason": f"budget {args.max_minutes} min reached before start"}
            E7.save_progress(PROGRESS, st)
            print(f"E7-1 [{i}/{len(wanted)}] {pid}: SLOW_CANDIDATE (budget reached) - not started", flush=True)
            continue
        print(f"\nE7-1 [{i-1}/{len(wanted)} | {int(100*(i-1)/len(wanted))}%]  "
              f"Pipeline: {pid} {E7.PIPELINE_DESC[pid]}  Stage: starting  "
              f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time()-T0))}", flush=True)
        st["pipelines"][pid] = {"status": "RUNNING", "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "desc": E7.PIPELINE_DESC[pid], "spec": SPEC[pid],
                                "git_head": st["protocol_commit"], "seed": E7.SEED}
        E7.save_progress(PROGRESS, st)
        t_c = time.time()
        try:
            r = run_one(pid, scope["bank_subset"], scope["test120"], dev)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            st["pipelines"][pid] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}",
                                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S")}
            E7.save_progress(PROGRESS, st)
            continue
        if r.get("status") != "OK":
            st["pipelines"][pid] = {"status": "FAILED", "reason": r.get("reason", ""),
                                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S")}
            E7.save_progress(PROGRESS, st)
            continue
        dt = time.time() - t_c
        done_times.append(dt)
        rows = r["rows"]
        # pixel metrics only on NG images that actually have a mask (E4-D1 known issue)
        masks = r.get("masks")
        if masks is not None:
            idx = [n for n, x in enumerate(rows)
                   if x["kind"] == "Good" or (x["kind"] == "NG" and x.get("mask_path", "")
                                              and Path(x["mask_path"]).exists())]
        else:
            idx = []
        m = E7.evaluate(rows, r.get("maps"), None, want_pixel=False)
        pix = r.get("pixel") or {}
        m["pixel_auroc"] = pix.get("pixel_auroc", float("nan"))
        m["aupro"] = pix.get("aupro", float("nan"))
        m["n_masked_ng"] = pix.get("n_masked_ng", 0)
        E7.write_csv(RAW / f"{pid}_per_image.csv",
                     [{k: v for k, v in x.items() if k in
                       ("specimen", "kind", "illumination", "score")} for x in rows])
        payload = {"pipeline": pid, "desc": E7.PIPELINE_DESC[pid], "spec": SPEC[pid],
                   "metrics": m, "stages": r["stages"], "runtime_s": round(dt, 1),
                   "git_head": st["protocol_commit"], "protocol_sha256": st["protocol_sha256"],
                   "seed": E7.SEED, "frozen_view": E7.FROZEN_VIEW,
                   "bank_n": len(scope["bank_subset"]), "test_n": len(scope["test120"]),
                   "has_maps": r.get("maps") is not None}
        (RAW / f"{pid}.json").write_text(json.dumps(payload, indent=2, default=float))
        st["pipelines"][pid] = {"status": "DONE", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "runtime_s": round(dt, 1), "desc": E7.PIPELINE_DESC[pid],
                                "spec": SPEC[pid], "metrics": m,
                                "artifact": f"results/e7_1/raw/{pid}.json",
                                "git_head": st["protocol_commit"], "seed": E7.SEED}
        E7.save_progress(PROGRESS, st)
        flag = "  SLOW_CANDIDATE(>15min)" if dt > 900 else ""
        print(f"  E7-1 [{i}/{len(wanted)} | {int(100*i/len(wanted))}%] {pid} DONE  "
              f"AUROC={m['image_auroc']:.5f} AUPR={m['image_aupr']:.5f} d'={m['d_prime']:.4f}  "
              f"{dt:.0f}s{flag}", flush=True)
    analyze(st)
    print(f"\nE7-1 total = {time.strftime('%H:%M:%S', time.gmtime(time.time()-T0))}", flush=True)
    return 0


# ------------------------------------------------------------------ analysis
def analyze(st: dict) -> None:
    got = {}
    for pid in ORDER:
        p = RAW / f"{pid}.json"
        if p.exists():
            got[pid] = json.loads(p.read_text())
    if "P00" not in got:
        print("[analyze] P00 baseline missing - cannot compute deltas")
        return
    base = got["P00"]["metrics"]
    metrics, rows = {}, []
    for pid in ORDER:
        if pid not in got:
            pr = st["pipelines"].get(pid, {})
            rows.append({"pipeline": pid, "desc": E7.PIPELINE_DESC[pid], "status": pr.get("status", "PENDING"),
                         "image_auroc": float("nan"), "d_auroc": float("nan"), "image_aupr": float("nan"),
                         "d_prime": float("nan"), "runtime_s": pr.get("runtime_s", float("nan")),
                         "decision": "NOT_RUN", "tier": "-"})
            continue
        m = got[pid]["metrics"]
        if pid == "P00":
            d = {"tier": "REF", "decision": "REFERENCE"}
            metrics[pid] = {"image_auroc": m["image_auroc"], "decision": "REFERENCE", "d_auroc": 0.0}
        else:
            d = E7.tier(m["image_auroc"] - base["image_auroc"],
                        m["image_aupr"] - base["image_aupr"], m["d_prime"] - base["d_prime"])
            metrics[pid] = {"image_auroc": m["image_auroc"], "decision": d["decision"],
                            "d_auroc": m["image_auroc"] - base["image_auroc"]}
        rows.append({"pipeline": pid, "desc": E7.PIPELINE_DESC[pid], "status": "DONE",
                     "image_auroc": m["image_auroc"], "d_auroc": m["image_auroc"] - base["image_auroc"],
                     "image_aupr": m["image_aupr"],
                     "d_aupr": m["image_aupr"] - base["image_aupr"],
                     "d_prime": m["d_prime"], "d_dprime": m["d_prime"] - base["d_prime"],
                     "pixel_auroc": m.get("pixel_auroc", float("nan")),
                     "aupro": m.get("aupro", float("nan")),
                     "runtime_s": got[pid]["runtime_s"], "tier": d["tier"], "decision": d["decision"]})
    E7.write_csv(OUT / "summary.csv", rows)

    syn = []
    for comp in E7.SYNERGY_PARENT:
        if comp in got and E7.SYNERGY_PARENT[comp][0] in got and E7.SYNERGY_PARENT[comp][1] in got:
            s = E7.synergy(metrics, comp)
            if comp in metrics:
                s["decision_AB"] = metrics[comp]["decision"]
            syn.append(s)
    E7.write_csv(OUT / "synergy.csv", syn)

    gate = E7.gate_e7_2(metrics)
    (OUT / "gate_e7_2.json").write_text(json.dumps(gate, indent=2, default=float))
    st["analysis"] = {"baseline": {k: base[k] for k in ("image_auroc", "image_aupr", "d_prime")},
                      "tiers": {r["pipeline"]: r["decision"] for r in rows},
                      "synergy": syn, "gate": gate}
    E7.save_progress(PROGRESS, st)

    print("\n" + "=" * 118)
    print(f"{'ID':5s} {'pipeline':30s} {'AUROC':>8s} {'dAUROC':>9s} {'AUPR':>8s} {'dAUPR':>9s} "
          f"{'dprime':>8s} {'runtime':>8s}  DECISION")
    print("-" * 118)
    for r in sorted(rows, key=lambda x: (-(x["image_auroc"] if np.isfinite(x["image_auroc"]) else -1))):
        if not np.isfinite(r["image_auroc"]):
            print(f"{r['pipeline']:5s} {r['desc']:30s} {'--':>8s} {'--':>9s} {'--':>8s} {'--':>9s} "
                  f"{'--':>8s} {'--':>8s}  {r['decision']} ({r['status']})")
            continue
        print(f"{r['pipeline']:5s} {r['desc']:30s} {r['image_auroc']:8.5f} {r['d_auroc']:+9.5f} "
              f"{r['image_aupr']:8.5f} {r['d_aupr']:+9.5f} {r['d_prime']:8.4f} "
              f"{r['runtime_s']:7.0f}s  {r['decision']}")
    print("\n[composition synergy]  A | B | A+B | synergy_gain | positive")
    for s in syn:
        print(f"  {s['A']} | {s['B']} | {s['A_plus_B']} | {s['synergy_gain']:+.5f} | "
              f"{'POSITIVE_COMPOSITION' if s['positive_composition'] else 'no'}  "
              f"(A {s['d_auroc_A']:+.5f}, B {s['d_auroc_B']:+.5f}, AB {s['d_auroc_AB']:+.5f})")
    print(f"\n[E7-2 gate] ADVANCE={gate['advance']} KEEP={gate['keep']} BACKUP={gate['backup']}")
    print(f"            STOP={gate['stop']}")
    print(f"            SELECTED={gate['selected']}  proceed={gate['proceed_to_e7_2']}")


if __name__ == "__main__":
    raise SystemExit(main())
