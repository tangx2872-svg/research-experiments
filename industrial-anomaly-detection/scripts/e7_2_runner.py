#!/usr/bin/env python
"""E7-2 — cross-view medium validation of the E7-1 survivors.

Protocol: docs/E7_1_FROZEN_PROTOCOL.md section 15 (view draw frozen with VIEW_SEED=20261009).

ENGINEERING OPTIMISATION (no method change): the PatchCore-family model is fitted on the normal bank
only and the bank is view-independent, so each pipeline is fitted ONCE and then scored on every view.
Fitting per view would produce byte-identical models.
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
import e7_1_runner as R1  # noqa: E402

OUT = ROOT / "results" / "e7_2"
RAW = OUT / "raw"
PROGRESS = OUT / "progress.json"
ALL_VIEWS = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]


def frozen_views() -> list:
    """Primary view 120 + 3 mechanically drawn views (VIEW_SEED), drawn BEFORE any result is read."""
    remaining = sorted(v for v in ALL_VIEWS if v != E7.FROZEN_VIEW)
    extra = sorted(random.Random(E7.VIEW_SEED).sample(remaining, 3))
    return [E7.FROZEN_VIEW] + extra


# ---------------------------------------------------------------- fit once / score many
def fit_patchcore(bank: list, dev, tag: str):
    from experiment5a_model import FAlphaDualLayerPatchcore
    from anomalib.engine import Engine
    import experiment1b_defect_sensitivity as e1b
    t0 = time.time()
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank], batch_size=16,
                                   num_workers=0, seed=E7.SEED, name=f"e72_{tag}")
    model = FAlphaDualLayerPatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                     coreset_sampling_ratio=0.1, num_neighbors=9,
                                     alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(OUT / "fit" / tag))
    engine.fit(model=model, datamodule=dm)
    tm = model.model; tm.eval(); e1b._move_model_to_device(tm, dev)
    print(f"  E7-2 [{tag}] fit {time.time()-t0:.0f}s coreset={int(tm.memory_bank.shape[0])}", flush=True)
    return tm, {"fit_s": round(time.time() - t0, 1), "coreset_size": int(tm.memory_bank.shape[0])}


def score_patchcore(tm, test: list, dev, tag: str) -> list:
    import experiment1b_defect_sensitivity as e1b
    bar = E7.Bar(tag, "scoring", len(test))
    rows = []
    for i, r in enumerate(test, 1):
        s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        if not np.isfinite(s):
            raise FloatingPointError(f"{tag}: non-finite score")
        rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                     "score": float(s)})
        if i % 200 == 0 or i == len(test):
            bar(i)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None, help="default: E7-1 gate selection")
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    views = frozen_views()
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(f"E7-2 - cross-view medium validation   views={views} (VIEW_SEED={E7.VIEW_SEED})")
    print("=" * 100, flush=True)

    # candidate set comes mechanically from the E7-1 gate
    gate = json.loads((ROOT / "results" / "e7_1" / "gate_e7_2.json").read_text())
    cands = [c.strip() for c in args.candidates.split(",")] if args.candidates else gate["selected"]
    print(f"  E7-1 gate selected: {gate['selected']}  -> E7-2 candidates: {cands}", flush=True)
    if not cands:
        print("  [STOP] E7-1 gate returned 0 candidates - E7-2 must not start.")
        return 0
    if len(cands) > 4:
        print("  [STOP] gate returned >4 candidates - protocol section 11 forbids this.")
        return 3

    scope = E7.load_scope()
    bank = scope["bank_source"]                      # FULL normal bank (1200), protocol section 15
    print(f"  full bank = {len(bank)} images", flush=True)
    test_by_view = {v: [r for r in scope["test_all"] if r["view"] == v] for v in views}
    for v in views:
        assert len(test_by_view[v]) == 700, (v, len(test_by_view[v]))

    kinds = sorted({R1.SPEC[p]["image"] for p in (["P00"] + cands)} - {None})
    if kinds:
        todo = bank + [r for v in views for r in test_by_view[v]]
        print(f"\n[stage] caches {kinds} for {len(todo)} source images", flush=True)
        bar = E7.Bar("cache", "build", len(todo), unit="img")
        t0 = time.time()
        for k in kinds:
            E7.ensure_cache(k, todo, bar)
        print(f"  caches ready in {time.time()-t0:.1f}s", flush=True)

    st = E7.progress_json(PROGRESS)
    st.update({"views": views, "VIEW_SEED": E7.VIEW_SEED, "candidates": cands,
               "bank_n": len(bank), "protocol_commit": "af56dd3",
               "protocol_sha256": E7.sha256(ROOT / "docs" / "E7_1_FROZEN_PROTOCOL.md")})
    E7.save_progress(PROGRESS, st)

    T0 = time.time()
    for pid in ["P00"] + cands:
        need = []
        for v in views:
            p = RAW / f"{pid}_view{v}.json"
            if not (p.exists() and not args.force):
                need.append(v)
        if not need:
            print(f"E7-2 {pid}: all views already DONE - skipped", flush=True)
            continue
        if (time.time() - T0) / 60.0 > args.max_minutes:
            st["pipelines"][f"{pid}"] = {"status": "SLOW_CANDIDATE",
                                         "reason": f"E7-2 budget {args.max_minutes} min reached"}
            E7.save_progress(PROGRESS, st)
            print(f"E7-2 {pid}: SLOW_CANDIDATE (budget) - not started", flush=True)
            continue
        kind = R1.SPEC[pid]["image"]
        bank_r = R1.reroute(bank, kind)
        t0 = time.time()
        try:
            tm, fitinfo = fit_patchcore(bank_r, dev, f"{pid}")
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            st["pipelines"][pid] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            E7.save_progress(PROGRESS, st)
            continue
        for v in need:
            test_r = R1.reroute(test_by_view[v], kind)
            try:
                rows = score_patchcore(tm, test_r, dev, f"{pid}/v{v}")
            except Exception as exc:  # noqa: BLE001
                import traceback; traceback.print_exc()
                st["pipelines"][f"{pid}@{v}"] = {"status": "FAILED", "reason": str(exc)[:200]}
                E7.save_progress(PROGRESS, st)
                continue
            m = E7.evaluate(rows)
            E7.write_csv(RAW / f"{pid}_view{v}_per_image.csv", rows)
            (RAW / f"{pid}_view{v}.json").write_text(json.dumps(
                {"pipeline": pid, "view": v, "metrics": m, "fit": fitinfo,
                 "runtime_s": round(time.time() - t0, 1), "bank_n": len(bank), "test_n": len(rows),
                 "seed": E7.SEED, "git_head": "af56dd3"}, indent=2, default=float))
            st["pipelines"][f"{pid}@{v}"] = {"status": "DONE", "metrics": m, "view": v,
                                             "artifact": f"results/e7_2/raw/{pid}_view{v}.json"}
            E7.save_progress(PROGRESS, st)
            print(f"  E7-2 [{pid}@{v}] AUROC={m['image_auroc']:.5f} AUPR={m['image_aupr']:.5f} "
                  f"d'={m['d_prime']:.4f}", flush=True)
        del tm
        torch.cuda.empty_cache()
    analyze(views, cands, st)
    print(f"\nE7-2 total = {time.strftime('%H:%M:%S', time.gmtime(time.time()-T0))}", flush=True)
    return 0


# ---------------------------------------------------------------- analysis (protocol section 15)
def analyze(views: list, cands: list, st: dict) -> None:
    got = {}
    for pid in ["P00"] + cands:
        for v in views:
            p = RAW / f"{pid}_view{v}.json"
            if p.exists():
                got[(pid, v)] = json.loads(p.read_text())
    rows = []
    print("\n" + "=" * 104)
    print(f"{'candidate':10s} " + " ".join(f"{'v'+v:>9s}" for v in views) +
          f" {'meanAUROC':>10s} {'meanDAUROC':>11s} {'medDAUROC':>10s} {'pos/4':>6s} "
          f"{'worstAUROC':>11s} {'worstDAU':>9s}  VERDICT")
    print("-" * 104)
    for pid in cands:
        per, d_auc = [], []
        for v in views:
            c = got.get((pid, v)); b = got.get(("P00", v))
            if c and b:
                per.append(c["metrics"]["image_auroc"])
                d_auc.append(c["metrics"]["image_auroc"] - b["metrics"]["image_auroc"])
            else:
                per.append(float("nan")); d_auc.append(float("nan"))
        dd = np.array(d_auc, dtype=float)
        pos = int(np.nansum(dd > 0))
        mean_d = float(np.nanmean(dd)) if np.isfinite(dd).any() else float("nan")
        rec = {"pipeline": pid, "view_auroc": per, "view_d_auroc": d_auc,
               "mean_auroc": float(np.nanmean(per)), "median_auroc": float(np.nanmedian(per)),
               "mean_d_auroc": mean_d, "median_d_auroc": float(np.nanmedian(dd)),
               "positive_views": pos, "n_views": len(views),
               "worst_view": views[int(np.nanargmin(per))], "worst_auroc": float(np.nanmin(per)),
               "worst_d_auroc": float(np.nanmin(dd))}
        # finalist rule (protocol section 15)
        syn_pos = True
        sp = E7.SYNERGY_PARENT.get(pid)
        if sp:      # A+B composition: require a positive E7-1 synergy gain
            try:
                s1 = {r["pipeline"]: r for r in
                      __import__("csv").DictReader(open(ROOT / "results" / "e7_1" / "summary.csv"))}
                syn_pos = float(s1[pid]["d_auroc"]) > max(float(s1[sp[0]]["d_auroc"]),
                                                          float(s1[sp[1]]["d_auroc"]))
            except Exception:  # noqa: BLE001
                syn_pos = False
        if mean_d >= 0.015 and pos >= 3:
            v = "FINALIST (Path A)"
        elif mean_d >= 0.010 and pos >= 3 and syn_pos:
            v = "FINALIST (Path B)"
        elif mean_d > 0:
            v = "HOLD (unstable / below finalist bar)"
        else:
            v = "REJECTED"
        rec["verdict"] = v
        rec["synergy_positive"] = bool(syn_pos)
        rows.append(rec)
        print(f"{pid:10s} " + " ".join(f"{x:9.5f}" for x in per) +
              f" {rec['mean_auroc']:10.5f} {mean_d:+11.5f} {rec['median_d_auroc']:+10.5f} "
              f"{pos}/{len(views):<4d} {rec['worst_auroc']:11.5f} {rec['worst_d_auroc']:+9.5f}  {v}")
    (OUT / "cross_view.json").write_text(json.dumps(rows, indent=2, default=float))
    fin = [r["pipeline"] for r in rows if r["verdict"].startswith("FINALIST")]
    st["analysis"] = {"per_view": {f"{p}@{v}": got[(p, v)]["metrics"]["image_auroc"]
                                   for (p, v) in got},
                      "cross_view": rows, "finalists": fin[:2]}
    E7.save_progress(PROGRESS, st)
    print(f"\n[E7-2 finalists] {fin[:2] if fin else 'NONE'}")
    print("[E7-2] baseline per view: " +
          ", ".join(f"{v}={got[('P00', v)]['metrics']['image_auroc']:.5f}" for v in views
                    if ("P00", v) in got))


if __name__ == "__main__":
    raise SystemExit(main())
