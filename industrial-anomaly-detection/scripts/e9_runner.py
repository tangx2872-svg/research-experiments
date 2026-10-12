#!/usr/bin/env python
"""E9 - Real-Illumination Robust Module Screening runner (smoke -> round1 -> round2).

Reuses the frozen E7 PatchCore (`e7_pipelines.run_patchcore`) verbatim. Feature-level
candidates (F01 / A1) are implemented by monkey-patching `generate_embedding` to L2-normalise
each patch feature at runtime; the coreset / kNN / max-pool scoring stay frozen.

Scope: 300-image bank (E7-1 seed-0 rule), test @ view 120. Baseline = E8 Original P00.
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import e7_common as E7  # noqa: E402
import e7_pipelines as PL  # noqa: E402
import e8_common as E8  # noqa: E402
import e9_common as E9  # noqa: E402

OUT = ROOT / "results" / "e9"
RAW = OUT / "raw"


# ------------------------------------------------------------------ feature monkeypatch
def _patch_feature_norm():
    import experiment5a_model as E5M
    orig = E5M.FAlphaDualLayerPatchcoreModel.generate_embedding

    def gen(self, features):
        return F.normalize(orig(self, features), dim=1)
    E5M.FAlphaDualLayerPatchcoreModel.generate_embedding = gen
    return orig


def _unpatch(orig):
    import experiment5a_model as E5M
    E5M.FAlphaDualLayerPatchcoreModel.generate_embedding = orig


# ------------------------------------------------------------------ baseline
def baseline_auroc(category: str) -> float:
    p = ROOT / "results" / "e8" / "raw" / f"P00_{category}_seed0.json"
    d = json.loads(p.read_text())
    return float(d["per_view"]["120"]["image_auroc"])


def baseline_dprime(category: str) -> float:
    p = ROOT / "results" / "e8" / "raw" / f"P00_{category}_seed0.json"
    d = json.loads(p.read_text())
    return float(d["per_view"]["120"]["d_prime"])


# ------------------------------------------------------------------ prep
def build_input_cache(kind: str, records: list, bar=None) -> int:
    """Apply an input transform to L0_256 sources and write the E9 cache."""
    from e9_common import apply_kind, image_cache_path, E9_CACHE_ROOT
    written = 0
    for i, r in enumerate(records, 1):
        dst = image_cache_path(kind, r)
        if dst.exists():
            continue
        src = E8.CACHE_ROOT / "l0_256" / Path(r["image_path"]).relative_to(E7.LD.DATA_ROOT)
        rgb = np.asarray(Image.open(src).convert("RGB"))
        out = apply_kind(kind, rgb)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst)
        written += 1
        if bar is not None and i % 500 == 0:
            bar(i, extra=f"{kind} written={written}")
    return written


def build_retinex(records: list, bar=None) -> int:
    """Retinex from L0_448 source (Car/Cube originals are not local). Bit-identical to E7's
    `_load_rgb448(original)` because L0_448 is that exact resize, stored losslessly."""
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    model, save_fn = E7._retinex_model()
    written = 0
    for i, r in enumerate(records, 1):
        rel = Path(r["image_path"]).relative_to(E7.LD.DATA_ROOT)
        dst = E7.CACHE_ROOT / "retinex" / rel
        if dst.exists():
            continue
        src = E8.CACHE_ROOT / "l0_448" / rel
        arr = np.asarray(Image.open(src).convert("RGB"))
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            R, _L, _ = model(t)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_fn(R, str(dst))
        written += 1
        if bar is not None and i % 200 == 0:
            bar(i, extra=f"retinex written={written}")
    return written


def prep(categories: list, tracker: E8.EtaTracker) -> None:
    """Build E9 input caches + Retinex for Car/Cube (Bird retinex already cached)."""
    for cat in categories:
        sc = E9.round1_scope(cat)
        recs = sc["bank"] + sc["test"]
        # input transforms
        for kind in E9.INPUT_KINDS:
            t0 = time.time()
            n = build_input_cache(kind, recs)
            print(f"[E9 PREP] {cat} {kind}: wrote {n} (missing) in {time.time()-t0:.0f}s "
                  f"disk {E8.free_gb():.1f} GB", flush=True)
        # retinex from L0_448 (Bird cached in E7; Car/Cube built here)
        need = [r for r in recs if not (E7.CACHE_ROOT / "retinex" /
                Path(r["image_path"]).relative_to(E7.LD.DATA_ROOT)).exists()]
        if need:
            print(f"[E9 PREP] {cat} retinex: {len(need)} missing -> building (GPU)", flush=True)
            build_retinex(need, bar=None)
        else:
            print(f"[E9 PREP] {cat} retinex: already cached", flush=True)
        tracker.heartbeat_if_due(minutes=5)


# ------------------------------------------------------------------ run one candidate
def run_candidate(cid: str, scope: dict, dev, tag: str) -> dict:
    spec = E9.CANDIDATES[cid]
    kind = spec["image_kind"] or "l0_256"
    bank_r = E9.reroute(scope["bank"], kind)
    test_r = E9.reroute(scope["test"], kind)
    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        return {"status": "FAILED", "reason": f"{len(missing)} cache images missing (prep first)",
                "example": missing[:2]}

    seed = scope.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    orig = _patch_feature_norm() if spec["feature_norm"] else None
    try:
        res = PL.run_patchcore(bank_r, test_r, dev, tag, want_maps=False)
    finally:
        if orig is not None:
            _unpatch(orig)

    rows = res["rows"]
    met = E7.evaluate(rows, want_pixel=False)
    met["runtime_s"] = round(res["stages"].get("total_s", float("nan")), 1)
    met["fit_s"] = round(res["stages"].get("fit_s", float("nan")), 1)
    met["coreset_size"] = res["stages"].get("coreset_size")
    met["peak_vram_mb"] = res["stages"].get("peak_vram_alloc_mb")
    return {"status": "OK", "metrics": met, "rows": rows}


# ------------------------------------------------------------------ smoke
def smoke(dev) -> int:
    print("[E9 SMOKE] Bird / seed0 / tiny subset (12 bank + 24 test) per candidate", flush=True)
    scope = E9.smoke_scope("Bird")
    ok = 0
    for cid in E9.ORDER:
        t0 = time.time()
        out = run_candidate(cid, scope, dev, f"E9smoke_{cid}")
        el = time.time() - t0
        if out["status"] != "OK":
            print(f"  [E9 SMOKE] {cid}: FAILED  {out.get('reason','')}", flush=True)
            continue
        m = out["metrics"]
        fin = np.isfinite(m["image_auroc"]) and np.isfinite(m["d_prime"])
        print(f"  [E9 SMOKE] {cid}: {'PASS' if fin else 'FAIL(inf)'}  AUROC={m['image_auroc']:.4f} "
              f"d'={m['d_prime']:.4f}  n_good={m['n_good']} n_def={m['n_defect']}  "
              f"fit={m['fit_s']:.0f}s VRAM={m['peak_vram_mb']:.0f}MB  {el:.1f}s", flush=True)
        ok += int(fin)
    print(f"[E9 SMOKE] {ok}/{len(E9.ORDER)} PASS", flush=True)
    return 0 if ok == len(E9.ORDER) else 2


# ------------------------------------------------------------------ round 1
def round1(dev) -> int:
    units = [(cid, cat) for cat in E9.ROUND1_CATS for cid in E9.ORDER]
    tracker = E8.EtaTracker(len(units), "E9-Round1")
    tracker.stage_announce("Round 1 (3 cats x 7 candidates, seed0, view120)", len(units) * 2.5)
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, cat in units:
        up = RAW / f"{cid}_{cat}_seed0.json"
        if up.exists():
            print(f"[E9 R1] SKIP {cid}/{cat} (already done)", flush=True)
            continue
        name = f"{cid}/{cat}"
        tracker.start_unit()
        print(f"\n[E9 R1] START {name}", flush=True)
        scope = E9.round1_scope(cat); scope["seed"] = 0
        out = run_candidate(cid, scope, dev, f"E9_{cid}_{cat}_s0")
        payload = {"candidate": cid, "category": cat, "seed": 0, "view": E9.ROUND1_VIEW,
                   "family": E9.CANDIDATES[cid]["family"],
                   "baseline_auroc": baseline_auroc(cat),
                   "baseline_dprime": baseline_dprime(cat)}
        payload.update(out)
        if out["status"] == "OK":
            payload["d_auroc"] = out["metrics"]["image_auroc"] - payload["baseline_auroc"]
            payload["d_dprime"] = out["metrics"]["d_prime"] - payload["baseline_dprime"]
        E8.save_json(up, payload)
        if out["status"] == "OK":
            m = out["metrics"]
            tracker.finish_unit(name, extra=f"AUROC={m['image_auroc']:.4f} "
                                            f"dA={payload['d_auroc']:+.4f} d'={m['d_prime']:.4f}")
        else:
            tracker.finish_unit(name, extra=f"FAILED: {out.get('reason','')[:50]}")
        tracker.heartbeat_if_due()
    print("\n[E9 R1] done", flush=True)
    return 0


# ------------------------------------------------------------------ round 2
def round2(dev, top2: list) -> int:
    import e8_fetch as EF
    cats = E9.ROUND1_CATS
    seeds = [0, 1, 2]
    views = ["120", "090"]
    units = [(cid, cat, s) for cat in cats for s in seeds for cid in top2]
    tracker = E8.EtaTracker(len(units), "E9-Round2")
    tracker.stage_announce(f"Round 2 (top2={top2}, 3 cats x 3 seeds x 2 views)", len(units) * 2.5)
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, cat, s in units:
        up = RAW / f"{cid}_{cat}_seed{s}.json"
        if up.exists():
            print(f"[E9 R2] SKIP {cid}/{cat}/s{s}", flush=True)
            continue
        name = f"{cid}/{cat}/s{s}"
        tracker.start_unit()
        scope = EF.build_scope(cat)
        bank = E8.bank_for(scope["pool"], s)
        test = [r for r in scope["test"] if r["view"] in views]
        out = run_candidate(cid, {"category": cat, "bank": bank, "test": test, "seed": s},
                            dev, f"E9_{cid}_{cat}_s{s}")
        payload = {"candidate": cid, "category": cat, "seed": s, "views": views,
                   "family": E9.CANDIDATES[cid]["family"]}
        payload.update(out)
        if out["status"] == "OK":
            per_view = {}
            for r, rec in zip(out["rows"], test):
                per_view.setdefault(rec["view"], []).append(r)
            pv = {}
            for v in views:
                if v not in per_view:
                    continue
                m = E7.evaluate(per_view[v], want_pixel=False)
                bp = ROOT / "results" / "e8" / "raw" / f"P00_{cat}_seed{s}.json"
                bd = json.loads(bp.read_text())["per_view"][v]
                m["baseline_auroc"] = bd["image_auroc"]
                m["baseline_dprime"] = bd["d_prime"]
                m["d_auroc"] = m["image_auroc"] - bd["image_auroc"]
                m["d_dprime"] = m["d_prime"] - bd["d_prime"]
                pv[v] = m
            payload["per_view"] = pv
            payload["mean_d_auroc"] = float(np.mean([pv[v]["d_auroc"] for v in pv]))
            payload["mean_d_dprime"] = float(np.mean([pv[v]["d_dprime"] for v in pv]))
            tracker.finish_unit(name, extra=f"mean_dA={payload['mean_d_auroc']:+.4f}")
        else:
            tracker.finish_unit(name, extra=f"FAILED {out.get('reason','')[:40]}")
        E8.save_json(up, payload)
        tracker.heartbeat_if_due()
    print("\n[E9 R2] done", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke", help="prep|smoke|round1|round2")
    ap.add_argument("--top2", default="", help="comma-separated candidate ids for round2")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    print(f"[E9] GPU {torch.cuda.get_device_properties(0).name} | stage={args.stage} | "
          f"disk {E8.free_gb():.1f} GB", flush=True)

    if args.stage == "prep":
        tr = E8.EtaTracker(1, "E9-PREP")
        prep(E9.ROUND1_CATS, tr)
        return 0
    if args.stage == "smoke":
        return smoke(dev)
    if args.stage == "round1":
        return round1(dev)
    if args.stage == "round2":
        top2 = [c.strip() for c in args.top2.split(",") if c.strip()] or ["A0", "I01"]
        return round2(dev, top2)
    print("unknown stage", args.stage)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
