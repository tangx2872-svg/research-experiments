#!/usr/bin/env python
"""E8 - core matrix runner (Stage B gate + Stage C matrix) and Extensions A/B/C.

Protocol: `docs/E8_FROZEN_PROTOCOL.md` (commit f332ee1).

One unit = one (method, category, seed) fit, scored on all 4 frozen views.
The E7 frozen pipeline is reused verbatim (`e7_pipelines.run_patchcore`, `e7_common.evaluate`).
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

import e7_common as E7  # noqa: E402
import e7_pipelines as PL  # noqa: E402
import e8_common as E8  # noqa: E402
import e8_fetch as F  # noqa: E402


class Bar(E7.Bar):
    """Same bar, E8 label (display only)."""

    def step(self, done=None, extra=""):
        self.done = done if done is not None else self.done + 1
        el = time.time() - self.t0
        rate = self.done / el if el > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else float("inf")
        a, r = E8.vram()
        print(f"  E8 [{self.tag}] {self.stage} {self.done}/{self.total} "
              f"{100.0*self.done/max(1,self.total):5.1f}%  elapsed={E8.hms(el)}  "
              f"ETA={E8.hms(remain)}  finish~{E8.clock(remain)}  {rate:6.1f} {self.unit}/s  "
              f"VRAM {a:.0f}/{r:.0f} MB  disk {E8.free_gb():.1f} GB {extra}", flush=True)


E7.Bar = Bar


# ------------------------------------------------------------------ unit execution
def reroute(records: list, kind: str) -> list:
    return [dict(r, image_path=str(E8.cache_path(kind, r))) for r in records]


def run_unit(method: str, category: str, seed: int, dev) -> dict:
    """Fit once, score all 4 frozen views, evaluate per view."""
    kind = E8.METHOD_KIND[method]
    scope = F.build_scope(category)
    bank = E8.bank_for(scope["pool"], seed)
    test = scope["test"]

    bank_r, test_r = reroute(bank, kind), reroute(test, kind)
    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        return {"status": "FAILED",
                "reason": f"{len(missing)} cache images missing (Stage A incomplete)",
                "example": missing[:2]}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tag = f"E8_{method}_{category}_s{seed}"
    t0 = time.time()
    res = PL.run_patchcore(bank_r, test_r, dev, tag, want_maps=True)
    total_s = time.time() - t0

    if len(res["rows"]) != len(test):
        return {"status": "FAILED", "reason": f"row count {len(res['rows'])} != test {len(test)}"}

    per_view = {}
    rows_by_view = {}
    for i, rec in enumerate(test):
        v = rec["view"]
        rows_by_view.setdefault(v, {"rows": [], "maps": [], "masks": []})
        rows_by_view[v]["rows"].append(dict(res["rows"][i], specimen=rec["specimen"],
                                            kind=rec["kind"], illumination=rec["illumination"],
                                            view=v))
        rows_by_view[v]["maps"].append(res["maps"][i])
        rows_by_view[v]["masks"].append(res["masks"][i])

    for v in E8.VIEWS:
        if v not in rows_by_view:
            continue
        b = rows_by_view[v]
        m = E7.evaluate(b["rows"], b["maps"], b["masks"], want_pixel=True)
        m["view"] = v
        m["n_records"] = len(b["rows"])
        m["good_mean"] = float(np.mean([r["score"] for r in b["rows"] if r["kind"] == "Good"]))
        m["ng_mean"] = float(np.mean([r["score"] for r in b["rows"] if r["kind"] == "NG"]))
        per_view[v] = m

    return {"status": "OK", "method": method, "category": category, "seed": seed,
            "view": E8.VIEWS, "bank_size": len(bank), "test_size": len(test),
            "bank_specimens": sorted({r["specimen"] for r in bank}),
            "image_kind": kind, "per_view": per_view,
            "runtime_s": round(total_s, 1),
            "fit_s": round(res["stages"].get("fit_s", float("nan")), 1),
            "scoring_s": round(res["stages"].get("scoring_s", float("nan")), 1),
            "coreset_size": res["stages"].get("coreset_size"),
            "peak_vram_mb": res["stages"].get("peak_vram_alloc_mb"),
            "protocol_sha256": E8.sha_of_protocol() if hasattr(E8, "sha_of_protocol") else None,
            "git_head": git_head()}


def git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def protocol_sha() -> str:
    return E7.sha256(E8.PROTOCOL) if E8.PROTOCOL.exists() else "missing"


# ------------------------------------------------------------------ stage drivers
def unit_list(methods, categories, seeds) -> list:
    return [(m, c, s) for c in categories for s in seeds for m in methods]


def execute_units(units: list, dev, tracker: E8.EtaTracker, st: dict, label: str = "") -> dict:
    """Run every requested unit in order; atomic per-unit save, resume-aware, deadline-aware."""
    done = E8.scan_done()
    results = {}
    for (m, c, s) in units:
        key = E8.unit_key(m, c, s)
        if key in done:
            print(f"[E8] SKIP COMPLETED {m}/{c}/seed{s}", flush=True)
            results[key] = done[key]
            continue
        if tracker.deadline_exceeded():
            print(f"[E8] HARD DEADLINE reached - not starting {m}/{c}/seed{s}", flush=True)
            break
        if E8.free_gb() < E8.MIN_FREE_GB:
            print(f"[E8] DISK GUARD: only {E8.free_gb():.1f} GB free - stopping before "
                  f"{m}/{c}/seed{s} (state saved)", flush=True)
            break
        name = f"{m}/{c}/seed{s}"
        print(f"\n[E8] UNIT START {name}  kind={E8.METHOD_KIND[m]}  "
              f"({label}) deadline in {tracker.time_to_deadline_h():.2f} h", flush=True)
        tracker.start_unit()
        st.setdefault("units", {})[key] = {"status": "RUNNING", "method": m, "category": c,
                                           "seed": s, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
        E8.save_progress(st)
        payload = {"method": m, "category": c, "seed": s}
        try:
            out = run_unit(m, c, s, dev)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        payload.update(out)
        if out.get("status") == "OK":
            payload["protocol_sha256"] = protocol_sha()
            E8.save_json(E8.unit_path(m, c, s), payload)
            st["units"][key] = {"status": "DONE", "method": m, "category": c, "seed": s,
                                "runtime_s": out["runtime_s"],
                                "mean_auroc": float(np.mean([v["image_auroc"]
                                                             for v in out["per_view"].values()])),
                                "end": time.strftime("%Y-%m-%d %H:%M:%S")}
            E8.save_progress(st)
            tracker.finish_unit(name, extra=f"runtime={out['runtime_s']:.0f}s "
                                            f"mean_auroc={st['units'][key]['mean_auroc']:.5f}")
            results[key] = payload
        else:
            st["units"][key] = {"status": "FAILED", "method": m, "category": c, "seed": s,
                                "reason": out.get("reason", "")}
            E8.save_progress(st)
            E8.log_failure({"method": m, "category": c, "seed": s, "exception": out.get("reason", "")})
            tracker.finish_unit(name, extra=f"FAILED: {out.get('reason','')[:60]}")
        tracker.heartbeat_if_due()
    return results


def collect_cells(methods, categories, seeds) -> dict:
    """Load per-view metrics from disk into per_cell[(method, category, seed, view)]."""
    cells = {}
    for (m, c, s) in unit_list(methods, categories, seeds):
        p = E8.unit_path(m, c, s)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if d.get("status") != "OK":
            continue
        for v, met in d.get("per_view", {}).items():
            cells[(m, c, s, v)] = met
    return cells


# ------------------------------------------------------------------ Stage B gate (protocol s3.3)
def stage_b_verdict(cells: dict) -> dict:
    rows = {}
    for m in E8.METHODS:
        got = cells.get((m, "Bird", 0, "120"))
        ref = E8.E7_1_REF[m]
        if not got:
            return {"verdict": "FAIL", "reason": f"missing Bird/seed0/view120 for {m}"}
        rows[m] = {"auroc": got["image_auroc"], "ref": ref["image_auroc"],
                   "dev": got["image_auroc"] - ref["image_auroc"],
                   "d_prime": got["d_prime"], "d_prime_ref": ref["d_prime"]}
    max_dev = max(abs(v["dev"]) for v in rows.values())
    order = sorted(rows, key=lambda m: -rows[m]["auroc"])
    ref_order = sorted(E8.METHODS, key=lambda m: -E8.E7_1_REF[m]["image_auroc"])
    ranking_ok = order == ref_order
    means = {m: float(np.mean([cells[(m, "Bird", 0, v)]["image_auroc"] for v in E8.VIEWS
                              if (m, "Bird", 0, v) in cells])) for m in E8.METHODS}
    if max_dev <= 0.010 and ranking_ok:
        v = "PASS"
    elif max_dev <= 0.025 and ranking_ok:
        v = "WARNING"
    else:
        v = "FAIL"
    return {"verdict": v, "max_abs_dev": max_dev, "ranking_observed": order,
            "ranking_reference": ref_order, "ranking_preserved": ranking_ok,
            "per_method": rows, "bird_seed0_4view_mean": means,
            "e7_2_reference_means": {m: float(np.mean(E8.E7_2_REF[m])) for m in E8.METHODS},
            "note": "ranking well-ordered means P03 >= P02 >= P00 by measured AUROC"}


# ------------------------------------------------------------------ safety gate (protocol s3.3 / s5)
def safety_gate(categories: list) -> dict:
    g = {}
    g["protocol_committed"] = (E8.PROTOCOL.exists(), f"{E8.PROTOCOL.name} sha={protocol_sha()[:16]}")
    prep = E8.OUT / "prep.json"
    ok_prep = prep.exists()
    detail = "prep.json found"
    if ok_prep:
        d = json.loads(prep.read_text())
        missing = [c for c in categories if c not in d or "gate" not in d[c]
                   or not all(v["pass"] for v in d[c]["gate"].values())]
        ok_prep = not missing
        detail = f"gate passed for {list(d)}" if ok_prep else f"gate missing/failed for {missing}"
    g["stage_a_gate"] = (ok_prep, detail)
    import subprocess
    try:
        out = subprocess.check_output(["pgrep", "-af", "e8_runner|e8_fetch|e7_1_runner|e5_1_runner"],
                                      text=True).strip()
        mine = str(__import__("os").getpid())
        others = [ln for ln in out.splitlines() if ln.split()[0] not in (mine,)]
        busy = [ln for ln in others if "e8_runner.py" in ln]
    except subprocess.CalledProcessError:
        busy = []
    g["no_other_runner"] = (not busy, f"other runners: {busy if busy else 'none'}")
    g["disk"] = (E8.free_gb() >= E8.MIN_FREE_GB, f"{E8.free_gb():.1f} GB free (min {E8.MIN_FREE_GB})")
    g["deadline"] = (not E8.EtaTracker(1).deadline_exceeded(),
                     f"deadline in {E8.EtaTracker(1).time_to_deadline_h():.2f} h")
    return g


# ------------------------------------------------------------------ extensions
EXTC = {"C125": ("clahe_c125", 1.25), "C150": ("clahe_mild", 1.50),
        "C175": ("clahe_c175", 1.75), "C200": ("clahe_medium", 2.00),
        "C225": ("clahe_c225", 2.25)}


def build_extc_caches(category: str, tracker: E8.EtaTracker) -> None:
    """Extension C: new CLAHE clip levels, produced from the stored l0_448 source."""
    from PIL import Image
    scope = F.build_scope(category)
    recs = scope["pool"] + scope["test"]
    todo = []
    for name, (kind, clip) in EXTC.items():
        if kind in ("clahe_mild", "clahe_medium"):
            continue
        for r in recs:
            if not (E8.CACHE_ROOT / kind / F.rel_of(r)).exists():
                todo.append((kind, clip, F.rel_of(r)))
    print(f"[E8 extC] {category}: building {len(todo)} new CLAHE caches "
          f"for clip levels {sorted({c for _, c, _ in todo})}", flush=True)
    for i, (kind, clip, rel) in enumerate(todo, 1):
        src = E8.CACHE_ROOT / "l0_448" / rel
        if not src.exists():
            continue
        arr = np.asarray(Image.open(src).convert("RGB"))
        dst = E8.CACHE_ROOT / kind / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(E7._apply_clahe(arr, clip)).save(dst)
        if i % 200 == 0:
            print(f"[E8 extC] {category}: {i}/{len(todo)} caches  disk {E8.free_gb():.1f} GB",
                  flush=True)
    tracker.heartbeat_if_due(minutes=10)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core", help="comma list: b,c,extA,extB,extC")
    ap.add_argument("--categories", default=",".join(E8.CATEGORIES_CORE))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default=",".join(E8.METHODS))
    ap.add_argument("--ext-seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[E8] GPU {props.name} {props.total_memory/2**30:.1f} GiB | protocol "
          f"{protocol_sha()[:16]} | git {git_head()[:8]} | stages={stages}", flush=True)

    gate = safety_gate(cats)
    for k, (ok, det) in gate.items():
        print(f"  safety {k}: {'PASS' if ok else 'FAIL'}  {det}", flush=True)
    if not all(ok for ok, _ in gate.values()):
        print("[E8] SAFETY GATE FAILED - not starting. Recorded as BLOCKED.", flush=True)
        E8.save_json(E8.OUT / "blocked.json",
                     {"gate": {k: v[1] for k, v in gate.items()},
                      "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 5

    st = E8.progress_json()
    st["stage"] = stages
    st["protocol_sha256"] = protocol_sha()
    st["git_head"] = git_head()
    st["categories"] = cats
    E8.save_progress(st)

    if "b" in stages:
        tracker = E8.EtaTracker(len(E8.METHODS), "E8-StageB")
        tracker.stage_announce("B (Bird regression gate)", 15)
        execute_units(unit_list(methods, ["Bird"], [0]), dev, tracker, st, "Stage B")
        v = stage_b_verdict(collect_cells(methods, ["Bird"], [0]))
        E8.save_json(E8.OUT / "stage_b.json", v)
        print(f"[E8] STAGE B GATE = {v['verdict']}  (max |dev| = {v['max_abs_dev']:.5f}, "
              f"ranking preserved = {v['ranking_preserved']})", flush=True)
        for m, r in v["per_method"].items():
            print(f"    {m}: auroc={r['auroc']:.5f} ref={r['ref']:.5f} dev={r['dev']:+.5f} | "
                  f"d'={r['d_prime']:.5f} ref={r['d_prime_ref']:.5f}", flush=True)
        print(f"    Bird seed0 4-view means: {v['bird_seed0_4view_mean']}", flush=True)
        if v["verdict"] == "FAIL":
            print("[E8] STAGE B = FAIL -> STOP E8 (do not run the night on a wrong pipeline).",
                  flush=True)
            return 6

    if "c" in stages:
        units = unit_list(methods, cats, seeds)
        tracker = E8.EtaTracker(len(units), "E8-StageC")
        tracker.stage_announce("C (core matrix)", len(units) * 2.5)
        execute_units(units, dev, tracker, st, "Stage C")
        cv = E8.core_verdict(collect_cells(methods, cats, seeds))
        E8.save_json(E8.OUT / "verdict.json", cv)
        print(f"[E8] CORE VERDICT = {cv['E8']} (primary {cv['primary']})", flush=True)
        for m, e in cv["per_method"].items():
            print(f"    {m}: mean dAUROC={e['cells']['mean']:+.5f} "
                  f"CI[{e['bootstrap_ci']['lo']:+.4f},{e['bootstrap_ci']['hi']:+.4f}] "
                  f"pos_cat={e['positive_categories']}/{e['n_categories']} "
                  f"mean dd'={e['mean_d_dprime']:+.4f} -> {e['verdict']}", flush=True)

    if "extA" in stages:
        ext = E8.CATEGORIES_EXT
        remaining = E8.EtaTracker(1).time_to_deadline_h()
        print(f"[E8 extA] breadth extension: {ext} | {remaining:.2f} h to deadline", flush=True)
        units = unit_list(methods, ext, seeds)
        tracker = E8.EtaTracker(len(units), "E8-extA")
        tracker.stage_announce("extA (additional categories)", len(units) * 2.5)
        execute_units(units, dev, tracker, st, "extA")
        cv = E8.core_verdict(collect_cells(methods, cats + ext, seeds))
        E8.save_json(E8.OUT / "verdict_extA.json", cv)
        print(f"[E8 extA] verdict over {cats+ext} = {cv['E8']}", flush=True)

    if "extB" in stages:
        cv = E8.core_verdict(collect_cells(methods, cats, seeds))
        winner = cv["primary"]
        ext_seeds = [int(s) for s in args.ext_seeds.split(",") if s.strip()]
        use_methods = ["P00"] + ([winner] if winner != "P00" else [])
        units = unit_list(use_methods, cats, ext_seeds)
        tracker = E8.EtaTracker(len(units), "E8-extB")
        tracker.stage_announce(f"extB (winner {winner}, seeds {ext_seeds})", len(units) * 2.5)
        print(f"[E8 extB] winner={winner} methods={use_methods} seeds={ext_seeds}", flush=True)
        execute_units(units, dev, tracker, st, "extB")
        cv2 = E8.core_verdict(collect_cells(use_methods, cats, ext_seeds))
        E8.save_json(E8.OUT / "verdict_extB.json", cv2)
        print(f"[E8 extB] verdict with {len(ext_seeds)} seeds = {cv2['E8']}", flush=True)

    if "extC" in stages:
        cv = E8.core_verdict(collect_cells(methods, cats, seeds))
        winner = cv["primary"]
        for name, (kind, clip) in EXTC.items():
            E8.METHOD_KIND[name] = kind
        tracker = E8.EtaTracker(1, "E8-extC")
        extc_cats = cats[:1] if cv["E8"] == "STOP" else cats
        for c in extc_cats:
            build_extc_caches(c, tracker)
        use_methods = list(EXTC.keys())
        units = unit_list(use_methods, extc_cats, [0, 1])
        tracker = E8.EtaTracker(len(units), "E8-extC")
        tracker.stage_announce("extC (CLIP neighbourhood)", len(units) * 2.5)
        print(f"[E8 extC] winner={winner} grid={sorted(EXTC)} cats={extc_cats}", flush=True)
        execute_units(units, dev, tracker, st, "extC")
        E8.save_json(E8.OUT / "verdict_extC.json",
                     {"grid": {k: v[1] for k, v in EXTC.items()}, "categories": extc_cats,
                      "cells": collect_cells(use_methods, extc_cats, [0, 1])})
    print(f"\n[E8] done. artifacts in {E8.OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
