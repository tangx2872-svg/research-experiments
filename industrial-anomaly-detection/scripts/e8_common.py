#!/usr/bin/env python
"""E8 - shared frozen scope, bank rule, caches, progress/ETA, verdict.

Protocol: `docs/E8_FROZEN_PROTOCOL.md` (frozen at commit f332ee1 BEFORE any E8 result).

This module implements the frozen E8 design. It reuses the E7 frozen pipeline
(`e7_pipelines.run_patchcore`, `e7_common.evaluate`) verbatim - it does not redesign it.
"""
from __future__ import annotations

import json
import random
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402

# ------------------------------------------------------------------ frozen design (protocol s2)
CATEGORIES_CORE = ["Bird", "Car", "Cube"]
CATEGORIES_EXT = ["Dice", "Doll", "Holder"]
METHODS = ["P00", "P02", "P03"]
PRIMARY = "P03"
SEEDS = [0, 1, 2]
VIEWS = ["120", "090", "240", "330"]
BANK_POOL = 1200
BANK_SIZE = 300
BANK_SUBSET_SEED = E7.BANK_SUBSET_SEED
CACHE_ROOT = E7.CACHE_ROOT
CACHE_RES = E7.CACHE_RES

METHOD_KIND = {"P00": "l0_256", "P02": "clahe_mild", "P03": "clahe_medium"}
METHOD_DESC = {"P00": "Original PatchCore", "P02": "CLAHE mild(1.5)->PatchCore",
               "P03": "CLAHE medium(2.0)->PatchCore"}

E7_1_REF = {"P00": {"image_auroc": 0.78089, "d_prime": 1.22579},
            "P02": {"image_auroc": 0.79458, "d_prime": 1.21570},
            "P03": {"image_auroc": 0.79591, "d_prime": 1.21751}}
E7_2_REF = {"P00": [0.76963, 0.82964, 0.76890, 0.75465],
            "P02": [0.79543, 0.81060, 0.78714, 0.79276],
            "P03": [0.81062, 0.81536, 0.80133, 0.79992]}

OUT = ROOT / "results" / "e8"
RAW = OUT / "raw"
PROTOCOL = ROOT / "docs" / "E8_FROZEN_PROTOCOL.md"
HARD_DEADLINE = None      # deadline voided (user directive 2026-10-12): exit on GO/HOLD/STOP only
MIN_FREE_GB = 3.0


# ------------------------------------------------------------------ progress / ETA (protocol s5)
def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def vram() -> tuple:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(str(path)).free / 2**30


def gpu_util() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except Exception:  # noqa: BLE001
        return "n/a"


def hms(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0:
        return "n/a"
    return time.strftime("%H:%M:%S", time.gmtime(sec))


def clock(sec: float) -> str:
    if not np.isfinite(sec):
        return "n/a"
    return time.strftime("%H:%M", time.localtime(time.time() + sec))


class EtaTracker:
    """Dynamic ETA from real completed-unit wall-clock (EMA). One unit = one fit+4-view scoring."""

    def __init__(self, total_units: int, label: str = "E8", status_path=None):
        self.total, self.label = total_units, label
        self.status_path = status_path or (OUT / "status.json")
        self.t0 = time.time()
        self.times = []
        self.unit_t0 = time.time()
        self.last_beat = time.time()
        self._last = None

    def start_unit(self) -> None:
        self.unit_t0 = time.time()

    def finish_unit(self, name: str, extra: str = "") -> None:
        self.times.append(time.time() - self.unit_t0)
        self._last = name
        self.report(name, extra)

    def mean_unit(self) -> float:
        return float(np.mean(self.times)) if self.times else float("nan")

    def ema_unit(self) -> float:
        if not self.times:
            return float("nan")
        if len(self.times) < 3:
            return self.mean_unit()
        e = self.times[0]
        for t in self.times[1:]:
            e = 0.5 * t + 0.5 * e
        return e

    def remaining(self) -> float:
        e = self.ema_unit()
        return float("nan") if not np.isfinite(e) else e * max(0, self.total - len(self.times))

    def report(self, name: str, extra: str = "") -> None:
        done, el, rem = len(self.times), time.time() - self.t0, self.remaining()
        a, r = vram()
        print(f"[{self.label}] {done}/{self.total} {100.0*done/max(1,self.total):5.1f}% | "
              f"elapsed {hms(el)} | ETA {hms(rem)} | finish~{clock(rem)} | "
              f"avg/unit={hms(self.mean_unit())} | current={name} | GPU={gpu_util()} | "
              f"VRAM {a:.0f}/{r:.0f} MB | RAM {ram_mb():.0f} MB | disk {free_gb():.1f} GB | {extra}",
              flush=True)
        self.last_beat = time.time()
        self._write_status(name)

    def heartbeat_if_due(self, minutes: int = 5, extra: str = "") -> None:
        if time.time() - self.last_beat < minutes * 60:
            return
        done, el, rem = len(self.times), time.time() - self.t0, self.remaining()
        print(f"\n[E8 HEARTBEAT]\ncompleted = {done}/{self.total}\nelapsed   = {hms(el)}\n"
              f"ETA       = {hms(rem)}  (finish ~{clock(rem)})\nGPU util  = {gpu_util()}\n"
              f"VRAM      = {vram()[0]:.0f} MB alloc / {vram()[1]:.0f} MB reserved\n"
              f"disk free = {free_gb():.1f} GB\nRAM       = {ram_mb():.0f} MB\n"
              f"failures  = {count_failures()}\n{extra}\n", flush=True)
        self.last_beat = time.time()
        self._write_status(self._last)

    def _write_status(self, current: str = None) -> None:
        import os
        done = len(self.times)
        el = time.time() - self.t0
        rem = self.remaining()
        try:
            save_json(self.status_path, {
                "stage": self.label, "completed": done, "total": self.total,
                "percent": round(100.0 * done / max(1, self.total), 2),
                "elapsed_sec": round(el, 1),
                "eta_sec": (round(rem, 1) if np.isfinite(rem) else None),
                "current_unit": current, "last_completed_unit": self._last,
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid()})
        except Exception:  # noqa: BLE001
            pass

    def stage_announce(self, stage: str, est_min: float) -> None:
        print(f"\n=== Stage {stage} | estimated {est_min/60:.1f} h | "
              f"expected finish {clock(est_min*60)} ===\n", flush=True)

    def deadline_exceeded(self) -> bool:
        return False if HARD_DEADLINE is None else time.time() >= HARD_DEADLINE

    def time_to_deadline_h(self) -> float:
        return float("inf") if HARD_DEADLINE is None else (HARD_DEADLINE - time.time()) / 3600.0


# ------------------------------------------------------------------ persistence (protocol s5)
def progress_json() -> dict:
    p = OUT / "progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"experiment": "E8", "units": {}, "stage": "init"}


def save_json(path: Path, obj) -> None:
    """Atomic write: .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


def save_progress(st: dict) -> None:
    save_json(OUT / "progress.json", st)


def log_failure(rec: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "failures.jsonl", "a") as f:
        f.write(json.dumps(dict(rec, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                           default=float) + "\n")


def count_failures() -> int:
    p = OUT / "failures.jsonl"
    return sum(1 for _ in open(p)) if p.exists() else 0


def unit_path(method: str, category: str, seed: int) -> Path:
    return RAW / f"{method}_{category}_seed{seed}.json"


def unit_key(method: str, category: str, seed: int) -> str:
    return f"{method}|{category}|{seed}"


def scan_done() -> dict:
    """Resume support: map unit_key -> payload for every already-OK unit."""
    done = {}
    if not RAW.exists():
        return done
    for f in sorted(RAW.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("status") == "OK":
            done[unit_key(d["method"], d["category"], int(d["seed"]))] = d
    return done


# ------------------------------------------------------------------ scope + bank (protocol s2.3)
def load_scope(category: str, require_exists: bool = True) -> dict:
    """Per-category frozen scope: 1200-image bank pool + unreduced 4-view test set.

    `require_exists=False` is used for categories whose originals are not on local disk
    (E8 serves them from the photometric caches instead) - the official metadata is still
    the sole source of split/label/view/illumination.
    """
    recs = LD.load_records(category=category, require_exists=require_exists)
    est = LD.estimate_scope(recs)
    pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    test = sorted([r for r in LD.select(recs, views=VIEWS) if r["split"] == "test"],
                  key=lambda r: r["image_path"])
    return {"category": category, "recs": recs, "est": est, "pool": pool, "test": test}


def bank_for(pool: list, seed: int) -> list:
    """Frozen bank rule: sorted(random.Random(20261009+seed).sample(range(len(pool)), 300))."""
    n = len(pool)
    size = min(BANK_SIZE, n)
    pick = sorted(random.Random(BANK_SUBSET_SEED + seed).sample(range(n), size))
    return [pool[i] for i in pick]


def sanity_scope(scope: dict, seed: int = 0) -> dict:
    pool, test, cat = scope["pool"], scope["test"], scope["category"]
    bank = bank_for(pool, seed)
    bs = {r["specimen"] for r in bank}
    ts = {r["specimen"] for r in test}
    return {
        "S1_bank_pool": (len(pool) == BANK_POOL, f"{cat}: pool={len(pool)} (expect {BANK_POOL})"),
        "S1b_bank_size": (len(bank) == BANK_SIZE, f"{cat}: bank(seed {seed})={len(bank)} (expect {BANK_SIZE})"),
        "S2_test_unreduced": (len(test) == 2800, f"{cat}: test@4views={len(test)} (expect 2800)"),
        "S3_normal_only": (all(r["kind"] == "Good" and r["split"] == "train" for r in bank),
                           "bank = train-split Good images only"),
        "S4_no_leakage": (not (bs & ts), f"|bank specs n test specs|={len(bs & ts)}"),
        "S5_views": (sorted({r["view"] for r in test}) == sorted(VIEWS),
                     f"{cat}: views={sorted({r['view'] for r in test})}"),
        "S6_illum": (sorted({r["illumination"] for r in test}) == [f"{i:02d}" for i in range(1, 11)],
                     f"{cat}: illum={sorted({r['illumination'] for r in test})}"),
    }


# ------------------------------------------------------------------ caches (protocol s2.7)
def cache_path(kind: str, rec: dict) -> Path:
    return CACHE_ROOT / kind / Path(rec["image_path"]).relative_to(LD.DATA_ROOT)


# ------------------------------------------------------------------ verdict (protocol s3.5)
def agg(values: list) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def bootstrap_ci(values: list, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 2:
        return {"lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return {"lo": float(np.quantile(means, alpha / 2)), "hi": float(np.quantile(means, 1 - alpha / 2))}


def core_verdict(per_cell: dict) -> dict:
    """per_cell[(method, category, seed, view)] -> metric dict. Decision = f(protocol s3.5)."""
    out = {}
    cats = sorted({k[1] for k in per_cell})
    seeds = sorted({k[2] for k in per_cell})
    for m in ("P02", "P03"):
        cell_d, cat_d, seed_d, view_d, pres_d, aupr_d = [], {}, {}, {}, [], []
        pos_cat_seed = 0
        tot_cat_seed = 0
        for cat in cats:
            for s in seeds:
                cs = []
                for v in VIEWS:
                    a, b = per_cell.get((m, cat, s, v)), per_cell.get(("P00", cat, s, v))
                    if not a or not b:
                        continue
                    d = a["image_auroc"] - b["image_auroc"]
                    cell_d.append(d)
                    cs.append(d)
                    cat_d.setdefault(cat, []).append(d)
                    seed_d.setdefault(s, []).append(d)
                    view_d.setdefault(v, []).append(d)
                    for key, bucket in (("d_prime", pres_d), ("image_aupr", aupr_d)):
                        x, y = a.get(key), b.get(key)
                        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                            bucket.append(x - y)
                if cs:
                    tot_cat_seed += 1
                    if float(np.mean(cs)) > 0:
                        pos_cat_seed += 1
        cat_mean = {c: agg(v)["mean"] for c, v in cat_d.items()}
        seed_mean = {s: agg(v)["mean"] for s, v in seed_d.items()}
        view_mean = {v: agg(x)["mean"] for v, x in view_d.items()}
        e = {"cells": agg(cell_d), "bootstrap_ci": bootstrap_ci(cell_d),
             "per_category_mean_d_auroc": cat_mean, "per_seed_mean_d_auroc": seed_mean,
             "per_view_mean_d_auroc": view_mean,
             "mean_d_dprime": agg(pres_d)["mean"], "n_dprime": len(pres_d),
             "mean_d_aupr": agg(aupr_d)["mean"], "n_aupr": len(aupr_d),
             "positive_categories": sum(1 for x in cat_mean.values() if x > 0),
             "n_categories": len(cat_mean),
             "positive_cat_seed": pos_cat_seed, "n_cat_seed": tot_cat_seed,
             "positive_views": sum(1 for x in view_mean.values() if x > 0), "n_views": len(view_mean),
             "worst_category": (min(cat_mean, key=cat_mean.get) if cat_mean else None),
             "worst_category_d": (min(cat_mean.values()) if cat_mean else float("nan")),
             "best_category": (max(cat_mean, key=cat_mean.get) if cat_mean else None),
             "best_category_d": (max(cat_mean.values()) if cat_mean else float("nan")),
             "worst_view": (min(view_mean, key=view_mean.get) if view_mean else None),
             "worst_view_d": (min(view_mean.values()) if view_mean else float("nan")),
             "seed_consistent": bool(seed_mean) and all(x > 0 for x in seed_mean.values())}
        mean = e["cells"]["mean"]
        safe = np.isfinite(mean)
        if safe and mean > 0 and e["positive_categories"] >= max(2, -(-e["n_categories"] // 3 * 2)) \
                and e["mean_d_dprime"] > -0.10 and e["seed_consistent"]:
            e["verdict"] = "GO"
        elif safe and mean > 0:
            e["verdict"] = "HOLD"
        else:
            e["verdict"] = "STOP"
        e["go_rule_note"] = (f"mean dAUROC={mean:+.4f} (need >0), "
                             f"positive categories={e['positive_categories']}/{e['n_categories']} "
                             f"(need >=2), mean dd'={e['mean_d_dprime']:+.4f} (need >-0.10), "
                             f"seed consistent={e['seed_consistent']}")
        out[m] = e
    order = {"GO": 2, "HOLD": 1, "STOP": 0}
    best = max(out, key=lambda m: (order[out[m]["verdict"]],
                                   out[m]["cells"]["mean"] if np.isfinite(out[m]["cells"]["mean"]) else -9))
    return {"per_method": out, "primary": best, "primary_verdict": out[best]["verdict"],
            "E8": out[best]["verdict"]}
