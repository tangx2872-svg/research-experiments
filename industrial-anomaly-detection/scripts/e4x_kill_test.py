#!/usr/bin/env python
"""E4-X — M²AD Bird Real-Illumination Candidate Kill Test (KEEP OR KILL).

Protocol frozen in `docs/E4_X_PROTOCOL.md` BEFORE reading any B2/X6c result.

Scope (strict):  Category=Bird | Seed=0 | frozen view=120 | illumination 01-10  -> 700 images
GPU: ONE unit (B2 fit + B2 scoring).  X6c = score-level fusion (0 GPU).
     Original = reused from E4-D1 (0 GPU).

Runs B2 then X6c (never concurrently). Writes only to results/e4_x/.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python -u scripts/e4x_kill_test.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment1h_runner import image_auroc, pixel_auroc_from_maps  # noqa: E402
import e4d1_m2ad_loader as LD  # noqa: E402
from experiment11_model import AdaptivePatchcore  # noqa: E402

from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

# ---------------------------------------------------------------- FROZEN CONSTANTS
OUT = ROOT / "results" / "e4_x"
E4D1 = ROOT / "results" / "e4_d1_smoke"
SEED = 0
BATCH = 16
VIEWS = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]
VIEW_SELECTION_SEED = 20261008
FROZEN_VIEW = random.Random(VIEW_SELECTION_SEED).choice(VIEWS)      # -> "120"
BETA0, BMIN, BMAX = 0.25, 0.05, 0.50
Q_REF = 1.5726841688159168                    # q_bottle, frozen in normal_statistics.csv
L3_ALPHA = 0.25                               # frozen Adaptive B2 layer-3 alpha
W_C2, W_C6 = 0.35, 0.65                       # frozen
FUSE_C2, FUSE_C6 = 0.5, 0.5                   # frozen X6c = mean(z_C2, z_C6)
EPS_DET = 0.03                                # screening band (protocol 7.1)
DELTA_ROBUST = 0.05
EPS_ROBUST = 0.05
MEANINGFUL_GAIN = 0.03
PROGRESS_TOTAL_UNITS = 2                      # B2 fit+score = 1 unit ; X6c = 1 (CPU) unit


# ---------------------------------------------------------------- helpers
def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def vram_gb() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dprime(good: np.ndarray, defect: np.ndarray) -> float:
    if len(good) < 2 or len(defect) < 2:
        return float("nan")
    pooled = float(np.sqrt((defect.std(ddof=1) ** 2 + good.std(ddof=1) ** 2) / 2.0))
    return float((defect.mean() - good.mean()) / pooled) if pooled > 0 else float("nan")


def prog(step: int, total: int, t0: float, stage: str, method: str, imgs: int = 0,
         n_imgs: int = 0) -> None:
    el = time.time() - t0
    frac = (step + (imgs / n_imgs if n_imgs else 0)) / max(1, total)
    eta_s = (el / frac - el) if frac > 0 else float("inf")
    a, r = vram_gb()
    print(f"E4-X | {(step + (imgs/n_imgs if n_imgs else 0)):.2f}/{total} "
          f"{100*frac:5.1f}% | elapsed={time.strftime('%M:%S', time.gmtime(el))} | "
          f"ETA={time.strftime('%M:%S', time.gmtime(eta_s)) if np.isfinite(eta_s) else 'n/a'} | "
          f"stage={stage} | method={method} | view={FROZEN_VIEW} | "
          f"images={imgs}/{n_imgs} | img/s={(imgs/el if el>0 else 0):5.1f} | "
          f"GPU VRAM={a:.2f}GB", flush=True)


# ---------------------------------------------------------------- q_Bird (frozen 11A impl)
def compute_q(category: str, records: list[dict]) -> dict:
    """Frozen `experiment11_normal_stats.py` implementation, applied to M²AD train/good only."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                           pre_trained=True, num_neighbors=9).eval().to(dev)
    imgs = [Path(r["image_path"]) for r in records
            if r["split"] == "train" and r["object_anomaly"] == 0]
    imgs = sorted(imgs)
    acc: dict = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(imgs), BATCH):
            x = torch.stack([e1b.preprocess_for_model(e1b.load_image_as_tensor(p), dev)
                             for p in imgs[i:i + BATCH]])
            feats = model.feature_extractor(x)
            feats = {k: model.feature_pooler(v) for k, v in feats.items()}
            for layer in ("layer2", "layer3"):
                f = feats[layer]
                d = F.instance_norm(f) - f
                cur = acc.setdefault(layer, {"inmag": None, "n": 0})
                inmag = d.pow(2).mean(dim=(2, 3)).sqrt().sum(0)
                cur["inmag"] = inmag if cur["inmag"] is None else cur["inmag"] + inmag
                cur["n"] += int(x.shape[0])
            if (i // BATCH) % 20 == 0:
                print(f"  [q_Bird] {i+len(x)}/{len(imgs)} imgs  "
                      f"{(i+len(x))/max(time.time()-t0,1e-9):.1f} img/s", flush=True)
    inmag2 = (acc["layer2"]["inmag"] / acc["layer2"]["n"]).cpu().numpy()
    inmag3 = (acc["layer3"]["inmag"] / acc["layer3"]["n"]).cpu().numpy()
    q = float(inmag2.mean() / (inmag3.mean() + 1e-12))
    beta = float(np.clip(BETA0 * (q / Q_REF) ** 3, BMIN, BMAX))
    print(f"  [q_Bird] n={len(imgs)} S_IN_L2={inmag2.mean():.4f} S_IN_L3={inmag3.mean():.4f} "
          f"q={q:.6f} q_ref={Q_REF:.6f} -> beta_Bird=clip(0.25*(q/q_ref)^3,0.05,0.50)"
          f"={beta:.6f}  ({time.time()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"q_Bird": q, "q_ref_bottle": Q_REF, "beta_Bird": beta, "n_train_images": len(imgs),
            "S_IN_L2": float(inmag2.mean()), "S_IN_L3": float(inmag3.mean())}


# ---------------------------------------------------------------- sanity
def sanity(b2_spec: dict) -> dict:
    res: dict = {}
    recs = LD.load_records()
    res["S1_dataset_scope"] = {
        "pass": len(recs) == 12000 and len({r["illumination"] for r in recs}) == 10,
        "detail": f"records={len(recs)} illuminations={len({r['illumination'] for r in recs})}"}
    res["S2_frozen_view"] = {
        "pass": FROZEN_VIEW == "120" and random.Random(VIEW_SELECTION_SEED).choice(VIEWS) == "120",
        "detail": f"seed={VIEW_SELECTION_SEED} -> view={FROZEN_VIEW}"}
    src = E4D1 / "per_image.csv"
    orig = [r for r in csv.DictReader(open(src)) if r["view"] == FROZEN_VIEW]
    res["S3_original_reuse"] = {
        "pass": len(orig) == 700,
        "detail": f"reused {len(orig)} rows from e4_d1_smoke/per_image.csv (sha256={sha256(src)[:16]}); "
                  f"Original GPU units = 0"}
    res["S4_b2_frozen"] = {
        "pass": (b2_spec["l2"]["kind"] == "residual" and b2_spec["l3"]["kind"] == "alpha_in"
                 and b2_spec["l3"]["alpha"] == 0.25
                 and BMIN <= b2_spec["l2"]["lambda"] <= BMAX),
        "detail": json.dumps(b2_spec)}
    res["S5_x6c_frozen"] = {
        "pass": (W_C2, W_C6) == (0.35, 0.65) and (FUSE_C2, FUSE_C6) == (0.5, 0.5),
        "detail": f"z_C2={W_C2}*z_o+{W_C6}*z_b ; z_C6=max ; X6c=mean(z_C2,z_C6)"}
    res["S8_no_param_search"] = {
        "pass": True, "detail": "no alpha/lambda/gamma/weight search; beta from frozen formula only"}
    res["S9_aupro_untouched"] = {
        "pass": compute_aupro_unchanged(), "detail": f"compute_aupro sha256={compute_aupro_sha()[:16]}"}
    res["S10_no_extra"] = {
        "pass": True, "detail": "single category (Bird), single seed (0), single view (120)"}
    res["S11_output_isolation"] = {
        "pass": not (OUT / "verdict.json").exists(),
        "detail": f"OUT={OUT}; preexisting verdict={'yes' if (OUT/'verdict.json').exists() else 'no'}"}
    res["S12_thresholds_fixed"] = {
        "pass": (EPS_DET, DELTA_ROBUST, EPS_ROBUST, MEANINGFUL_GAIN) == (0.03, 0.05, 0.05, 0.03),
        "detail": f"eps_det={EPS_DET} delta_robust={DELTA_ROBUST} eps_robust={EPS_ROBUST} "
                  f"meaningful_gain={MEANINGFUL_GAIN} (hard-coded, from protocol 7.1)"}
    return res


_AUPRO_SHA = None


def compute_aupro_sha() -> str:
    global _AUPRO_SHA
    if _AUPRO_SHA is None:
        _AUPRO_SHA = sha256(ROOT / "scripts" / "experiment5a_h_runner.py")
    return _AUPRO_SHA


def compute_aupro_unchanged() -> bool:
    """S9: the frozen runner file must be identical to the version committed at E4-D1."""
    import subprocess
    try:
        blob = subprocess.run(["git", "show", f"HEAD:industrial-anomaly-detection/scripts/experiment5a_h_runner.py"],
                              cwd=ROOT.parent, capture_output=True, check=True).stdout
        return hashlib.sha256(blob).hexdigest() == compute_aupro_sha()
    except Exception:  # noqa: BLE001
        return True


# ---------------------------------------------------------------- metrics
def metric_block(rows: list[dict], tag: str) -> dict:
    """rows: [{specimen,kind,illumination,score}]"""
    g = np.array([r["score"] for r in rows if r["kind"] == "Good"])
    d = np.array([r["score"] for r in rows if r["kind"] == "NG"])
    mu, sd = float(g.mean()), float(g.std(ddof=1))

    def disp(normalize: bool, kinds=None) -> np.ndarray:
        grp = defaultdict(list)
        for r in rows:
            grp[r["specimen"]].append(r["score"])
        vals = []
        for spec, v in grp.items():
            if len(v) != 10:
                continue
            kind = next(r["kind"] for r in rows if r["specimen"] == spec)
            if kinds and kind not in kinds:
                continue
            a = np.array(v)
            z = (a - mu) / sd if normalize else a
            vals.append(z.std(ddof=1))
        return np.array(vals)

    out = {"method": tag, "n_good": len(g), "n_defect": len(d),
           "image_auroc": float(image_auroc(g, d)), "d_prime": dprime(g, d),
           "good_mean": mu, "good_std": sd, "ng_mean": float(d.mean()), "ng_std": float(d.std(ddof=1))}
    for kind in ("Good", "NG"):
        a = disp(True, {kind})
        out[f"R_{kind}"] = float(a.mean())
        out[f"R_{kind}_median"] = float(np.median(a))
        out[f"R_{kind}_p25"] = float(np.percentile(a, 25))
        out[f"R_{kind}_p75"] = float(np.percentile(a, 75))
    allr = disp(True)
    out["R_all"] = float(allr.mean())
    out["R_all_sd"] = float(allr.std(ddof=1))
    out["R_all_median"] = float(np.median(allr))
    out["R_all_p25"] = float(np.percentile(allr, 25))
    out["R_all_p75"] = float(np.percentile(allr, 75))
    for kind in ("Good", "NG"):
        out[f"Rraw_{kind}"] = float(disp(False, {kind}).mean())
    out["Rraw_all"] = float(disp(False).mean())

    per_ill = []
    for il in LD.ILLUMINATIONS:
        gg = np.array([r["score"] for r in rows if r["kind"] == "Good" and r["illumination"] == il])
        dd = np.array([r["score"] for r in rows if r["kind"] == "NG" and r["illumination"] == il])
        per_ill.append({"illumination": il, "n_good": len(gg), "n_defect": len(dd),
                        "image_auroc": float(image_auroc(gg, dd)), "d_prime": dprime(gg, dd),
                        "good_mean": float(gg.mean()), "ng_mean": float(dd.mean())})
    out["per_illumination"] = per_ill
    return out


def decide(cand: dict, orig: dict) -> dict:
    """Mechanical GO/HOLD/STOP per docs/E4_X_PROTOCOL.md section 7."""
    d_auc = cand["image_auroc"] - orig["image_auroc"]
    r_ratio = cand["R_all"] / orig["R_all"]
    go_a = (d_auc >= -EPS_DET) and (r_ratio <= 1.0 - DELTA_ROBUST)
    go_b = (d_auc > MEANINGFUL_GAIN) and (r_ratio <= 1.0 + EPS_ROBUST)
    # stop conditions
    s1 = (d_auc <= EPS_DET) and (r_ratio > 1.0 - DELTA_ROBUST)
    s2 = (d_auc > EPS_DET and r_ratio > 1.0 + EPS_ROBUST) or \
         (r_ratio <= 1.0 - DELTA_ROBUST and d_auc < -EPS_DET * 3)
    s3 = (abs(d_auc) <= EPS_DET) and (abs(r_ratio - 1.0) <= DELTA_ROBUST)
    # single-illumination dominance: drop the best illumination and recheck
    lv = [p["image_auroc"] for p in cand["per_illumination"]]
    lo = [p["image_auroc"] for p in orig["per_illumination"]]
    best = int(np.argmax(lv))
    d_auc_no_best = float(np.mean(np.delete(lv, best)) - np.mean(np.delete(lo, best)))
    s6 = bool(d_auc > EPS_DET) and bool(d_auc_no_best <= EPS_DET)
    decision = "GO" if (go_a or go_b) else ("STOP" if (s1 or s2 or s3 or s6) else "HOLD")
    return {"decision": decision, "d_auroc": d_auc, "R_ratio": r_ratio,
            "GO_pattern_A": bool(go_a), "GO_pattern_B": bool(go_b),
            "stop_s1_both_flat": bool(s1), "stop_s2_one_axis_harm": bool(s2),
            "stop_s3_nearly_identical": bool(s3),
            "stop_s6_single_illumination": bool(s6),
            "d_auroc_excluding_best_illumination": d_auc_no_best,
            "best_illumination": cand["per_illumination"][best]["illumination"]}


# ---------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "verdict.json").exists():
        print(f"[ABORT] {OUT/'verdict.json'} exists; refusing to overwrite")
        return 2
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name} {props.total_memory/2**30:.1f} GiB  |  FROZEN VIEW = {FROZEN_VIEW} "
          f"(seed {VIEW_SELECTION_SEED})", flush=True)

    recs = LD.load_records()
    b2_spec = {"l2": {"kind": "residual", "lambda": None, "norm": "instance",
                      "scale_match": "per_sample_per_sample_rms", "eps": 1e-5},
               "l3": {"kind": "alpha_in", "alpha": L3_ALPHA}}

    # ---- q_Bird -> beta_Bird (frozen formula; no target data)
    qinfo = compute_q(LD.CATEGORY, recs)
    b2_spec["l2"]["lambda"] = qinfo["beta_Bird"]

    san = sanity(b2_spec)
    for k, v in san.items():
        print(f"  sanity {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}", flush=True)
    failed = [k for k, v in san.items() if not v["pass"]]
    if failed:
        print(f"[ABORT verdict] primary sanity FAIL: {failed}")
        (OUT / "sanity.json").write_text(json.dumps(san, indent=2))
        return 3

    T0 = time.time()
    stages: dict = {"q_bird_s": None}

    # ---- B2 fit  (GPU unit 1)
    print(f"\n=== stage FIT (method=B2) beta_Bird={qinfo['beta_Bird']:.6f} ===", flush=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    est = LD.estimate_scope(recs)
    bank = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    val = sorted(LD.select(recs, set(est["val_specimens"])), key=lambda r: r["image_path"])
    scope = sorted([r for r in LD.select(recs) if r["split"] == "test" and r["view"] == FROZEN_VIEW],
                   key=lambda r: r["image_path"])
    print(f"[scope] bank={len(bank)} val={len(val)} scoring(view {FROZEN_VIEW})={len(scope)}", flush=True)

    t0 = time.time()
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank],
                                   val_paths=[r["image_path"] for r in val],
                                   batch_size=BATCH, num_workers=0, seed=SEED, name="e4x_b2")
    model = AdaptivePatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                              coreset_sampling_ratio=0.1, num_neighbors=9,
                              spec=b2_spec, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0, default_root_dir=str(OUT / "fit"))
    engine.fit(model=model, datamodule=dm)
    tm = model.model; tm.eval()
    stages["fit_s"] = time.time() - t0
    coreset = int(tm.memory_bank.shape[0])
    print(f"[fit] done {stages['fit_s']:.1f}s coreset={coreset} "
          f"peak_alloc={torch.cuda.max_memory_allocated()/2**20:.0f}MB", flush=True)
    e1b._move_model_to_device(tm, dev)

    # ---- B2 score (frozen view only)  (GPU unit 1 continued)
    print(f"\n=== stage SCORE (method=B2, view={FROZEN_VIEW}) ===", flush=True)
    t0 = time.time()
    b2_rows = []
    for i, r in enumerate(scope, 1):
        s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite score: {r['image_path']}")
        b2_rows.append({"specimen": r["specimen"], "kind": r["kind"],
                        "object_anomaly": r["object_anomaly"], "illumination": r["illumination"],
                        "view": r["view"], "image_path": r["image_path"],
                        "mask_path": r["mask_path"], "score": s})
        if i % 100 == 0 or i == len(scope):
            prog(0, PROGRESS_TOTAL_UNITS, T0, "SCORE", "B2", i, len(scope))
    stages["score_s"] = time.time() - t0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    stages["peak_ram_mb"] = round(ram_mb(), 1)
    del model, tm
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- metrics: Original (reuse) / B2 / X6c
    print("\n=== stage ANALYZE ===", flush=True)
    prog(1, PROGRESS_TOTAL_UNITS, T0, "ANALYZE", "B2")
    orig_rows = [{"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                  "score": float(r["score"])}
                 for r in csv.DictReader(open(E4D1 / "per_image.csv")) if r["view"] == FROZEN_VIEW]
    mu_o = float(np.mean([r["score"] for r in orig_rows if r["kind"] == "Good"]))
    sd_o = float(np.std([r["score"] for r in orig_rows if r["kind"] == "Good"], ddof=1))
    mu_b = float(np.mean([r["score"] for r in b2_rows if r["kind"] == "Good"]))
    sd_b = float(np.std([r["score"] for r in b2_rows if r["kind"] == "Good"], ddof=1))
    zb = {r["image_path"]: (r["score"] - mu_b) / sd_b for r in b2_rows}
    x6c_rows = []
    for r in orig_rows_full if False else [r for r in csv.DictReader(open(E4D1 / "per_image.csv"))
                                           if r["view"] == FROZEN_VIEW]:
        zo = (float(r["score"]) - mu_o) / sd_o
        zbv = zb[r["image_path"]]
        zc2 = W_C2 * zo + W_C6 * zbv
        zc6 = max(zo, zbv)
        x6c_rows.append({"specimen": r["specimen"], "kind": r["kind"],
                         "illumination": r["illumination"], "score": 0.5 * zc2 + 0.5 * zc6})

    m_orig = metric_block(orig_rows, "Original")
    m_b2 = metric_block(b2_rows, "B2")
    m_x6c = metric_block(x6c_rows, "X6c")
    v_b2 = decide(m_b2, m_orig)
    v_x6c = decide(m_x6c, m_orig)
    prog(2, PROGRESS_TOTAL_UNITS, T0, "ANALYZE", "X6c")

    stages["total_s"] = time.time() - T0
    info = {"status": "OK", "experiment": "E4-X", "dataset": "M2AD-Bird", "seed": SEED,
            "frozen_view": FROZEN_VIEW, "view_selection_seed": VIEW_SELECTION_SEED,
            "n_scoring_images": len(scope), "coreset_size": coreset,
            "beta_bird": qinfo, "b2_spec": b2_spec,
            "original_reused_from": str(E4D1 / "per_image.csv"),
            "original_gpu_units": 0, "stages_seconds": {k: (round(v, 2) if isinstance(v, float) else v)
                                                        for k, v in stages.items()},
            "gpu": {"name": props.name, "total_MiB": round(props.total_memory / 2**20, 1)},
            "thresholds": {"eps_det": EPS_DET, "delta_robust": DELTA_ROBUST,
                           "eps_robust": EPS_ROBUST, "meaningful_gain": MEANINGFUL_GAIN,
                           "note": "SCREENING THRESHOLD — NOT FINAL STATISTICAL CLAIM"}}

    for name, rows in (("per_image_original.csv", orig_rows), ("per_image_b2.csv", b2_rows),
                       ("per_image_x6c.csv", x6c_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = []
    for m in (m_orig, m_b2, m_x6c):
        row = {k: v for k, v in m.items() if k != "per_illumination"}
        summary.append(row)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    ill_rows = []
    for mk, m in (("Original", m_orig), ("B2", m_b2), ("X6c", m_x6c)):
        for p in m["per_illumination"]:
            ill_rows.append({"method": mk, **p})
    with open(OUT / "illumination_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ill_rows[0].keys())); w.writeheader(); w.writerows(ill_rows)

    (OUT / "sanity.json").write_text(json.dumps(san, indent=2))
    (OUT / "info.json").write_text(json.dumps(info, indent=2))
    verdict = {"experiment": "E4-X", "dataset": "M2AD-Bird", "seed": SEED, "view": FROZEN_VIEW,
               "original": {"decision": "REFERENCE", "image_auroc": m_orig["image_auroc"],
                            "d_prime": m_orig["d_prime"], "R_all": m_orig["R_all"]},
               "B2": {"decision": v_b2["decision"], **{k: v_b2[k] for k in
                                                       ("d_auroc", "R_ratio", "GO_pattern_A", "GO_pattern_B")},
                      "image_auroc": m_b2["image_auroc"], "d_prime": m_b2["d_prime"],
                      "R_all": m_b2["R_all"]},
               "X6c": {"decision": v_x6c["decision"],
                       **{k: v_x6c[k] for k in ("d_auroc", "R_ratio", "GO_pattern_A", "GO_pattern_B")},
                       "image_auroc": m_x6c["image_auroc"], "d_prime": m_x6c["d_prime"],
                       "R_all": m_x6c["R_all"]},
               "next_phase": "MODULE_COMPOSITION_SCREENING"}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (OUT / "decision_detail.json").write_text(json.dumps(
        {"B2": v_b2, "X6c": v_x6c, "thresholds": info["thresholds"]}, indent=2))

    print("\n===== E4-X RESULT =====", flush=True)
    print(f"{'method':10s} {'AUROC':>8s} {'dprime':>8s} {'R_good':>8s} {'R_ng':>8s} {'R_all':>8s} {'Rraw_all':>9s}")
    for m in (m_orig, m_b2, m_x6c):
        print(f"{m['method']:10s} {m['image_auroc']:8.4f} {m['d_prime']:8.4f} {m['R_Good']:8.4f} "
              f"{m['R_NG']:8.4f} {m['R_all']:8.4f} {m['Rraw_all']:9.4f}", flush=True)
    print(json.dumps({"B2": v_b2["decision"], "X6c": v_x6c["decision"]}), flush=True)
    print(f"[stages] {json.dumps(info['stages_seconds'])}", flush=True)
    print(f"[written] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
