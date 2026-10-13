#!/usr/bin/env python
"""E12-B - Lightweight Illumination-Adaptive Multi-Level Fusion Screening.

Frozen protocol: results/e12b/e12b_protocol.json

Candidates (frozen DINOv2-B block9 + final layer patch tokens, 1-NN-max scoring):
  S0 = final-layer only                     (REUSED baseline, no recompute)
  S1 = P4-A1 fixed concat(l11, l9)          (REUSED historical result, no rerun)
  S2 = global scalar gate                   (1 trainable param)
  S3 = channel gate                         (768 trainable params)
  S4 = sample-adaptive two-scale SE gate    (73874 trainable params)

Normal-only training: for each bank normal image, mild illumination augmentation
(brightness/contrast/gamma +/-10%) -> 6 variants. Loss = L_consistency + 0.1*L_identity,
where L_consistency = 1 - cos(F_fused(x), F_fused(T_illum(x))) and
L_identity = ||F_fused(x) - F_ref(x)||^2 with F_ref = normalize(0.5*n(f9)+0.5*n(ffinal)).

DINO forward is done ONCE and cached in RAM (no backward through DINO). Features are
NOT persisted to disk (too large); resume skips completed cells and re-extracts per category.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import (raw_test_map, official_metrics, route,  # noqa: E402
                           nn_image_scores)
from e11_p4_screen import _prep  # noqa: E402  (bit-exact preprocessing for S0/S1 reproduction)

# ------------------------------------------------------------------ DINOv2-B loader (offline)
# torch.hub.load("facebookresearch/dinov2", ...) hangs on unreachable github.com at startup,
# so load the cached repo + cached checkpoint directly (no network, identical weights/arch).
_HUB_REPO = "/root/.cache/torch/hub/facebookresearch_dinov2_main"
_model_cache = {}


def load_dinov2_local(dev):
    if "vitb" in _model_cache:
        return _model_cache["vitb"]
    if _HUB_REPO not in sys.path:
        sys.path.insert(0, _HUB_REPO)
    from dinov2.hub.backbones import dinov2_vitb14  # noqa: E402
    model = dinov2_vitb14(pretrained=True)   # loads cached dinov2_vitb14_pretrain.pth locally
    model.eval().to(dev)
    _model_cache["vitb"] = model
    return model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_PATCH = 1024
D = 768
ALPHA = 0.1
BATCH_IMGS = 8
MAX_STEPS = 500
PATIENCE_STEPS = 20
EVAL_EVERY = 5
AUG_LEVELS = [0.9, 1.1]
AUG_FAMILIES = ["brightness", "contrast", "gamma"]
LR = {"S2": 0.01, "S3": 0.01, "S4": 0.001}
TRAIN_N = 240
VAL_N = 60
SMOKE_TRAIN_N = 20
SMOKE_STEPS = 20
TIMEOUT_PER_CANDIDATE_S = 20 * 60

OUTDIR = ROOT / "results" / "e12b"
PARTIAL = OUTDIR / "e12b_partial.json"
TRAIN_CSV = OUTDIR / "e12b_training.csv"
SCREEN_CSV = OUTDIR / "e12b_screening.csv"
P4_CSV = ROOT / "results" / "e11" / "p4_module_screening.csv"


# ------------------------------------------------------------------ photometric augmentation
def _aug(t: torch.Tensor, family: str, level: float) -> torch.Tensor:
    if family == "brightness":
        return (t * level).clamp(0.0, 1.0)
    if family == "contrast":
        return ((t - 0.5) * level + 0.5).clamp(0.0, 1.0)
    if family == "gamma":
        return t.clamp(min=1e-6).pow(1.0 / level).clamp(0.0, 1.0)
    raise ValueError(family)


AUG_LIST = [(f, lv) for f in AUG_FAMILIES for lv in AUG_LEVELS]  # 6 variants


# ------------------------------------------------------------------ feature extraction
def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def extract_l9_l11(model, path: str, dev):
    """-> (f9, ffinal) each (1024, 768) float32 numpy, bit-exact with E11-P4 l9/l11."""
    img = Image.open(path).convert("RGB")
    with torch.inference_mode():
        t = _prep(img, 448, model, dev)
        layers = model.get_intermediate_layers(t.unsqueeze(0), n=5, norm=True)
        f9 = layers[2][0].float().cpu().numpy()   # block9
        ff = layers[4][0].float().cpu().numpy()   # block11 (final)
    return f9, ff


def extract_l9_l11_aug(model, path: str, dev, family: str, level: float):
    """-> (f9_aug, ff_aug) for one augmented image."""
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                    T.ToTensor()])
    t = tf(img)
    t = _aug(t, family, level)
    t = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(t)
    ps = model.patch_size
    t = t[:, :t.shape[1] - t.shape[1] % ps, :t.shape[2] - t.shape[2] % ps].to(dev)
    with torch.inference_mode():
        layers = model.get_intermediate_layers(t.unsqueeze(0), n=5, norm=True)
        f9 = layers[2][0].float().cpu().numpy()
        ff = layers[4][0].float().cpu().numpy()
    return f9, ff


def extract_originals(model, recs_r, dev, log_every=200):
    """-> (f9, ff) each (N, 1024, 768) float32 numpy."""
    f9s, ffs = [], []
    for i, r in enumerate(recs_r, 1):
        f9, ff = extract_l9_l11(model, r["image_path"], dev)
        f9s.append(f9)
        ffs.append(ff)
        if i % log_every == 0:
            print(f"    extract orig {i}/{len(recs_r)}", flush=True)
    return np.stack(f9s, axis=0), np.stack(ffs, axis=0)


def extract_augmented(model, recs_r, dev, log_every=300):
    """-> (f9, ff) each (N, 6, 1024, 768) float32 numpy (6 = AUG_LIST variants)."""
    N = len(recs_r)
    K = len(AUG_LIST)
    f9s = np.empty((N, K, N_PATCH, D), dtype="float32")
    ffs = np.empty((N, K, N_PATCH, D), dtype="float32")
    for i, r in enumerate(recs_r):
        for k, (fam, lv) in enumerate(AUG_LIST):
            f9, ff = extract_l9_l11_aug(model, r["image_path"], dev, fam, lv)
            f9s[i, k] = f9
            ffs[i, k] = ff
        if (i + 1) % log_every == 0:
            print(f"    extract aug {i+1}/{N}", flush=True)
    return f9s, ffs


# ------------------------------------------------------------------ gates
class ScalarGate(nn.Module):
    """S2: single scalar sigmoid gate."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def fuse(self, raw9, rawf):
        n9, nf = _norm(raw9), _norm(rawf)
        lam = torch.sigmoid(self.theta)
        return _norm(lam * n9 + (1.0 - lam) * nf)


class ChannelGate(nn.Module):
    """S3: per-channel sigmoid gate."""
    def __init__(self, dim=D):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim))

    def fuse(self, raw9, rawf):
        n9, nf = _norm(raw9), _norm(rawf)
        lam = torch.sigmoid(self.theta)
        return _norm(lam * n9 + (1.0 - lam) * nf)


class SEGate(nn.Module):
    """S4: sample-adaptive two-scale SE gate (global-pooled block9+final -> MLP -> [w9, wfinal])."""
    def __init__(self, dim=D, hidden=D // 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()

    def fuse(self, raw9, rawf):
        M = raw9.shape[0]
        n_img = M // N_PATCH
        n9, nf = _norm(raw9), _norm(rawf)
        g9 = raw9.reshape(n_img, N_PATCH, -1).mean(dim=1)
        gf = rawf.reshape(n_img, N_PATCH, -1).mean(dim=1)
        w = torch.softmax(self.mlp(torch.cat([g9, gf], dim=1)), dim=1)  # (n_img, 2)
        w9 = w[:, 0:1].repeat_interleave(N_PATCH, dim=0)
        wf = w[:, 1:2].repeat_interleave(N_PATCH, dim=0)
        return _norm(w9 * n9 + wf * nf)


def make_gate(candidate: str, dev) -> nn.Module:
    g = {"S2": ScalarGate, "S3": ChannelGate, "S4": SEGate}[candidate]()
    return g.to(dev)


def n_params(gate) -> int:
    return sum(p.numel() for p in gate.parameters())


def gate_weights(gate, candidate: str, raw9, rawf, dev):
    """Return a human-readable summary of learned weights."""
    with torch.inference_mode():
        if candidate == "S2":
            lam = float(torch.sigmoid(gate.theta).item())
            return {"lambda": lam, "w_final": 1.0 - lam, "w_block9": lam}
        if candidate == "S3":
            lam = torch.sigmoid(gate.theta).detach().cpu().numpy()
            return {"lambda_mean": float(lam.mean()), "lambda_min": float(lam.min()),
                    "lambda_max": float(lam.max()),
                    "frac_gt_half": float((lam > 0.5).mean())}
        # S4: mean w9 / wfinal over provided images
        t9 = torch.from_numpy(raw9).to(dev)
        tf_ = torch.from_numpy(rawf).to(dev)
        n_img = raw9.shape[0] // N_PATCH
        g9 = t9.reshape(n_img, N_PATCH, -1).mean(dim=1)
        gf = tf_.reshape(n_img, N_PATCH, -1).mean(dim=1)
        w = torch.softmax(gate.mlp(torch.cat([g9, gf], dim=1)), dim=1).detach().cpu().numpy()
        return {"w_block9_mean": float(w[:, 0].mean()), "w_final_mean": float(w[:, 1].mean()),
                "w_block9_std": float(w[:, 0].std()), "w_final_std": float(w[:, 1].std())}


# ------------------------------------------------------------------ training
def collapse_stats(gate, f9, ff, dev, n_sample=50):
    """-> {mean_norm, std_fused} computed on a sample (M*1024, 768)."""
    n = min(n_sample, f9.shape[0])
    t9 = torch.from_numpy(f9[:n].reshape(n * N_PATCH, D)).to(dev)
    tf_ = torch.from_numpy(ff[:n].reshape(n * N_PATCH, D)).to(dev)
    with torch.inference_mode():
        fused = gate.fuse(t9, tf_)
        norm = fused.norm(dim=1).mean().item()
        std = fused.std(dim=0).mean().item()
    return {"mean_norm": float(norm), "std_fused": float(std)}


def consistency_loss(gate, f9, ff, aug_f9, aug_ff, dev, alpha=ALPHA):
    """-> (L_cons, L_identity, total) on the full given set (chunked)."""
    N = f9.shape[0]
    K = aug_f9.shape[1]
    cons, ident = 0.0, 0.0
    cnt = 0
    with torch.inference_mode():
        for i in range(N):
            r9 = torch.from_numpy(f9[i].reshape(N_PATCH, D)).to(dev)
            rf = torch.from_numpy(ff[i].reshape(N_PATCH, D)).to(dev)
            F_orig = gate.fuse(r9, rf)
            n9, nf = _norm(r9), _norm(rf)
            F_ref = _norm(0.5 * n9 + 0.5 * nf)
            ident += float(((F_orig - F_ref) ** 2).sum(dim=1).mean().item())
            for k in range(K):
                a9 = torch.from_numpy(aug_f9[i, k].reshape(N_PATCH, D)).to(dev)
                af = torch.from_numpy(aug_ff[i, k].reshape(N_PATCH, D)).to(dev)
                F_aug = gate.fuse(a9, af)
                cons += float((1.0 - (F_orig * F_aug).sum(dim=1)).mean().item())
                cnt += 1
    cons /= max(1, cnt)
    ident /= max(1, N)
    return {"cons": cons, "identity": ident, "total": cons + alpha * ident}


def train_gate(gate, train_f9, train_ff, aug_f9, aug_ff,
               val_f9, val_ff, val_aug_f9, val_aug_ff, dev, lr, seed, smoke):
    """Returns training report dict. Gate is mutated in place."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    N = train_f9.shape[0]
    K = aug_f9.shape[1]
    max_steps = SMOKE_STEPS if smoke else MAX_STEPS

    before = collapse_stats(gate, train_f9, train_ff, dev)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    report = {"steps_used": 0, "best_val_cons": float("nan"), "final_cons": float("nan"),
              "grad_norm": 0.0, "grad_present": False, "loss_finite": True}

    for step in range(max_steps):
        idx = rng.integers(0, N, size=BATCH_IMGS)
        aidx = rng.integers(0, K, size=BATCH_IMGS)
        r9 = torch.from_numpy(train_f9[idx].reshape(BATCH_IMGS * N_PATCH, D)).to(dev)
        rf = torch.from_numpy(train_ff[idx].reshape(BATCH_IMGS * N_PATCH, D)).to(dev)
        a9 = torch.from_numpy(aug_f9[idx, aidx].reshape(BATCH_IMGS * N_PATCH, D)).to(dev)
        af = torch.from_numpy(aug_ff[idx, aidx].reshape(BATCH_IMGS * N_PATCH, D)).to(dev)

        opt.zero_grad()
        F_orig = gate.fuse(r9, rf)
        F_aug = gate.fuse(a9, af)
        n9, nf = _norm(r9), _norm(rf)
        F_ref = _norm(0.5 * n9 + 0.5 * nf)
        L_cons = 1.0 - (F_orig * F_aug).sum(dim=1).mean()
        L_id = ((F_orig - F_ref) ** 2).sum(dim=1).mean()
        loss = L_cons + ALPHA * L_id
        if not torch.isfinite(loss):
            report["loss_finite"] = False
            break
        loss.backward()
        # grad check
        grads = [p.grad for p in gate.parameters() if p.grad is not None]
        report["grad_present"] = len(grads) > 0 and all(g is not None for g in grads)
        gn = float(sum((g ** 2).sum().item() for g in grads) ** 0.5) if grads else 0.0
        report["grad_norm"] = max(report["grad_norm"], gn)
        opt.step()

        if not smoke and (step + 1) % EVAL_EVERY == 0:
            v = consistency_loss(gate, val_f9, val_ff, val_aug_f9, val_aug_ff, dev)
            if v["cons"] < best_val - 1e-6:
                best_val = v["cons"]
                no_improve = 0
                best_state = {k: p.detach().clone() for k, p in gate.named_parameters()}
            else:
                no_improve += 1
            if no_improve * EVAL_EVERY >= PATIENCE_STEPS:
                break
        report["steps_used"] = step + 1

    if best_state is not None:
        gate.load_state_dict(best_state)

    after = collapse_stats(gate, train_f9, train_ff, dev)
    report.update({"before": before, "after": after,
                   "final_cons": float(consistency_loss(gate, train_f9, train_ff,
                                                        aug_f9, aug_ff, dev)["cons"]),
                   "best_val_cons": best_val if best_val < float("inf") else float("nan")})
    return report


# ------------------------------------------------------------------ scoring helpers
def fuse_to_numpy(gate, f9, ff, dev, chunk=100):
    """-> float32 numpy (N*1024, 768), L2-normalised fused features."""
    N = f9.shape[0]
    out = []
    with torch.inference_mode():
        for i in range(0, N, chunk):
            t9 = torch.from_numpy(f9[i:i + chunk].reshape(-1, D)).to(dev)
            tf_ = torch.from_numpy(ff[i:i + chunk].reshape(-1, D)).to(dev)
            out.append(gate.fuse(t9, tf_).cpu().numpy().astype("float32"))
    return np.concatenate(out, axis=0)


def norm_concat(f9, ff):
    x = np.concatenate([ff, f9], axis=-1)          # concat(l11, l9) = P4-A1 order
    x = x.reshape(-1, 2 * D).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def norm_single(f):
    x = f.reshape(-1, D).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ reference numbers
def load_p4_ref() -> dict:
    out = {}
    with open(P4_CSV) as f:
        for row in csv.DictReader(f):
            if row["module"] in ("baseline", "A1"):
                out.setdefault(row["category"], {})[row["module"]] = {
                    "auroc": float(row["auroc"]), "ap": float(row["ap"])}
    assert set(out) == {"Bird", "Car", "Cube"}, f"P4 CSV categories: {set(out)}"
    return out


# ------------------------------------------------------------------ run one category
def run_category(category, dev, smoke, ref) -> dict:
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]

    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    model = load_dinov2_local(dev)
    dino_grad_none = all(p.grad is None for p in model.parameters())

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)

    t0 = time.time()
    bank_f9, bank_ff = extract_originals(model, bank_r, dev)
    test_f9, test_ff = extract_originals(model, test_r, dev)
    extract_orig_s = time.time() - t0
    print(f"  extracted orig bank={bank_f9.shape} test={test_f9.shape} "
          f"in {extract_orig_s:.1f}s", flush=True)

    # S0 / S1 reproduction (always computed as an internal gate, not a re-run)
    bank_s0 = norm_single(bank_ff)
    test_s0 = norm_single(test_ff)
    scores_s0 = nn_image_scores(bank_s0, test_s0, dev)
    s0_met = official_metrics(_rows(test, scores_s0, rawmap, rel))
    bank_s1 = norm_concat(bank_f9, bank_ff)
    test_s1 = norm_concat(test_f9, test_ff)
    scores_s1 = nn_image_scores(bank_s1, test_s1, dev)
    s1_met = official_metrics(_rows(test, scores_s1, rawmap, rel))
    t1 = time.time()

    print(f"  S0 reprod: auroc={s0_met['auroc']:.5f} (ref {ref[category]['baseline']['auroc']:.5f}, "
          f"d={s0_met['auroc']-ref[category]['baseline']['auroc']:+.6f})", flush=True)
    print(f"  S1 reprod: auroc={s1_met['auroc']:.5f} (ref {ref[category]['A1']['auroc']:.5f}, "
          f"d={s1_met['auroc']-ref[category]['A1']['auroc']:+.6f})", flush=True)

    # augmented features for training (train/val split of bank normals)
    if smoke:
        train_n = SMOKE_TRAIN_N
        train_f9, train_ff = bank_f9[:train_n], bank_ff[:train_n]
        aug_f9, aug_ff = extract_augmented(model, bank_r[:train_n], dev)
        val_f9 = val_ff = val_aug_f9 = val_aug_ff = None
    else:
        train_f9, train_ff = bank_f9[:TRAIN_N], bank_ff[:TRAIN_N]
        val_f9, val_ff = bank_f9[TRAIN_N:TRAIN_N + VAL_N], bank_ff[TRAIN_N:TRAIN_N + VAL_N]
        aug_f9, aug_ff = extract_augmented(model, bank_r[:TRAIN_N + VAL_N], dev)
        val_aug_f9, val_aug_ff = aug_f9[TRAIN_N:], aug_ff[TRAIN_N:]
        aug_f9, aug_ff = aug_f9[:TRAIN_N], aug_ff[:TRAIN_N]
    extract_aug_s = time.time() - t1
    print(f"  extracted aug in {extract_aug_s:.1f}s", flush=True)

    out = {"category": category, "smoke": smoke, "dino_grad_none": bool(dino_grad_none),
           "extract_orig_s": round(extract_orig_s, 1),
           "extract_aug_s": round(extract_aug_s, 1),
           "s0_reprod": {"auroc": s0_met["auroc"], "ap": s0_met["ap"]},
           "s1_reprod": {"auroc": s1_met["auroc"], "ap": s1_met["ap"]},
           "candidates": {}}

    for cand in ("S2", "S3", "S4"):
        tc = time.time()
        torch.manual_seed(0)
        gate = make_gate(cand, dev)
        rep = train_gate(gate, train_f9, train_ff, aug_f9, aug_ff,
                         val_f9, val_ff, val_aug_f9, val_aug_ff, dev, LR[cand], seed=0,
                         smoke=smoke)
        bank_fused = fuse_to_numpy(gate, bank_f9, bank_ff, dev)
        test_fused = fuse_to_numpy(gate, test_f9, test_ff, dev)
        scores = nn_image_scores(bank_fused, test_fused, dev)
        met = official_metrics(_rows(test, scores, rawmap, rel))
        wf9 = (val_f9 if val_f9 is not None else train_f9).reshape(-1, D)
        wff = (val_ff if val_ff is not None else train_ff).reshape(-1, D)
        w = gate_weights(gate, cand, wf9, wff, dev)
        out["candidates"][cand] = {
            "metrics": met, "trainable_params": n_params(gate),
            "gate_weights": w, "training": rep,
            "score_s": round(time.time() - tc, 1)}
        print(f"  {cand}: auroc={met['auroc']:.5f} ap={met['ap']:.5f} "
              f"steps={rep['steps_used']} params={n_params(gate)} "
              f"cons_final={rep['final_cons']:.5f} weights={w}", flush=True)
        del gate, bank_fused, test_fused
        torch.cuda.empty_cache()

    del model, bank_f9, bank_ff, test_f9, test_ff, aug_f9, aug_ff
    torch.cuda.empty_cache()
    return out


def _rows(test, scores, rawmap, rel):
    rows = []
    for r, s in zip(test, scores):
        m = rawmap[rel(r)]
        rows.append({"specimen": r["specimen"], "kind": r["kind"],
                     "illumination": r["illumination"], "view": "120", "score": float(s),
                     "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})
    return rows


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ref = load_p4_ref()

    print(f"[E12-B] cats={cats} smoke={args.smoke} | disk {E8.free_gb():.1f} GB | "
          f"GPU {E8.gpu_util()}", flush=True)
    print(f"[E12-B] S0 reuse (P4 baseline): " + " ".join(
        f"{c}={ref[c]['baseline']['auroc']:.5f}" for c in cats), flush=True)
    print(f"[E12-B] S1 reuse (P4 A1):      " + " ".join(
        f"{c}={ref[c]['A1']['auroc']:.5f}" for c in cats), flush=True)

    partial = {"experiment": "E12-B", "cells": {}, "ref": ref, "sanity": {}}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    last_out = None
    for ci, cat in enumerate(cats):
        learned = ["S2", "S3", "S4"]
        done_keys = {f"{cat}|{c}" for c in learned} & set(partial.get("cells", {}).keys())
        if not args.smoke and done_keys == set(f"{cat}|{c}" for c in learned):
            print(f"[E12-B] SKIP {cat} (all learned cells done)", flush=True)
            continue

        print(f"\n[E12-B] RUN {cat}", flush=True)
        try:
            out = run_category(cat, dev, args.smoke, ref)
            last_out = out
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[E12-B] {cat} FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.smoke:
                return 1
            continue

        if not args.smoke:
            # S0/S1 reproduction gate
            d0 = out["s0_reprod"]["auroc"] - ref[cat]["baseline"]["auroc"]
            d1 = out["s1_reprod"]["auroc"] - ref[cat]["A1"]["auroc"]
            print(f"[E12-B] {cat} reprod gate: S0 d={d0:+.6f} S1 d={d1:+.6f}", flush=True)
            if abs(d0) > 1e-4 or abs(d1) > 1e-4:
                print(f"[E12-B] FATAL: {cat} S0/S1 did not reproduce; stopping", flush=True)
                return 2
            partial["sanity"][cat] = {"s0_reprod": out["s0_reprod"], "s1_reprod": out["s1_reprod"]}
            for c in learned:
                cell = out["candidates"][c]
                partial["cells"][f"{cat}|{c}"] = {
                    "auroc": cell["metrics"]["auroc"], "ap": cell["metrics"]["ap"],
                    "trainable_params": cell["trainable_params"],
                    "gate_weights": cell["gate_weights"],
                    "steps_used": cell["training"]["steps_used"],
                    "best_val_cons": cell["training"]["best_val_cons"],
                    "final_cons": cell["training"]["final_cons"],
                    "std_fused_before": cell["training"]["before"]["std_fused"],
                    "std_fused_after": cell["training"]["after"]["std_fused"],
                    "loss_finite": cell["training"]["loss_finite"]}
            tmp = PARTIAL.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(partial, indent=2, default=float))
            tmp.replace(PARTIAL)

        el = time.time() - t_start
        done = len(partial.get("cells", {}))
        total = len(cats) * 3
        per = el / max(1, ci + 1)
        eta = per * (len(cats) - (ci + 1))
        print(f"[E12-B] PROGRESS {done}/{total} cells | elapsed={el/60:.1f}m | "
              f"ETA={eta/60:.1f}m | disk {E8.free_gb():.1f} GB", flush=True)

    if args.smoke:
        print("\n[E12-B] SMOKE CHECK SUMMARY", flush=True)
        ok = True
        o = last_out
        checks = {}
        if o is None:
            checks["0_run"] = False
            ok = False
        else:
            checks["0_run"] = True
            cands = o["candidates"]
            checks["1_loss_finite"] = all(cands[c]["training"]["loss_finite"] for c in ("S2", "S3", "S4"))
            checks["2_gate_grad"] = all(cands[c]["training"]["grad_present"] for c in ("S2", "S3", "S4"))
            checks["3_dino_grad_none"] = bool(o["dino_grad_none"])
            checks["4_no_collapse"] = all(cands[c]["training"]["after"]["std_fused"] > 1e-4
                                          for c in ("S2", "S3", "S4"))
            checks["5_scoring_finite"] = all(np.isfinite(cands[c]["metrics"]["auroc"])
                                             and np.isfinite(cands[c]["metrics"]["ap"])
                                             for c in ("S2", "S3", "S4"))
            checks["6_S0_matches"] = abs(o["s0_reprod"]["auroc"] - ref[o["category"]]["baseline"]["auroc"]) <= 1e-4
            checks["7_S1_matches"] = abs(o["s1_reprod"]["auroc"] - ref[o["category"]]["A1"]["auroc"]) <= 1e-4
        for k, v in checks.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
            ok = ok and v
        print(f"[E12-B] SMOKE {'PASS' if ok else 'FAIL'} (no checkpoint/CSV written)", flush=True)
        return 0 if ok else 1

    write_training_csv(partial)
    write_screening_csv(partial, ref)
    print(f"\n[E12-B] done -> {SCREEN_CSV}", flush=True)
    return 0


def write_training_csv(partial):
    w = csv.writer(open(TRAIN_CSV, "w", newline=""))
    w.writerow(["category", "candidate", "trainable_params", "steps_used", "best_val_cons",
                "final_cons", "std_fused_before", "std_fused_after", "loss_finite", "gate_weights"])
    for cat in ("Bird", "Car", "Cube"):
        for c in ("S2", "S3", "S4"):
            cell = partial.get("cells", {}).get(f"{cat}|{c}")
            if not cell:
                continue
            w.writerow([cat, c, cell["trainable_params"], cell["steps_used"],
                        cell["best_val_cons"], cell["final_cons"],
                        cell["std_fused_before"], cell["std_fused_after"],
                        cell["loss_finite"], json.dumps(cell["gate_weights"])])


def write_screening_csv(partial, ref):
    w = csv.writer(open(SCREEN_CSV, "w", newline=""))
    w.writerow(["category", "view", "config", "auroc", "ap", "d_auroc", "d_ap", "status"])
    for cat in ("Bird", "Car", "Cube"):
        s0 = ref[cat]["baseline"]
        s1 = ref[cat]["A1"]
        w.writerow([cat, "120", "S0", f"{s0['auroc']:.5f}", f"{s0['ap']:.5f}",
                    "+0.00000", "+0.00000", "reused_p4"])
        w.writerow([cat, "120", "S1", f"{s1['auroc']:.5f}", f"{s1['ap']:.5f}",
                    f"{s1['auroc']-s0['auroc']:+.5f}", f"{s1['ap']-s0['ap']:+.5f}", "reused_p4"])
        for c in ("S2", "S3", "S4"):
            cell = partial.get("cells", {}).get(f"{cat}|{c}")
            if not cell:
                w.writerow([cat, "120", c, "", "", "", "", "MISSING"])
                continue
            w.writerow([cat, "120", c, f"{cell['auroc']:.5f}", f"{cell['ap']:.5f}",
                        f"{cell['auroc']-s0['auroc']:+.5f}", f"{cell['ap']-s0['ap']:+.5f}", "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
