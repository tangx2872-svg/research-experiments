#!/usr/bin/env python
"""E5-1 - Broad Module Composition Screening runner.

Protocol: docs/E5_1_FROZEN_PROTOCOL.md (frozen BEFORE any candidate result; commit e96cd5a).

Frozen scope: M2AD Bird / seed 0 / view 120 / bank 1200 / scoring 700 / embedding 1536x32x32 / IMAGE_SIZE 256.

Candidates and frozen provenance:
  C01 PIAD-Retinex  input photometric      official github.com/Kaichen-Yang/piad_baseline @67b816b
  C06 SimpleNet     post-concat            official github.com/DonaldRR/SimpleNet @351a2b8
  C07 ReConPatch    post-concat            FROM-PAPER MINIMAL (arXiv:2305.16713)
  C08 CRAD          memory representation  ADAPTED FROM CRAD github.com/tae-mo/CRAD @b5a1c472
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "SimpleNet"))
sys.path.insert(0, str(ROOT / "third_party" / "CRAD"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e4x_common as C  # noqa: E402
import e5_1_metrics as M  # noqa: E402
import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment1h_runner import image_auroc  # noqa: E402

OUT = ROOT / "results" / "e5_1"
RAW = OUT / "raw"
PROGRESS = OUT / "progress.json"
FROZEN_PROTOCOL = ROOT / "docs" / "E5_1_FROZEN_PROTOCOL.md"
ORIG_PER_IMAGE = ROOT / "results" / "e4_x" / "per_image_original.csv"
RETINEX = ROOT / "data" / "m2ad" / "e5_1_retinex"

SEED = 0
BATCH = 16
FROZEN_VIEW = "120"
EMB_CH = 1536
EMB_HW = 32
ILLUMINATIONS = C.ILLUMINATIONS


def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def vram() -> tuple:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256(p: Path) -> str:
    return C.sha256(p)


def progress_json() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except json.JSONDecodeError:
            pass
    return {"experiment": "E5-1", "candidates": {}}


def save_progress(st: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(st, indent=2, default=float))


class Bar:
    def __init__(self, cand, stage, total, unit="img"):
        self.cand, self.stage, self.total, self.unit = cand, stage, total, unit
        self.t0 = time.time()

    def __call__(self, done, extra=""):
        el = time.time() - self.t0
        rate = done / el if el > 0 else 0.0
        remain = (self.total - done) / rate if rate > 0 else float("inf")
        fin = time.strftime("%H:%M", time.localtime(time.time() + remain)) if np.isfinite(remain) else "n/a"
        a, r = vram()
        print(f"  E5-1 [{self.cand}] {self.stage} {done}/{self.total} "
              f"{100.0*done/max(1,self.total):5.1f}%  "
              f"elapsed={time.strftime('%H:%M:%S', time.gmtime(el))}  "
              f"ETA={time.strftime('%H:%M:%S', time.gmtime(remain)) if np.isfinite(remain) else 'n/a'}"
              f"  finish~{fin}  {rate:6.1f} {self.unit}/s  "
              f"VRAM alloc={a:7.1f}MB res={r:7.1f}MB  RAM={ram_mb():7.1f}MB {extra}", flush=True)


def load_scope() -> dict:
    recs = LD.load_records()
    est = LD.estimate_scope(recs)
    bank = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    val = sorted(LD.select(recs, set(est["val_specimens"])), key=lambda r: r["image_path"])
    score = sorted([r for r in LD.select(recs, views=[FROZEN_VIEW]) if r["split"] == "test"],
                   key=lambda r: r["image_path"])
    return {"recs": recs, "est": est, "bank": bank, "val": val, "score": score}


def sanity_s1_s3(scope: dict) -> dict:
    res = {}
    res["S1_scope"] = {"pass": (len(scope["bank"]), len(scope["val"]), len(scope["score"])) == (1200, 240, 700),
                       "detail": f"bank={len(scope['bank'])} val={len(scope['val'])} "
                                 f"score={len(scope['score'])} view={FROZEN_VIEW} EXPECT 1200/240/700"}
    ill = sorted({r["illumination"] for r in scope["score"]})
    res["S1b_illum"] = {"pass": ill == ILLUMINATIONS, "detail": f"illuminations={ill}"}
    res["S3_original_asset"] = {"pass": ORIG_PER_IMAGE.exists(),
                                "detail": f"sha256={sha256(ORIG_PER_IMAGE) if ORIG_PER_IMAGE.exists() else 'MISSING'}"}
    bs = {r["specimen"] for r in scope["bank"]}
    ss = {r["specimen"] for r in scope["score"]}
    res["S7_train_test_separation"] = {"pass": not (bs & ss), "detail": f"|bank n score|={len(bs & ss)} (expect 0)"}
    res["S8_normal_only_bank"] = {"pass": all(r["kind"] == "Good" and r["split"] == "train" for r in scope["bank"]),
                                  "detail": "bank = train split Good specimens only"}
    return res


def build_torch_model():
    from experiment5a_model import FAlphaDualLayerPatchcoreModel
    tm = FAlphaDualLayerPatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                       pre_trained=True, num_neighbors=9, alpha_l2=0.0, alpha_l3=0.0)
    tm.eval()
    return tm


@torch.no_grad()
def embed_records(tm, records, dev, cand, stage, tag, image_key="image_path") -> torch.Tensor:
    out = []
    bar = Bar(cand, stage, len(records))
    for i, r in enumerate(records, 1):
        img = e1b.load_image_as_tensor(Path(r[image_key]))
        inp = e1b.preprocess_for_model(img, dev).unsqueeze(0)
        feats = tm.feature_extractor(inp)
        emb = tm.generate_embedding(feats)
        if tuple(emb.shape[1:]) != (EMB_CH, EMB_HW, EMB_HW):
            raise RuntimeError(f"{tag} sanity S4 embedding shape {tuple(emb.shape)}")
        if not torch.isfinite(emb).all():
            raise RuntimeError(f"{tag} sanity S5 non-finite embedding at {r[image_key]}")
        out.append(emb[0].to(torch.float32).cpu())
        if i % 100 == 0 or i == len(records):
            bar(i)
    return torch.stack(out)


def _patch_flat(emb: torch.Tensor) -> torch.Tensor:
    n, c, h, w = emb.shape
    return emb.permute(0, 2, 3, 1).reshape(n, h * w, c)


# --------------------------------------------------------------------- C01
def run_c01(scope: dict, dev) -> dict:
    """PIAD-Retinex: reflectance images -> refit frozen PatchCore -> score."""
    from experiment5a_model import FAlphaDualLayerPatchcore
    from anomalib.engine import Engine

    def rp(rec, sub):
        return RETINEX / sub / Path(rec["image_path"]).relative_to(LD.DATA_ROOT)

    bank = [dict(r, image_path=str(rp(r, "bank"))) for r in scope["bank"]]
    val = [dict(r, image_path=str(rp(r, "val"))) for r in scope["val"]]
    score = [dict(r, image_path=str(rp(r, "score"))) for r in scope["score"]]
    missing = [r["image_path"] for r in bank + score if not Path(r["image_path"]).exists()]
    if missing:
        return {"status": "FAILED",
                "reason": f"retinex images missing ({len(missing)}); run e5_1_retinex_prep.py first",
                "example": missing[:3]}

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    stages = {}
    T0 = time.time()

    t0 = time.time()
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank],
                                   val_paths=[r["image_path"] for r in val],
                                   batch_size=BATCH, num_workers=0, seed=SEED, name="e51_c01")
    stages["datamodule_setup_s"] = time.time() - t0

    model = FAlphaDualLayerPatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                     coreset_sampling_ratio=0.1, num_neighbors=9,
                                     alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(OUT / "fit" / "C01"))
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    stages["fit_s"] = time.time() - t0
    tm = model.model; tm.eval(); e1b._move_model_to_device(tm, dev)
    stages["coreset_size"] = int(tm.memory_bank.shape[0])
    print(f"  E5-1 [C01] fit done {stages['fit_s']:.1f}s coreset={stages['coreset_size']}", flush=True)

    bar = Bar("C01", "scoring", len(score))
    rows = []
    t0 = time.time()
    for i, r in enumerate(score, 1):
        s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        if not np.isfinite(s):
            raise FloatingPointError(f"C01 non-finite score {r['image_path']}")
        rows.append({"specimen": r["specimen"], "kind": r["kind"],
                     "illumination": r["illumination"], "score": s})
        if i % 100 == 0 or i == len(score):
            bar(i)
    stages["scoring_s"] = time.time() - t0
    stages["total_s"] = time.time() - T0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    stages["peak_ram_mb"] = round(ram_mb(), 1)
    return {"status": "OK", "per_image": rows, "stages_seconds": stages, "tau_val": None,
            "reused": ["image list", "view=120", "official split", "labels", "masks"],
            "recomputed": ["Retinex reflectance (bank 1200 + scoring 700)", "PatchCore coreset", "all scores"]}


def run_c06(bank_emb, test_emb, dev, _unused=None) -> dict:
    """SimpleNet (official code): Projection adaptor + Discriminator; synthetic anomalies only."""
    import importlib
    sp = str(ROOT / "third_party" / "SimpleNet")
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)
    for _m in ("utils", "common", "metrics", "backbones", "resnet", "simplenet", "datasets"):
        sys.modules.pop(_m, None)
    sn = importlib.import_module("simplenet")

    pretrain_dim = target_dim = EMB_CH
    meta_epochs, gan_epochs = 40, 4
    noise_std, mix_noise = 0.015, 1
    dsc_layers, dsc_hidden, dsc_margin, dsc_lr = 2, 1024, 0.5, 0.0002
    pre_proj, lr = 1, 1e-3

    torch.manual_seed(SEED)
    proj = sn.Projection(target_dim, target_dim, pre_proj, 0).to(dev)
    disc = sn.Discriminator(target_dim, n_layers=dsc_layers, hidden=dsc_hidden).to(dev)
    p_opt = torch.optim.AdamW(proj.parameters(), lr=lr * .1)
    d_opt = torch.optim.Adam(disc.parameters(), lr=dsc_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, meta_epochs * gan_epochs, dsc_lr * .4)

    bank_pf = _patch_flat(bank_emb)
    n_img = bank_pf.shape[0]
    bs = 8
    total_steps = meta_epochs * gan_epochs * int(np.ceil(n_img / bs))
    bar = Bar("C06", "fitting", total_steps, unit="step")
    step, losses = 0, []
    for _me in range(meta_epochs):
        for _ge in range(gan_epochs):
            perm = torch.randperm(n_img)
            for k in range(0, n_img, bs):
                idx = perm[k:k + bs]
                true_feats = proj(bank_pf[idx].to(dev).reshape(-1, pretrain_dim))
                nidx = torch.randint(0, mix_noise, torch.Size([true_feats.shape[0]]), device=dev)
                one_hot = F.one_hot(nidx, num_classes=mix_noise).to(torch.float32)
                noise = torch.stack([torch.normal(0, noise_std * 1.1 ** kk, true_feats.shape, device=dev)
                                     for kk in range(mix_noise)], dim=1)
                noise = (noise * one_hot.unsqueeze(-1)).sum(1)
                fake_feats = true_feats + noise
                d_opt.zero_grad(); p_opt.zero_grad()
                sc = disc(torch.cat([true_feats, fake_feats]))
                ts_, fs_ = sc[:len(true_feats)], sc[len(true_feats):]
                th = dsc_margin
                loss = torch.clip(-ts_ + th, min=0).mean() + torch.clip(fs_ + th, min=0).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("C06 non-finite loss (sanity S5)")
                loss.backward(); p_opt.step(); d_opt.step(); sched.step()
                losses.append(float(loss.detach())); step += 1
                if step % 200 == 0 or step == total_steps:
                    bar(step, f"loss={np.mean(losses[-200:]):.4f}")
    proj.eval(); disc.eval()
    RAW.mkdir(parents=True, exist_ok=True)
    torch.save({"discriminator": disc.state_dict(), "pre_projection": proj.state_dict()}, RAW / "C06_weights.pt")

    bar = Bar("C06", "scoring", test_emb.shape[0])
    test_pf = _patch_flat(test_emb)
    out = []
    with torch.no_grad():
        for i in range(test_pf.shape[0]):
            f = proj(test_pf[i].to(dev))
            out.append(float((-disc(f)).max().cpu()))
            if (i + 1) % 100 == 0:
                bar(i + 1)
    return {"status": "OK", "scores": out, "train_loss_last200": float(np.mean(losses[-200:])),
            "reused": ["frozen wide_resnet50_2 embedding (bank + scoring)"],
            "recomputed": ["SimpleNet Projection + Discriminator fit", "discriminator scores"],
            "implementation": "OFFICIAL SimpleNet code (adaptor + discriminative component)",
            "hyperparams": {"pretrain_embed_dimension": pretrain_dim, "target_embed_dimension": target_dim,
                            "meta_epochs": meta_epochs, "gan_epochs": gan_epochs, "noise_std": noise_std,
                            "mix_noise": mix_noise, "dsc_layers": dsc_layers, "dsc_hidden": dsc_hidden,
                            "dsc_margin": dsc_margin, "dsc_lr": dsc_lr, "pre_proj": pre_proj, "lr": lr,
                            "batch_size": bs}}


def run_c07(bank_emb, test_emb, dev, _unused=None) -> dict:
    """ReConPatch - FROM-PAPER MINIMAL implementation (arXiv:2305.16713)."""
    F_DIM = 512
    NEG_M, ALPHA, K_NN = 1.0, 0.5, 5
    EPOCHS, LR, WD = 120, 1e-5, 1e-2
    EMA_GAMMA = 0.999
    PATCH_SAMPLE = 128
    BS = 8

    torch.manual_seed(SEED)
    f_net = torch.nn.Sequential(torch.nn.Linear(EMB_CH, F_DIM)).to(dev)
    g_net = torch.nn.Sequential(torch.nn.Linear(F_DIM, F_DIM), torch.nn.ReLU(),
                                torch.nn.Linear(F_DIM, F_DIM)).to(dev)
    f_bar, g_bar = copy.deepcopy(f_net).eval(), copy.deepcopy(g_net).eval()
    for p in list(f_bar.parameters()) + list(g_bar.parameters()):
        p.requires_grad = False
    opt = torch.optim.Adam(list(f_net.parameters()) + list(g_net.parameters()), lr=LR, weight_decay=WD)

    bank_pf = _patch_flat(bank_emb)
    n_img = bank_pf.shape[0]
    steps_per_ep = int(np.ceil(n_img / BS))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps_per_ep)
    bar = Bar("C07", "fitting", EPOCHS * steps_per_ep, unit="step")
    step, losses = 0, []
    for _ep in range(EPOCHS):
        perm = torch.randperm(n_img)
        for k in range(0, n_img, BS):
            idx = perm[k:k + BS]
            sel = torch.randint(0, bank_pf.shape[1], (len(idx), PATCH_SAMPLE))
            p = torch.stack([bank_pf[j][sel[t]] for t, j in enumerate(idx)]).to(dev)
            B, S, _ = p.shape
            flat = p.reshape(B * S, EMB_CH)
            z = g_net(f_net(flat))
            with torch.no_grad():
                z_bar = g_bar(f_bar(flat))
            N = z.shape[0]
            d2 = torch.cdist(z_bar, z_bar, p=2) ** 2
            sigma = d2.detach().median().clamp_min(1e-8)
            w_pair = torch.exp(-d2 / sigma)
            with torch.no_grad():
                knn = d2.topk(min(K_NN, N - 1), largest=False).indices
                mask_nk = torch.zeros_like(d2, dtype=torch.bool)
                mask_nk.scatter_(1, knn, True)
                inter = (mask_nk.unsqueeze(1) & mask_nk.unsqueeze(0)).sum(-1).to(d2.dtype)
                w_ctx = 0.5 * (inter / float(min(K_NN, N - 1)) + (inter / float(min(K_NN, N - 1))).t())
                w = ALPHA * w_pair + (1 - ALPHA) * w_ctx
            dz = torch.cdist(z, z, p=2)
            delta = dz / dz.detach().mean().clamp_min(1e-8)
            loss = (w * delta ** 2 + (1 - w) * torch.clamp(NEG_M - delta, min=0) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("C07 non-finite loss (sanity S5)")
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pb, npb in ((f_net, f_bar), (g_net, g_bar)):
                    for a, b in zip(npb.parameters(), pb.parameters()):
                        a.mul_(EMA_GAMMA).add_(b.detach(), alpha=1 - EMA_GAMMA)
            losses.append(float(loss.detach())); step += 1
            if step % 500 == 0 or step == EPOCHS * steps_per_ep:
                bar(step, f"loss={np.mean(losses[-500:]):.4f}")

    f_net.eval()
    coreset_ratio, K = 0.01, 9
    RAW.mkdir(parents=True, exist_ok=True)
    torch.save({"f": f_net.state_dict(), "g": g_net.state_dict()}, RAW / "C07_weights.pt")
    with torch.no_grad():
        flat_all = bank_pf.reshape(-1, EMB_CH)
        outs = []
        for k in range(0, flat_all.shape[0], 65536):
            outs.append(f_net(flat_all[k:k + 65536].to(dev)).cpu())
        bank_f = torch.cat(outs, 0)
        n_keep = max(1, int(bank_f.shape[0] * coreset_ratio))
        gsel = torch.randperm(bank_f.shape[0], generator=torch.Generator().manual_seed(SEED))[:n_keep]
        mem = bank_f[gsel].to(dev)
        bar = Bar("C07", "scoring", test_emb.shape[0])
        test_pf = _patch_flat(test_emb)
        out = []
        for i in range(test_pf.shape[0]):
            q = f_net(test_pf[i].to(dev))
            d = torch.cdist(q, mem, p=2)
            dk, _ = d.topk(K, largest=False)
            out.append(float(dk[:, 0].max().cpu()))
            if (i + 1) % 100 == 0:
                bar(i + 1)
    return {"status": "OK", "scores": out, "train_loss_last500": float(np.mean(losses[-500:])),
            "reused": ["frozen wide_resnet50_2 embedding (bank + scoring)"],
            "recomputed": ["ReConPatch f/g + EMA fit (from-paper minimal)", "modulated-space kNN scores"],
            "implementation": "FROM-PAPER MINIMAL (no official implementation located)",
            "hyperparams": {"f_dim": F_DIM, "k": K_NN, "margin": NEG_M, "alpha": ALPHA, "epochs": EPOCHS,
                            "lr": LR, "weight_decay": WD, "ema_gamma": EMA_GAMMA,
                            "patch_sample_per_image": PATCH_SAMPLE, "batch_images": BS,
                            "optimizer": "Adam (AdamP unavailable; protocol D2)", "coreset_ratio": coreset_ratio}}


def run_c08(bank_emb, test_emb, dev, _unused=None) -> dict:
    """CRAD - ADAPTED FROM CRAD (official class, adapted to our frozen feature space)."""
    import torch.nn as nn
    cp = str(ROOT / "third_party" / "CRAD")
    if cp in sys.path:
        sys.path.remove(cp)
    sys.path.insert(0, cp)
    for _m in ("utils", "models", "datasets", "tools", "simplenet", "common"):
        sys.modules.pop(_m, None)
    from models.reconstructions.crad import CRAD
    from models.reconstructions.mobilenetv3 import MobileBottleneck

    ch = EMB_CH
    layers, local_resol, global_resol = 11, 8, 4
    mse_lamb, cos_lamb, mse_coef, noise_std = 0.3, 0.7, 0.05, 0.05
    ch_exp = 864
    EPOCHS, GRID_LR, NET_LR, CLIP, STEP, GAMMA = 50, 0.1, 0.001, 0.1, 40, 0.1
    BS = 16

    torch.manual_seed(SEED)
    model = CRAD(layers=layers, frozen_layers=[], initializer=None,
                 inplanes=[ch], feature_size=(EMB_HW, EMB_HW), save_recon=False,
                 mse_lamb=mse_lamb, cos_lamb=cos_lamb, mse_coef=mse_coef,
                 noise_std=noise_std, ch_exp=ch_exp,
                 local_resol=local_resol, global_resol=global_resol).to(dev)
    model.layer = nn.Sequential(*[MobileBottleneck(2 * ch, 2 * ch, 3, 1, ch_exp, True, "HS")
                                  for _ in range(layers)]).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  E5-1 [C08] CRAD params = {n_param/1e6:.1f}M (ch={ch}, ch_exp={ch_exp}, layers={layers})", flush=True)

    net_p = [p for n, p in model.named_parameters() if not n.startswith("query")]
    opt = torch.optim.Adam([{"params": list(model.query.parameters()), "lr": GRID_LR},
                            {"params": net_p, "lr": NET_LR}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=GAMMA)

    n_img = bank_emb.shape[0]
    spe = int(np.ceil(n_img / BS))
    steps = EPOCHS * spe
    bar = Bar("C08", "fitting", steps, unit="step")
    step, losses = 0, []
    model.train()
    for _ep in range(EPOCHS):
        perm = torch.randperm(n_img)
        for k in range(0, n_img, BS):
            x = bank_emb[perm[k:k + BS]].to(dev)
            out = model({"feature_align": x})
            loss = F.mse_loss(out["feature_rec"], out["feature_align"])
            if not torch.isfinite(loss):
                raise FloatingPointError("C08 non-finite loss (sanity S5)")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            losses.append(float(loss.detach())); step += 1
            if step % 25 == 0 or step == steps:
                bar(step, f"loss={np.mean(losses[-25:]):.4f}")
        sched.step()
        print(f"    E5-1 [C08] epoch {_ep+1}/{EPOCHS} done step={step}/{steps} "
              f"epoch_loss={np.mean(losses[-spe:]):.5f}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RAW / "C08_weights.pt")

    model.eval()
    bar = Bar("C08", "scoring", test_emb.shape[0])
    out = []
    with torch.no_grad():
        for i in range(test_emb.shape[0]):
            r = model({"feature_align": test_emb[i:i + 1].to(dev)})
            out.append(float(r["pred"].reshape(-1).max().cpu()))
            if (i + 1) % 100 == 0:
                bar(i + 1)
    return {"status": "OK", "scores": out, "train_loss_last_n": float(np.mean(losses[-25:])),
            "n_params_M": round(n_param / 1e6, 1),
            "reused": ["frozen wide_resnet50_2 embedding (bank + scoring)"],
            "recomputed": ["CRAD grid + MobileBottleneck fit", "CRAD reconstruction-error scores"],
            "implementation": "ADAPTED FROM CRAD (official class; adapted feature space)",
            "hyperparams": {"ch": ch, "feature_size": [EMB_HW, EMB_HW], "local_resol": local_resol,
                            "global_resol": global_resol, "ch_exp": ch_exp, "layers": layers,
                            "mse_lamb": mse_lamb, "cos_lamb": cos_lamb, "mse_coef": mse_coef,
                            "noise_std": noise_std, "epochs": EPOCHS, "grid_lr": GRID_LR,
                            "net_lr": NET_LR, "clip_max_norm": CLIP, "step_size": STEP, "gamma": GAMMA,
                            "batch_size": BS}}


# --------------------------------------------------------------------- metrics + persist
def load_original_rows() -> list:
    with open(ORIG_PER_IMAGE) as f:
        return [{"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                 "score": float(r["score"])} for r in csv.DictReader(f)]


def evaluate_and_store(cand, rows, orig_rows, extra, protocol_commit) -> dict:
    metrics = C.metric_block(rows, cand)
    orig_metrics = C.metric_block(orig_rows, "Original")
    dec = M.decide_e5_1(metrics, orig_metrics)
    l1oo = M.leave_one_illumination_out(rows, orig_rows, ILLUMINATIONS, image_auroc)
    if dec["decision"] == "GO" and any(x["advantage_lost"] for x in l1oo):
        dec["SINGLE-CONDITION-DRIVEN"] = True
    dec["leave_one_illumination_out_lost"] = [x["dropped_illumination"] for x in l1oo if x["advantage_lost"]]

    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / f"{cand}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["specimen", "kind", "illumination", "score"])
        w.writeheader(); w.writerows(rows)
    payload = {"candidate": cand, "metrics": metrics, "original_metrics": orig_metrics,
               "decision": dec, "leave_one_illumination_out": l1oo, "extra": extra,
               "git_head": git_head(), "protocol_commit": protocol_commit, "seed": SEED,
               "frozen_view": FROZEN_VIEW, "protocol_sha256": sha256(FROZEN_PROTOCOL)}
    (RAW / f"{cand}.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


RUNNERS = {"C01": run_c01, "C06": run_c06, "C07": run_c07, "C08": run_c08}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="C01")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    wanted = [c.strip() for c in args.candidates.split(",") if c.strip()]

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name} {props.total_memory/2**30:.1f} GiB", flush=True)

    protocol_commit = git_head()
    scope = load_scope()
    san = sanity_s1_s3(scope)
    for k, v in san.items():
        print(f"  sanity {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}", flush=True)
    failed = [k for k, v in san.items() if not v["pass"]]
    if failed:
        print(f"[ABORT] sanity FAIL: {failed}")
        return 3

    st = progress_json()
    st.setdefault("candidates", {})
    st["protocol_commit"] = protocol_commit
    st["protocol_sha256"] = sha256(FROZEN_PROTOCOL)
    st["seed"] = SEED
    st["frozen_view"] = FROZEN_VIEW
    st["sanity"] = {k: v["pass"] for k, v in san.items()}
    save_progress(st)

    orig_rows = load_original_rows()
    om = C.metric_block(orig_rows, "Original")
    print(f"[reference] Original AUROC={om['image_auroc']:.5f} d'={om['d_prime']:.5f} R_all={om['R_all']:.5f}",
          flush=True)

    need_embed = [c for c in wanted if c in ("C06", "C07", "C08")]
    bank_emb = test_emb = None
    if need_embed:
        tm = build_torch_model()
        e1b._move_model_to_device(tm, dev)
        t0 = time.time()
        bank_emb = embed_records(tm, scope["bank"], dev, "SHARED", "embed(bank)", "embed")
        test_emb = embed_records(tm, scope["score"], dev, "SHARED", "embed(scoring)", "embed")
        print(f"  E5-1 [SHARED] embeddings bank={tuple(bank_emb.shape)} test={tuple(test_emb.shape)} "
              f"in {time.time()-t0:.1f}s RAM={ram_mb():.1f}MB", flush=True)
        del tm
        torch.cuda.empty_cache()

    total = len(wanted)
    T_ALL = time.time()
    for ci, cand in enumerate(wanted, 1):
        prev = st["candidates"].get(cand, {})
        if prev.get("status") == "DONE" and not args.force:
            print(f"E5-1 [{ci}/{total}] {cand}: already DONE - skipped", flush=True)
            continue
        print(f"\nE5-1 [{ci-1}/{total} | {int(100*(ci-1)/total)}%]  Candidate: {cand}  Stage: starting", flush=True)
        st["candidates"][cand] = {"status": "RUNNING", "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "git_head": protocol_commit, "seed": SEED, "frozen_view": FROZEN_VIEW,
                                  "protocol_sha256": sha256(FROZEN_PROTOCOL)}
        save_progress(st)
        t_c = time.time()
        try:
            r = RUNNERS[cand](scope, dev) if cand == "C01" else RUNNERS[cand](bank_emb, test_emb, dev)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            st["candidates"][cand] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}",
                                      "end_time": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_progress(st)
            print(f"  E5-1 [{cand}] FAILED: {exc}", flush=True)
            continue

        if r.get("status") != "OK":
            st["candidates"][cand] = {"status": r.get("status", "HOLD"), "reason": r.get("reason", ""),
                                      "end_time": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_progress(st)
            print(f"  E5-1 [{cand}] {r.get('status')}: {r.get('reason')}", flush=True)
            continue

        if "per_image" in r:
            rows = r["per_image"]
        else:
            rows = [{"specimen": scope["score"][i]["specimen"], "kind": scope["score"][i]["kind"],
                     "illumination": scope["score"][i]["illumination"], "score": r["scores"][i]}
                    for i in range(len(scope["score"]))]
        extra = {k: v for k, v in r.items() if k not in ("per_image", "scores")}
        payload = evaluate_and_store(cand, rows, orig_rows, extra, protocol_commit)
        d = payload["decision"]
        st["candidates"][cand] = {
            "status": "DONE", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t_c, 1),
            "metrics": {k: payload["metrics"][k] for k in ("image_auroc", "d_prime", "R_all", "R_Good", "R_NG")},
            "decision": d["decision"], "d_auroc": d["d_auroc"], "d_dprime": d["d_dprime"], "R_ratio": d["R_ratio"],
            "artifact": f"results/e5_1/raw/{cand}.json", "git_head": protocol_commit, "seed": SEED,
            "frozen_view": FROZEN_VIEW, "protocol_sha256": sha256(FROZEN_PROTOCOL)}
        save_progress(st)
        print(f"  E5-1 [{ci}/{total} | {int(100*ci/total)}%] {cand} DONE  "
              f"AUROC={payload['metrics']['image_auroc']:.5f} (dA={d['d_auroc']:+.5f})  "
              f"d'={payload['metrics']['d_prime']:.5f} (dd={d['d_dprime']:+.5f})  "
              f"R_ratio={d['R_ratio']:.4f}  -> {d['decision']}", flush=True)

    print(f"\nE5-1 total wall = {time.strftime('%H:%M:%S', time.gmtime(time.time()-T_ALL))}", flush=True)
    print(f"[written] {RAW}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
