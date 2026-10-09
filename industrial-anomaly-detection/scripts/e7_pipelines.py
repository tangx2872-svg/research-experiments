#!/usr/bin/env python
"""E7 — the 14 frozen pipelines. Implements the upstream design; does not design it.

Protocol: `docs/E7_1_FROZEN_PROTOCOL.md`. Provenance is recorded there (section 3).
Score direction convention: HIGHER = more anomalous, for every pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==================================================================== PatchCore family
def run_patchcore(bank: list, test: list, dev, tag: str, want_maps: bool = True) -> dict:
    """Frozen E4-D1/E5-1 PatchCore (wide_resnet50_2, layer2+layer3, 256, coreset 0.1, k=9, max-pool)."""
    from experiment5a_model import FAlphaDualLayerPatchcore
    from anomalib.engine import Engine
    import experiment1b_defect_sensitivity as e1b

    stages = {}
    T0 = time.time()
    t0 = time.time()
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank], batch_size=16,
                                   num_workers=0, seed=E7.SEED, name=f"e7_{tag}")
    stages["setup_s"] = time.time() - t0

    model = FAlphaDualLayerPatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                     coreset_sampling_ratio=0.1, num_neighbors=9,
                                     alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(E7.ROOT / "results" / "e7_1" / "fit" / tag))
    bar = E7.Bar(tag, "fit", 1, unit="fit")
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    stages["fit_s"] = time.time() - t0
    tm = model.model; tm.eval(); e1b._move_model_to_device(tm, dev)
    stages["coreset_size"] = int(tm.memory_bank.shape[0])
    bar.step(1, extra=f"fit={stages['fit_s']:.0f}s coreset={stages['coreset_size']}")

    bar = E7.Bar(tag, "scoring", len(test))
    rows, maps, masks = [], [], []
    t0 = time.time()
    for i, r in enumerate(test, 1):
        s, amap = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        if not np.isfinite(s):
            raise FloatingPointError(f"{tag}: non-finite score for {r['image_path']}")
        rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                     "score": float(s), "image_path": r["image_path"]})
        if want_maps:
            maps.append(amap)
            masks.append(e1b.load_mask(Path(r["mask_path"])) if r["mask_path"]
                         else np.zeros((1, 1), dtype=np.uint8))
        if i % 100 == 0 or i == len(test):
            bar(i)
    stages["scoring_s"] = time.time() - t0
    stages["total_s"] = time.time() - T0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)

    # optional pixel-level extras; only where a real mask exists (E4-D1 known AUPRO issue)
    pixel = {"pixel_auroc": float("nan"), "aupro": float("nan"), "n_masked_ng": 0}
    try:
        good_maps = [m for m, r in zip(maps, test) if r["kind"] == "Good"]
        def_maps = [m for m, msk, r in zip(maps, masks, test)
                    if r["kind"] == "NG" and np.asarray(msk).size > 4]
        def_masks = [msk for msk, r in zip(masks, test)
                     if r["kind"] == "NG" and np.asarray(msk).size > 4]
        pixel["n_masked_ng"] = len(def_masks)
        if good_maps and def_maps:
            from experiment1h_runner import pixel_auroc_from_maps
            pixel["pixel_auroc"] = float(pixel_auroc_from_maps(good_maps, def_maps, def_masks))
    except Exception as exc:  # noqa: BLE001
        pixel["pixel_error"] = f"{type(exc).__name__}: {str(exc)[:70]}"
    try:
        if good_maps and def_maps:
            from experiment5a_h_runner import compute_aupro
            pixel["aupro"] = float(compute_aupro(good_maps, def_maps, def_masks))
    except Exception as exc:  # noqa: BLE001
        pixel["aupro_error"] = f"{type(exc).__name__}: {str(exc)[:70]}"

    del tm
    torch.cuda.empty_cache()
    return {"rows": rows, "maps": maps, "masks": masks, "pixel": pixel, "stages": stages,
            "family": "patchcore", "image_kind": tag}


# ==================================================================== DINOv2 family (AnomalyDINO)
def _dinov2_prepare(model, path: str, dev):
    """AnomalyDINO `prepare_image`, verbatim semantics: smaller-edge 448 BICUBIC + ImageNet norm + crop."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size=E7.CACHE_RES, interpolation=T.InterpolationMode.BICUBIC,
                             antialias=True),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    t = tf(img)
    _, h, w = t.shape
    ps = model.patch_size
    cw, ch = w - w % ps, h - h % ps
    t = t[:, :ch, :cw]
    return t, (ch // ps, cw // ps)


def mean_top1p(distances: np.ndarray) -> float:
    """AnomalyDINO `src/post_eval.py::mean_top1p`, verbatim."""
    flat = distances.flatten()
    n = int(len(flat) * 0.01)
    if n == 0:
        return float(np.max(flat))
    return float(np.mean(np.sort(flat)[::-1][:n]))


def run_dinov2(model_name: str, bank: list, test: list, dev, tag: str) -> dict:
    """AnomalyDINO: L2-normalised kNN, k=1, image score = mean of the top-1% patch distances."""
    stages = {}
    T0 = time.time()
    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(dev)
    stages["model_load_s"] = time.time() - t0

    bar = E7.Bar(tag, "bank-embed", len(bank))
    feats = []
    with torch.inference_mode():
        for i, r in enumerate(bank, 1):
            t, _ = _dinov2_prepare(model, r["image_path"], dev)
            tok = model.get_intermediate_layers(t.unsqueeze(0).to(dev))[0].squeeze()
            feats.append(tok.float().cpu().numpy())
            if i % 25 == 0 or i == len(bank):
                bar(i)
    bank_f = np.concatenate(feats, axis=0).astype("float32")
    n_bank = bank_f.shape[0]
    del feats
    # L2 normalise (faiss.normalize_L2 semantics)
    bank_f /= np.maximum(np.linalg.norm(bank_f, axis=1, keepdims=True), 1e-12)
    bank_t = torch.from_numpy(bank_f).to(dev)
    stages["n_bank_patches"] = int(n_bank)

    bar = E7.Bar(tag, "scoring", len(test))
    rows, maps = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(test, 1):
            t, grid = _dinov2_prepare(model, r["image_path"], dev)
            tok = model.get_intermediate_layers(t.unsqueeze(0).to(dev))[0].squeeze().float()
            tok = tok / tok.norm(dim=1, keepdim=True).clamp_min(1e-12)
            # cosine distance to the 1-NN == squared-L2/2 of a normalised pair; computed in chunks
            best = None
            for k in range(0, n_bank, 65536):
                sim = tok @ bank_t[k:k + 65536].T                      # (P, chunk)
                m = sim.max(dim=1).values
                best = m if best is None else torch.maximum(best, m)
            dist = (1.0 - best).clamp_min(0.0)                          # cosine distance
            score = mean_top1p(dist.cpu().numpy())
            if not np.isfinite(score):
                raise FloatingPointError(f"{tag}: non-finite score for {r['image_path']}")
            rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                         "score": float(score), "image_path": r["image_path"]})
            maps.append(dist.reshape(grid).cpu().numpy())
            if i % 100 == 0 or i == len(test):
                bar(i)
    stages["scoring_s"] = time.time() - t0
    stages["total_s"] = time.time() - T0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    del model, bank_t
    torch.cuda.empty_cache()
    return {"rows": rows, "maps": maps, "masks": None, "stages": stages,
            "family": "dinov2", "model_name": model_name, "image_kind": tag}


# ==================================================================== SoftPatch family
def _softpatch_modules():
    """Import the official SoftPatch weight/sampler code (no re-implementation)."""
    import importlib.util
    base = ROOT / "third_party" / "SoftPatch"
    saved = {k: sys.modules.pop(k, None) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    spec_s = importlib.util.spec_from_file_location("sp_sampler", base / "src" / "sampler.py")
    sampler = importlib.util.module_from_spec(spec_s); spec_s.loader.exec_module(sampler)
    spec_p = importlib.util.spec_from_file_location("sp_softpatch", base / "src" / "softpatch.py")
    # softpatch.py imports `sampler` / `common` / `backbones` as top-level modules
    sys.modules["sampler"] = sampler
    sys.path.insert(0, str(base / "src"))
    sp_mod = importlib.util.module_from_spec(spec_p)
    try:
        spec_p.loader.exec_module(sp_mod)
    finally:
        pass
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
    return sp_mod, sampler


def _patchcore_embed(model, paths: list, dev, tag: str, bar) -> np.ndarray:
    """Frozen PatchCore embedding as a 2-D array (N*P, C), row = n*P + p, p = h*W + w."""
    import experiment1b_defect_sensitivity as e1b
    outs = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            img = e1b.load_image_as_tensor(Path(p))
            inp = e1b.preprocess_for_model(img, dev).unsqueeze(0)
            emb = model.generate_embedding(model.feature_extractor(inp))[0]      # (C,H,W)
            outs.append(emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).cpu().numpy())  # (P,C)
            if i % 50 == 0 or i == len(paths):
                bar(i)
    return np.concatenate(outs, axis=0).astype("float32")


def run_softpatch(weight_method: str, bank: list, test: list, dev, tag: str) -> dict:
    """Official SoftPatch mechanism (nearest / LOF) on the shared PatchCore representation."""
    from experiment5a_model import FAlphaDualLayerPatchcoreModel
    import experiment1b_defect_sensitivity as e1b

    assert weight_method in ("nearest", "lof"), "registry forbids a third weighting"
    THRESHOLD, SAMPLING_RATIO = 0.15, 0.1
    LOF_K = 5          # effective official value (main.py passes `LOF_k`, which load() ignores)
    stages = {}
    T0 = time.time()
    torch.manual_seed(E7.SEED); np.random.seed(E7.SEED)

    sp_mod, sampler_mod = _softpatch_modules()
    torch_model = FAlphaDualLayerPatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                                pre_trained=True, num_neighbors=9,
                                                alpha_l2=0.0, alpha_l3=0.0)
    torch_model.eval(); e1b._move_model_to_device(torch_model, dev)

    bar = E7.Bar(tag, "bank-embed", len(bank))
    t0 = time.time()
    feat = _patchcore_embed(torch_model, [r["image_path"] for r in bank], dev, tag, bar)
    stages["bank_embed_s"] = time.time() - t0
    n_bank, dim = len(bank), feat.shape[1]
    patches_per_image = feat.shape[0] // n_bank
    stages["n_bank_patches"] = int(feat.shape[0])

    sp = sp_mod.SoftPatch(torch.device("cuda"))
    sp.device = torch.device("cuda")
    sp.weight_method = weight_method
    sp.lof_k = LOF_K
    sp.threshold = THRESHOLD
    side = int(round(patches_per_image ** 0.5))
    assert side * side == patches_per_image, f"non-square patch grid {patches_per_image}"
    sp.feature_shape = (side, side)   # (H, W) of the embedding grid, as `_compute_patch_weight` expects
    # official SoftPatch replaces any incoming sampler with WeightedGreedyCoresetSampler
    # (softpatch.py:85) -- that is the class which consumes the soft sampling weights
    sampler = sampler_mod.WeightedGreedyCoresetSampler(SAMPLING_RATIO, torch.device("cuda"))
    sp.featuresampler = sampler

    t0 = time.time()
    with torch.no_grad():
        patch_weight = sp._compute_patch_weight(feat)                 # (N, P) official weights
        pw = patch_weight.reshape(-1)
        thr = torch.quantile(pw, 1 - THRESHOLD)
        sampling_weight = torch.where(pw > thr, 0, 1)
        sampler.set_sampling_weight(sampling_weight)
        pw = pw.clamp(min=0)
        sample_features, sample_indices = sampler.run(feat)
        coreset_weight = pw[sample_indices].cpu().numpy()
    stages["weight_coreset_s"] = time.time() - t0
    stages["coreset_size"] = int(sample_features.shape[0])
    mem = torch.from_numpy(np.asarray(sample_features, dtype="float32")).to(dev)
    cw = torch.from_numpy(np.asarray(coreset_weight, dtype="float32")).to(dev)
    print(f"  E7 [{tag}] weight={weight_method} coreset={stages['coreset_size']} "
          f"(from {feat.shape[0]} patches) weights[min={cw.min():.4g} max={cw.max():.4g}]", flush=True)

    bar = E7.Bar(tag, "scoring", len(test))
    rows, maps = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(test, 1):
            img = e1b.load_image_as_tensor(Path(r["image_path"]))
            inp = e1b.preprocess_for_model(img, dev).unsqueeze(0)
            emb = torch_model.generate_embedding(torch_model.feature_extractor(inp))[0]   # (C,H,W)
            H, W = emb.shape[1], emb.shape[2]
            q = emb.permute(1, 2, 0).reshape(-1, emb.shape[0])                            # (P,C)
            d = torch.cdist(q, mem, p=2)                                                 # official FaissNN L2
            dist, idx = d.min(dim=1)                                                     # anomaly_score_num_nn = 1
            s_patch = dist * cw[idx]                                                     # official soft weight
            score = float(s_patch.max().cpu())                                           # patch_maker.score = max
            if not np.isfinite(score):
                raise FloatingPointError(f"{tag}: non-finite score for {r['image_path']}")
            rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                         "score": score, "image_path": r["image_path"]})
            maps.append(s_patch.reshape(H, W).cpu().numpy())
            if i % 100 == 0 or i == len(test):
                bar(i)
    stages["scoring_s"] = time.time() - t0
    stages["total_s"] = time.time() - T0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    del torch_model, mem, cw
    torch.cuda.empty_cache()
    return {"rows": rows, "maps": maps, "masks": None, "stages": stages,
            "family": "softpatch", "weight_method": weight_method, "image_kind": tag}
