#!/usr/bin/env python
"""E9 - Real-Illumination Robust Module Screening: candidate registry + transforms.

Protocol-free screening (no pre-registered GO threshold); the E8 lesson applies:
Bird-only gains are STOP. Round 1 must cover Bird+Car+Cube.

Frozen scope: 300-image bank (E7-1 rule seed 0), test @ view 120, 700 images.
Baseline = E8 Original P00 (L0_256 input), already scored for 3 cats x 3 seeds x 4 views.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import e8_common as E8  # noqa: E402
import e8_fetch as EF  # noqa: E402
import e7_common as E7  # noqa: E402

ROUND1_CATS = ["Bird", "Car", "Cube"]
ROUND1_SEED = 0
ROUND1_VIEW = "120"
BANK_SIZE = E8.BANK_SIZE          # 300

# ------------------------------------------------------------------ candidate registry
# family: input | feature | combo
CANDIDATES = {
    "A0": {"family": "retinex", "name": "Retinex->PatchCore",
           "desc": "PIAD official reflectance R (URetinex-Net) then frozen PatchCore",
           "image_kind": "retinex", "feature_norm": False},
    "I01": {"family": "input", "name": "Gray-World->PatchCore",
            "desc": "per-channel mean scaled to global mean (color constancy)",
            "image_kind": "e9_grayworld", "feature_norm": False},
    "I02": {"family": "input", "name": "Shades-of-Gray(p=6)->PatchCore",
            "desc": "p-norm illuminant estimate (generalised Gray-World)",
            "image_kind": "e9_shades6", "feature_norm": False},
    "I03": {"family": "input", "name": "Gamma(g=0.8)->PatchCore",
            "desc": "exposure gamma correction (I^0.8)",
            "image_kind": "e9_gamma08", "feature_norm": False},
    "I04": {"family": "input", "name": "Homomorphic->PatchCore",
            "desc": "log-FFT high-pass suppresses low-freq illumination (D0=32)",
            "image_kind": "e9_homomorphic", "feature_norm": False},
    "F01": {"family": "feature", "name": "L2-feature-norm",
            "desc": "unit-normalise each 1536-d patch feature (Euclidean->cosine kNN)",
            "image_kind": None, "feature_norm": True},
    "A1": {"family": "combo", "name": "Retinex+L2-feature-norm",
           "desc": "reflectance R input + L2 feature normalisation",
           "image_kind": "retinex", "feature_norm": True},
}
ORDER = ["A0", "I01", "I02", "I03", "I04", "F01", "A1"]

INPUT_KINDS = {"e9_grayworld", "e9_shades6", "e9_gamma08", "e9_homomorphic"}
E9_CACHE_ROOT = ROOT / "data" / "m2ad" / "e9_cache"


# ------------------------------------------------------------------ input transforms (numpy, uint8 in -> uint8 out)
def _clip(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def gray_world(rgb):
    f = rgb.astype(np.float32)
    mean = f.reshape(-1, 3).mean(axis=0)
    g = float(mean.mean())
    scale = g / (mean + 1e-6)
    return _clip(f * scale[None, None, :])


def shades_of_gray(rgb, p=6):
    f = rgb.astype(np.float32) / 255.0
    e = (f ** p).reshape(-1, 3).mean(axis=0) ** (1.0 / p)
    e = e / (np.linalg.norm(e) + 1e-6)
    return _clip(f * 255.0 / (e[None, None, :] + 1e-6))


def gamma(rgb, g=0.8):
    f = rgb.astype(np.float32) / 255.0
    return _clip(255.0 * (f ** g))


def homomorphic(rgb, d0=32, gamma_h=1.2, gamma_l=0.6, c=1):
    out = np.empty_like(rgb, dtype=np.float32)
    rows, cols = rgb.shape[:2]
    u = np.arange(cols) - cols // 2
    v = np.arange(rows) - rows // 2
    U, V = np.meshgrid(u, v)
    D2 = U ** 2 + V ** 2
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * D2 / (d0 ** 2))) + gamma_l
    for ch in range(3):
        img = rgb[:, :, ch].astype(np.float32) / 255.0
        f = np.fft.fftshift(np.fft.fft2(np.log1p(img)))
        rec = np.fft.ifft2(np.fft.ifftshift(f * H)).real
        out[:, :, ch] = np.expm1(rec) * 255.0
    return _clip(out)


TRANSFORMS = {
    "e9_grayworld": gray_world,
    "e9_shades6": lambda x: shades_of_gray(x, 6),
    "e9_gamma08": lambda x: gamma(x, 0.8),
    "e9_homomorphic": lambda x: homomorphic(x),
}


def apply_kind(kind: str, rgb: np.ndarray) -> np.ndarray:
    return TRANSFORMS[kind](rgb)


# ------------------------------------------------------------------ scope
def round1_scope(category: str) -> dict:
    sc = EF.build_scope(category)
    bank = E8.bank_for(sc["pool"], ROUND1_SEED)
    test = [r for r in sc["test"] if r["view"] == ROUND1_VIEW]
    return {"category": category, "bank": bank, "test": test}


def smoke_scope(category: str, n_bank: int = 12, n_test: int = 24) -> dict:
    sc = EF.build_scope(category)
    bank = sorted(E8.bank_for(sc["pool"], ROUND1_SEED))[:n_bank]
    test = sorted([r for r in sc["test"] if r["view"] == ROUND1_VIEW])[:n_test]
    return {"category": category, "bank": bank, "test": test}


# ------------------------------------------------------------------ image cache routing
def image_cache_path(kind: str, rec: dict) -> Path:
    rel = Path(rec["image_path"]).relative_to(E7.LD.DATA_ROOT)
    if kind == "retinex":
        return E7.CACHE_ROOT / "retinex" / rel
    if kind == "l0_256":
        return E8.CACHE_ROOT / "l0_256" / rel
    return E9_CACHE_ROOT / kind / rel


def reroute(records: list, kind: str | None) -> list:
    if kind is None:
        return [dict(r) for r in records]  # feature-level: consume L0_256 (same as P00)
    return [dict(r, image_path=str(image_cache_path(kind, r))) for r in records]
