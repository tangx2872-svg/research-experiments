"""Experiment 8B — P0 asset audit (CPU-only, produces no target result).

A2: 1J normal banks (15 cat-seed x 2 layers): shapes + per-channel sigma distribution
A3: 1J defect-side raw features (npz) coverage + shapes
A4: illumination-side feature-level assets (score-level does not count)
A5: GT mask / test coverage + mask->feature-grid defect patch coverage

Outputs results/experiment8b/audit/asset_audit.{json,md}
Usage: python -u scripts/experiment8b_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
OUT = ROOT / "results" / "experiment8b" / "audit"
BANKS = ROOT / "results" / "experiment_1j" / "banks"
RAW_1J = ROOT / "results" / "experiment_1j" / "raw"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
LAYERS = {"layer2": 512, "layer3": 1024}
FROZEN_PRIMARY = {
    ("cable", "bent_wire"), ("hazelnut", "print"), ("bottle", "contamination"),
    ("screw", "manipulated_front"), ("screw", "thread_side"), ("screw", "thread_top"),
    ("grid", "glue"), ("grid", "metal_contamination"), ("grid", "thread"),
}
OVERLAP_DEFECT = 0.5


def load_mask_256(mask_path: Path) -> np.ndarray:
    m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if m.shape != (256, 256):
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def mask_overlap_grid(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        h0, h1 = int(i * h / gh), int((i + 1) * h / gh)
        for j in range(gw):
            w0, w1 = int(j * w / gw), int((j + 1) * w / gw)
            out[i, j] = float(mask[h0:h1, w0:w1].mean())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    bank_info, sigmas = {}, []
    for cat in CATEGORIES:
        for seed in SEEDS:
            for layer, cdim in LAYERS.items():
                p = BANKS / f"{cat}_seed{seed}_{layer}.npy"
                if not p.exists():
                    bank_info[f"{cat}_seed{seed}_{layer}"] = "MISSING"
                    continue
                b = np.load(p)
                assert b.shape[1] == cdim, (p, b.shape)
                sd = b.astype(np.float64).std(axis=0, ddof=1)
                sigmas.append(sd)
                bank_info[f"{cat}_seed{seed}_{layer}"] = {
                    "shape": list(b.shape), "dtype": str(b.dtype),
                    "sigma_min": float(sd.min()), "sigma_median": float(np.median(sd)),
                    "sigma_max": float(sd.max()),
                    "n_ch_sigma_lt_1e-3": int((sd < 1e-3).sum()),
                    "n_ch_sigma_lt_1e-6": int((sd < 1e-6).sum()),
                }
    all_sd = np.concatenate(sigmas) if sigmas else np.array([])
    res["normal_banks"] = {
        "n_assets_expected": 30,
        "n_assets_found": sum(1 for v in bank_info.values() if v != "MISSING"),
        "per_asset": bank_info,
        "sigma_pooled": {
            "n": int(all_sd.size),
            "min": float(all_sd.min()) if all_sd.size else None,
            "p01": float(np.percentile(all_sd, 1)) if all_sd.size else None,
            "median": float(np.median(all_sd)) if all_sd.size else None,
            "max": float(all_sd.max()) if all_sd.size else None,
            "n_lt_1e-3": int((all_sd < 1e-3).sum()) if all_sd.size else 0,
            "n_lt_1e-6": int((all_sd < 1e-6).sum()) if all_sd.size else 0,
        },
    }

    npz_by_cell, shape_check = {}, {}
    for cat in CATEGORIES:
        for seed in SEEDS:
            for layer, cdim in LAYERS.items():
                n = 0
                for dt_dir in sorted((RAW_1J / cat).glob("*")):
                    d = dt_dir / f"seed{seed}" / layer
                    if d.exists():
                        n += len(list(d.glob("*.npz")))
                npz_by_cell[f"{cat}_seed{seed}_{layer}"] = n
                cand = sorted((RAW_1J / cat).glob(f"*/seed{seed}/{layer}/*.npz"))
                if cand:
                    a = np.load(cand[0])["feat"]
                    shape_check[f"{cat}_seed{seed}_{layer}"] = [list(a.shape), str(a.dtype)]
                    assert a.shape[0] == cdim, (cand[0], a.shape)
    a0 = a1 = 0
    for p in RAW_1J.rglob("*.npz"):
        a0 += p.stem.endswith("_a0")
        a1 += p.stem.endswith("_a1")
    res["defect_side_1j"] = {
        "n_npz_found": int(sum(npz_by_cell.values())), "per_cell": npz_by_cell,
        "shape_samples": shape_check, "alpha_a0": a0, "alpha_a1": a1,
    }

    feats = [p for p in (ROOT / "results").rglob("*.npy")] + [p for p in (ROOT / "results").rglob("*.npz")]
    res["illumination_side"] = {
        "all_feature_files_in_results": len(feats),
        "feature_files_outside_1j": [str(p.relative_to(ROOT)) for p in feats
                                     if "experiment_1j" not in str(p)][:20],
        "q3_has_npy_or_npz": bool(list((ROOT / "results/experiment_7a_o_q3").rglob("*.npy"))
                                  or list((ROOT / "results/experiment_7a_o_q3").rglob("*.npz"))),
        "q4_has_npy_or_npz": bool(list((ROOT / "results/experiment_7a_o_q4").rglob("*.npy"))
                                  or list((ROOT / "results/experiment_7a_o_q4").rglob("*.npz"))),
        "q0_has_npy_or_npz": bool(list((ROOT / "results/experiment_7a_o").rglob("*.npy"))
                                  or list((ROOT / "results/experiment_7a_o").rglob("*.npz"))),
        "verdict": ("NO feature-level illumination asset: brightness/gamma perturbation only stored as "
                    "per-image anomaly scores in 7A-O Q3/Q4; no paired feature representation exists"),
    }

    cov = {}
    for cat in CATEGORIES:
        dts = e1b.discover_defect_types(cat)
        entry = {"defect_types": {},
                 "n_test_good": len(list((DATA_ROOT / cat / "test" / "good").glob("*.png")))}
        for dt in dts:
            imgs = sorted((DATA_ROOT / cat / "test" / dt).glob("*.png"))
            n_mask_ok = n_empty = ov2_zero = ov3_zero = nb = 0
            for p in imgs:
                mp = DATA_ROOT / cat / "ground_truth" / dt / f"{p.stem}_mask.png"
                if not mp.exists():
                    continue
                m = load_mask_256(mp)
                if m.sum() == 0:
                    n_empty += 1
                    continue
                n_mask_ok += 1
                if (cat, dt) in FROZEN_PRIMARY:
                    if (mask_overlap_grid(m, 32, 32) > OVERLAP_DEFECT).sum() == 0:
                        ov2_zero += 1
                    if (mask_overlap_grid(m, 16, 16) > OVERLAP_DEFECT).sum() == 0:
                        ov3_zero += 1
                    nb += 1
            entry["defect_types"][dt] = {
                "n_images": len(imgs), "n_mask_present": n_mask_ok, "n_empty_mask": n_empty,
                "frozen_primary": (cat, dt) in FROZEN_PRIMARY, "n_primary_checked": nb,
                "n_l2_zero_defect_patch": ov2_zero, "n_l3_zero_defect_patch": ov3_zero,
            }
        cov[cat] = entry
    res["data_coverage"] = cov

    (OUT / "asset_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    sp = res["normal_banks"]["sigma_pooled"]
    lines = ["# Experiment 8B — P0 asset audit", "",
             f"- normal bank assets: {res['normal_banks']['n_assets_found']}/30"]
    lines.append("- per-channel sigma (30 banks, {} ch): min={:.3g} p01={:.3g} median={:.3g} max={:.3g}; "
                 "n(<1e-3)={} n(<1e-6)={}".format(
                     sp["n"], sp["min"], sp["p01"], sp["median"], sp["max"],
                     sp["n_lt_1e-3"], sp["n_lt_1e-6"]))
    lines.append(f"- 1J defect-side npz: {res['defect_side_1j']['n_npz_found']} (a0={a0}, a1={a1})")
    lines.append(f"- illumination-side: {res['illumination_side']['verdict']}")
    lines.append(f"- feature files in results outside 1J: {res['illumination_side']['feature_files_outside_1j']}")
    lines += ["", "## frozen primary 9 defects: images with zero defect patch (overlap>0.5)"]
    for cat, e in cov.items():
        for dt, d in e["defect_types"].items():
            if d["frozen_primary"]:
                lines.append(f"- {cat}/{dt}: imgs={d['n_images']} mask_ok={d['n_mask_present']} "
                             f"empty_mask={d['n_empty_mask']} l2_zero={d['n_l2_zero_defect_patch']} "
                             f"l3_zero={d['n_l3_zero_defect_patch']}")
    lines += ["", "## test coverage"]
    for cat, e in cov.items():
        lines.append(f"- {cat}: good={e['n_test_good']} "
                     f"defects={[(k, v['n_images']) for k, v in e['defect_types'].items()]}")
    (OUT / "asset_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[audit] written to {OUT}")


if __name__ == "__main__":
    main()
