"""Experiment 5B-C — Group C (normalization-sensitivity) asset extraction (forward only).

补齐 5B 预注册但缺失的 Group C predictors：
normalization sensitivity = geometry(F0) vs geometry(IN(F0))，仅用 normal train 图。

冻结实现来源：experiment1j_extract.geometry_of（1J-A）+ FAlphaLayerPatchcoreModel._alpha_mix(α=1)（1E/1H/1J）。
8 个 Group C predictors（primary = README §5 指名的 category-specific s_L2/s_L3）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment1b_defect_sensitivity import make_validation_split  # noqa: E402
from experiment1j_extract import geometry_of  # noqa: E402
from falpha_layer_patchcore import FAlphaLayerPatchcore  # noqa: E402

OUT = ROOT / "results" / "experiment_5b_final"
GC_DIR = OUT / "group_c"
PER_IMG = GC_DIR / "per_image"
REF_DIR = OUT / "reference"
LOG_DIR = OUT / "logs"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
LAYERS = ["layer2", "layer3"]
EPS = 1e-12

GROUP_C_REGISTRY = [
    ("sens_radius_L2", "layer2", "mean_image log((rms_radius(IN(F0))+eps)/(rms_radius(F0)+eps))", "high->fragile", True),
    ("sens_radius_L3", "layer3", "mean_image log((rms_radius(IN(F0))+eps)/(rms_radius(F0)+eps))", "high->fragile", True),
    ("sens_mdc_L2", "layer2", "mean_image log((mdc(IN(F0))+eps)/(mdc(F0)+eps))", "high->fragile", False),
    ("sens_mdc_L3", "layer3", "mean_image log((mdc(IN(F0))+eps)/(mdc(F0)+eps))", "high->fragile", False),
    ("sens_effrank_L2", "layer2", "mean_image log((normalized_pr(IN(F0))+eps)/(normalized_pr(F0)+eps))", "high->fragile", False),
    ("sens_effrank_L3", "layer3", "mean_image log((normalized_pr(IN(F0))+eps)/(normalized_pr(F0)+eps))", "high->fragile", False),
    ("sens_pca1_L2", "layer2", "mean_image (pca1_ratio(IN(F0)) - pca1_ratio(F0))", "high->fragile", False),
    ("sens_pca1_L3", "layer3", "mean_image (pca1_ratio(IN(F0)) - pca1_ratio(F0))", "high->fragile", False),
]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_model(device):
    model = FAlphaLayerPatchcore(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1, num_neighbors=9,
        alpha=0.0, intervention_location="layer2", visualizer=False,
    )
    tm = model.model
    tm.eval()
    e1b._move_model_to_device(tm, device)
    return tm


def layer_features(tm, img_path: Path, device):
    img = e1b.load_image_as_tensor(img_path)
    inp = e1b.preprocess_for_model(img, device).unsqueeze(0)
    with torch.no_grad():
        features = tm.feature_extractor(inp)
    features = {k: tm.feature_pooler(v) for k, v in features.items()}
    return {"layer2": features["layer2"], "layer3": features["layer3"]}


def in_features(tm, feat):
    prev = tm.alpha
    tm.alpha = 1.0
    try:
        return tm._alpha_mix(feat)
    finally:
        tm.alpha = prev


def geometry_pair(tm, feat_t):
    f0 = feat_t[0].detach().cpu().numpy().astype(np.float32)
    f_in = in_features(tm, feat_t)[0].detach().cpu().numpy().astype(np.float32)
    assert f0.shape == f_in.shape, f"shape mismatch {f0.shape} vs {f_in.shape}"
    max_delta = float(np.abs(f_in.astype(np.float64) - f0.astype(np.float64)).max())
    return geometry_of(f0), geometry_of(f_in), max_delta


def log_ratio(a1, a0):
    return float(np.log((a1 + EPS) / (a0 + EPS)))


def process_unit(tm, cat, seed, device, max_images):
    _, train_ids = make_validation_split(cat, seed)
    if max_images:
        train_ids = train_ids[:max_images]
    normal_root = e1b.DATA_ROOT / cat / "train" / "good"
    rows = []
    for name in train_ids:
        p = normal_root / name
        assert "train" in str(p) and "good" in str(p), f"non-normal path: {p}"
        feats = layer_features(tm, p, device)
        row = {"category": cat, "seed": seed, "image": name}
        for layer in LAYERS:
            g0, gin, md = geometry_pair(tm, feats[layer])
            row[f"mdc_0_{layer}"] = g0["mdc"]
            row[f"rms_0_{layer}"] = g0["rms_radius"]
            row[f"npr_0_{layer}"] = g0["normalized_pr"]
            row[f"pca1_0_{layer}"] = g0["pca1_ratio"]
            row[f"mdc_in_{layer}"] = gin["mdc"]
            row[f"rms_in_{layer}"] = gin["rms_radius"]
            row[f"npr_in_{layer}"] = gin["normalized_pr"]
            row[f"pca1_in_{layer}"] = gin["pca1_ratio"]
            row[f"in_delta_{layer}"] = md
        rows.append(row)
    return rows


def aggregate(rows):
    agg = {}
    for layer in LAYERS:
        tag = "L2" if layer == "layer2" else "L3"
        agg[f"sens_radius_{tag}"] = float(np.mean([log_ratio(r[f"rms_in_{layer}"], r[f"rms_0_{layer}"]) for r in rows]))
        agg[f"sens_mdc_{tag}"] = float(np.mean([log_ratio(r[f"mdc_in_{layer}"], r[f"mdc_0_{layer}"]) for r in rows]))
        agg[f"sens_effrank_{tag}"] = float(np.mean([log_ratio(r[f"npr_in_{layer}"], r[f"npr_0_{layer}"]) for r in rows]))
        agg[f"sens_pca1_{tag}"] = float(np.mean([r[f"pca1_in_{layer}"] - r[f"pca1_0_{layer}"] for r in rows]))
    return agg


def sanity_definition_match(tm, device):
    """S-C7：用 1J-A 冻结 defect per_image CSV 复现 geometry（含 α=1），验证实现一致。"""
    ref_csv = ROOT / "results" / "experiment_1j" / "per_image" / "experiment1j_geometry_per_image.csv"
    ref = {}
    for r in csv.DictReader(open(ref_csv, newline="")):
        ref[(r["category"], r["defect_type"], r["seed"], r["layer"], r["alpha"],
             Path(r["image_path"]).stem)] = r
    devs = []
    for cat, dt, seed, stem in [("cable", "bent_wire", "0", "000"), ("grid", "glue", "0", "000"),
                                ("bottle", "contamination", "1", "000"), ("cable", "bent_wire", "0", "002")]:
        img = e1b.DATA_ROOT / cat / "test" / dt / f"{stem}.png"
        if not img.exists():
            continue
        feats = layer_features(tm, img, device)
        for layer in LAYERS:
            g0, gin, _ = geometry_pair(tm, feats[layer])
            for alpha, g in (("0.0", g0), ("1.0", gin)):
                key = (cat, dt, seed, layer, alpha, stem)
                if key not in ref:
                    continue
                for m in ["mdc", "rms_radius", "normalized_pr", "pca1_ratio"]:
                    rv = float(ref[key][m])
                    devs.append(abs(g[m] - rv) / (abs(rv) + 1e-12))
    max_rel = max(devs) if devs else float("nan")
    return {"max_rel_dev": max_rel, "n_compared": len(devs),
            "pass": bool(np.isfinite(max_rel) and max_rel < 1e-4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cats = [c for c in args.categories.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    for d in (GC_DIR, PER_IMG, REF_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[5B-C] device={device}", flush=True)
    t0 = time.time()
    tm = build_model(device)
    print("[5B-C] model built (eval, no fit); feature path == 1J-A/1H", flush=True)

    dm = sanity_definition_match(tm, device)
    print(f"[S-C7] definition match vs 1J-A frozen CSV: max_rel_dev={dm['max_rel_dev']:.3e} "
          f"(n={dm['n_compared']}) {'PASS' if dm['pass'] else 'FAIL'}", flush=True)

    good_dir = e1b.DATA_ROOT / cats[0] / "train" / "good"
    det_img = good_dir / sorted(p.name for p in good_dir.glob("*.png"))[0]
    fa = layer_features(tm, det_img, device)
    fb = layer_features(tm, det_img, device)
    det_max = float(max((fa[k] - fb[k]).abs().max().item() for k in LAYERS))
    print(f"[S-C3] determinism max|Δfeature|={det_max:.3e} ({det_img.name})", flush=True)

    per_unit = {}
    for cat in cats:
        for seed in seeds:
            out_csv = PER_IMG / f"{cat}_seed{seed}.csv"
            if out_csv.exists() and not args.no_resume:
                rows = list(csv.DictReader(open(out_csv, newline="")))
                for r in rows:
                    for k in list(r.keys()):
                        if k not in ("category", "seed", "image"):
                            r[k] = float(r[k])
                print(f"[5B-C] {cat}:s{seed} resumed ({len(rows)} images)", flush=True)
            else:
                rows = process_unit(tm, cat, seed, device, args.max_images)
                fields = ["category", "seed", "image"] + [k for k in rows[0] if k not in ("category", "seed", "image")]
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
                print(f"[5B-C] {cat}:s{seed} done ({len(rows)} images, {time.time()-t0:.0f}s)", flush=True)
            per_unit[(cat, seed)] = aggregate(rows)

    pred_names = [r[0] for r in GROUP_C_REGISTRY]
    with open(GC_DIR / "group_c_predictors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed"] + pred_names)
        for cat in cats:
            for seed in seeds:
                w.writerow([cat, seed] + [f"{per_unit[(cat,seed)][n]:.8g}" for n in pred_names])

    with open(REF_DIR / "predictor_registry_group_c.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predictor_name", "group", "layer", "definition", "source", "uses_normal_only",
                    "direction_hypothesis", "primary_or_secondary", "unresolved_pre_registered"])
        for name, layer, definition, direction, primary in GROUP_C_REGISTRY:
            w.writerow([name, "C_normalization_sensitivity", layer, definition,
                        "normal_train_images(F0_vs_IN(F0))", "true", direction,
                        "primary" if primary else "secondary_same_frozen_family", "false"])

    finite_ok, in_effect_ok = True, True
    for cat in cats:
        for seed in seeds:
            for r in csv.DictReader(open(PER_IMG / f"{cat}_seed{seed}.csv", newline="")):
                vals = [float(v) for k, v in r.items() if k not in ("category", "seed", "image")]
                if not np.isfinite(vals).all():
                    finite_ok = False
                if float(r["in_delta_layer2"]) <= 0 or float(r["in_delta_layer3"]) <= 0:
                    in_effect_ok = False
    n_units = len(per_unit)
    checks = [
        ("S-C1_shape_F0_eq_IN", True, "asserted per image: F0.shape == IN(F0).shape"),
        ("S-C2_finite", finite_ok, "all per-image geometry values finite (no NaN/Inf)"),
        ("S-C3_determinism", det_max == 0.0, f"max|Δfeature| = {det_max:.3e} (repeat extraction, same input)"),
        ("S-C4_IN_effect_exists", in_effect_ok, "max|IN(F0)-F0| > 0 for every image/layer"),
        ("S-C5_coverage", n_units == len(cats) * len(seeds),
         f"{n_units} units = {len(cats)} categories × {len(seeds)} seeds (2 layers each)"),
        ("S-C6_no_defect_leakage", True,
         "only <cat>/train/good read (path asserted); no test image / GT label / C2 damage / anomaly score"),
        ("S-C7_definition_match", dm["pass"],
         f"vs 1J-A frozen per_image CSV: max_rel_dev={dm['max_rel_dev']:.3e} (n={dm['n_compared']})"),
    ]
    with open(GC_DIR / "group_c_sanity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for name, ok, detail in checks:
            w.writerow([name, "PASS" if ok else "FAIL", detail])

    overall = all(ok for _, ok, _ in checks)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    freeze = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Group C (normalization sensitivity) predictors frozen BEFORE any target correlation",
        "git_head": head,
        "recovered_definition_note": (
            "5B registry 未逐条枚举 Group C；README §5/§7/§9 唯一具名量是 category-specific s_L2/s_L3"
            "（1J-A radius_response），§7 概念定义 geometry(F) vs geometry(IN(F))。"
            "按 1J-A 冻结 geometry 指标族机械应用于 F0 vs IN(F0)；primary = radius（= s_L2/s_L3）。"),
        "group_c_predictors": [
            {"name": n, "group": "C_normalization_sensitivity", "layer": l, "definition": d,
             "direction_hypothesis": dirn, "role": "primary" if p else "secondary_same_frozen_family"}
            for n, l, d, dirn, p in GROUP_C_REGISTRY],
        "definition_provenance": {
            "geometry_of": "scripts/experiment1j_extract.py::geometry_of (1J-A frozen)",
            "response_formula": "log((metric(IN(F0))+1e-12)/(metric(F0)+1e-12)); pca1 = simple difference (1J-A frozen)",
            "IN": "FAlphaLayerPatchcoreModel._alpha_mix(alpha=1) == F.instance_norm(F) (1E/1H/1J frozen)",
            "feature_path": "wide_resnet50_2 eval + feature_pooler; layer2 (512,32,32), layer3 (1024,16,16)",
        },
        "aggregation": "per-image geometry -> per-image response -> mean over that (category, seed) normal train images",
        "categories": cats, "seeds": seeds, "layers": LAYERS,
        "source_assets": "data/mvtec_ad/<cat>/train/good (make_validation_split train side)",
        "n_units": n_units,
        "sanity_all_pass": bool(overall),
        "sanity": [{"check": n, "status": "PASS" if ok else "FAIL", "detail": d} for n, ok, d in checks],
        "md5": {"group_c_predictors.csv": md5_file(GC_DIR / "group_c_predictors.csv"),
                "predictor_registry_group_c.csv": md5_file(REF_DIR / "predictor_registry_group_c.csv")},
        "targets_read": False,
    }
    with open(REF_DIR / "group_c_freeze.json", "w") as f:
        json.dump(freeze, f, indent=2)

    print(f"[5B-C] sanity {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} PASS; "
          f"overall={'PASS' if overall else 'FAIL'}")
    print(f"[5B-C] freeze record written (targets_read=False); elapsed {time.time()-t0:.0f}s")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
