"""Experiment 5A — Geometry-Guided Layer-wise Normalization：正式 runner。

11 个 configuration（协议冻结）：
  B0–B4  uniform per-layer α ∈ {0, 0.25, 0.5, 0.75, 1.0}
  G1–G3  geometry-guided (α_L2=0.604·A, α_L3=A), A ∈ {0.5, 0.75, 1.0}
  C1–C3  mean-α control（uniform，ᾱ = 对应 G 的 simple mean α）

每 config（全部 seed=0）：
  1. fit：FAlphaDualLayerPatchcore，189 张 train good（1B split），coreset 0.1，k=9
  2. τ_val = 20 张 validation good 分数 max（1B 协议，禁止用 test 定阈值）
  3. clean test：20 good + 63 defect（broken_large/broken_small/contamination）
  4. shifted：20 good × 4 photometric shifts（apply_photometric，[0,1] 空间，
     ImageNet normalize 之前）
  每 config 完成立即持久化（status.json + per_image CSV append），支持断点续跑。

Sanity（运行前注册，见 experiments/experiment5a/README.md §6）：
  S1 dual(0,0) vs 原始 PatchcoreModel embedding max_abs_diff < 1e-6（GPU 前必查）
  S2 NaN/Inf 扫描（逐 config）
  S3 rule hash 前后一致（运行前记录 md5，结束复查）
  S4 bank source：train_ids hash 逐 config 一致 + coreset size 一致
  S5 apply_photometric 数值范围 ∈[0,1]，brightness0.7(1.0)=0.7
  S6 geometry_rule.json frozen_before_run == true
  S7 config 总数 == 11
  S8 数据量：val=20 / train=189 / test good=20 / defect=20+22+21
  S10 mean-α control ᾱ 与对应 G 的 simple mean α 一致（|Δ|<1e-3）

输出（results/experiment_5a/）：
  raw/per_image_scores.csv / raw/val_scores.csv / logs/status.json
  summary/sanity_checks.csv（运行结束时写）

用法：
  python scripts/experiment5a_runner.py                  # 全部 11 个
  python scripts/experiment5a_runner.py --configs B0     # 单个（调试）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment1_illumination_tradeoff as e1  # noqa: E402  (apply_photometric 原样复用)
from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel  # noqa: E402

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
OUT_DIR = ROOT / "results" / "experiment_5a"
RAW_DIR = OUT_DIR / "raw"
SUMMARY_DIR = OUT_DIR / "summary"
LOGS_DIR = OUT_DIR / "logs"
RULE_JSON = OUT_DIR / "geometry_rule.json"

CATEGORY = "bottle"
SEED = 0
RATIO = 0.603651  # 冻结：s_L3 / s_L2（geometry_rule.json）

SHIFTS = [
    ("brightness_0.7", "brightness", 0.7),
    ("brightness_1.3", "brightness", 1.3),
    ("gamma_0.7", "gamma", 0.7),
    ("gamma_1.3", "gamma", 1.3),
]

# 11 个 configuration（协议冻结，禁止运行时改）
def build_configs() -> list[dict]:
    cfgs = [
        {"name": f"B{i}", "kind": "uniform", "alpha_l2": a, "alpha_l3": a}
        for i, a in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    for i, A in enumerate([0.5, 0.75, 1.0], start=1):
        a2 = A * RATIO
        cfgs.append({"name": f"G{i}", "kind": "guided", "alpha_l2": a2, "alpha_l3": A})
        mean_a = (a2 + A) / 2.0
        cfgs.append({"name": f"C{i}", "kind": "mean_control", "alpha_l2": mean_a, "alpha_l3": mean_a})
    return cfgs


def hash_ids(ids: list[str]) -> str:
    return hashlib.md5(",".join(sorted(ids)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sanity S1：dual(0,0) 与原始 PatchcoreModel embedding 数值等价
# ---------------------------------------------------------------------------
def sanity_alpha0_equivalence() -> tuple[bool, float]:
    val_ids, _ = e1b.make_validation_split(category=CATEGORY, seed=SEED)
    img = e1b.load_image_as_tensor(DATA_ROOT / CATEGORY / "train" / "good" / val_ids[0])
    x = e1b.preprocess_for_model(img, torch.device("cpu")).unsqueeze(0)

    base = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                          pre_trained=True, num_neighbors=9)
    dual = FAlphaDualLayerPatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                         pre_trained=True, num_neighbors=9,
                                         alpha_l2=0.0, alpha_l3=0.0)
    base.eval(); dual.eval()
    with torch.no_grad():
        feats = base.feature_extractor(x)
        e_base = base.generate_embedding(feats)
        e_dual = dual.generate_embedding(feats)
    diff = float((e_base - e_dual).abs().max().item())
    return diff < 1e-6, diff


# ---------------------------------------------------------------------------
# Sanity S5：apply_photometric 数值检查
# ---------------------------------------------------------------------------
def sanity_photometric() -> tuple[bool, list[str]]:
    msgs = []
    t = torch.tensor([[[0.0, 0.5, 1.0]]])  # (1,1,3)
    ok = True
    for name, itype, level in SHIFTS:
        r = e1.apply_photometric(t, itype, level)
        if r.min().item() < 0.0 or r.max().item() > 1.0:
            ok = False; msgs.append(f"{name}: out of [0,1] ({r.min():.3f},{r.max():.3f})")
    r = e1.apply_photometric(t, "brightness", 0.7)
    # float32 精度：|float32(0.7) − float64(0.7)| ≈ 1.2e-8，容差取 1e-6
    if abs(r[0, 0, 2].item() - 0.7) > 1e-6:
        ok = False; msgs.append(f"brightness0.7(1.0)={r[0,0,2].item():.9f} != 0.7")
    if abs(e1.apply_photometric(t, "identity", 1.0).sum().item() - 1.5) > 1e-6:
        ok = False; msgs.append("identity 不是恒等")
    if ok:
        msgs = ["photometric range/identity OK"]
    return ok, msgs


# ---------------------------------------------------------------------------
# fit（与 1H fit_layer_model 相同流程，仅模型不同）
# ---------------------------------------------------------------------------
def fit_dual_model(alpha_l2: float, alpha_l3: float, train_ids: list[str], logs_dir: Path):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    datamodule = MVTecAD(root=str(DATA_ROOT), category=CATEGORY,
                         train_batch_size=16, eval_batch_size=16, num_workers=0, seed=SEED)
    datamodule.setup()

    train_root = DATA_ROOT / CATEGORY / "train" / "good"
    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    model = FAlphaDualLayerPatchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        alpha_l2=alpha_l2,
        alpha_l3=alpha_l3,
        visualizer=False,
    )
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    default_root_dir=str(logs_dir))
    engine.fit(model=model, datamodule=datamodule)
    torch_model = model.model
    torch_model.eval()
    return model, torch_model


# ---------------------------------------------------------------------------
# 单 config 执行
# ---------------------------------------------------------------------------
def run_config(cfg: dict, val_ids: list[str], train_ids: list[str],
               good_paths: list[Path], defect_paths: dict[str, list[Path]],
               device: torch.device) -> tuple[list[dict], dict]:
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logs_dir = LOGS_DIR / f"fit_{cfg['name']}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lightning_model, torch_model = fit_dual_model(
        cfg["alpha_l2"], cfg["alpha_l3"], train_ids, logs_dir)
    e1b._move_model_to_device(torch_model, device)

    coreset_size = "not_measured"
    try:
        if hasattr(torch_model, "memory_bank") and isinstance(torch_model.memory_bank, torch.Tensor):
            coreset_size = int(torch_model.memory_bank.shape[0])
    except Exception:
        pass

    rows: list[dict] = []

    def score_img(img: torch.Tensor, subset: str, shift: str, defect_type: str, path: Path):
        s, _ = e1b.predict_one(torch_model, img, device)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite score: {cfg['name']} {path}")
        rows.append({
            "config": cfg["name"], "kind": cfg["kind"],
            "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            "subset": subset, "shift": shift, "defect_type": defect_type,
            "image_path": str(path), "score": s,
        })

    # 1) validation good → τ_val
    val_scores = []
    for vid in val_ids:
        p = DATA_ROOT / CATEGORY / "train" / "good" / vid
        img = e1b.load_image_as_tensor(p)
        s, _ = e1b.predict_one(torch_model, img, device)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite val score: {cfg['name']} {p}")
        val_scores.append(s)
        rows.append({
            "config": cfg["name"], "kind": cfg["kind"],
            "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            "subset": "val", "shift": "none", "defect_type": "good",
            "image_path": str(p), "score": s,
        })
    tau_val = float(max(val_scores))

    # 2) clean test good
    for p in good_paths:
        score_img(e1b.load_image_as_tensor(p), "clean_good", "none", "good", p)

    # 3) clean test defect
    for dt, paths in defect_paths.items():
        for p in paths:
            score_img(e1b.load_image_as_tensor(p), "clean_defect", "none", dt, p)

    # 4) shifted good（只对 good；Axis A）
    for shift_name, itype, level in SHIFTS:
        for p in good_paths:
            img = e1b.load_image_as_tensor(p)
            img_s = e1.apply_photometric(img, itype, level)
            score_img(img_s, "shift_good", shift_name, "good", p)

    peak_mb = "not_measured"
    if torch.cuda.is_available():
        try:
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    del lightning_model, torch_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    info = {
        "status": "OK", "runtime_seconds": round(time.time() - t0, 2),
        "coreset_size": coreset_size, "tau_val": tau_val,
        "peak_gpu_memory_allocated_mb": peak_mb,
        "n_rows": len(rows),
    }
    return rows, info


def append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None,
                        help="只跑指定 config（如 B0 G1）；默认全部 11 个")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_cfgs = build_configs()
    assert len(all_cfgs) == 11, "config count != 11"

    # ---------- S3：rule hash 运行前记录 ----------
    rule_md5_before = hashlib.md5(RULE_JSON.read_bytes()).hexdigest()

    # ---------- S6：frozen flag ----------
    rule = json.loads(RULE_JSON.read_text())
    frozen_ok = bool(rule.get("frozen_before_run"))

    # ---------- S10：mean-α control 与 guided simple mean ----------
    s10_msgs, s10_ok = [], True
    for i in (1, 2, 3):
        g = next(c for c in all_cfgs if c["name"] == f"G{i}")
        c = next(cc for cc in all_cfgs if cc["name"] == f"C{i}")
        gm = (g["alpha_l2"] + g["alpha_l3"]) / 2.0
        if abs(gm - c["alpha_l2"]) > 1e-9 or c["alpha_l2"] != c["alpha_l3"]:
            s10_ok = False
            s10_msgs.append(f"C{i} α={c['alpha_l2']:.6f} != G{i} mean {gm:.6f}")
        elif abs(round(gm, 3) - float(f"0.{'401' if i == 1 else '601' if i == 2 else '802'}")) > 1e-9:
            s10_ok = False
            s10_msgs.append(f"C{i} rounded α != protocol (0.401/0.601/0.802)")
    if s10_ok:
        s10_msgs = [f"C{i} == mean(G{i}) (3dp = 0.401/0.601/0.802) OK" for i in (1, 2, 3)]

    # ---------- S1：α=0 embedding 等价（GPU 前必查） ----------
    print("[sanity] S1 alpha=(0,0) embedding equivalence ...", flush=True)
    s1_ok, s1_diff = sanity_alpha0_equivalence()
    print(f"[sanity] S1 max_abs_diff = {s1_diff:.3e} -> {'PASS' if s1_ok else 'FAIL'}", flush=True)
    if not s1_ok:
        raise SystemExit("S1 FAIL：α=0 不等价，STOP（不得进入正式 5A）")

    # ---------- S5：photometric 数值 ----------
    s5_ok, s5_msgs = sanity_photometric()
    print(f"[sanity] S5 {'PASS' if s5_ok else 'FAIL'}: {s5_msgs}", flush=True)

    # ---------- S8：数据量 ----------
    val_ids, train_ids = e1b.make_validation_split(category=CATEGORY, seed=SEED)
    defect_types = e1b.discover_defect_types(CATEGORY)
    good_paths = sorted((DATA_ROOT / CATEGORY / "test" / "good").glob("*.png"))
    defect_paths = {dt: sorted((DATA_ROOT / CATEGORY / "test" / dt).glob("*.png"))
                    for dt in defect_types}
    n_defect = sum(len(v) for v in defect_paths.values())
    s8_ok = (len(val_ids) == 20 and len(train_ids) == 189 and len(good_paths) == 20
             and n_defect == 63 and sorted(defect_paths) == ["broken_large", "broken_small", "contamination"])
    print(f"[sanity] S8 val={len(val_ids)} train={len(train_ids)} good={len(good_paths)} "
          f"defect={n_defect} ({ {k: len(v) for k, v in defect_paths.items()} }) "
          f"-> {'PASS' if s8_ok else 'FAIL'}", flush=True)

    train_hash = hash_ids(train_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}", flush=True)

    # ---------- resume：读 status.json ----------
    status_path = LOGS_DIR / "status.json"
    status: dict = {}
    if status_path.exists():
        status = json.loads(status_path.read_text())

    per_image_csv = RAW_DIR / "per_image_scores.csv"
    todo = [c for c in all_cfgs
            if args.configs is None or c["name"] in args.configs]
    skipped = [c["name"] for c in todo if status.get(c["name"], {}).get("status") == "OK"]
    todo = [c for c in todo if status.get(c["name"], {}).get("status") != "OK"]
    if skipped:
        print(f"[resume] skip completed: {skipped}", flush=True)

    fields = list(build_config_row(all_cfgs[0]).keys())

    for cfg in todo:
        print(f"\n[{cfg['name']} kind={cfg['kind']} a_l2={cfg['alpha_l2']:.4f} "
              f"a_l3={cfg['alpha_l3']:.4f}] start", flush=True)
        try:
            rows, info = run_config(cfg, val_ids, train_ids, good_paths, defect_paths, device)
        except Exception as exc:  # 单 unit 失败：记录，不覆盖已完成结果
            status[cfg["name"]] = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
            status_path.write_text(json.dumps(status, indent=2))
            print(f"[{cfg['name']}] FAILED: {exc}", flush=True)
            raise
        append_csv(per_image_csv, rows, fields)
        status[cfg["name"]] = {
            "status": "OK",
            "kind": cfg["kind"], "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            **info,
            "train_ids_hash": train_hash,
        }
        status_path.write_text(json.dumps(status, indent=2))
        print(f"[{cfg['name']}] OK rows={info['n_rows']} "
              f"runtime={info['runtime_seconds']}s tau_val={info['tau_val']:.4f} "
              f"peak={info['peak_gpu_memory_allocated_mb']}MB", flush=True)

    # ---------- S3 复查 + S4 + S7 + S2 ----------
    rule_md5_after = hashlib.md5(RULE_JSON.read_bytes()).hexdigest()
    s3_ok = rule_md5_before == rule_md5_after
    s7_ok = sum(1 for k, v in status.items() if v.get("status") == "OK") == 11
    s4_ok, s4_msgs = True, []
    hashes = {v.get("train_ids_hash") for v in status.values() if v.get("status") == "OK"}
    sizes = {v.get("coreset_size") for v in status.values() if v.get("status") == "OK"}
    if len(hashes) > 1:
        s4_ok = False; s4_msgs.append(f"train_ids hash 不一致: {hashes}")
    if len(sizes) > 1:
        s4_ok = False; s4_msgs.append(f"coreset size 不一致: {sizes}")
    if s4_ok:
        s4_msgs = [f"train_ids_hash={list(hashes)} coreset_size={list(sizes)} 一致 OK"]

    # S2：全部 score 有限
    s2_ok, s2_msgs = True, ["all scores finite OK"]
    if per_image_csv.exists():
        with open(per_image_csv, newline="") as f:
            all_rows = list(csv.DictReader(f))
        bad = [r for r in all_rows if not np.isfinite(float(r["score"]))]
        if bad:
            s2_ok = False; s2_msgs = [f"{len(bad)} non-finite scores"]

    s9_ok, s9_msgs = True, []
    if (abs(rule.get("sensitivity_l2_family_level", -1) - 0.2670) > 5e-4
            or abs(rule.get("sensitivity_l3_family_level", -1) - 0.1612) > 5e-4
            or abs(rule.get("alpha_ratio_l2_over_l3", -1) - 0.6037) > 5e-4):
        s9_ok = False
        s9_msgs.append(f"rule constants mismatch: {rule.get('sensitivity_l2_family_level')}, "
                       f"{rule.get('sensitivity_l3_family_level')}, {rule.get('alpha_ratio_l2_over_l3')}")
    else:
        s9_msgs = [f"s_L2={rule['sensitivity_l2_family_level']:.4f} "
                   f"s_L3={rule['sensitivity_l3_family_level']:.4f} "
                   f"ratio={rule['alpha_ratio_l2_over_l3']:.4f} 与协议一致 OK"]
    if abs(RATIO - rule.get("alpha_ratio_l2_over_l3", -1)) > 1e-6:
        s9_ok = False
        s9_msgs.append(f"runner RATIO {RATIO} != rule ratio {rule.get('alpha_ratio_l2_over_l3')}")

    checks = [
        ("S1_alpha0_embedding_equivalence", s1_ok, f"max_abs_diff={s1_diff:.3e}"),
        ("S2_nan_inf_scan", s2_ok, "; ".join(s2_msgs)),
        ("S3_rule_hash_unchanged", s3_ok, f"{rule_md5_before[:12]}... -> {rule_md5_after[:12]}..."),
        ("S4_bank_source_consistency", s4_ok, "; ".join(s4_msgs)),
        ("S5_photometric_numeric_range", s5_ok, "; ".join(s5_msgs)),
        ("S6_rule_frozen_before_run", frozen_ok, f"frozen_before_run={frozen_ok}"),
        ("S7_config_count_11", s7_ok, f"OK configs={sum(1 for v in status.values() if v.get('status')=='OK')}"),
        ("S8_data_counts", s8_ok, "val=20 train=189 good=20 defect=63"),
        ("S9_rule_constants_match_protocol", s9_ok, "; ".join(s9_msgs)),
        ("S10_mean_alpha_control_match", s10_ok, "; ".join(s10_msgs)),
    ]
    with open(SUMMARY_DIR / "sanity_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for name, ok, detail in checks:
            w.writerow([name, "PASS" if ok else "FAIL", detail])

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n[done] sanity: {len(checks) - n_fail}/{len(checks)} PASS", flush=True)
    if n_fail:
        raise SystemExit(f"{n_fail} sanity check(s) FAILED — 结果不得进入正式分析")


def build_config_row(cfg: dict) -> dict:
    return {
        "config": cfg["name"], "kind": cfg["kind"],
        "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
        "subset": "", "shift": "", "defect_type": "", "image_path": "", "score": "",
    }


if __name__ == "__main__":
    main()
