"""Experiment 9B — Second Method Screening (M7-M11) · Round-0 runner。

复用 `experiment5a_h_runner.run_config` 的完整评估路径（bank / coreset 0.1 / kNN 9 /
4 个冻结光照偏移 / scoring / per_image.csv），只把 `fit_dual_model` monkey-patch 成 9B
的候选构造器 —— 与 7A-O / 8B / 9A 完全同一注入方式。

候选（冻结，见 results/experiment_9b_screening/config.json）：
  M7  M7_L2a{L2}_L3a{L3}       逐层 α-IN，α_L2 != α_L3（6 configs）
  M8  M8_gate_ab025_b{050,100} train-only s_c 的 z-map 软门，a_bar=0.25
  M9  M9_sw_gamma{1,2}         train-only s_c 的 p90-power 软门
  M11 M11_L2heavy_b{050,100}   (a_bar_L2, a_bar_L3) = (0.5, 0.0) + 逐层 channel 门
  M10 = REUSE（7A-O Family B dual_step），本 runner 不跑
  smoke-only: SMOKE_orig_a000 / SMOKE_gate_const025（等价性校验，非候选）

用法：
  python -u scripts/experiment9b_runner.py --round round0 --worker-tag w0 --units "bottle:0:M7_L2a025_L3a000,cable:0:M8_gate_ab025_b050"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment5a_h_runner as h  # noqa: E402
import experiment9b_model as m9b  # noqa: E402
import experiment_progress as ep  # noqa: E402
from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel  # noqa: E402
from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

EXP_ROOT = ROOT / "results" / "experiment_9b_screening"
CACHE_DIR = EXP_ROOT / "cache"
LOG_DIR = EXP_ROOT / "logs"
REF_DIR = EXP_ROOT / "reference"

# ---- 冻结的候选定义（参数在 config.json 中；此处不得引入未冻结取值）----
def _z(a_bar, beta):
    return ("m8_zmap", {"a_bar": a_bar, "beta": beta})


def _p(gamma):
    return ("m9_powmap", {"gamma": gamma})


# gate 类配置没有单一标量 α；沿用 7A-O 的既有约定：alpha_l2 = alpha_l3 = -1.0（哨兵，
# 表示「非 uniform-α 干预」），真实门统计记录在 module_meta.json 的 gate_mean_l2/l3。
GATE_ALPHA_SENTINEL = -1.0

CONFIGS = {
    # M7 — Layer-Selective Normalization
    "M7_L2a000_L3a025": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.25},
    "M7_L2a000_L3a050": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.5},
    "M7_L2a025_L3a000": {"kind": "alpha", "alpha_l2": 0.25, "alpha_l3": 0.0},
    "M7_L2a025_L3a050": {"kind": "alpha", "alpha_l2": 0.25, "alpha_l3": 0.5},
    "M7_L2a050_L3a000": {"kind": "alpha", "alpha_l2": 0.5, "alpha_l3": 0.0},
    "M7_L2a050_L3a025": {"kind": "alpha", "alpha_l2": 0.5, "alpha_l3": 0.25},
    # M8 — Normal-Only Channel Gate（z-map, a_bar = 0.25 = 6A/6B 历史 knee）
    "M8_gate_ab025_b050": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _z(0.25, 0.5), "gate_l3": _z(0.25, 0.5)},
    "M8_gate_ab025_b100": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _z(0.25, 1.0), "gate_l3": _z(0.25, 1.0)},
    # M9 — Soft Channel Weighting（p90-power map）
    "M9_sw_gamma1": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _p(1.0), "gate_l3": _p(1.0)},
    "M9_sw_gamma2": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _p(2.0), "gate_l3": _p(2.0)},
    # M11 — Layer x Channel Gate
    "M11_L2heavy_b050": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _z(0.5, 0.5), "gate_l3": _z(0.0, 0.5)},
    "M11_L2heavy_b100": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL, "alpha_l3": GATE_ALPHA_SENTINEL, "gate_l2": _z(0.5, 1.0), "gate_l3": _z(0.0, 1.0)},
    # smoke-only（不做候选判定）
    "SMOKE_orig_a000": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.0, "smoke": True},
    # 同 runner 内的 α 对照（P3 sanity 扩展，非候选）：用于判别 constant-gate vs α-IN
    "SMOKE_uniform_a025": {"kind": "alpha", "alpha_l2": 0.25, "alpha_l3": 0.25, "smoke": True},
    "SMOKE_gate_const025": {"kind": "gate", "alpha_l2": GATE_ALPHA_SENTINEL,
                            "alpha_l3": GATE_ALPHA_SENTINEL,
                            "gate_l2": ("const", {"value": 0.25}),
                            "gate_l3": ("const", {"value": 0.25}), "smoke": True},
}

CANDIDATES = [k for k, v in CONFIGS.items() if not v.get("smoke")]
CAPTURED: dict = {}


def unit_id(category: str, seed: int, name: str) -> str:
    return f"{category}:{seed}:{name}"


# ---------------------------------------------------------------------------
# P3 sanity（CPU，GPU 之前）
# ---------------------------------------------------------------------------
def pre_run_sanity(units, round_name: str) -> dict:
    import experiment1b_defect_sensitivity as e1b

    res = {}
    res["S1_cuda"] = (torch.cuda.is_available(), f"cuda={torch.cuda.is_available()}")
    res["S2_output_isolated"] = ("experiment_9b_screening" in str(EXP_ROOT),
                                f"out root = {EXP_ROOT.relative_to(ROOT)}")
    res["S3_configs_known"] = (all(u[2] in CONFIGS for u in units),
                              f"configs={sorted({u[2] for u in units})}")
    res["S4_no_uniform_M7_in_grid"] = (
        all(not (c.startswith("M7") and CONFIGS[c]["alpha_l2"] == CONFIGS[c]["alpha_l3"])
            for c in CONFIGS), "M7 grid 不含 α_L2 == α_L3（uniform 被排除）")

    # ---- 用真实图像提取 feature，验证 gate 路径 ----
    val, train = e1b.make_validation_split("bottle", 0)
    img = e1b.load_image_as_tensor(ROOT / "data" / "mvtec_ad" / "bottle" / "train" / "good" / train[0])
    x = e1b.preprocess_for_model(img, torch.device("cpu")).unsqueeze(0)
    kw = dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
              num_neighbors=9)
    with torch.no_grad():
        base = PatchcoreModel(**kw).eval()
        feats = base.feature_extractor(x)
        feats = {k: base.feature_pooler(v) for k, v in feats.items()}
        e_base = base.generate_embedding({k: v.clone() for k, v in feats.items()})

        m_alpha0 = FAlphaDualLayerPatchcoreModel(**kw, alpha_l2=0.0, alpha_l3=0.0).eval()
        e_alpha0 = m_alpha0.generate_embedding({k: v.clone() for k, v in feats.items()})

        m_gate0 = m9b.ChannelGatePatchcoreModel(**kw, gate_l2=None, gate_l3=None).eval()
        e_gate_none = m_gate0.generate_embedding({k: v.clone() for k, v in feats.items()})
        m_gatez = m9b.ChannelGatePatchcoreModel(
            **kw, gate_l2=np.zeros(512, np.float32), gate_l3=np.zeros(1024, np.float32)).eval()
        e_gate_zero = m_gatez.generate_embedding({k: v.clone() for k, v in feats.items()})

        m_alpha25 = FAlphaDualLayerPatchcoreModel(**kw, alpha_l2=0.25, alpha_l3=0.25).eval()
        e_alpha25 = m_alpha25.generate_embedding({k: v.clone() for k, v in feats.items()})
        g25, _ = m9b.make_gate(np.zeros(512, np.float32), "const", value=0.25)
        g25b, _ = m9b.make_gate(np.zeros(1024, np.float32), "const", value=0.25)
        m_gate25 = m9b.ChannelGatePatchcoreModel(**kw, gate_l2=g25, gate_l3=g25b).eval()
        e_gate25 = m_gate25.generate_embedding({k: v.clone() for k, v in feats.items()})

    d1 = float((e_base - e_alpha0).abs().max())
    d2 = float((e_base - e_gate_none).abs().max())
    d3 = float((e_base - e_gate_zero).abs().max())
    d4 = float((e_alpha25 - e_gate25).abs().max())
    res["S5_gate0_equals_original"] = (max(d1, d2, d3) < 1e-6,
                                       f"max|d|: alpha0={d1:.2e} gateNone={d2:.2e} gateZero={d3:.2e}")
    res["S6_const_gate_equals_uniform_alpha"] = (d4 < 1e-6,
                                                 f"const gate 0.25 vs alpha=0.25: max|d|={d4:.2e}")
    res["S7_embedding_shape"] = (tuple(e_base.shape) == (1, 1536, 32, 32),
                                 f"embedding shape = {tuple(e_base.shape)}")
    res["S8_no_nan_inf_embedding"] = (bool(torch.isfinite(e_base).all() and torch.isfinite(e_gate25).all()),
                                      "embeddings finite")
    # gate 结构：layer3 gate 只作用于 layer3 块（在 upsample 之前）
    g3 = np.zeros(1024, np.float32)
    g3[:] = 0.5
    m_g3 = m9b.ChannelGatePatchcoreModel(**kw, gate_l2=np.zeros(512, np.float32), gate_l3=g3).eval()
    with torch.no_grad():
        e_g3 = m_g3.generate_embedding({k: v.clone() for k, v in feats.items()})
    l2_block_same = float((e_g3[:, :512] - e_base[:, :512]).abs().max())
    l3_block_diff = float((e_g3[:, 512:] - e_base[:, 512:]).abs().max())
    res["S9_layer_gate_isolation"] = (l2_block_same == 0.0 and l3_block_diff > 0,
                                      f"L2 block Δ={l2_block_same:.2e}, L3 block Δ={l3_block_diff:.2e}")
    # gate 范围合法
    s_dummy = np.abs(np.random.RandomState(0).randn(512)) + 0.1
    grange = True
    for mp, kws in (("m8_zmap", dict(a_bar=0.25, beta=1.0)), ("m8_zmap", dict(a_bar=0.5, beta=1.0)),
                    ("m9_powmap", dict(gamma=2.0))):
        a, _ = m9b.make_gate(s_dummy, mp, **kws)
        grange = grange and bool(a.min() >= 0.0 and a.max() <= 1.0)
    res["S10_gate_range_legal"] = (grange, "all frozen gate maps produce values in [0, 1]")

    # S11: s_c cache 非退化（per-channel std > 0）+ 只来自 train/good
    import experiment9b_model as _m
    import numpy as _np
    s11_ok, s11_det = True, []
    for cat, seed, _ in units:
        _val, _tr = e1b.make_validation_split(cat, seed)
        _h = _m.sensitivity_hash(cat, seed, _tr)
        _npz = CACHE_DIR / f"illum_sensitivity_{cat}_seed{seed}_{_h}.npz"
        if not _npz.exists():
            s11_det.append(f"{cat}: cache missing (will estimate, train-only)")
            continue
        _z = _np.load(_npz)
        _js = json.loads((CACHE_DIR / f"illum_sensitivity_{cat}_seed{seed}_{_h}.json").read_text())
        for ln, key in (("layer2", "s2"), ("layer3", "s3")):
            sd = float(_z[key].std())
            if sd <= 0.0:
                s11_ok = False
            s11_det.append(f"{cat}/{ln}: std={sd:.4g} n_train={_js['n_train']} src={_js['source'][:18]}")
    res["S11_sensitivity_non_degenerate_train_only"] = (s11_ok, "; ".join(s11_det)[:300])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"sanity_pre_{round_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (okk, d) in res.items():
            w.writerow([k, "PASS" if okk else "FAIL", d])
    return res


# ---------------------------------------------------------------------------
# fit：与 5A-H / 7A-O / 9A 同构，只换模型类
# ---------------------------------------------------------------------------
def patched_fit_dual_model(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir):
    name = CAPTURED["config"]
    device = CAPTURED["device"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    datamodule = MVTecAD(root=str(h.DATA_ROOT), category=category,
                         train_batch_size=16, eval_batch_size=16, num_workers=0, seed=seed)
    datamodule.setup()
    keep = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    cfg = CONFIGS[name]
    common = dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                  coreset_sampling_ratio=0.1, num_neighbors=9, visualizer=False)
    t0 = time.time()
    if cfg["kind"] == "gate":
        # s_c 只用 train/good 估计（落盘复用；每个 (cat, seed) 一次）
        s, meta = m9b.load_or_estimate_sensitivity(CACHE_DIR, category, seed, train_ids,
                                                   device, h.DATA_ROOT)
        CAPTURED["sensitivity_meta"] = {k: meta[k] for k in
                                        ("category", "seed", "hash", "n_train", "reused", "source")}
        CAPTURED["sensitivity_stats"] = meta["stats"]
        gates, gstats = {}, {}
        for lname, key in (("layer2", "gate_l2"), ("layer3", "gate_l3")):
            map_name, kws = cfg[key]
            a, st = m9b.make_gate(s[lname], map_name, **kws)
            gates[lname] = a
            gstats[lname] = st
        CAPTURED["gate_stats"] = gstats
        CAPTURED["gate_mean_l2"] = float(gates["layer2"].mean())
        CAPTURED["gate_mean_l3"] = float(gates["layer3"].mean())
        model = m9b.ChannelGatePatchcore(**common, gate_l2=gates["layer2"], gate_l3=gates["layer3"])
    else:
        model = FAlphaDualLayerPatchcore(**common, alpha_l2=alpha_l2, alpha_l3=alpha_l3)
    CAPTURED["prep_seconds"] = round(time.time() - t0, 2)

    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    default_root_dir=str(logs_dir))
    engine.fit(model=model, datamodule=datamodule)
    tm = model.model
    tm.eval()
    try:
        CAPTURED["embed_dim"] = int(tm.memory_bank.shape[1])
    except Exception:
        CAPTURED["embed_dim"] = -1
    return model, tm


h.fit_dual_model = patched_fit_dual_model


def write_progress_json(round_name: str) -> dict:
    shards, alive = ep.load_shards(EXP_ROOT, round_name)
    est = ep.estimate(shards, alive_workers=alive)
    out = {"round": round_name, "experiment": "9B",
           "completed": est["completed"], "total": est["total"],
           "remaining": est["remaining"], "failed": est["failed"],
           "current_unit": est["last_unit"], "workers_active": est["workers_active"],
           "elapsed_seconds": round(est["elapsed"], 1),
           "avg_unit_seconds": est["avg_unit_seconds"],
           "eta_seconds": est["eta_seconds"], "eta_basis": est["eta_basis"],
           "throughput_units_per_min": est["throughput_units_per_min"],
           "estimated_finish_time": (time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(est["estimated_finish"]))
                                     if est["estimated_finish"] else None),
           "per_worker": [{"worker": j.get("worker"), "pid": j.get("pid"),
                           "done": len(j.get("units", [])), "of": j.get("units_total"),
                           "current": j.get("current")} for j in shards],
           "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}
    (EXP_ROOT / "progress.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    import experiment1b_defect_sensitivity as e1b

    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 category:seed:CONFIG")
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    args = ap.parse_args()

    units = []
    for tok in args.units.split(","):
        c, s, n = tok.strip().split(":")
        units.append((c, int(s), n))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{args.worker_tag}] PID={os.getpid()} round={args.round_name} units={units}", flush=True)

    sres = pre_run_sanity(units, f"{args.round_name}_{args.worker_tag}")
    for k, (okk, d) in sres.items():
        print(f"[sanity {args.worker_tag}] {k}: {'PASS' if okk else 'FAIL'} ({d})", flush=True)
    fails = [k for k, v in sres.items() if not v[0]]
    if fails:
        raise SystemExit(f"[{args.worker_tag}] pre-run sanity FAILED: {fails} — STOP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (LOG_DIR / f"runner_pid_{args.round_name}_{args.worker_tag}.txt").write_text(str(os.getpid()))

    uid_list = [unit_id(*u) for u in units]
    prog = ep.Progress(EXP_ROOT, args.round_name, uid_list, worker=args.worker_tag)
    write_progress_json(args.round_name)

    for category, seed, name in units:
        uid = unit_id(category, seed, name)
        cfg = dict(CONFIGS[name])
        cfg["name"] = name
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20, f"{category} val != 20"
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((h.DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((h.DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = h.expected_rows(category, good_paths, defect_paths)
        cfg_dir = EXP_ROOT / "raw" / args.round_name / category / f"seed_{seed}" / f"config_{name}"
        print(f"\n[{args.worker_tag}] unit {uid} rows={exp_rows} train={len(train_ids)}", flush=True)

        if h.unit_done(cfg_dir, exp_rows):
            info = json.loads((cfg_dir / "info.json").read_text())
            print(f"[{uid}] resume skip (already OK)", flush=True)
            prog.unit_end(uid, float(info.get("runtime_seconds") or 0.0), status="ok",
                          detail="resume-skip")
            write_progress_json(args.round_name)
            continue

        cfg_dir.mkdir(parents=True, exist_ok=True)
        lock = cfg_dir / "run.lock"
        lock.write_text(str(os.getpid()))
        CAPTURED.clear()
        CAPTURED.update({"config": name, "device": device, "round": args.round_name})
        prog.unit_start(uid)
        write_progress_json(args.round_name)
        t0 = time.time()
        status, detail = "ok", ""
        try:
            h.run_config(cfg, category, seed, val_ids, train_ids,
                         good_paths, defect_paths, device, cfg_dir)
        except Exception as exc:
            status = "failed"
            detail = f"{type(exc).__name__}: {exc}"
            (cfg_dir / "info.json").write_text(json.dumps({"status": f"FAILED: {detail}"}))
            print(f"[{uid}] FAILED {detail}", flush=True)
        finally:
            lock.unlink(missing_ok=True)

        info = {}
        if (cfg_dir / "info.json").exists():
            try:
                info = json.loads((cfg_dir / "info.json").read_text())
            except Exception:
                info = {}
        meta = {"experiment": "9B", "unit": uid, "config": name, "round": args.round_name,
                "candidate": name in CANDIDATES,
                "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
                         if k not in ("name",)},
                "gate_stats": CAPTURED.get("gate_stats"),
                "sensitivity_meta": CAPTURED.get("sensitivity_meta"),
                "gate_mean_l2": CAPTURED.get("gate_mean_l2"),
                "gate_mean_l3": CAPTURED.get("gate_mean_l3"),
                "embedding_dim": CAPTURED.get("embed_dim", -1),
                "prep_seconds": CAPTURED.get("prep_seconds"),
                "memory_bank_size": info.get("coreset_size"),
                "runtime_seconds": info.get("runtime_seconds"),
                "wallclock_seconds": round(time.time() - t0, 2),
                "peak_gpu_memory_allocated_mb": info.get("peak_gpu_memory_allocated_mb"),
                "declared_alpha_l2": cfg.get("alpha_l2"), "declared_alpha_l3": cfg.get("alpha_l3"),
                "status": status, "detail": detail}
        (cfg_dir / "module_meta.json").write_text(json.dumps(meta, indent=2))
        prog.unit_end(uid, time.time() - t0, status=status, detail=detail)
        write_progress_json(args.round_name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prog.close()
    write_progress_json(args.round_name)
    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
