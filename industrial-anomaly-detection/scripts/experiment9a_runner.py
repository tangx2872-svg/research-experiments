"""Experiment 9A — Method Screening v2 · Round 0/1 runner。

复用 `experiment5a_h_runner.run_config` 的**完整评估路径**（bank / coreset(0.1) /
kNN(9) / illumination (±30% brightness/gamma) / scoring / 指标 / per_image.csv），
只把 `fit_dual_model` monkey-patch 成 9A 的候选模型构造器 —— 与 7A-O / 8B 同一注入方式。

候选（每个只有一个 frozen 配置）：
  B0_original                   α_l2=α_l3=0                     （路径等价性对照）
  M1_illum_standard             per-image 光照标准化到 train 参考统计（α=0）
  M3_nuis_k16                   逐层投影掉 illumination 差分 top-16 方向（α=0）
  M4_equiv_lam050_a040091275    α=0.200456375 = (1−λ)·α', λ=0.5, α'=0.40091275
  B1_fixed_050                  α=0.5（frozen fixed-normalization baseline）

用法：
  python -u scripts/experiment9a_runner.py --round round0 --units "bottle:0:B0_original,bottle:0:M1_illum_standard" --worker-tag w0
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

import experiment9a_model as m9  # noqa: E402
import experiment5a_h_runner as h  # noqa: E402
import experiment_progress as ep  # noqa: E402
from experiment5a_model import FAlphaDualLayerPatchcore  # noqa: E402
from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

EXP_ROOT = ROOT / "results" / "experiment_9a_screening"
CACHE_DIR = EXP_ROOT / "cache"
LOG_DIR = EXP_ROOT / "logs"
REF_DIR = EXP_ROOT / "reference"

LAMBDA_M4 = 0.5                 # 冻结：7A-O Family A / 6A 项目标准 λ
ALPHA_ROBUST_M4 = 0.40091275    # 冻结：项目强 baseline（B2_uniform）
ALPHA_M4_EQUIV = (1.0 - LAMBDA_M4) * ALPHA_ROBUST_M4   # = 0.200456375

CONFIGS = {
    "B0_original": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.0},
    "B1_fixed_050": {"kind": "alpha", "alpha_l2": 0.5, "alpha_l3": 0.5},
    "M4_equiv_lam050_a040091275": {"kind": "alpha",
                                   "alpha_l2": ALPHA_M4_EQUIV, "alpha_l3": ALPHA_M4_EQUIV},
    "M1_illum_standard": {"kind": "m1", "alpha_l2": 0.0, "alpha_l3": 0.0},
    "M3_nuis_k16": {"kind": "m3", "alpha_l2": 0.0, "alpha_l3": 0.0},
    # M4 sanity 对照：字面 residual 组合形式（α'=0.40091275, λ=0.5），应与上面逐位等价
    "M4_resid_lam050_a040091275": {"kind": "m4resid",
                                   "alpha_l2": ALPHA_ROBUST_M4, "alpha_l3": ALPHA_ROBUST_M4,
                                   "lam": LAMBDA_M4},
}

CAPTURED: dict = {}


def unit_id(category: str, seed: int, name: str) -> str:
    return f"{category}:{seed}:{name}"


def pre_run_sanity(units, round_name: str) -> dict:
    res = {}
    res["S1_cuda"] = (torch.cuda.is_available(), f"cuda={torch.cuda.is_available()}")
    res["S2_output_isolated"] = ("experiment_9a_screening" in str(EXP_ROOT),
                                 f"out root = {EXP_ROOT.relative_to(ROOT)}")
    res["S3_configs_known"] = (all(u[2] in CONFIGS for u in units),
                              f"configs={sorted({u[2] for u in units})}")
    res["S4_m4_alpha_exact"] = (abs(ALPHA_M4_EQUIV - 0.200456375) < 1e-12,
                                f"(1-{LAMBDA_M4})*{ALPHA_ROBUST_M4} = {ALPHA_M4_EQUIV!r}")
    # S5: M1 闭式正确性（CPU，随机张量）
    torch.manual_seed(0)
    mu_ref, sd_ref = [0.5, 0.4, 0.3], [0.2, 0.25, 0.15]
    mdl = m9.M1IllumStandardModel.__new__(m9.M1IllumStandardModel)
    torch.nn.Module.__init__(mdl)
    mdl.register_buffer("imagenet_mean", torch.tensor(m9.IMAGENET_MEAN).view(1, 3, 1, 1))
    mdl.register_buffer("imagenet_std", torch.tensor(m9.IMAGENET_STD).view(1, 3, 1, 1))
    mdl.register_buffer("mu_ref", torch.tensor(mu_ref).view(1, 3, 1, 1))
    mdl.register_buffer("sigma_ref", torch.tensor(sd_ref).view(1, 3, 1, 1))
    x = torch.rand(2, 3, 8, 8)
    y = (x - mdl.imagenet_mean) / mdl.imagenet_std
    yp = mdl.m1_correct(y)
    xp = yp * mdl.imagenet_std + mdl.imagenet_mean          # 反归一化回 [0,1] 空间
    err_mu = float((xp.mean(dim=(2, 3)) - torch.tensor(mu_ref)).abs().max())
    err_sd = float((xp.std(dim=(2, 3), unbiased=False) -
                    torch.tensor(sd_ref).view(1, 3).to(xp.dtype)).abs().max())
    res["S5_m1_closed_form"] = (err_mu < 1e-4 and err_sd < 1e-4,
                                f"max|mu' - mu_ref|={err_mu:.2e}, max|sd' - sd_ref|={err_sd:.2e}")
    # S6: M3 投影幂等 + 正交性（CPU，随机基）
    U = torch.linalg.qr(torch.randn(16, 4)).Q
    F = torch.randn(1, 16, 3, 3)
    P = m9.M3NuisanceSuppressModel._project(F, U)
    P2 = m9.M3NuisanceSuppressModel._project(P, U)
    resid = float((U.transpose(0, 1) @ P.reshape(1, 16, -1)[0]).abs().max())
    res["S6_m3_projection"] = (float((P - P2).abs().max()) < 1e-5 and resid < 1e-5,
                               f"idempotent={float((P - P2).abs().max()):.2e}, "
                               f"max|U^T P|={resid:.2e}")
    # S7: 冻结光照定义（与 5A-H 相同实现）
    t = torch.tensor([[[0.0, 0.5, 1.0]]])
    import experiment1_illumination_tradeoff as e1
    ok = all(e1.apply_photometric(t, it, lv).min().item() >= 0.0 for _, it, lv in m9.SHIFTS)
    res["S7_shifts_frozen"] = (ok and [s[0] for s in m9.SHIFTS] ==
                               ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"],
                               f"shifts={[s[0] for s in m9.SHIFTS]}, k_nuis={m9.K_NUISANCE}")
    # S8: M4 residual 组合形式 == 直接 effective-α（CPU，走 M4 类真实方法）
    mdl4 = m9.M4ResidualCompensationModel.__new__(m9.M4ResidualCompensationModel)
    torch.nn.Module.__init__(mdl4)
    mdl4.lam = LAMBDA_M4
    torch.manual_seed(0)
    f8 = torch.randn(2, 8, 5, 5)
    a_eff = (1.0 - LAMBDA_M4) * ALPHA_ROBUST_M4
    out_resid = mdl4._residual(f8, ALPHA_ROBUST_M4)
    out_direct = (1.0 - a_eff) * f8 + a_eff * torch.nn.functional.instance_norm(f8)
    d8 = float((out_resid - out_direct).abs().max())
    res["S8_m4_residual_algebra"] = (d8 < 1e-6,
                                     f"max|residual - direct(a_eff={a_eff!r})|={d8:.2e}")

    (LOG_DIR).mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"sanity_pre_{round_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (okk, d) in res.items():
            w.writerow([k, "PASS" if okk else "FAIL", d])
    return res


# ---------------------------------------------------------------------------
# fit：与 5A-H / 7A-O 同构，只换模型类
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
    if cfg["kind"] == "m1":
        mu, sd = m9.reference_stats(category, train_ids, h.DATA_ROOT)
        CAPTURED["m1_ref"] = {"mu_ref": mu, "sigma_ref": sd}
        model = m9.M1IllumStandardPatchcore(**common, mu_ref=mu, sigma_ref=sd)
    elif cfg["kind"] == "m3":
        basis, meta = m9.load_or_estimate_basis(CACHE_DIR, category, seed, train_ids,
                                                device, h.DATA_ROOT)
        CAPTURED["m3_basis_meta"] = meta
        model = m9.M3NuisanceSuppressPatchcore(**common, basis_l2=basis["layer2"],
                                               basis_l3=basis["layer3"])
    elif cfg["kind"] == "m4resid":
        model = m9.M4ResidualCompensationPatchcore(**common, alpha_prime=alpha_l2,
                                                   lam=cfg["lam"])
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
    out = {
        "round": round_name,
        "completed": est["completed"], "total": est["total"],
        "remaining": est["remaining"], "failed": est["failed"],
        "current_candidate": (est["last_unit"] or "").split(":")[-1] or None,
        "current_unit": est["last_unit"], "workers_active": est["workers_active"],
        "elapsed_seconds": round(est["elapsed"], 1),
        "avg_unit_seconds": est["avg_unit_seconds"],
        "eta_seconds": est["eta_seconds"], "eta_basis": est["eta_basis"],
        "estimated_finish_time": (time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(est["estimated_finish"]))
                                  if est["estimated_finish"] else None),
        "remaining_units": [j.get("unit_ids", []) for j in shards],
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_worker": [{ "worker": j.get("worker"), "pid": j.get("pid"),
                        "done": len(j.get("units", [])), "of": j.get("units_total"),
                        "current": j.get("current")} for j in shards],
    }
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
        print(f"\n[{args.worker_tag}] unit {uid} rows={exp_rows} "
              f"train={len(train_ids)} cfg={cfg}", flush=True)

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
        except Exception as exc:  # 单 unit 失败不丢其他结果
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
        meta = {"experiment": "9A", "unit": uid, "config": name, "round": args.round_name,
                "spec": {k: v for k, v in cfg.items()},
                "embedding_dim": CAPTURED.get("embed_dim", -1),
                "prep_seconds": CAPTURED.get("prep_seconds"),
                "memory_bank_size": info.get("coreset_size"),
                "runtime_seconds": info.get("runtime_seconds"),
                "wallclock_seconds": round(time.time() - t0, 2),
                "peak_gpu_memory_allocated_mb": info.get("peak_gpu_memory_allocated_mb"),
                "declared_alpha_l2": cfg["alpha_l2"], "declared_alpha_l3": cfg["alpha_l3"],
                "lambda_m4": LAMBDA_M4 if name.startswith("M4") else None,
                "alpha_robust_m4": ALPHA_ROBUST_M4 if name.startswith("M4") else None,
                "m4_alpha_prime": (cfg["alpha_l2"] if cfg["kind"] == "m4resid" else None),
                "m4_lambda": (cfg.get("lam") if cfg["kind"] == "m4resid" else None),
                "m4_effective_alpha": ((1.0 - cfg["lam"]) * cfg["alpha_l2"]
                                       if cfg["kind"] == "m4resid" else None),
                "m1_reference_stats": CAPTURED.get("m1_ref"),
                "m3_basis_meta": CAPTURED.get("m3_basis_meta"),
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
