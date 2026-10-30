#!/usr/bin/env python
"""E26: AnomalyDINO 官方实现在 M2AD 上的本协议配置。

官方来源：third_party/AnomalyDINO @ b9d1c2648e3a5247437d4d953d907a8f3d994457
本脚本只负责**数据寻址 + 评测**；特征处理、检索与得分聚合全部沿用官方代码：

  参考特征 : src.backbones.DINOv2Wrapper.prepare_image / extract_features
             + src.utils.augment_image（8 角度旋转）
  检索     : L2_normalized + k=1（等价实现，见下）
  图像得分 : src.post_eval.mean_top1p（前 1% patch 距离均值）

后端替换（仅依赖层，非算法替换）：
  官方用 faiss.IndexFlatL2(faiss-gpu)。本机 faiss 1.15.1 为 CPU-only 且禁止安装依赖，
  故用 PyTorch fp32 计算同样的精确 1-NN：
      特征均已 L2 归一化 → faiss 的 squared-L2 = 2 − 2cos，官方随后 /2 得 1 − cos；
      本实现直接取 max cos 后算 1 − cos。等价性由 --equiv 与官方 faiss-CPU 路径逐图对比。

输入（--input）：
  cache（默认）：data/m2ad/e7_cache/l0_448 —— 官方 448 短边 BICUBIC + antialias 的等价结果。
                 M²AD 原图统一 1024×1024（docs/E4_M2AD_INTEGRITY_AUDIT.md:96），
                 故 l0_448 与官方 transforms.Resize(448) 输出一致（已验证，见 backend_equivalence.json）。
  orig        ：1024² 原图（服务器仅 Bird 类别保留原图）。

用法：
  python run_anomalydino_m2ad.py --cells Bird:0
  python run_anomalydino_m2ad.py --all
  python run_anomalydino_m2ad.py --equiv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.hub as _th

# 离线 shim：避免 torch.hub 的在线校验（断网时会阻塞 ~130 s 并可能失败）
_HUB = Path(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))
if _HUB.is_dir():
    _orig_hub_load = _th.load

    def _offline_load(repo_or_dir, model, *args, **kw):
        if repo_or_dir == "facebookresearch/dinov2":
            repo_or_dir, kw = str(_HUB), {**kw, "source": "local"}
        return _orig_hub_load(repo_or_dir, model, *args, **kw)

    _th.load = _offline_load

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDINO"))

from src.backbones import get_model        # noqa: E402  官方
from src.utils import augment_image        # noqa: E402  官方
from src.post_eval import mean_top1p       # noqa: E402  官方

import e8_common as E8                     # noqa: E402
import e8_fetch as F                        # noqa: E402
import e4d1_m2ad_loader as LD               # noqa: E402

OUT = ROOT / "results" / "e26_anomalydino"
SCORES = OUT / "scores"
WORK = OUT / "work"
CELLS_E22 = ROOT / "results" / "e22" / "cells"

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
VIEW = "120"
BANK_N = 300
ANGLES = 8
RES = 448


def rel_key(rec) -> str:
    return Path(rec["image_path"]).relative_to(LD.DATA_ROOT).as_posix()


def raw_meta(cat: str) -> dict:
    meta = json.loads((ROOT / "data/m2ad/jsons_extract/meta_unsupervised.json").read_text())
    return {r["img_path"]: r for r in meta["test"][cat]}


def sha256_of(seq) -> str:
    h = hashlib.sha256()
    for s in seq:
        h.update(str(s).encode())
        h.update(b"\n")
    return h.hexdigest()


def src_path(rec, input_mode: str) -> Path:
    if input_mode == "cache":
        return Path(E8.cache_path("l0_448", rec))
    return Path(rec["image_path"])


def build_items(cat: str, seed: int, input_mode: str = "cache"):
    """返回 refs / items，元素均为 dict（含输入路径 src 与原始路径 orig）。"""
    scope = F.build_scope(cat)
    bank = E8.bank_for(scope["pool"], seed)
    assert len(bank) == BANK_N, f"bank size {len(bank)} != {BANK_N}"
    rawmap = raw_meta(cat)
    test = [r for r in scope["test"] if r["view"] == VIEW]
    refs = [{"src": src_path(r, input_mode), "orig": Path(r["image_path"]),
             "rel": rel_key(r)} for r in bank]
    items = []
    for r in test:
        m = rawmap[rel_key(r)]
        det = m["detectable"] if m["detectable"] is not False else False
        items.append({"src": src_path(r, input_mode), "orig": Path(r["image_path"]),
                      "label": int(m["image_anomaly"]), "detectable": det,
                      "specimen": r["specimen"], "illumination": r["illumination"]})
    return refs, items


def make_tree(cat: str, seed: int, refs, items, tag: str = ""):
    """官方目录布局（符号链接）。幂等，且**不做任何批量删除**。"""
    base = WORK / f"{tag}{cat}_s{seed}"
    (base / cat / "train" / "good").mkdir(parents=True, exist_ok=True)
    (base / cat / "test" / "good").mkdir(parents=True, exist_ok=True)
    (base / cat / "test" / "anomaly").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(refs):
        r["name"] = f"{i:04d}__{r['src'].name}"
        dst = base / cat / "train" / "good" / r["name"]
        if not dst.exists():
            os.symlink(r["src"], dst)
        r["path"] = dst
    for i, t in enumerate(items):
        t["name"] = f"{i:04d}__{t['specimen']}_{t['illumination']}.png"
        t["sub"] = "good" if t["label"] == 0 else "anomaly"
        dst = base / cat / "test" / t["sub"] / t["name"]
        if not dst.exists():
            os.symlink(t["src"], dst)
        t["path"] = dst
    return base


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def read_rgb(p) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"imread failed: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def run_cell(model, cat: str, seed: int, device: str, verbose=True,
             input_mode: str = "cache") -> dict:
    refs, items = build_items(cat, seed, input_mode)
    make_tree(cat, seed, refs, items, tag=f"{input_mode}_")
    t0 = time.time()

    bank = None
    n_patch = None
    cur = 0
    for i, r in enumerate(refs):
        img = read_rgb(r["src"])
        for aug in augment_image(img, augmentation="rotate"):
            ten, grid = model.prepare_image(aug)
            f = np.atleast_2d(model.extract_features(ten).astype("float32"))
            assert f.shape[0] == grid[0] * grid[1], (f.shape, grid)
            if bank is None:
                n_patch = f.shape[0]
                bank = np.empty((len(refs) * ANGLES * n_patch, f.shape[1]), dtype="float32")
            bank[cur:cur + f.shape[0]] = f
            cur += f.shape[0]
        if verbose and (i + 1) % 100 == 0:
            print(f"    ref {i+1}/{len(refs)}  {time.time()-t0:.0f}s", flush=True)
    bank = l2norm(bank)
    bank_t = torch.from_numpy(bank).to(device)
    n_bank_vec = bank_t.shape[0]
    t_bank = time.time() - t0

    t1 = time.time()
    for i, t in enumerate(items):
        img = read_rgb(t["src"])
        ten, _g = model.prepare_image(img)
        q = torch.from_numpy(l2norm(np.atleast_2d(
            model.extract_features(ten).astype("float32")))).to(device)
        best = torch.full((q.shape[0],), -2.0, device=device)
        for j in range(0, n_bank_vec, 200_000):
            sim = q @ bank_t[j:j + 200_000].T
            best = torch.maximum(best, sim.max(dim=1).values)
        t["score"] = float(mean_top1p((1.0 - best).cpu().numpy()))
        if verbose and (i + 1) % 100 == 0:
            print(f"    test {i+1}/{len(items)}  {time.time()-t1:.0f}s", flush=True)

    del bank_t, bank
    torch.cuda.empty_cache()

    # ---------- 身份核对
    cell_e22 = json.loads((CELLS_E22 / f"{cat}_seed{seed}.json").read_text())
    by_key = {(t["specimen"], t["illumination"]): t for t in items}
    assert len(by_key) == len(items), "duplicate (specimen,illumination)"
    keys_e22 = list(zip(list(cell_e22["specimen"]), list(cell_e22["illumination"])))
    items = [by_key[k] for k in keys_e22]
    keep = [t for t in items if t["detectable"] is not False]
    n_ok = sum(1 for t in keep if t["label"] == 0)
    n_ng = sum(1 for t in keep if t["label"] == 1)
    bh = sha256_of([r["orig"].as_posix() for r in refs])
    ident = {
        "n_test": len(items) == cell_e22["n_test"],
        "n_ok": n_ok == cell_e22["n_ok"],
        "n_ng": n_ng == cell_e22["n_ng"],
        "n_filtered": (len(items) - len(keep)) == cell_e22["n_filtered"],
        "sample_set_identical": set(keys_e22) == set((t["specimen"], t["illumination"])
                                                     for t in items),
        "labels_identical": [t["label"] for t in items] == list(cell_e22["labels"]),
        "detectable_identical": [bool(t["detectable"] is not False) for t in items]
                                == [bool(x) for x in cell_e22["detectable_raw"]],
        "bank_identical": bh == cell_e22["bank_hash"],
    }
    return {"category": cat, "seed": seed, "protocol": "e26_v1",
            "method": "AnomalyDINO-official-agreement",
            "backbone": "dinov2_vits14", "resolution": RES,
            "knn": "L2_normalized k=1", "agg": "mean_top1p",
            "masking": False, "rotation_angles": ANGLES, "input_mode": input_mode,
            "n_ref_images": len(refs), "n_bank_vectors": int(n_bank_vec),
            "n_patch": int(n_patch),
            "n_test": len(items), "n_ok": n_ok, "n_ng": n_ng,
            "n_filtered": len(items) - len(keep),
            "bank_hash": bh, "identity_check": ident,
            "identity_ok": all(ident.values()),
            "t_bank_s": round(t_bank, 1), "t_test_s": round(time.time() - t1, 1),
            "refs": [{"name": r["name"], "src": r["src"].as_posix(),
                      "orig": r["orig"].as_posix(), "rel": r["rel"]} for r in refs],
            "tests": [{"name": t["name"], "src": t["src"].as_posix(),
                       "orig": t["orig"].as_posix(), "label": t["label"],
                       "detectable": (t["detectable"] is not False),
                       "specimen": t["specimen"], "illumination": t["illumination"],
                       "score": t["score"]} for t in items],
            "argv": sys.argv, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def run_equivalence_check(model, device: str, n_ref=30, n_test=5) -> dict:
    """缩小设定：官方 run_anomaly_detection(faiss-CPU) vs 本脚本 torch 后端。"""
    from src.detection import run_anomaly_detection

    cat, seed = "Bird", 0
    refs, items = build_items(cat, seed, "orig")          # 用原图，最忠实
    good = [x for x in items if x["label"] == 0][:n_test]
    anom = [x for x in items if x["label"] == 1][:n_test]
    refs, items = refs[:n_ref], good + anom
    base = make_tree(cat, seed, refs, items, tag="_equiv_")
    plots = OUT / "logs" / "equiv_plots"
    plots.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    official, _mb, _it = run_anomaly_detection(
        model=model, object_name=cat, data_root=str(base), n_ref_samples=-1,
        object_anomalies={cat: ["anomaly"]}, plots_dir=str(plots), save_examples=False,
        masking=False, mask_ref_images=False, rotation=True,
        knn_metric="L2_normalized", knn_neighbors=1, faiss_on_cpu=True, seed=seed,
        save_patch_dists=False, save_tiffs=False)
    t_official = time.time() - t0

    t0 = time.time()
    bank = None
    for r in refs:
        img = read_rgb(r["src"])
        for aug in augment_image(img, augmentation="rotate"):
            ten, _g = model.prepare_image(aug)
            f = np.atleast_2d(model.extract_features(ten).astype("float32"))
            if bank is None:
                bank = np.empty((len(refs) * ANGLES * f.shape[0], f.shape[1]), "float32")
                cur = 0
            bank[cur:cur + f.shape[0]] = f
            cur += f.shape[0]
    bank_t = torch.from_numpy(l2norm(bank)).to(device)
    diffs = []
    for t in items:
        img = read_rgb(t["src"])
        ten, _g = model.prepare_image(img)
        q = torch.from_numpy(l2norm(np.atleast_2d(
            model.extract_features(ten).astype("float32")))).to(device)
        best = torch.full((q.shape[0],), -2.0, device=device)
        for j in range(0, bank_t.shape[0], 200_000):
            sim = q @ bank_t[j:j + 200_000].T
            best = torch.maximum(best, sim.max(dim=1).values)
        mine = float(mean_top1p((1.0 - best).cpu().numpy()))
        key = f"{t['sub']}/{t['name']}"
        if key in official:
            diffs.append(abs(mine - official[key]))
    return {"n_ref": len(refs), "n_angles": ANGLES, "n_test": len(items),
            "n_bank_vectors": int(bank_t.shape[0]),
            "max_abs_diff": float(max(diffs)) if diffs else None,
            "n_compared": len(diffs),
            "t_official_faiss_cpu_s": round(t_official, 1),
            "t_torch_backend_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--input", default="cache", choices=["cache", "orig"])
    a = ap.parse_args()

    SCORES.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    model = get_model("dinov2_vits14", a.device, RES)

    if a.equiv:
        r = run_equivalence_check(model, a.device)
        print("[EQUIV]", json.dumps(r, ensure_ascii=False))
        (OUT / "backend_equivalence.json").write_text(json.dumps(r, indent=1))
        return 0

    if a.all:
        todo = [(c, s) for c in CATS for s in SEEDS]
    elif a.cells:
        todo = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells]
    else:
        raise SystemExit("use --all or --cells Cat:seed")

    fails = 0
    for cat, seed in todo:
        dst = SCORES / f"{cat}_seed{seed}.json"
        if dst.exists():
            print(f"[skip] {dst.name}", flush=True)
            continue
        print(f"[run ] {cat} seed={seed} input={a.input}", flush=True)
        try:
            out = run_cell(model, cat, seed, a.device, input_mode=a.input)
        except Exception:
            fails += 1
            print(f"[FAIL] {cat} seed={seed}\n{traceback.format_exc()}", flush=True)
            continue
        dst.write_text(json.dumps(out))
        print(f"[done] {dst.name} identity_ok={out['identity_ok']} "
              f"n={out['n_ok']}/{out['n_ng']} t_bank={out['t_bank_s']}s "
              f"t_test={out['t_test_s']}s", flush=True)
    print(f"[END] fails={fails}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
