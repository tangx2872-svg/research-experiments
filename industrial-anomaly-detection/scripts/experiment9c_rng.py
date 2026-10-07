"""Experiment 9C — Matched-RNG helpers（Protocol V2）。

目标：消除「不同方法代码路径消耗不同数量 RNG → KCenterGreedy 选择不同 coreset」的评测不公。

设计要点（见 results/experiment_9c_rng_calibration/config.json 的 replay_point_frozen）：
  * replay 点 = `KCenterGreedy.select_coreset_idxs` 入口（wrapper 注入，不改 anomalib 源码）
  * 控制 torch CPU / torch CUDA / NumPy global / python random（后两者含入快照；审计显示 python random 未被 coreset 消耗）
  * 使用「canonical 快照 + 恢复」而不是裸 manual_seed：快照由 (category, seed) 常量唯一确定，
    与上游构造路径完全无关，且可对 C1/C2 证明「fit 前 RNG state 一致」
  * 不触碰：模型数学、feature 值、train split、coreset ratio、kNN、illumination、test/defect 数据
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

# 模块级上下文：由 runner 在每个 unit 前设置
_CTX: dict = {"unit": None, "category": None, "seed": None, "protocol": "v2", "out_dir": None}
_EVIDENCE: list = []
_ORIG = None
_INSTALLED = False


# ---------------------------------------------------------------------------
# RNG 状态：捕获 / 恢复 / 哈希 / 探测
# ---------------------------------------------------------------------------
def capture_rng_state() -> dict:
    st = {"torch_cpu": torch.get_rng_state(),
          "numpy": np.random.get_state(),
          "python": random.getstate(),
          "cuda": None}
    if torch.cuda.is_available():
        try:
            st["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            st["cuda"] = None
    return st


def restore_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch_cpu"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _sha(b) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()[:16]


def state_hashes(st: dict | None = None) -> dict:
    st = st or capture_rng_state()
    out = {"torch_cpu": _sha(st["torch_cpu"].numpy().tobytes()),
           "numpy": _sha(np.asarray(st["numpy"][1]).tobytes()),
           "numpy_pos": int(st["numpy"][2]),
           "python": _sha(repr(st["python"]).encode())}
    if st.get("cuda") is not None:
        out["cuda"] = "_".join(_sha(s.numpy().tobytes()) for s in st["cuda"])
    else:
        out["cuda"] = None
    return out


def canonical_state(seed: int) -> dict:
    """(category, seed) 唯一确定的 canonical 快照：seed 四条流后捕获。"""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    return capture_rng_state()


def probe_rng(n: int = 3) -> dict:
    """非破坏性探测：save → draw → restore（不干扰真实 coreset 采样）。"""
    st = capture_rng_state()
    pr = {"torch_rand": [float(x) for x in torch.rand(n)],
          "torch_randint": [int(x) for x in torch.randint(0, 1000000, (n,))],
          "numpy_rand": [float(x) for x in np.random.rand(n)]}
    if torch.cuda.is_available():
        pr["cuda_rand"] = [float(x) for x in torch.rand(n, device="cuda").cpu()]
    restore_rng_state(st)
    return pr


def tensor_sha256(t: torch.Tensor, chunk: int = 20000) -> str:
    h = hashlib.sha256()
    x = t.detach()
    for i in range(0, int(x.shape[0]), chunk):
        h.update(x[i:i + chunk].contiguous().cpu().numpy().tobytes())
    h.update(str(tuple(x.shape)).encode())
    return h.hexdigest()


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float()
    return {"shape": list(x.shape), "dtype": str(t.dtype), "device": str(t.device),
            "sum_abs": float(x.abs().sum().item()), "mean": float(x.mean().item())}


# ---------------------------------------------------------------------------
# 上下文与证据落盘
# ---------------------------------------------------------------------------
def set_context(**kw) -> None:
    _CTX.update(kw)


def get_context() -> dict:
    return dict(_CTX)


def get_evidence() -> list:
    return list(_EVIDENCE)


def clear_evidence() -> None:
    _EVIDENCE.clear()


def _flush() -> None:
    od = _CTX.get("out_dir")
    if not od:
        return
    p = Path(od)
    p.mkdir(parents=True, exist_ok=True)
    unit = str(_CTX.get("unit") or "unit").replace(":", "_")
    (p / ("rng_evidence_%s.json" % unit)).write_text(json.dumps(
        {"unit": _CTX.get("unit"), "category": _CTX.get("category"), "seed": _CTX.get("seed"),
         "protocol": _CTX.get("protocol"), "n_select_calls": len(_EVIDENCE),
         "select_calls": _EVIDENCE}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Protocol V2：在 coreset selection 入口恢复 canonical RNG state
# ---------------------------------------------------------------------------
def _patched_select_coreset_idxs(self):  # noqa: ANN001
    global _ORIG
    proto = str(_CTX.get("protocol") or "v2")
    rec: dict = {"protocol": proto, "ts": time.time(),
                 "n_observations": int(self.n_observations),
                 "coreset_size": int(self.coreset_size),
                 "embedding_device": str(self.embedding.device),
                 "embedding_stats": tensor_stats(self.embedding)}
    if proto == "v2":
        rec["rng_hash_before_restore"] = state_hashes()
        rec["rng_probe_before_restore"] = probe_rng()
        restore_rng_state(canonical_state(int(_CTX["seed"])))
        rec["rng_hash_after_restore"] = state_hashes()
        rec["rng_probe_after_restore"] = probe_rng()      # 非破坏性：draw 后恢复
        rec["rng_hash_at_real_sampling"] = state_hashes()
    else:
        rec["rng_hash_v1_before_selection"] = state_hashes()
        rec["rng_probe_v1_before_selection"] = probe_rng()

    idxs = _ORIG(self)                                    # ← 真实 coreset 选择

    idx_arr = np.asarray(idxs, dtype=np.int64)
    bank = self.embedding[torch.as_tensor(idx_arr, device=self.embedding.device)]
    rec["coreset"] = {"n": int(idx_arr.size), "first20": [int(i) for i in idx_arr[:20]],
                      "last5": [int(i) for i in idx_arr[-5:]],
                      "sha256": hashlib.sha256(idx_arr.tobytes()).hexdigest(),
                      "sorted_first20": [int(i) for i in sorted(idx_arr.tolist())[:20]]}
    rec["memory_bank"] = {"sha256": tensor_sha256(bank), "stats": tensor_stats(bank)}
    _EVIDENCE.append(rec)
    _flush()
    return idxs


def install_matched_rng(seed: int, out_dir, protocol: str = "v2") -> None:
    """注入 wrapper（只做一次）；protocol='v1' 时不恢复 RNG（保持历史行为）。"""
    global _ORIG, _INSTALLED
    from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy

    if not _INSTALLED:
        _ORIG = KCenterGreedy.select_coreset_idxs
        KCenterGreedy.select_coreset_idxs = _patched_select_coreset_idxs
        _INSTALLED = True
    _CTX.update({"seed": int(seed), "out_dir": str(out_dir), "protocol": protocol})


# ---------------------------------------------------------------------------
# 可复现的 RNG 审计（生成 summary/rng_audit.json）
# ---------------------------------------------------------------------------
def run_rng_audit(seed: int = 0, n: int = 4000, d: int = 64, extra: int = 5) -> dict:
    from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy

    g = torch.Generator().manual_seed(7)
    emb = torch.randn(n, d, generator=g).abs()

    def _select(extra_consumption: int = 0, replay: bool = False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        for _ in range(extra_consumption):
            torch.randn(3)
            np.random.rand(3)
        for _ in range(extra_consumption):
            torch.randn(3)
            np.random.rand(3)
        if replay:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        return KCenterGreedy(embedding=emb, sampling_ratio=0.1).select_coreset_idxs()

    # 1) 哪些流被消耗
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    h0 = state_hashes()
    KCenterGreedy(embedding=emb, sampling_ratio=0.1).select_coreset_idxs()
    h1 = state_hashes()
    consumed = {k: (h0[k] != h1[k]) for k in ("torch_cpu", "numpy", "python", "cuda")}

    base = _select(0, False)
    v1_after_extra = _select(extra, False)
    v2_replay = _select(extra, True)
    return {"seed": seed, "synthetic_embedding": [n, d], "coreset_size": len(base),
            "streams_consumed_by_coreset": consumed,
            "same_seed_same_coreset": base == _select(0, False),
            "extra_rng_consumption_changes_coreset": base != v1_after_extra,
            "replay_before_coreset_restores_equality": base == v2_replay,
            "n_extra_consumption_steps_simulated": extra,
            "note": ("torch/numpy 被 coreset 消耗；python random 未被消耗。"
                     "V1 的差异来自上游消耗数量不同；在 coreset 入口重放 RNG 可完全恢复一致。")}


# ---------------------------------------------------------------------------
# V2 的第二个 replay 点：Engine.fit 入口（固定 train DataLoader 的 shuffle 行序）
# ---------------------------------------------------------------------------
_ORIG_ENGINE_FIT = None
_FIT_INSTALLED = False
FIT_EVIDENCE: list = []


def _flush_fit() -> None:
    od = _CTX.get("out_dir")
    if not od:
        return
    p = Path(od)
    p.mkdir(parents=True, exist_ok=True)
    unit = str(_CTX.get("unit") or "unit").replace(":", "_")
    (p / ("fit_replay_evidence_%s.json" % unit)).write_text(json.dumps(
        {"unit": _CTX.get("unit"), "protocol": _CTX.get("protocol"),
         "reason": ("train DataLoader 使用 shuffle=True 且无显式 generator -> 行序由全局 RNG 决定；"
                    "在 Engine.fit 入口 replay 可固定行序，使 embedding 逐位一致"),
         "calls": FIT_EVIDENCE}, indent=2, ensure_ascii=False))


def _patched_engine_fit(self, *args, **kwargs):  # noqa: ANN001
    global _ORIG_ENGINE_FIT
    proto = str(_CTX.get("protocol") or "v2")
    if proto == "v2" and _CTX.get("seed") is not None:
        rec = {"ts": time.time(), "rng_hash_before_fit_replay": state_hashes()}
        restore_rng_state(canonical_state(int(_CTX["seed"])))
        rec["rng_hash_after_fit_replay"] = state_hashes()
        rec["rng_probe_after_fit_replay"] = probe_rng()
        FIT_EVIDENCE.append(rec)
        _flush_fit()
    return _ORIG_ENGINE_FIT(self, *args, **kwargs)


def install_fit_replay() -> None:
    """注入 Engine.fit 入口 replay（只做一次；protocol=v1 时不做任何事）。"""
    global _ORIG_ENGINE_FIT, _FIT_INSTALLED
    from anomalib.engine import Engine

    if not _FIT_INSTALLED:
        _ORIG_ENGINE_FIT = Engine.fit
        Engine.fit = _patched_engine_fit
        _FIT_INSTALLED = True


def dataloader_order_after(seed: int, extra: int, replay: bool, category: str = "bottle",
                           root: str = "data/mvtec_ad") -> list:
    """审计用：模拟上游 RNG 消耗对 train 行序的影响（CPU）。"""
    from pathlib import Path as _P

    from anomalib.data import MVTecAD

    import experiment1b_defect_sensitivity as e1b

    _val, tr = e1b.make_validation_split(category, seed)
    keep = {n for n in tr}
    torch.manual_seed(seed)
    np.random.seed(seed)
    for _ in range(extra):
        torch.randn(3)
        np.random.rand(3)
    if replay:
        torch.manual_seed(seed)
        np.random.seed(seed)
    dm = MVTecAD(root=root, category=category, train_batch_size=16, eval_batch_size=16,
                 num_workers=0, seed=seed)
    dm.setup()
    td = dm.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(_P(str(p)).name) in keep)].reset_index(drop=True)
    td._samples = df
    dl = dm.train_dataloader()
    return [_P(str(b["image_path"][0])).name for b in dl]
