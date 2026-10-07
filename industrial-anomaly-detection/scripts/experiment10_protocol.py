"""Experiment 10 — Corrected-189 train filter（修 9B-R §11 发现的协议属性）。

问题：`Engine.fit` 内部会再次 `dm.setup(stage="fit")`，重建 train_data，
使 `experiment5a_h_runner.run_config` 的 train 过滤失效 -> memory bank 实际含全部 209 张
（含 20 张 val），与注释「validation 不进 memory bank」矛盾。

修法（最小侵入、不改历史代码）：包住 `Engine.fit`，在它内部再次 setup 之后**重新施加** train 过滤；
并对 `Patchcore.training_step` 计数以**证明**实际嵌入图像数 = 189。
本模块只被 Experiment 10 使用；历史实验 raw 不受影响、不被覆盖。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

_CTX: dict = {"train_ids": None, "category": None, "seed": None, "out_dir": None,
              "mode": "corrected189", "filter_log": [], "image_count": None}
_INSTALLED = {"fit": False, "count": False}


def set_context(**kw) -> None:
    _CTX.update(kw)


def get_context() -> dict:
    return dict(_CTX)


def get_evidence() -> dict:
    return {"mode": _CTX.get("mode"), "category": _CTX.get("category"), "seed": _CTX.get("seed"),
            "n_train_ids": (len(_CTX["train_ids"]) if _CTX.get("train_ids") else None),
            "filter_log": list(_CTX.get("filter_log") or []),
            "images_embedded_at_fit": _CTX.get("image_count")}


def _write(out_dir, fname: str, rec: dict) -> None:
    try:
        d = Path(out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[warn] protocol evidence write failed: {exc}", flush=True)


def _apply_filter(dm) -> None:
    keep = _CTX.get("train_ids")
    if not keep:
        return
    keep = set(keep)
    td = getattr(dm, "train_data", None)
    if td is None or not hasattr(td, "_samples"):
        return
    df = td._samples
    before = len(df)
    df2 = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep)].reset_index(drop=True)
    td._samples = df2
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df2)
    _CTX.setdefault("filter_log", []).append(
        {"ts": time.time(), "n_before": int(before), "n_after": int(len(df2)),
         "kept_all_train_ids": bool(len(df2) == len(keep))})


def _wrap_datamodule(dm) -> None:
    if dm is None or getattr(dm, "_exp10_wrapped", False):
        return
    orig_setup = dm.setup

    def setup_filtered(stage=None):
        orig_setup(stage)
        _apply_filter(dm)          # Lightning 再次 setup 之后立刻重新过滤

    dm.setup = setup_filtered
    dm._exp10_wrapped = True
    _apply_filter(dm)


def install_corrected_train_filter() -> None:
    """包住 Engine.fit：过滤其 datamodule，并在 fit 结束后记录实际嵌入图像数。"""
    from anomalib.engine import Engine

    if not _INSTALLED["fit"]:
        prev_fit = Engine.fit

        def fit(self, *args, **kwargs):
            dm = kwargs.get("datamodule")
            if dm is None:
                for v in list(args) + list(kwargs.values()):
                    if hasattr(v, "setup") and hasattr(v, "train_data"):
                        dm = v
                        break
            _CTX["image_count"] = None
            _CTX["filter_log"] = []
            if _CTX.get("mode") == "corrected189":
                _wrap_datamodule(dm)
            out = prev_fit(self, *args, **kwargs)
            if _CTX.get("out_dir"):
                unit = str(_CTX.get("category") or "") + "_" + str(_CTX.get("seed") or "")
                _write(_CTX["out_dir"], f"train_filter_evidence_{unit}.json", get_evidence())
            return out

        Engine.fit = fit
        _INSTALLED["fit"] = True

    if not _INSTALLED["count"]:
        from anomalib.models.image.patchcore.lightning_model import Patchcore

        prev_ts = Patchcore.training_step

        def training_step(self, batch, *a, **kw):
            if _CTX.get("mode") == "corrected189":
                n = int(batch.image.shape[0]) if hasattr(batch, "image") else 0
                _CTX["image_count"] = int((_CTX.get("image_count") or 0) + n)
            return prev_ts(self, batch, *a, **kw)

        Patchcore.training_step = training_step
        _INSTALLED["count"] = True
