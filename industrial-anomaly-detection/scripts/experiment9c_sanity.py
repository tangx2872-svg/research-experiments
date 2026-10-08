"""Experiment 9C — P4 CPU sanity（GPU 之前必须全 PASS）。用法：python -u scripts/experiment9c_sanity.py"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment9c_rng as r9c  # noqa: E402
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy  # noqa: E402

EXP = ROOT / "results" / "experiment_9c_rng_calibration"
SUM = EXP / "summary"


def _emb(n=3000, d=64):
    g = torch.Generator().manual_seed(7)
    return torch.randn(n, d, generator=g).abs()


def main() -> None:
    checks = []
    emb = _emb()

    # S1: 同一快照 restore 后连续 torch.rand 完全一致
    st = r9c.canonical_state(0)
    r9c.restore_rng_state(st)
    a = r9c.probe_rng(4)
    r9c.restore_rng_state(st)
    b = r9c.probe_rng(4)
    checks.append(("S1_snapshot_restore_reproduces_rng_stream", a == b, f"probe equal={a == b}"))

    # S2: 不同构造路径消耗 RNG 后，restore 快照 -> 完全一致
    r9c.restore_rng_state(r9c.canonical_state(0))
    torch.randn(5)
    np.random.rand(5)
    torch.rand(3, device="cuda") if torch.cuda.is_available() else None
    c = r9c.probe_rng(4)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    d = r9c.probe_rng(4)
    r9c.restore_rng_state(r9c.canonical_state(0))
    e = r9c.probe_rng(4)
    # 正确语义：任意消耗后（未 restore）探测必须不同（证明探测敏感）；
    # 而 restore canonical 快照后必须与 canonical 完全一致（证明重放有效）。
    checks.append(("S2_canonical_restore_after_arbitrary_consumption",
                   (d == e) and (c != e),
                   f"restore-canonical reproduces canonical: {d == e}; "
                   f"un-restored stream differs (probe sensitive): {c != e}"))

    # S3: NumPy / python RNG replay exact
    r9c.restore_rng_state(r9c.canonical_state(0))
    n1 = np.random.rand(5).tolist()
    p1 = random.random()
    r9c.restore_rng_state(r9c.canonical_state(0))
    n2 = np.random.rand(5).tolist()
    p2 = random.random()
    checks.append(("S3_numpy_python_replay_exact", n1 == n2 and p1 == p2,
                   f"numpy_equal={n1 == n2} python_equal={p1 == p2}"))

    # S4: helper 不修改 input tensor
    t = emb.clone()
    h0 = r9c.tensor_sha256(emb)
    r9c.tensor_stats(emb)
    r9c.tensor_sha256(emb)
    checks.append(("S4_helper_does_not_modify_input", torch.equal(t, emb) and h0 == r9c.tensor_sha256(emb),
                   "embedding unchanged by tensor_stats/tensor_sha256"))

    # S5: V1 模式不改变原行为（wrapper 在 protocol=v1 下不做 restore -> 与未包装函数同结果）
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    orig = KCenterGreedy.select_coreset_idxs
    ref = KCenterGreedy(embedding=emb, sampling_ratio=0.1).select_coreset_idxs()
    r9c.install_matched_rng(seed=0, out_dir=str(EXP / "raw" / "_sanity"), protocol="v1")
    r9c.set_context(unit="bottle:0:SANITY_V1", category="bottle", seed=0,
                    out_dir=str(EXP / "raw" / "_sanity"))
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    v1 = KCenterGreedy(embedding=emb, sampling_ratio=0.1).select_coreset_idxs()
    checks.append(("S5_v1_mode_preserves_original_behaviour", ref == v1,
                   f"v1-wrapped == unwrapped: {ref == v1}"))

    # S6: V2 模式下，上游额外消耗 RNG 也得到相同 coreset（核心机制）
    def run_v2(extra: int):
        r9c.install_matched_rng(seed=0, out_dir=str(EXP / "raw" / "_sanity"), protocol="v2")
        r9c.set_context(unit=f"bottle:0:SANITY_V2_{extra}", category="bottle", seed=0,
                        out_dir=str(EXP / "raw" / "_sanity"))
        torch.manual_seed(0)
        np.random.seed(0)
        for _ in range(extra):
            torch.randn(3)
            np.random.rand(3)
        return KCenterGreedy(embedding=emb, sampling_ratio=0.1).select_coreset_idxs()

    x0, x5, x11 = run_v2(0), run_v2(5), run_v2(11)
    KCenterGreedy.select_coreset_idxs = orig   # 复原
    checks.append(("S6_v2_matched_rng_removes_upstream_rng_divergence",
                   x0 == x5 == x11, f"extra=0/5/11 identical: {x0 == x5 == x11}"))

    # S7: V2 的第二个 replay 点（Engine.fit 入口）固定 train DataLoader shuffle 行序
    try:
        o0 = r9c.dataloader_order_after(0, 0, False)
        o3 = r9c.dataloader_order_after(0, 3, False)
        o3r = r9c.dataloader_order_after(0, 3, True)
        checks.append(("S7_fit_entry_replay_fixes_train_dataloader_order",
                       (o0 != o3) and (o0 == o3r),
                       f"extra-consumption changes row order: {o0 != o3}; "
                       f"fit-entry replay restores it: {o0 == o3r}"))
    except Exception as exc:  # 审计失败必须显式 FAIL，不能静默
        checks.append(("S7_fit_entry_replay_fixes_train_dataloader_order", False,
                       f"error {type(exc).__name__}: {exc}"))

    audit = r9c.run_rng_audit(seed=0, n=2000, d=48)
    (SUM / "rng_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    SUM.mkdir(parents=True, exist_ok=True)
    with open(SUM / "sanity_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for name, ok, det in checks:
            w.writerow([name, "PASS" if ok else "FAIL", det])
    for name, ok, det in checks:
        print(("PASS " if ok else "FAIL ") + name + " | " + det)
    print("-> sanity %d/%d PASS" % (sum(1 for _, ok, _ in checks if ok), len(checks)))
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("P4 sanity FAILED — refuse to run GPU")


if __name__ == "__main__":
    main()
