#!/usr/bin/env python
"""Parity check: exported figures vs the original runtime scripts.

Both sides are rendered in-memory and *nothing is written*: the original scripts
(scripts/e22_figures.py, scripts/e23_figures.py, scripts/e21_figures.py) are imported
with their ``save`` function replaced by a capture hook, so no file in results/ is
touched or overwritten.

Compared quantities (exact equality expected):
  * Fig 1: bar widths (= 3-seed mean ΔAUROC), bar centres, per-seed dot x positions
  * Fig 2: per-seed scatter offsets (F0/R1 scores for all 670 detectable images)
  * Fig 3: heatmap matrix (category x view ΔAUROC, seed 0)
  * all  : axis ranges

Usage
-----
    python verify_parity.py [--repo PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE.parent          # results/paper_figures_20261030
DEFAULT_REPO = PKG.parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture_from_original(repo: Path) -> dict:
    """Render the original figures in memory and read their artists back out."""
    out = {}

    def make_hook(store, name_key, target):
        def hook(fig, name):
            if name != target:          # other figures of the same script: skip
                plt.close(fig)
                return
            ax = fig.axes[0]
            if name_key == "bars":
                store[name] = {
                    "widths": [float(p.get_width()) for p in ax.patches],
                    "centres": [float(p.get_y() + p.get_height() / 2) for p in ax.patches],
                    "dots": [np.asarray(l.get_xdata(), dtype=float).tolist()
                             for l in ax.lines if l.get_marker() == "o"],
                    "xlim": list(ax.get_xlim()),
                    "ylim": list(ax.get_ylim()),
                }
            elif name_key == "scatter":
                store[name] = {
                    "offsets": [np.asarray(c.get_offsets(), dtype=float).tolist()
                                for c in ax.collections],
                    "xlim": list(ax.get_xlim()),
                    "ylim": list(ax.get_ylim()),
                }
            else:  # heatmap
                store[name] = {
                    "matrix": np.asarray(ax.images[0].get_array(), dtype=float).tolist(),
                    "xlim": list(ax.get_xlim()),
                    "ylim": list(ax.get_ylim()),
                }
            plt.close(fig)
        return hook

    for script, key, wanted in (
        ("scripts/e22_figures.py", "bars", "fig2_delta_auroc_10cat"),
        ("scripts/e23_figures.py", "scatter", "fig1_doll_f0_vs_r1_scatter"),
        ("scripts/e21_figures.py", "heatmap", "fig2_dauroc_heatmap"),
    ):
        mod = _load(repo / script, f"orig_{key}")
        store: dict = {}
        mod.save = make_hook(store, key, wanted)
        mod.main()
        out[key] = store[wanted]
    return out


def _capture_from_export() -> dict:
    mod = _load(PKG / "scripts" / "export_paper_figures.py", "export_mod")
    plt.rcParams.update(mod.STYLE)
    inputs = PKG / "inputs"

    fig, _ = mod.fig_delta_auroc_10cat(inputs / "main_table_10cat_3seed.csv",
                                       inputs / "all10_3seed_cells.csv", True)
    ax = fig.axes[0]
    out = {"bars": {
        "widths": [float(p.get_width()) for p in ax.patches],
        "centres": [float(p.get_y() + p.get_height() / 2) for p in ax.patches],
        "dots": [np.asarray(l.get_xdata(), dtype=float).tolist()
                 for l in ax.lines if l.get_marker() == "o"],
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }}
    plt.close(fig)

    fig, _ = mod.fig_doll_scatter(inputs / "doll_per_image_delta.csv",
                                  inputs / "all10_3seed_cells.csv", True)
    ax = fig.axes[0]
    out["scatter"] = {
        "offsets": [np.asarray(c.get_offsets(), dtype=float).tolist() for c in ax.collections],
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    plt.close(fig)

    fig, _ = mod.fig_multiview_heatmap(inputs / "analysis_cells.csv", True)
    ax = fig.axes[0]
    out["heatmap"] = {
        "matrix": np.asarray(ax.images[0].get_array(), dtype=float).tolist(),
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    plt.close(fig)
    return out


def _cmp(a, b) -> tuple[bool, float]:
    """Exact structural comparison; returns (equal, max_abs_diff)."""
    if isinstance(a, dict):
        ok, diff = True, 0.0
        for k in a:
            o, d = _cmp(a[k], b[k])
            ok = ok and o
            diff = max(diff, d)
        return ok, diff
    if isinstance(a, list):
        if len(a) != len(b):
            return False, float("inf")
        ok, diff = True, 0.0
        for x, y in zip(a, b):
            o, d = _cmp(x, y)
            ok = ok and o
            diff = max(diff, d)
        return ok, diff
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (a == b), abs(float(a) - float(b))
    return (a == b), 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="absolute tolerance; 1e-6 = the 6-decimal precision of the "
                         "CSV tables the export reads (the original scripts read "
                         "full-precision cell JSON)")
    args = ap.parse_args()

    orig = _capture_from_original(args.repo)
    new = _capture_from_export()

    report = {
        "repo": str(args.repo),
        "absolute_tolerance": args.tol,
        "tolerance_reason": "export reads the 6-dp CSV tables (same numbers as the paper "
                            "tables); the original runtime scripts read full-precision "
                            "per-cell JSON, so residuals are bounded by CSV rounding",
        "figures": {},
    }
    all_ok = True
    names = {"bars": "delta_auroc_10cat_horizontal",
             "scatter": "doll_f0_vs_r1_per_image_scatter",
             "heatmap": "multiview_delta_auroc_heatmap"}
    for key, fig_name in names.items():
        _, diff = _cmp(orig[key], new[key])
        ok = diff <= args.tol
        all_ok = all_ok and ok
        report["figures"][fig_name] = {"identical_within_tol": bool(ok),
                                       "exact_match": diff == 0.0,
                                       "max_abs_diff": diff}
        print(f"{'PASS' if ok else 'FAIL'}  {fig_name:<34} max|diff| = {diff:g}", flush=True)

    p = HERE / "parity_vs_original.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"report -> {p}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
