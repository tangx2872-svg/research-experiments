"""Experiment 16 — P5（50-unit 表）+ P6（seed-block）+ P7（bootstrap & tail）+ P8（robustness cost）。全 CPU。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402
import experiment16_fusion as F16  # noqa: E402

E16 = F16.E16
ANA = E16 / "analysis"
BLOCKS = [("A_seeds0-2", (0, 1, 2)), ("B_seeds3-4", (3, 4)),
          ("C_seeds5-6", (5, 6)), ("D_seeds7-9", (7, 8, 9))]
METHODS = ["Original", "Adaptive B2", "C2", "C6", "X6c"]
COMPLEXITY = {"Original": "low (1 unit)", "Adaptive B2": "low (1 unit, category-adaptive beta)",
              "C2": "low (1 fixed weight)", "C6": "low (max, 1 rule)", "X6c": "medium (mean of 2 rules)"}
B = 10000
RNG = np.random.RandomState(16)


def build():
    """-> per_unit rows：每 (cat,seed) 的 Original 基准 + 4 个方法。"""
    rows = []
    for cat in F16.CATS:
        for seed in F16.SEEDS10:
            po, pb = F16.source(cat, seed, "o"), F16.source(cat, seed, "b")
            if po is None or pb is None:
                continue
            mo = M.metrics_of({(cat, seed, "o"): po})[(cat, seed, "o")]
            mb = M.metrics_of({(cat, seed, "b"): pb})[(cat, seed, "b")]
            rec = {"category": cat, "seed": seed,
                   "orig_R": mo["mean_abs_delta_z"], "orig_D": mo["mean_dprime"],
                   "orig_auroc": mo["image_auroc"]}
            vb = M.verdict(mo, mb)
            rec.update({"B2_dR": vb["d_absz"], "B2_ddp": vb["d_dprime"],
                        "B2_PASS": vb["PASS"], "B2_cat": vb["catastrophic"],
                        "B2_R": mb["mean_abs_delta_z"], "B2_D": mb["mean_dprime"]})
            for cid in ("C2", "C6", "X6c"):
                u = F16.fused_unit(cat, seed, cid)
                mc = M.metrics_of({(cat, seed, cid): u})[(cat, seed, cid)]
                v = M.verdict(mo, mc)
                rec.update({f"{cid}_dR": v["d_absz"], f"{cid}_ddp": v["d_dprime"],
                            f"{cid}_PASS": v["PASS"], f"{cid}_cat": v["catastrophic"],
                            f"{cid}_R": mc["mean_abs_delta_z"], f"{cid}_D": mc["mean_dprime"]})
            rows.append(rec)
    return rows


KEYMAP = {"Adaptive B2": "B2"}          # row 里的字段前缀（B2 与 METHODS 名不同）


def agg(rows, key) -> dict:
    key = KEYMAP.get(key, key)
    dr = np.array([r[f"{key}_dR"] for r in rows]) if key != "Original" else np.array([0.0] * len(rows))
    dd = np.array([r[f"{key}_ddp"] for r in rows]) if key != "Original" else np.array([0.0] * len(rows))
    P = np.array([bool(r[f"{key}_PASS"]) for r in rows]) if key != "Original" else np.zeros(len(rows), bool)
    C = np.array([bool(r[f"{key}_cat"]) for r in rows]) if key != "Original" else np.zeros(len(rows), bool)
    R = np.array([r[f"{key}_R"] for r in rows]) if key != "Original" else np.array([r["orig_R"] for r in rows])
    D = np.array([r[f"{key}_D"] for r in rows]) if key != "Original" else np.array([r["orig_D"] for r in rows])
    return {"n": len(rows), "PASS": int(P.sum()), "pass_rate": float(P.mean()),
            "cat": int(C.sum()), "cat_rate": float(C.mean()),
            "mean_dR": float(dr.mean()), "mean_ddp": float(dd.mean()),
            "median_ddp": float(np.median(dd)), "worst_ddp": float(dd.min()),
            "absz": float(R.mean()), "dprime": float(D.mean()), "arr_ddp": dd, "arr_dR": dr,
            "arr_P": P.astype(float), "arr_C": C.astype(float), "arr_D": D, "arr_R": R}


def boot_ci(x: np.ndarray) -> tuple:
    bs = [np.mean(x[RNG.randint(0, len(x), len(x))]) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    bs = [np.mean(d[RNG.randint(0, len(d), len(d))]) for _ in range(B)]
    return {"point": float(d.mean()),
            "CI": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "sig": bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)}


def main() -> None:
    ANA.mkdir(parents=True, exist_ok=True)
    rows = build()
    n_units = len(rows)
    print("units built: %d (expect 50)" % n_units)
    A = {k: agg(rows, k) for k in METHODS}

    # ---------- P5 final table / leaderboard ----------
    fields = ["method", "PASS", "n_units", "pass_rate", "catastrophic", "cat_rate",
              "mean_ddp", "median_ddp", "worst_ddp", "mean_dR", "absz", "dprime",
              "complexity", "evidence"]
    tab = []
    for k in METHODS:
        a = A[k]
        tab.append({"method": k, "PASS": a["PASS"], "n_units": a["n"], "pass_rate": round(a["pass_rate"], 4),
                    "catastrophic": a["cat"], "cat_rate": round(a["cat_rate"], 4),
                    "mean_ddp": round(a["mean_ddp"], 4), "median_ddp": round(a["median_ddp"], 4),
                    "worst_ddp": round(a["worst_ddp"], 4), "mean_dR": round(a["mean_dR"], 4),
                    "absz": round(a["absz"], 4), "dprime": round(a["dprime"], 4),
                    "complexity": COMPLEXITY[k], "evidence": "E3+ (5 cats x 10 seeds)"})
    with open(ANA / "final_50unit_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(tab)
    with open(ANA / "final_leaderboard.csv", "w", newline="") as f:
        order = sorted(tab, key=lambda r: (-r["PASS"], r["catastrophic"], -r["worst_ddp"]))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(order)
    kf = ["category", "seed", "orig_R", "orig_D", "orig_auroc"] +          [f"{m}_{x}" for m in ("B2", "C2", "C6", "X6c") for x in ("dR", "ddp", "PASS", "cat", "R", "D")]
    with open(ANA / "per_unit_50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kf)
        w.writeheader(); w.writerows(rows)

    # ---------- P6 seed blocks ----------
    blocks = []
    for name, seeds in BLOCKS:
        sub = [r for r in rows if r["seed"] in seeds]
        rec = {"block": name, "seeds": ",".join(map(str, seeds)), "n_units": len(sub)}
        for k in METHODS:
            a = agg(sub, k)
            rec[f"{k}_PASS"] = f"{a['PASS']}/{a['n']}"
            rec[f"{k}_cat"] = a["cat"]
            rec[f"{k}_worst_ddp"] = round(a["worst_ddp"], 4)
            rec[f"{k}_mean_dR"] = round(a["mean_dR"], 4)
        blocks.append(rec)
    keys = list(blocks[0].keys())
    with open(ANA / "seed_block_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(blocks)

    # ---------- P7 bootstrap + tail ----------
    boot = {}
    for k in METHODS:
        a = A[k]
        boot[k] = {"PASS": a["PASS"], "n_units": a["n"],
                   "pass_rate": round(a["pass_rate"], 4),
                   "pass_rate_CI": [round(x, 4) for x in boot_ci(a["arr_P"])],
                   "cat": a["cat"], "cat_rate": round(a["cat_rate"], 4),
                   "mean_ddp": round(a["mean_ddp"], 4),
                   "mean_ddp_CI": [round(x, 4) for x in boot_ci(a["arr_ddp"])],
                   "mean_dR": round(a["mean_dR"], 4),
                   "mean_dR_CI": [round(x, 4) for x in boot_ci(a["arr_dR"])],
                   "worst_ddp": round(a["worst_ddp"], 4)}
    for k in ("C2", "C6", "Adaptive B2"):
        boot[f"X6c_vs_{k}"] = {
            "dPASS_rate": round(paired(A["X6c"]["arr_P"], A[k]["arr_P"])["point"], 4),
            "dPASS_rate_CI": [round(x, 4) for x in paired(A["X6c"]["arr_P"], A[k]["arr_P"])["CI"]],
            "dmean_ddp": round(paired(A["X6c"]["arr_ddp"], A[k]["arr_ddp"])["point"], 4),
            "dmean_ddp_CI": [round(x, 4) for x in paired(A["X6c"]["arr_ddp"], A[k]["arr_ddp"])["CI"]],
            "dmean_dR": round(paired(A["X6c"]["arr_dR"], A[k]["arr_dR"])["point"], 4),
            "dmean_dR_CI": [round(x, 4) for x in paired(A["X6c"]["arr_dR"], A[k]["arr_dR"])["CI"]],
        }
    boot["X6c_vs_Original"] = {
        "dPASS_rate": round(paired(A["X6c"]["arr_P"], A["Original"]["arr_P"])["point"], 4),
        "dmean_ddp": round(paired(A["X6c"]["arr_ddp"], A["Original"]["arr_ddp"])["point"], 4),
        "dmean_dR": round(paired(A["X6c"]["arr_dR"], A["Original"]["arr_dR"])["point"], 4)}
    tail = {}
    for k in METHODS:
        dd = np.sort(A[k]["arr_ddp"])
        worst = [round(float(dd[i]), 4) for i in (0, 1, 2)]              # worst 1 / 2 / 3
        t = {"worst_1": worst[0], "worst_2": worst[1], "worst_3": worst[2],
             "worst_5": round(float(np.sort(A[k]["arr_ddp"])[:5].mean()), 4)}
        if k != "Original":
            for ref in ("Adaptive B2", "C2", "C6"):
                if ref == k:
                    continue
                r5 = np.sort(A[ref]["arr_ddp"])[:5]
                t[f"worst_5_vs_{ref}"] = paired(np.sort(A[k]["arr_ddp"])[:5], r5)
        tail[k] = t
    json.dump({"bootstrap": boot, "tail": tail}, open(ANA / "statistics.json", "w"),
              indent=2, ensure_ascii=False)

    # ---------- P8 robustness cost ----------
    cost = []
    for k in METHODS:
        if k == "Original":
            continue
        a = A[k]
        rel = paired(A[k]["arr_dR"], A["Adaptive B2"]["arr_dR"])
        cost.append({"method": k, "robustness_mean_absz": round(a["absz"], 4),
                     "preservation_mean_dprime": round(a["dprime"], 4),
                     "tail_safety_worst_ddp": round(a["worst_ddp"], 4),
                     "catastrophic": a["cat"],
                     "robustness_cost_vs_B2_point": round(rel["point"], 4),
                     "robustness_cost_vs_B2_CI": [round(x, 4) for x in rel["CI"]],
                     "complexity": COMPLEXITY[k]})
    with open(ANA / "robustness_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cost[0].keys()))
        w.writeheader(); w.writerows(cost)

    # ---------- console ----------
    print("=" * 122)
    print("P5 — FINAL 50-UNIT TABLE (%d units)" % n_units)
    print("=" * 122)
    print("%-13s%8s%9s%6s%8s%10s%11s%10s%10s" % ("method", "PASS", "rate", "cat", "cat_rate",
                                                   "mean_ddp", "median_ddp", "worst_ddp", "mean_dR"))
    for r in tab:
        print("%-13s%8s%9.3f%6d%8.3f%10.4f%11.4f%10.4f%10.4f" % (
            r["method"], f"{r['PASS']}/{r['n_units']}", r["pass_rate"], r["catastrophic"],
            r["cat_rate"], r["mean_ddp"], r["median_ddp"], r["worst_ddp"], r["mean_dR"]))
    print("-" * 122)
    print("P6 — seed-block stability")
    for b in blocks:
        print("  %-11s n=%2d | " % (b["block"], b["n_units"]) + " | ".join(
            "%s %s(cat%d,worst%.3f)" % (k, b[f"{k}_PASS"], b[f"{k}_cat"], b[f"{k}_worst_ddp"])
            for k in METHODS))
    print("-" * 122)
    print("P7 — bootstrap (X6c vs ...)")
    for k in ("Adaptive B2", "C2", "C6"):
        v = boot[f"X6c_vs_{k}"]
        print("  X6c vs %-12s ΔPASS %+.3f %-18s | Δmean d′ %+.4f %-18s | Δmean ΔR %+.4f %s" % (
            k, v["dPASS_rate"], "[%.3f,%.3f]" % tuple(v["dPASS_rate_CI"]),
            v["dmean_ddp"], "[%.3f,%.3f]" % tuple(v["dmean_ddp_CI"]),
            v["dmean_dR"], "[%.3f,%.3f]" % tuple(v["dmean_dR_CI"])))
    print("-" * 122)
    print("P7 — tail (Δd′ worst 1/2/3/5-mean)")
    for k in METHODS:
        t = tail[k]
        print("  %-13s worst1 %+.4f | worst2 %+.4f | worst3 %+.4f | worst5mean %+.4f" % (
            k, t["worst_1"], t["worst_2"], t["worst_3"], t["worst_5"]))
    print("-" * 122)
    print("P8 — robustness cost vs B2")
    for c in cost:
        print("  %-13s absz %.4f | dprime %.4f | worst %+.4f | cat %d | ΔR vs B2 %+.4f %s" % (
            c["method"], c["robustness_mean_absz"], c["preservation_mean_dprime"],
            c["tail_safety_worst_ddp"], c["catastrophic"], c["robustness_cost_vs_B2_point"],
            "[%.4f,%.4f]" % tuple(c["robustness_cost_vs_B2_CI"])))


if __name__ == "__main__":
    main()
