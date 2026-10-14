#!/usr/bin/env python
"""E19 verification gate -- machine-readable audit of every E19 artifact.

Checks, in groups:
  SRC    source artifacts exist + sha256 recorded (provenance)
  BETA   results/e19/tables/beta_ablation.csv  ==  results/e17/e17_partial.json  (1e-12)
         and the frozen E17 advancement gates recomputed (tolerance 5e-5, see note)
  MAIN   results/e19/tables/main_results*.csv   ==  results/e18/main_table.csv  (1e-12)
         + bootstrap JSON + "empty cells stay empty"
  ILLUM  results/e19/tables/illum_groups.csv    ==  results/e18/illum_breakdown.csv (1e-12)
         + spread values cross-checked against results/e18/statistical_analysis.md
  FIG    every figure exists as non-trivial PNG + PDF
  NOTES  every notes file exists and carries its Source line
  IDEM   rerunning scripts/e19_figures.py reproduces byte-identical CSV/MD/notes
  GIT    E19 touched nothing outside {results/e19/**, scripts/e19_*, README.md, PROJECT_STATE.md}
         and the 6 known legacy worktree entries are still present

Writes results/e19/verification.csv. Exit code 1 if any check fails.
CPU-only; reads frozen artifacts, writes only inside results/e19/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
E16P = RES / "e16" / "e16_partial.json"
E17R = RES / "e17" / "e17_partial.json"
E17F = RES / "e17" / "final_method.json"
E18D = RES / "e18"
OUT = RES / "e19"
TAB = OUT / "tables"
FIG = OUT / "figures"
NOTES = OUT / "notes"

CATS = ["Bird", "Car", "Cube"]

KNOWN_LEGACY = {
    "industrial-anomaly-detection/monitor/exp_monitor.py",
    "industrial-anomaly-detection/results/e8/status.json",
    "industrial-anomaly-detection/monitor/events.jsonl",
    "industrial-anomaly-detection/scripts/_lockprobe.txt",
    "industrial-anomaly-detection/scripts/_probe2.py",
    "industrial-anomaly-detection/scripts/e11_p4_runner.py",
}
ALLOWED_FILES = {
    "industrial-anomaly-detection/README.md",
    "industrial-anomaly-detection/PROJECT_STATE.md",
    "industrial-anomaly-detection/.gitignore",
    "industrial-anomaly-detection/scripts/e19_figures.py",
    "industrial-anomaly-detection/scripts/e19_verify.py",
}
ALLOWED_PREFIX = "industrial-anomaly-detection/results/e19/"

rows: list[dict] = []


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(group, check, expected, actual, tolerance, ok, source):
    rows.append({
        "check_id": f"{group}-{sum(1 for r in rows if r['group'] == group) + 1:02d}",
        "group": group, "check": check,
        "expected": str(expected), "actual": str(actual),
        "tolerance": str(tolerance), "PASS": bool(ok), "source": source,
    })


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fget(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


# --------------------------------------------------------------------------
def check_src():
    files = [E16P, E17R, E17F, E18D / "main_table.csv", E18D / "illum_breakdown.csv",
             E18D / "bootstrap_f0_vs_ours.json", E18D / "native_scoring_appendix.csv",
             E18D / "statistical_analysis.md"]
    for p in files:
        ok = p.exists() and p.stat().st_size > 0
        rel = str(p.relative_to(ROOT))
        add("SRC", f"source artifact present: {rel}", "exists & non-empty",
            f"sha256:{sha(p)[:16]}" if ok else "MISSING", "-", ok, rel)


def check_beta():
    e17 = load_json(E17R)["cells"]
    fm = load_json(E17F)
    rows_b = read_rows(TAB / "beta_ablation.csv")
    add("BETA", "beta_ablation.csv row count", 15, len(rows_b), 0,
        len(rows_b) == 15, "results/e19/tables/beta_ablation.csv")

    by = {(r["category"], r["config"]): r for r in rows_b}
    cols = ["auroc", "ap", "d_prime", "robustness_feat", "robustness_score",
            "illum_shift", "std_ratio", "pairwise_cos"]
    for col in cols:
        worst = 0.0
        for c in CATS:
            for cfg in ["C0", "C1", "C2", "C3", "C4"]:
                src = e17[f"{c}|0"]["configs"][cfg][col]
                worst = max(worst, abs(fget(by[(c, cfg)], col) - src))
        add("BETA", f"beta_ablation.csv column '{col}' vs e17_partial.json (max|delta|)",
            0.0, worst, 1e-12, worst <= 1e-12,
            "results/e19/tables/beta_ablation.csv vs results/e17/e17_partial.json")

    worst = 0.0
    for c in CATS:
        cc = e17[f"{c}|0"]["configs"]
        for cfg in ["C0", "C1", "C2", "C3", "C4"]:
            worst = max(worst, abs(fget(by[(c, cfg)], "beta") - cc[cfg]["beta"]),
                        abs(fget(by[(c, cfg)], "steps_used") - cc[cfg]["steps_used"]))
            ok_flag = (str(by[(c, cfg)]["collapsed"]) == str(bool(cc[cfg]["collapsed"])))
            if not ok_flag:
                add("BETA", f"collapsed flag {c}/{cfg} matches e17_partial", True, False, 0,
                    False, "results/e17/e17_partial.json")
    add("BETA", "beta / steps_used match e17_partial.json (max|delta|)", 0.0, worst, 1e-9,
        worst <= 1e-9, "results/e17/e17_partial.json")

    # dAUROC_vs_F0 recompute
    e16 = load_json(E16P)["cells"]
    worst = 0.0
    for c in CATS:
        f0 = e16[f"{c}|0"]["f0"]
        for cfg in ["C0", "C1", "C2", "C3", "C4"]:
            v = e17[f"{c}|0"]["configs"][cfg]
            worst = max(worst,
                        abs(fget(by[(c, cfg)], "dAUROC_vs_F0") - (v["auroc"] - f0["auroc"])),
                        abs(fget(by[(c, cfg)], "dAP_vs_F0") - (v["ap"] - f0["ap"])))
    add("BETA", "delta vs F0 columns recompute (max|delta|)", 0.0, worst, 1e-12,
        worst <= 1e-12, "results/e19/tables/beta_ablation.csv vs results/e16/e16_partial.json")

    # frozen gates recomputed at full precision. NOTE: the frozen gates were computed
    # from the 5-decimal round1.csv record -> tolerance 5e-5, margin analysis in notes.
    gates = fm["advancement_gates"]
    for cfg in ["C1", "C2", "C3", "C4"]:
        bc_au = sum(e17[f"{c}|0"]["configs"][cfg]["auroc"]
                    - e17[f"{c}|0"]["configs"]["C0"]["auroc"] for c in ["Bird", "Car"]) / 2
        bc_ap = sum(e17[f"{c}|0"]["configs"][cfg]["ap"]
                    - e17[f"{c}|0"]["configs"]["C0"]["ap"] for c in ["Bird", "Car"]) / 2
        cube = (e17["Cube|0"]["configs"][cfg]["auroc"]
                - e17["Cube|0"]["configs"]["C0"]["auroc"])
        bc_rob = sum(e17[f"{c}|0"]["configs"][cfg]["robustness_score"]
                     - e17[f"{c}|0"]["configs"]["C0"]["robustness_score"]
                     for c in ["Bird", "Car"]) / 2
        for key, rec in (("bird_car_mean_dAUROC_vs_C0", bc_au),
                         ("bird_car_mean_dAP_vs_C0", bc_ap),
                         ("cube_dAUROC_vs_C0", cube),
                         ("bird_car_mean_d_rob_vs_C0", bc_rob)):
            d = abs(gates[cfg][key] - rec)
            add("BETA", f"frozen gate {cfg}/{key} recomputed from full precision",
                0.0, d, 5e-5, d <= 5e-5, "results/e17/final_method.json vs e17_partial.json")
    # smallest gate margin must dwarf the precision tolerance
    margin = min(abs(gates[c]["bird_car_mean_dAUROC_vs_C0"] - 0.003) for c in ["C1", "C2", "C3", "C4"])
    add("BETA", "smallest distance to the +0.003 gate exceeds precision tolerance (1e-4)",
        "> 1e-4", margin, 1e-4, margin > 1e-4, "results/e17/final_method.json")


def check_main():
    mt = {r["method"]: r for r in read_rows(E18D / "main_table.csv")}
    e19rows = read_rows(TAB / "main_results.csv")
    add("MAIN", "main_results.csv row count", 6, len(e19rows), 0, len(e19rows) == 6,
        "results/e19/tables/main_results.csv")
    worst, n_nonempty, n_empty = 0.0, 0, 0
    for r in e19rows:
        ref = mt[r["method"]]
        for c in CATS:
            for k in ("AUROC", "AP"):
                v, s = fget(r, f"{c}_{k}"), fget(ref, f"{c}_{k}")
                if s is None:
                    n_empty += 1
                    if v is not None:
                        worst = float("inf")
                else:
                    n_nonempty += 1
                    worst = max(worst, abs(v - s) if v is not None else float("inf"))
    add("MAIN", "non-empty cell count", 32, n_nonempty, 0, n_nonempty == 32,
        "results/e19/tables/main_results.csv")
    add("MAIN", "empty cells stay empty (never imputed)", 4, n_empty, 0, n_empty == 4,
        "results/e19/tables/main_results.csv")
    add("MAIN", "all cell values == e18/main_table.csv (max|delta|)", 0.0, worst, 1e-12,
        worst <= 1e-12, "results/e19/tables/main_results.csv vs results/e18/main_table.csv")

    bs = load_json(E18D / "bootstrap_f0_vs_ours.json")
    rows3 = read_rows(TAB / "main_results_3seed.csv")
    add("MAIN", "main_results_3seed.csv row count", 6, len(rows3), 0, len(rows3) == 6,
        "results/e19/tables/main_results_3seed.csv")
    worst = 0.0
    for r in rows3:
        c, m, tag = r["category"], r["metric"], r["metric"].lower()
        ref = mt["F0"]
        rou = mt["Ours (R1-ILA)"]
        pairs = [
            (float(r["F0_mean"]), float(ref[f"{c}_{m}_3seed_mean"])),
            (float(r["F0_std"]), float(ref[f"{c}_{m}_3seed_std"])),
            (float(r["Ours_mean"]), float(rou[f"{c}_{m}_3seed_mean"])),
            (float(r["Ours_std"]), float(rou[f"{c}_{m}_3seed_std"])),
            (float(r["delta_seedmean"]), bs[f"{c}|seedmean"][f"d_{tag}"]),
            (float(r["delta_seed0"]), bs[f"{c}|seed0"][f"d_{tag}"]),
            (float(r["ci_lo"]), bs[f"{c}|seedmean"][f"d_{tag}_ci"][0]),
            (float(r["ci_hi"]), bs[f"{c}|seedmean"][f"d_{tag}_ci"][1]),
        ]
        for a, b in pairs:
            worst = max(worst, abs(a - b))
    add("MAIN", "3-seed table + CI values == e18 artifacts (max|delta|)", 0.0, worst, 1e-12,
        worst <= 1e-12, "e18/main_table.csv + e18/bootstrap_f0_vs_ours.json")

    # markdown table cross-check (displayed values rounded to 5 dp)
    mdtxt = (TAB / "main_results.md").read_text(encoding="utf-8")
    worst = 0.0
    n_checked = 0
    for r in e19rows:
        m = re.search(r"^\| " + re.escape(r["method"]) + r" \| (.*?) \|$", mdtxt, re.M)
        if not m:
            worst = float("inf")
            continue
        cells = [x.strip() for x in m.group(1).split("|")]
        for i, c in enumerate(CATS):
            for j, k in enumerate(("AUROC", "AP")):
                v = fget(r, f"{c}_{k}")
                cell = cells[i * 2 + j]
                if v is None:
                    ok = cell == "—"
                    worst = worst if ok else float("inf")
                else:
                    worst = max(worst, abs(float(cell) - v))
                    n_checked += 1
    add("MAIN", "main_results.md display cells match CSV (max|delta| at 5 dp)",
        0.0, worst, 5e-6, worst <= 5e-6 and n_checked == 32,
        "results/e19/tables/main_results.md")


def check_illum():
    src = read_rows(E18D / "illum_breakdown.csv")
    e19rows = read_rows(TAB / "illum_groups.csv")
    add("ILLUM", "illum_groups.csv row count", 264, len(e19rows), 0, len(e19rows) == 264,
        "results/e19/tables/illum_groups.csv")
    srcset = {r["source"] for r in src}
    e19set = {r["source"] for r in e19rows}
    add("ILLUM", "unique source set preserved", len(srcset), len(e19set), 0,
        srcset == e19set, "results/e18/illum_breakdown.csv")
    bykey = {(r["source"], r["illumination"]): r for r in e19rows}
    for col in ["n_ok", "n_ng", "auroc", "ap", "good_mean", "good_std"]:
        worst = 0.0
        for r in src:
            worst = max(worst, abs(fget(bykey[(r["source"], r["illumination"])], col)
                                   - float(r[col])))
        add("ILLUM", f"illum_groups.csv column '{col}' vs e18 source (max|delta|)",
            0.0, worst, 1e-12, worst <= 1e-12,
            "results/e19/tables/illum_groups.csv vs results/e18/illum_breakdown.csv")
    # rel_good_mean recompute
    allref = {r["source"]: float(r["good_mean"]) for r in src if r["illumination"] == "ALL"}
    worst = 0.0
    for r in e19rows:
        worst = max(worst, abs(float(r["rel_good_mean"]) - float(r["good_mean"]) / allref[r["source"]]))
    add("ILLUM", "rel_good_mean recompute (max|delta|)", 0.0, worst, 1e-12, worst <= 1e-12,
        "results/e19/tables/illum_groups.csv")
    # spread cross-check against the E18 statistical_analysis.md reported values (4 dp display)
    md = (E18D / "statistical_analysis.md").read_text(encoding="utf-8")
    expected = {"Bird": (0.0358, 0.0178), "Car": (0.0889, 0.0692), "Cube": (0.0279, 0.0192)}
    groups = [f"{i:02d}" for i in range(1, 11)]
    worst = 0.0
    for c, (ef0, eou) in expected.items():
        if not re.search(r"^\| " + c + r" \| ", md, re.M):
            worst = float("inf")
            continue
        my = {}
        for meth in ("F0", "Ours"):
            src = f"{meth}|{c}|seed0"
            gm = [fget(bykey[(src, g)], "good_mean") for g in groups]
            my[meth] = max(gm) - min(gm)
        worst = max(worst, abs(my["F0"] - ef0), abs(my["Ours"] - eou))
    add("ILLUM", "spread values match results/e18/statistical_analysis.md (4 dp display)",
        0.0, worst, 5e-4, worst <= 5e-4 and worst != float("inf"),
        "results/e18/statistical_analysis.md §B.1 vs results/e19/tables/illum_groups.csv")


def check_fig_notes():
    stems = ["fig_beta_auroc", "fig_beta_decision", "fig_beta_collapse",
             "fig_illum_auroc", "fig_illum_drift", "fig_illum_context",
             "fig_main_auroc", "fig_main_forest", "fig_main_3seed"]
    for s in stems:
        png, pdf = FIG / f"{s}.png", FIG / f"{s}.pdf"
        ok = png.exists() and png.stat().st_size > 10_000 and pdf.exists() and pdf.stat().st_size > 8_000
        add("FIG", f"figure present & non-trivial: {s}.png/.pdf", "png>10kB & pdf>8kB",
            f"png={png.stat().st_size if png.exists() else 0}/pdf={pdf.stat().st_size if pdf.exists() else 0}",
            "-", ok, "results/e19/figures/")
    for n in ["e19_beta_ablation_notes.md", "e19_illum_groups_notes.md",
              "e19_main_results_notes.md"]:
        p = NOTES / n
        ok = p.exists() and "Source (read-only)" in p.read_text(encoding="utf-8")
        add("NOTES", f"notes present with Source line: {n}", "exists & has Source line",
            f"{p.stat().st_size} bytes" if p.exists() else "MISSING", "-", ok,
            f"results/e19/notes/{n}")


def check_idem_git():
    targets = [TAB / "beta_ablation.csv", TAB / "illum_groups.csv", TAB / "main_results.csv",
               TAB / "main_results.md", TAB / "main_results_3seed.csv",
               NOTES / "e19_beta_ablation_notes.md", NOTES / "e19_illum_groups_notes.md",
               NOTES / "e19_main_results_notes.md"]
    before = {t.name: sha(t) for t in targets}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "e19_figures.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    ok_run = r.returncode == 0
    add("IDEM", "scripts/e19_figures.py rerun exit code", 0, r.returncode, 0, ok_run,
        "scripts/e19_figures.py")
    for t in targets:
        same = ok_run and sha(t) == before[t.name]
        add("IDEM", f"rerun reproduces byte-identical: {t.name}",
            before[t.name][:16], sha(t)[:16] if t.exists() else "MISSING", "-", same,
            str(t.relative_to(ROOT)))

    st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                        capture_output=True, text=True)
    entries = []
    for line in st.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        entries.append((line[:2], path))
    unexpected = []
    for code, path in entries:
        if path in KNOWN_LEGACY or path in ALLOWED_FILES or path.startswith(ALLOWED_PREFIX):
            continue
        unexpected.append(f"{code} {path}")
    add("GIT", "no E19 write outside {results/e19/**, scripts/e19_*, README.md, PROJECT_STATE.md}",
        "[]", "; ".join(unexpected) if unexpected else "[]", "-", not unexpected,
        "git status --porcelain")
    missing_legacy = [p for p in KNOWN_LEGACY if not (ROOT.parent / p).exists()]
    add("GIT", "6 known legacy worktree paths still present (untouched)",
        6, 6 - len(missing_legacy), 0, not missing_legacy, "git status / workspace")
    hist = [f"{c} {p}" for c, p in entries
            if p.startswith("industrial-anomaly-detection/results/e1")
            and not p.startswith(ALLOWED_PREFIX)]
    add("GIT", "no modified tracked file under results/e1x (frozen history intact)",
        "[]", "; ".join(hist) if hist else "[]", "-", not hist,
        "git status --porcelain")


def main():
    for fn in (check_src, check_beta, check_main, check_illum, check_fig_notes, check_idem_git):
        fn()
    outp = OUT / "verification.csv"
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check_id", "group", "check", "expected",
                                          "actual", "tolerance", "PASS", "source"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(1 for r in rows if r["PASS"])
    n_fail = len(rows) - n_pass
    print(json.dumps({"total": len(rows), "pass": n_pass, "fail": n_fail,
                      "failed_checks": [r["check_id"] + " " + r["check"]
                                        for r in rows if not r["PASS"]]},
                     indent=2, ensure_ascii=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
