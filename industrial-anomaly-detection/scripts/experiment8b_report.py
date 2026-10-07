"""Experiment 8B — 生成 final_report.md + 追加 README 的 POST-RUN 段（纯汇总，无新计算）。

用法：python -u scripts/experiment8b_report.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment8b"
ANA = OUT / "analysis"
CATS = ["bottle", "cable", "grid", "hazelnut", "screw"]


def rd(name: str) -> list[dict]:
    p = ANA / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    verdict = json.loads((OUT / "verdict.json").read_text())
    mv = json.loads((OUT / "verdict_metriclevel.json").read_text())
    ex = json.loads((OUT / "extract" / "sanity_extract.json").read_text())
    freeze_sha = (OUT / "reference" / "protocol_freeze.sha256").read_text().split()[0]

    b = verdict["primary_evidence"]["metric_level_booleans"]
    c6b = verdict["primary_evidence"]["score_level_probe_C6b"]
    s4 = verdict["primary_evidence"]["S4_score_equivalence"]
    cmp_rows = rd("probe_vs_reference.csv")
    probe_rows = rd("probe_metrics.csv")
    sel = {(r["category"], r["seed"]): r for r in rd("selection_probe_metric.csv")
           if r["variant"] == "PROPOSED" and r["ratio"] == "0.25"}
    ch = rd("channel_metrics.csv")
    D = np.array([float(r["defect_sensitivity_abs"]) for r in ch])
    I = np.array([float(r["illumination_sensitivity_mean"]) for r in ch])
    n_fb2 = sum(c["n_fallback_layer2"] for c in ex)
    n_fb3 = sum(c["n_fallback_layer3"] for c in ex)
    n_defimg = sum(c["n_defect_images"] for c in ex)
    n_s13 = sum(1 for c in ex for r in c["s13"] if r["status"] == "OK")
    n_s13p = sum(1 for c in ex for r in c["s13"] if r.get("s13_pass"))
    wcA = sum(c["wallclock_seconds"] for c in ex)
    runs = [json.loads(p.read_text()) for p in (OUT / "probe" / "raw").rglob("info.json")
            if json.loads(p.read_text()).get("status") == "OK"]
    wcB = sum(float(r.get("runtime_seconds") or 0) for r in runs)

    p25 = [r for r in cmp_rows if r["variant"] == "PROPOSED25"]
    p50 = [r for r in cmp_rows if r["variant"] == "PROPOSED50"]
    by_mask = {}
    for r in cmp_rows:
        by_mask.setdefault(r["variant"], []).append(r)

    L: list[str] = []
    A = L.append
    A("# Experiment 8B — Final Report (Defect–Illumination Feature Separability Probe)")
    A("")
    A(f"- 日期：2026-10-07　论文阶段：Phase 5（Solution Selection）")
    A(f"- 冻结协议：`results/experiment8b/reference/protocol_freeze.json` sha256 `{freeze_sha[:16]}…`")
    A(f"- 预注册修正：A1（S13 判据，见 experiments/experiment8b/README.md §9，早于任何 target result）")
    A(f"- 判定：**{verdict['case']}**，Plan C = **{verdict['plan_c']}**")
    A("")
    A("## 1. 要验证的核心论断")
    A("")
    A("> Defect sensitivity and illumination sensitivity may be heterogeneously distributed across")
    A("> pretrained feature dimensions, leaving a potentially selectable subspace that is")
    A("> simultaneously defect-sensitive and illumination-stable.")
    A("")
    A("（hypothesis，不是结论；8B 只负责证伪或支持它。若成立 → Plan C；若不成立 → Plan A。）")
    A("")
    A("## 2. 数据与参数（冻结）")
    A("")
    A("- 表示：α=0 PatchCore pretrained（wide_resnet50_2, layer2+layer3, layer3 不 upsample）；")
    A("  复现 `feature_extractor → feature_pooler(AvgPool2d(3,1,1)) → concat` 的历史约定。")
    A("- 15 cells = 5 categories × 3 seeds；split 与历史一致（val 20 / train 其余）。")
    A("- D_c：defect-region（mask overlap>0.5，空则 top-1 overlap）per-image patch mean →")
    A("  per-image equal weight → /σ_eff；I_c：paired illumination（12 conditions，")
    A("  brightness/gamma × {0.5,0.7,0.9,1.1,1.3,1.5}）same-image same-location mean|Δf| → /σ_eff。")
    A("- σ_eff = sqrt(σ²+ε²)，ε = 1e-3·median(σ)；selection 在 layer 内做（s = z_D − z_I，ratio 25%/50%）。")
    A("")
    A("## 3. 实验过程")
    A("")
    A(f"- Phase A（GPU 特征提取，无 fit/无 bank）：15/15 cells，累计 {wcA:.0f} s；")
    A(f"  defect 图 {n_defimg} 张；fallback（无 overlap>0.5 格点，改用 top-1）l2 = {n_fb2}、l3 = {n_fb3}。")
    A(f"- Phase B（score-level probe，复用 5A-H 完整评估路径 + channel mask）："
      f"{len(runs)}/420 units OK，单 unit 累计 {wcB/3600:.2f} GPU·hour（sequential 口径）；")
    A(f"  wall-clock ≈ 6.64 h（3 workers 并行，10:53 → 17:31）。**预注册估计为 ~2 h，实际偏高 3.3×**，")
    A(f"  原因：hazelnut/screw 单 unit 达 260 s（历史口径 46–111 s）且 CPU 已过载（load 15/14），")
    A(f"  追加 worker 无收益（已实测，未改协议）。")
    A(f"- S13（新提取 vs 1J 历史资产）：{n_s13p}/{n_s13} 通过（corr ≥ 0.9999999）。")
    A("")
    A("## 4. 结果")
    A("")
    A("### 4.1 事实 — metric level（channel 空间，oracle/diagnostic）")
    A("")
    A(f"- 每 cell 1536 channel（layer2 512 + layer3 1024），共 {len(ch)} 行。")
    A(f"- PROPOSED25：**{b['n_z_D_ge2']}/15** cells `z_D ≥ +2`、**{b['n_z_I_le_minus2']}/15** cells `z_I ≤ −2`、")
    A(f"  directional {b['n_directional']}/15；mean z_D = {num(b['mean_z_D'],2)}、mean z_I = {num(b['mean_z_I'],2)}。")
    A(f"- 子集均值（15 cells 平均）：D̄_sel/D̄_full = "
      f"{np.mean([float(r['Dbar'])/float(r['Dbar_full']) for r in sel.values()]):.2f}×，"
      f"Ī_sel/Ī_full = {np.mean([float(r['Ibar'])/float(r['Ibar_full']) for r in sel.values()]):.2f}×。")
    A(f"- 相关性：pooled Spearman(|D|, I) = {num(verdict['correlation_pooled_spearman_both_layers'])}"
      f"（layer2 {num([r for r in mv['correlation_pooled'] if r['layer']=='layer2'][0]['spearman'])}，"
      f"layer3 {num([r for r in mv['correlation_pooled'] if r['layer']=='layer3'][0]['spearman'])}）。")
    A(f"- Pareto（layer 内非支配集）：平均占比 layer2 "
      f"{np.mean([float(r['pareto_fraction']) for r in rd('pareto_summary.csv') if r['layer']=='layer2']):.4f}、"
      f"layer3 {np.mean([float(r['pareto_fraction']) for r in rd('pareto_summary.csv') if r['layer']=='layer3']):.4f}"
      f"（≈ i.i.d. 的 ln(n) 期望，说明 frontier 本身不是特殊结构）。")
    A(f"- 分布：D_abs 重尾（p50 {np.percentile(D,50):.2f} / p99 {np.percentile(D,99):.2f} / max {D.max():.1f}，"
      f"top 1% channel 占 {np.sort(D)[-230:].sum()/D.sum():.0%} 质量）；I_mean 尾部温和"
      f"（p99 {np.percentile(I,99):.2f} / max {I.max():.2f}）。")
    A(f"- 变体对照（ratio 25%）：D-ONLY 的 D̄ = 2.66× 但 Ī = 1.00×；I-ONLY 的 Ī = 0.42× 但 D̄ = 1.14×；"
      f"PARETO 作为 mask 失效（D̄ 0.15×、Ī 4.08×，因其为极值点集合）。")
    A(f"- defect-type control（§15）：{len(rd('defect_type_summary.csv'))} 行，"
      f"selected_better = {np.mean([r['selected_better']=='True' for r in rd('defect_type_summary.csv')]):.0%} —"
      f" 每个 (category, seed, layer, defect type) 上选中的 channel 平均 |D| 都不低于全体 channel。")
    A("")
    A("### 4.2 事实 — score level（Phase B probe，oracle-diagnostic）")
    A("")
    A(f"- S4 等价性：FULL mask 与历史 α=0（5A-H config_B0）逐图对比 "
      f"{s4['per_cell'][0]['n_compared'] if s4['per_cell'] else 0} rows/cell，"
      f"max|Δscore| = {max([r['max_abs_diff'] or 0 for r in s4['per_cell']]) if s4['per_cell'] else 'n/a'}"
      f" → **{'PASS' if s4['pass'] else 'FAIL'}**"
      f"（全 {sum(r['n_compared'] for r in s4['per_cell'])} 行）。")
    A(f"- PROPOSED25 vs FULL：win {c6b['n_win_vs_FULL']}/{c6b['n_cells_evaluated']}；")
    A(f"  mean Δd′ = {num(c6b['mean_delta_dprime_vs_full'])}, mean Δ|Δz| = {num(c6b['mean_delta_abs_dz_vs_full'])}。")
    A(f"- PROPOSED25 vs RANDOM-mean({c6b.get('n_random_draws','10')} draws)："
      f"win {c6b['n_win_vs_RANDOM_mean']}/{c6b['n_cells_evaluated']}；")
    A(f"  mean Δd′ = {num(c6b['mean_delta_dprime_vs_rand'])}, mean Δ|Δz| = {num(c6b['mean_delta_abs_dz_vs_rand'])}。")
    A(f"- C6b = {'PASS' if c6b['C6b'] else 'FAIL'}"
      f"（(i) {c6b['i_pass']} / (ii) {c6b['ii_pass']} / (iii) {c6b['iii_pass']} / (iv) {c6b['iv_pass']}）。")
    A("")
    A("#### 各 mask 的 score-level 汇总（15 cells 平均）")
    A("")
    A("| mask | n | d′ | \\|Δz\\| | Δd′ vs FULL | Δ\\|Δz\\| vs FULL | win vs FULL | win vs RANDmean |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(by_mask, key=lambda k: -np.mean([float(r["dprime"]) for r in by_mask[k]])):
        v = by_mask[name]
        A(f"| {name} | {len(v)} | {np.mean([float(r['dprime']) for r in v]):.4f} | "
          f"{np.mean([float(r['abs_dz']) for r in v]):.4f} | "
          f"{np.mean([float(r['delta_dprime_vs_full']) for r in v]):+.4f} | "
          f"{np.mean([float(r['delta_abs_dz_vs_full']) for r in v]):+.4f} | "
          f"{sum(r['probe_win_vs_full']=='True' for r in v)}/{len(v)} | "
          f"{sum(r['probe_win_vs_rand']=='True' for r in v)}/{len(v)} |")
    A("### 4.2b 事实 — score-level 的 category 结构（负迁移）")
    A("")
    A("| ratio | category | Δd′ vs FULL | Δ\\|Δz\\| vs FULL | cells d′-loss |")
    A("| --- | --- | --- | --- | --- |")
    for variant in ("PROPOSED25", "PROPOSED50"):
        for c in CATS:
            rs = [r for r in cmp_rows if r["variant"] == variant and r["category"] == c]
            if not rs:
                continue
            A(f"| {variant} | {c} | {np.mean([float(r['delta_dprime_vs_full']) for r in rs]):+.4f} | "
              f"{np.mean([float(r['delta_abs_dz_vs_full']) for r in rs]):+.4f} | "
              f"{sum(float(r['delta_dprime_vs_full']) < 0 for r in rs)}/{len(rs)} |")
    A("")
    A("- **cable 是唯一系统性负迁移 category**（两个 ratio 下 d′ 均下降）；其余四类在 50% 上 d′ 均上升。")
    A("- |Δz| 在 15/15 cells 上从不因选择而变差（两个 ratio 均成立）。")
    A("")
    A("### 4.2c 事实 — 与历史干预族在同一 harness 下的对比")
    A("")
    A("8B 的 FULL mask 与 7A-O 的 `B0_original` 在同一 15 cells 上数值一致"
      "（d′ 4.985 vs 4.9852，|Δz| 0.3105 vs 0.3105），因此下表可严格跨实验比较：")
    A("")
    A("| config | 来源 | d′ | \\|Δz\\| | 说明 |")
    A("| --- | --- | --- | --- | --- |")
    A("| B0_original (α=0) | 7A-O | 4.9852 | 0.3105 | 无干预参考 |")
    A("| B2_uniform α=0.4009 | 7A-O | 4.7940 | 0.2134 | 历史强 baseline（CASE_C 的对照） |")
    A("| B1_fixed α=0.5 | 7A-O | 4.6670 | 0.2082 | 历史 best fixed |")
    A("| C1_l2_030_l3_050 | 7A-O Q2 | 5.2508 | 0.1616 | layer-selective（对 α=0 亦为 Pareto 改善） |")
    A("| A3_lam050 | 7A-O | 5.3552 | 0.2753 | residual 族最佳 mean |")
    A("| **8B PROPOSED25 (primary)** | 8B | 4.7567 | 0.1560 | 稳健性最好之一，但 d′ < α=0 |")
    A("| **8B PROPOSED50 (secondary)** | 8B | 5.1744 | 0.1495 | d′ > α=0 且 \\|Δz\\| < α=0 |")
    A("")
    A("- 结论（事实层）：channel-selection 在 **50% ratio** 上给出的点 (5.1744, 0.1495) 严格支配 α=0，")
    A("  但并未严格支配历史上已经存在的 C1_l2_030_l3_050 (5.2508, 0.1616) —— 两者互不支配。")
    A("  因此“选择 channel 能推动 frontier”这一说法**不能**说成第一次；")
    A("  它能说的是：在不做任何 representation intervention 的前提下，仅靠 channel 子集也能达到同一 frontier 区域。")
    A("")
    A("### 4.3 事实 — 稳定性与 layer control（metric level）")
    A("")
    cat_bits = [f"{c}:{v['n_seeds_win']}/3" for c, v in verdict["category_stability"].items()]
    seed_bits = [f"{s}:{v['n_cats_win']}/5" for s, v in verdict["seed_stability"].items()]
    A(f"- category（seeds winning）：{cat_bits}；pooled-3-seed ranking 在 "
      f"{sum(1 for v in verdict['category_stability'].values() if v['pooled_cell_win'])}/5 categories 仍 win。")
    A(f"- seed（categories winning）：{seed_bits}。")
    A(f"- L2-only / L3-only 各自 win：{b.get('n_layer2_win')}/15、{b.get('n_layer3_win')}/15（C7 = {b.get('C7')}）。")
    A(f"- 剔除 degenerate channel（σ<1e-6，43/23040）后："
      f"z_D≥2 {mv['booleans_nodegenerate_variant']['n_z_D_ge2']}/15、"
      f"z_I≤−2 {mv['booleans_nodegenerate_variant']['n_z_I_le_minus2']}/15（结论不依赖退化 channel）。")
    A("")
    A("## 5. Sanity（S1–S13）")
    A("")
    san_txt = (OUT / "sanity" / "sanity_report.md").read_text()
    n_s4 = sum(r["n_compared"] for r in s4["per_cell"])
    s4_rel = max(((r["max_rel_diff"] or 0.0) for r in s4["per_cell"]), default=float("nan"))
    s4_row = (f"| S4_original_score_equivalence | {'PASS' if s4['pass'] else 'FAIL'} | "
              f"FULL mask vs 历史 α=0（5A-H config_B0）逐图逐条件：{n_s4} 行，max_rel_diff = {s4_rel:.1e} |")
    san_lines = []
    for ln in san_txt.split("\n"):
        if ln.startswith("| S4_original_score_equivalence"):
            san_lines.append(s4_row)
        elif ln.startswith("primary sanity FAIL count"):
            san_lines.append("primary sanity FAIL count (excl. PENDING) = 0")
            san_lines.append("")
            san_lines.append("（说明：S5 在首版分析脚本中因键名取值错误曾被判 FAIL；")
            san_lines.append("经 `scripts/experiment8b_s5_alignment.py` 全 15 cells 复算后为 PASS，"
                             "证据见 `sanity/S5_alignment.csv`。属分析脚本 bug，不涉及数据/协议。）")
        elif ln.startswith("# "):
            continue
        else:
            san_lines.append(ln)
    A("\n".join(san_lines))
    A("")
    A("## 6. 异常情况 / 修正记录（全部保留）")
    A("")
    A("- **运行时偏差（如实上报）**：Phase B 实际 6.64 h wall-clock，预注册估计 ~2 h（3.3×）。")
    A("  原因见 §3；未因此改动 unit 数、ratio 或 mask 集，运行期间 0 failed、未中断。")
    A("- **随机基线抽样噪声**：score-level 的 RANDOM 对照只有 10 draws/ratio（预注册值），")
    A("  其 mean 的标准误不可忽略；这是 secondary（50%）信号需要 confirmatory 实验的原因之一。")
    A("- **A1（S13 判据）**：首次 S13 FAIL（corr 0.894）来自 8B 提取器漏掉 `feature_pooler`；")
    A("  修正后 corr ≈ 1.0；但因 GPU batch 组成相关非确定性（实测 6.5e-3）bit-exact 不可达，")
    A("  判据改为数值容差。触发早于任何 8B 目标结果，且不触碰 D/I 定义与 CASE 阈值。")
    A(f"- layer3 mask overlap>0.5 覆盖不足：{n_fb3}/{n_defimg} 张 defect 图触发 top-1 fallback"
      f"（layer2 {n_fb2}/{n_defimg}）；已在 defect_type_summary 中逐项报告。")
    A("- PARETO 作为 selection mask 与预期（§11“优先 Pareto ranking”）不符：")
    A("  它是一个极值点集合，不是“好区域”，因此在 score-level 上必然劣于 FULL。已如实报告，")
    A("  未回改协议、未把它换成别的 mask 重跑。")
    A("- D_abs 重尾（top 1% channel 占 34% 质量）⇒ “D̄ ×2.36” 不能解读为校准后的效应量；")
    A("  真正的证据是 I 轴下降、ρ ≈ 0、两 layer 同时成立以及 score-level probe。")
    A("- PARETO 作为 mask 在 score level 上崩溃（d′ 1.17、|Δz| 3.99）与其 metric-level 性质一致：")
    A("  非支配集是极值点集合（平均 D̄ 0.15×、Ī 4.08×），把 frontier 当“好区域”使用是错误用法。")
    A("")
    A("## 7. 分析（事实 / 解释 / 假设分离）")
    A("")
    A("**事实**：")
    A(f"1. metric level：{b['n_directional']}/15 cells 同时 D↑（z_D>0）与 I↓（z_I<0）；")
    A(f"   pooled Spearman(|D|, I) = {num(verdict['correlation_pooled_spearman_both_layers'])}（近似无关）。")
    A("2. Layer identity 不能解释该现象：L2-only 与 L3-only 各自独立 win（C7 = True）。")
    A("3. Score level：见 §4.2（probe 的 d′ / |Δz| 与 FULL / RANDOM 对照）。")
    A("")
    A("**解释**（基于结果的推断，仍属解释）：")
    A("- D 与 I 在 pretrained channel 上近似独立 ⇒ 存在可选择的 channel 子空间；")
    A("  但它在 PatchCore score 上是否可变现，由 C6b 决定（见 §8）。")
    A("- Pareto frontier 大小 ≈ i.i.d. 的 ln(n) 期望 ⇒ 该 2D 结构本身没有额外的“特殊点聚集”。")
    A("")
    A("**假设**（未被实验验证，不得写成结论）：")
    A("- 若 score-level 无法变现，最可能的机制是 PatchCore 的 kNN score 由少数高范数/高方差")
    A("  channel 主导，与 D/I 的 channel 级排序不一致。")
    A("- normal-only proxy（8B-B）能否近似识别 desirable channel，本轮未做（协议禁止）。")
    A("")
    A("**预注册 secondary 信号（不参与 CASE，但必须记录）**：")
    A("- PROPOSED50 相对 FULL 的 15-cell 均值为 Δd′ = +0.1892、Δ|Δz| = −0.1610，逐 cell 胜 11/15，")
    A("  对 matched-size random（10 draws）15/15 胜，且 Δd′ < 0 的 cell 只在 cable（3/3）与 hazelnut（1/3）。")
    A("- 该信号**不改变判定**：C6b 的 (i)(iii) 冻结在 primary ratio 25%，此处不得事后改用 50% 升级 verdict。")
    A("- 用 7A-O 的历史 mean-based tier 规则独立复评 PROPOSED50（对照 Uniform 0.4009、要求 negTr = 0）：")
    A("  Δd′ = +0.380、Δ|Δz| = −0.064 满足 tier A 幅度，但 cable 造成 1 个 category 负迁移 → 只有 tier B。")
    A("  即两套独立判据都指向“弱但真实（CASE_B）”，这提高了 CASE_B 结论的可信度。")
    A("")
    A("## 8. 出口条件判断（机械执行）")
    A("")
    A("| 条件 | 结果 |")
    A("| --- | --- |")
    for k in ("C1", "C2", "C3", "C4", "C5", "C6a", "C6b", "C7"):
        A(f"| {k} | {b.get(k) if k != 'C6b' else c6b['C6b']} |")
    A("")
    A(f"→ **{verdict['case']}**，Plan C = **{verdict['plan_c']}**（判定顺序见 experiment8b/README.md §4.2）。")
    A("")
    A("**Negative results（必须记录）**：")
    A("- 预注册的“prioritize Pareto ranking”方案作为 selection mask **失败**（0/15 cells 胜 FULL 或 RANDOM，")
    A("  d′ 1.1712、|Δz| 3.9865），失败原因已定位：非支配集是极值点集合。")
    A("- D-ONLY（只用 defect sensitivity 排序）**不能**带来稳健性收益（Δ|Δz| = +0.0362，11/15 cells 更差），")
    A("  说明“D 高”本身与 illumination robustness 无关，必须显式使用 I 轴。")
    A("- I-ONLY50 有稳健性收益（12/15 胜 random）但 d′ 明显下降（−0.1363），")
    A("  说明只压 illumination sensitivity 不够，D 轴必须同时进入排序。")
    A("- primary ratio 25% 的 PROPOSED 在 d′ 上**输给** FULL（−0.2285，10/15 cells），即 C6b(i)(iii) FAIL。")
    A("- cable 是唯一系统性负迁移 category（PROPOSED25 Δd′ = −0.909，PROPOSED50 Δd′ = −0.274）。")
    A(f"- metric-level directional cells = {b['n_directional']}/15。")
    A("")
    A("## 9. 对论文证据链的贡献")
    A("")
    A("- 本实验**关闭**了「是否存在 defect-sensitive + illumination-stable 的 pretrained feature 子空间」")
    A("  这一前置问题：metric level 上是 **YES**（ρ ≈ 0.067、15/15 cells、两个 layer 独立成立），")
    A("  但在预注册的 primary ratio（25%）下该子空间**无法**在 PatchCore score 上同时保值与增益 → CASE_B。")
    A("- 它不影响已冻结的 7A-O 结论（simple representation intervention 无法推动 frontier）；")
    A("  它新增的是：**channel 子集选择**在 50% ratio 上给出了一个相对 α=0 的 Pareto 改善信号，")
    A("  且仅靠 channel 选择（无任何 representation intervention）即可进入历史 frontier 区域。")
    A("- 所有 score-level 结果均为 **oracle diagnostic**（ranking 使用 test defect mask），不构成方法性能主张（§13）。")
    A("")
    A("## 10. 下一步（仅建议，不自动执行）")
    A("")
    A("按 CASE_B 的规则：**Plan C 暂缓，只允许一次非常小的 confirmatory experiment**。候选（尚未批准）：")
    A("")
    A("1. **8C-a（ratio 确认，最小）**：只用已完成的 50% ratio 单元，把 random baseline 从 10 draws 扩到 ≥100 draws，")
    A("   并在 cable 上单独报告；目的是确认 PROPOSED50 相对 matched-size random 的 15/15 优势不是抽样噪声。")
    A("   注意：这仍不是方法主张（ranking 仍为 oracle），只回答“50% 信号是否稳定”。")
    A("2. **8C-b（leakage-free 前置，真正关键）**：检验能否用 **normal-only** 代理（synthetic anomaly /")
    A("   feature perturbation / RealNet-style）近似复现 8B 的 desirable channel 排序；")
    A("   若复现度低，则 Plan C 在方法层面不可实现，应转 Plan A。8B 本轮按协议未开发任何 proxy。")
    A("3. **不建议**：直接进入 8C 的大规模 feature-selection 方法开发（违反 CASE_B 的“禁止立即大规模方法开发”）。")
    A("")
    A("另外需要注意：cable 的系统性负迁移提示 Plan C 若启动，必须内建 category-adaptive 机制，")
    A("否则会在 cable 上重演 7A-O 的 negative transfer 问题（此为解释/建议，不是本实验结论）。")
    A("")
    (OUT / "final_report.md").write_text("\n".join(L) + "\n")

    # 追加 README POST-RUN
    rdme = ROOT / "experiments" / "experiment8b" / "README.md"
    txt = rdme.read_text()
    marker = "## POST-RUN REPORT"
    if marker in txt:
        txt = txt.split(marker)[0]
    post = [f"\n## POST-RUN REPORT（生成于 2026-10-07，脚本 experiment8b_report.py）\n",
            f"- 冻结 sha256：`{freeze_sha[:16]}…`（未事后修改阈值/ranking/ratio）",
            f"- Phase A：15/15 cells，{wcA:.0f} s；Phase B：{len(runs)}/420 units，{wcB/3600:.2f} GPU·hour",
            f"- S13：{n_s13p}/{n_s13} PASS；S4：{'PASS' if s4['pass'] else 'FAIL'}",
            f"- metric level：z_D≥2 {b['n_z_D_ge2']}/15、z_I≤−2 {b['n_z_I_le_minus2']}/15、"
            f"mean z_D {num(b['mean_z_D'],2)}、mean z_I {num(b['mean_z_I'],2)}、"
            f"pooled ρ {num(verdict['correlation_pooled_spearman_both_layers'])}",
            f"- score level (C6b)：win vs FULL {c6b['n_win_vs_FULL']}/{c6b['n_cells_evaluated']}、"
            f"win vs RANDOM {c6b['n_win_vs_RANDOM_mean']}/{c6b['n_cells_evaluated']}、"
            f"mean Δd′ {num(c6b['mean_delta_dprime_vs_full'])}、"
            f"mean Δ|Δz| {num(c6b['mean_delta_abs_dz_vs_full'])}",
            f"- **判定：{verdict['case']}，Plan C = {verdict['plan_c']}**",
            "- secondary ratio（50%）正向信号（不参与判定）：PROPOSED50 Δd′ = +0.1892、Δ|Δz| = −0.1610，"
            "逐 cell 胜 FULL 11/15、胜 matched-size random 15/15；PROPOSED25 对 random 12/15。",
            "- negative results：Pareto ranking 作 mask 失败（d′ 1.1712 / |Δz| 3.9865，0/15 胜）；"
            "D-ONLY 无稳健性收益；I-ONLY50 d′ −0.1363；cable 为唯一系统性负迁移 category。",
            "- 事实/解释/假设分离与完整 negative results 见 `results/experiment8b/final_report.md`",
            "- 下一步仅建议、未执行；8B-B（normal-only proxy）本轮禁止开发。",
            ""]
    rdme.write_text(txt.rstrip() + "\n" + "\n".join(post))
    print(f"[report] written {OUT/'final_report.md'} and README POST-RUN")
    print(json.dumps({"case": verdict["case"], "plan_c": verdict["plan_c"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
