# PAPER EVIDENCE MAP — 论文逻辑 × 实验/图表证据对照

**用途**：把 Exp1–Exp16 的证据按论文叙事线组织；每项标注 supporting experiment / figure-table / strength / remaining gap。
**状态**：Exp16 结束时的最新版本（方法侧已到 Method FREEZE 决策点）。

| # | 论文论点（叙事线） | Supporting experiment | Figure / Table | Strength | Remaining gap |
|---|---|---|---|---|---|
| 1 | **现实问题 / motivation**：工业视觉在照明变化下性能下降 | Exp1（illumination trade-off）、Exp1H/1I（层级特异） | 根 README §5–7；Exp1/1H/1I README | 强（多类别多 seed） | 缺真实产线数据（E4） |
| 2 | **NDIL 现象**：照明变化主要影响 normal 分布而非缺陷可分性 | Exp1、Exp5A-H、Exp6A | 5A-H/6A 表 | 强 | 缺跨数据集复现 |
| 3 | **defect-dependent response**：α-IN 的鲁棒性收益与缺陷保持损失不对称 | Exp1H、Exp1I、Exp5A-H | 1H/1I 表 | 中–强 | 机制解释仍为经验性 |
| 4 | **size-confound exclusion**：损失不能仅由缺陷尺寸差异解释 | Exp5C、Exp1J | 5C/1J 表 | 中 | 仅限已测类别 |
| 5 | **cross-category evidence**：效应跨类别方向一致但幅度依赖类别 | Exp5A-H、Exp7A-O、Exp10/11 | 7A-O/10/11 leaderboard | 强 | cable/hazelnut 仍为难点 |
| 6 | **simple-property failure**：简单属性（s_c、q_c、RMS 等）无法解释 category dependency | Exp8B、Exp9A/9B | 8B/9B 相关表 | 中（negative result） | 仍是 negative，非正面机制 |
| 7 | **layer-specific evidence**：L2 与 L3 的作用不同（L2 偏 preservation、L3 偏 robustness） | Exp1H、Exp1I、Exp7A-O、Exp14（A5/A6/A3） | 7A-O 层级表；Exp14 §6 | 强 | 更细的 channel 级机制未确立 |
| 8 | **transmission-chain evidence**：扰动沿层级传播并被放大 | Exp1H/1I、Exp3（若在仓库） | 1H/1I 图 | 中 | 需更直接的链式测量 |
| 9 | **方法筛选**：固定 C2 → Adaptive B2 → Category×Channel（STOP）→ Dual-Path（代数否定）→ INSS（STOP）→ late fusion（胜出） | Exp10、Exp11、Exp12、Exp14、Exp15 | Exp10/11/12/14/15 leaderboard | **强**（统一协议、跨路径 bit-exact 核验） | 已完成 |
| 10 | **最终 mitigation**：late score fusion（C2/C6/X6c）在同等 PASS 下消除 catastrophic | **Exp16**（50 单元）、Exp14/15 | Exp16 Fig1–Fig5；final_50unit_table.csv | **强（E3+：5 类 × 10 seeds）** | **E4 跨数据集未做**（见 HELD_OUT_VALIDATION_PLAN.md） |
| 11 | **remaining limitation**：①无 held-out 数据集；②robustness cost 显著（fusion 的 ΔR 增益小于 B2）；③机制层面仍缺 defect↔illumination 的因果证据（Exp13-P 已证非特异） | Exp13-P、Exp15 §12、Exp16 §limitations | Exp13-P README；Exp15/16 limitations | 明确（诚实记录） | 需 E4 + 更细机制实验 |

## 负结果清单（论文必须报告，不可隐藏）

| 负结果 | 实验 | 结论 |
|---|---|---|
| 简单属性无法解释 category dependency | Exp8B/9A/9B | negative |
| Category × Channel gating 无效 | Exp12 | STOP（M1/M2/M3 全部硬性淘汰） |
| illumination-nuisance subspace 与 defect 方向**非特异**纠缠 | Exp13-P | 否决 defect-entanglement 叙事 |
| dual-path 线性融合 **代数上**只是 B2 强度重参数化 | Exp14 | 结构性否定（无 Pareto 改进） |
| Tiny INSS 不改善 robustness（λ 大时反而有害） | Exp14 | Family B 全家族 STOP |
| 层强度组合（X4a/X4b）被 B2 支配 | Exp15 | 淘汰 |
| normal-only category-adaptive 融合规则未超过固定权重 | Exp15 | honest negative |
| **B2 的 catastrophic 率被低估**（25→35 单元后为 11.4%） | Exp14/15 | 对冠军的修正 |
| **Exp14 的 top-1 选择存在 seed 过拟合**（C1 在 unseen seeds 退化） | Exp15/16 | 方法论警示 |

## 协议可信度证据（方法学资产）

| 证据 | 数值 |
|---|---|
| `E14_A5` ≡ Exp10 `P10_T1`（跨代码路径） | **max\|Δscore\| = 0.0（bit-exact）** |
| `E14_A1` ≡ Exp11 `E11_B2`（3 类） | **max\|Δscore\| = 0.0** |
| `E14_A6` ≡ Exp11 `E11_B2_L2only`（本次债务审计补核） | **max\|Δscore\| = 0.0** |
| 9C Matched-RNG V2（equivalent pair） | **NF 0.0202/0.0542 → 0.000000/0.000000** |
| Exp16 S1 历史重建一致性 | 15 个 (cat,seed,method) 三元组 max\|Δ\| ≤ 1e-9，PASS/catastrophic 标志完全一致 |
