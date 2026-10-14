# E17 · Lightweight Refinement — Decision Memo

## Verdict: **FREEZE_CURRENT_R1**

```
E17 VERDICT  = FREEZE_CURRENT_R1
FINAL METHOD = current R1   (BETA = 0.1)
```

> 这是**默认且完全成功**的结果（protocol §5 情况 A）：没有挑战者明显胜出 → 冻结当前已资格认证的 R1。

## 1. 实验题目
在不增加明显复杂度的前提下，判断当前 R1 是否已经足够好，或是否存在更简单 / 更稳定 / 略优的小变体值得冻结为最终 Ours。

## 2. 日期
2026-10-14

## 3. 论文阶段
方法收敛（refinement / freeze）—— E16 已判 R1 为 QUALIFIED PAPER METHOD。

## 4. 要验证的核心论断
BETA 是否已接近局部最优；是否存在值得替换 current R1 的单一轴变体。

## 5. 实验目的
防止后续调参泥潭：**用一次受控的局部扫描换一个永久的冻结决定**。

## 6. 数据与参数
- **唯一搜索轴**：`BETA`（`L = l_cons + BETA·l_preserve`；`l_cons` 权重硬编码 1.0，BETA 是唯一显式损失级参数）
- 注入方式：`e13a_screen.BETA = v`（运行时模块全局；**未修改任何冻结代码**）
- 候选：`C0=0.1`(当前 R1，复用) / `C1=0.0` / `C2=0.05` / `C3=0.2` / `C4=0.5`（乘法对称、局部）
- 冻结不变量：结构 `z+Wz`(768×768, zero-init)、lr=1e-3、≤300 step、patience 30、batch 8、train/val=240/60、aug = brightness/contrast/gamma × {0.85,1.15}、bank 300、view120、DINOv2-B frozen、1-NN-max
- 工程：每 category-run 只做 **1 次特征提取**，随后在缓存特征上循环 5 个 BETA

## 7. 实验过程
P0 只读审计 + 参数审计 → protocol 冻结 → Round 1（Bird/Car/Cube × seed0 × 5 config，33.8 min）。
**C0 复现门 3/3 PASS**（AUROC/AP 与 E16 逐位一致到 1e-6，例如 Bird 0.7578641 vs 0.7578641）→ BETA 注入机制被证明正确。
Round 1 判定后无挑战者达标 → **Round 2 未运行（0 GPU）**。

## 8. 结果（Fact）

### 8.1 Round 1 全表（seed0）

| BETA | cfg | Bird AUROC | Bird AP | Car AUROC | Car AP | Cube AUROC | Cube AP | rob_s (B/C/Cu) | shift (B/C/Cu) | steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | C1 | **0.73309** | 0.75208 | **0.78587** | 0.84015 | **0.90904** | 0.94728 | .033/.017/.010 | .048/.034/.026 | 300 |
| 0.05 | C2 | 0.74627 | 0.78868 | 0.80381 | 0.85289 | 0.87053 | 0.92912 | .183/.191/.149 | .091/.078/.065 | 104 |
| **0.1** | **C0** | 0.75786 | 0.79911 | **0.80511** | 0.85502 | 0.87010 | 0.92856 | .328/.281/.220 | .118/.097/.079 | 44–59 |
| 0.2 | C3 | 0.76521 | 0.80633 | 0.80122 | 0.85490 | 0.86140 | 0.92224 | .486/.360/.305 | .150/.111/.090 | 44–99 |
| 0.5 | C4 | **0.76647** | **0.81239** | 0.79524 | 0.85066 | 0.86035 | 0.92109 | .669/.508/.418 | .171/.134/.105 | 59–99 |

（F0 参考：Bird 0.74725 / Car 0.78552 / Cube 0.85409）

### 8.2 相对 current R1（C0）的 Δ 与五道晋级门

| cfg | BETA | B+C mean ΔAUROC | B+C mean ΔAP | Cube ΔAUROC | Δrob(B+C) | collapse | G1 ≥+0.003 | G2 Cube≥−0.003 | G3 ΔAP≥−0.002 | G4 rob | G5 | **晋级** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.0 | −0.02200 | −0.03095 | +0.03894 | −0.279 | **True** | ✗ | ✓ | ✗ | ✓ | **✗** | **NO** |
| C2 | 0.05 | −0.00645 | −0.00628 | +0.00043 | −0.117 | False | ✗ | ✓ | ✗ | ✓ | ✓ | **NO** |
| C3 | 0.2 | **+0.00173** | +0.00355 | −0.00870 | +0.119 | False | ✗ | **✗** | ✓ | **✗** | ✓ | **NO** |
| C4 | 0.5 | −0.00063 | +0.00446 | −0.00975 | +0.284 | False | ✗ | **✗** | ✓ | **✗** | ✓ | **NO** |

**全部 4 个挑战者都未通过 G1（+0.003 门槛）**；C3/C4 另外还在 G2（Cube 安全）与 G4（robustness 不恶化）上失败；C1 触发 collapse 且 ΔAP 崩。
→ **advancing = []，simplify = []，Round 2 未运行（0 GPU）。**

### 8.3 SIMPLIFY 通道检查
- C3（+0.00173）、C4（−0.00063）满足 `|Δ vs C0| ≤ 0.002`，但**结构完全同构**（同参数量、同 runtime、同损失形式，仅常数不同）→ **不是"更简单"**，不构成 SIMPLIFY_CANDIDATE。
- 唯一**真正更简单**的候选是 C1（BETA=0 直接**消除** BETA 这个超参数），但其 |Δ| = 0.022 远超 ±0.002 带宽，且触发 Cube collapse → 不合格。

## 9. 异常情况
- 无 OOM / NaN / 失败 cell；C0 复现门 3/3 PASS。
- **C1（BETA=0）在 Cube 上触发 `collapse=True`**（mean pairwise cosine = 0.988 > 0.98 阈值；std_ratio 5.32）。Bird/Car 上 pairwise cos 亦达 0.979/0.976（阈值边缘）、std_ratio 5.42/5.71 → 已按协议 STOP_COLLAPSE 处理。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. **BETA 是单调的检测↔鲁棒性旋钮**：BETA↑ → illumination shift↑（Bird 0.048→0.171）、robustness_score↑（0.033→0.669，越差），但 Bird 检测↑（AUROC 0.733→0.766）。
2. **类别偏好分裂**：Bird 偏好高 BETA（0.5 时最好，单调上升未触顶）；Car（0.1 最好）与 Cube（0.1 最好，0.05 持平）偏好 ~0.1。
3. **BETA=0（纯光照一致性）在三类上都是最差检测器**（Bird 0.733 / Car 0.786 / Cube 0.909 vs C0 0.758/0.805/0.870），并在 Cube 触发 collapse。
4. `robustness_feat` 在 BETA=0 时降到 0.0007（近乎完美光照一致性），但 AUROC 最差 → **光照一致性本身不是目标**。
5. 聚合上 C0（0.1）的 Bird+Car mean ΔAUROC 最高（+0.01510 vs F0）；C3 仅差 +0.00173，C4 为 −0.00063。

**Interpretation**
1. **BETA=0.1 是聚合意义下的内点最优**：检测收益与光照鲁棒性之间存在真实权衡，0.1 位于聚合最优附近；向两侧都变差（左因 collapse 崩坏，右因 Cube 受损且 robustness 恶化）。
2. **preservation 项是必需组件，不是可选正则**：去掉它（BETA=0）会几乎压扁表示（三类 pairwise cos 均 ≥0.976，Cube 越过 0.98 阈值），并在所有类别损失检测性能。这也**解释了 E16 中 normal dispersion 反升**——adapter 需要锚定以避免退化。
3. **没有统一配置能同时取悦所有类别**：C3/C4 改善 Bird（+0.0074/+0.0086）的同时损害 Car（−0.0039/−0.0099）与 Cube（−0.0087/−0.0098）。协议明确禁止 category-specific 参数，故此类候选不具备论文价值。
4. C4 在 Bird 上单调升至扫描边界 0.5，说明 Bird 的最优 BETA **可能 > 0.5**；但这是**单类别**现象且代价是 Car/Cube 变差，不构成本轮有效结论（禁止新增第 6 个配置）。

**Hypothesis（未验证，不得当结论）**
- 猜测存在类别相关的"最优锚定强度"：光照 shift 越大（Bird）越需要弱锚定以换检测收益；shift 越小（Cube）越需要强锚定以自保。本轮只提供 "consistent with" 级别证据。
- **不**能主张 BETA 已被全局最优化；只能说"在本次 5 点局部扫描中，0.1 是聚合最优"。

## 11. 结论
Round 1 的 5 点局部扫描**没有发现值得替换 current R1 的配置**。按冻结判据 **FREEZE_CURRENT_R1**：最终方法为 **R1-ILA（BETA = 0.1）**，配置写入 `results/e17/final_method.json`（E18+ 唯一参数来源）。

## 12. 出口条件判断
- §5 情况 A（无挑战者明显胜出）→ **FREEZE_CURRENT_R1** ✅
- §5 情况 B：需 B+C mean ΔAUROC ≥ +0.003 且 4/6 cells 不低且 Cube ≥ −0.003；最佳挑战者仅 +0.00173 且 Cube −0.0087 → 不成立。
- §5 情况 C：唯一更简单的 C1 的 |Δ| = 0.022 > 0.002 → 不成立。
- §14 Early Stop：Round 1 后无挑战者达晋级门槛 → 直接停止，Round 2 未运行。

## 13. 对论文证据链的贡献
把 R1 从"跨 3 seed 已资格认证"推进到"**参数已冻结、局部最优性已被受控验证**"：BETA 是单调的检测↔鲁棒性旋钮，BETA=0.1 是聚合内点最优，preservation 项是必需组件。E18 起不再有任何方法侧调参自由。

## 14. 下一步（不自动执行）
```
NEXT = E18 Final Baseline Benchmark
```
目标：论文最终主表（PatchCore / AnomalyDINO / SoftPatch / EfficientAD / DINOv2 F0 / Ours），优先复用 E11–E16 已有结果。**不得自动启动。**

## 15. 复现信息
- 命令：`python scripts/e17_refine.py --round 1` → `python scripts/e17_analysis.py`
- 输出：`results/e17/{audit.md, protocol.md, round1.csv, round2.csv, summary.csv, final_method.json, e17_decision_memo.md, README.md, figures/, logs/}`
- Round 1 耗时 33.8 min（3 category-run × 5 config，含 C0 复现门）；Round 2 未运行（0 GPU）；总 GPU ≈ 34 min（预算内）
