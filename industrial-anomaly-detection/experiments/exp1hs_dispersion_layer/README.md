# Experiment 1H-S — Dispersion-Aligned Layer Analysis

**日期**：2026-10-03
**依赖**：1H（layer selectivity，用 Δmean_defect 口径）
**状态**：✅ 完成 —— **判定 CASE B（Partial alignment）**；发现 1H 的 mean 口径层偏好是 score-level 现象，非 dispersion 来源
**成本**：**零 GPU、零训练**，纯读 1H 已产出的 `raw_results.csv`（逐 seed std_defect）

---

## 1. 动机

1H 发现了 layer2/layer3 的层选择性（bottle/cable → layer2、screw/grid → layer3），但它的「层偏好」按 **Δmean_defect**（anomaly score 的 level shift）定义；而 1E→1G 的主线是 **dispersion（Δstd_defect）**。口径不统一是证据链里最大的缝。1H-S 用 std_defect 把 1H 重新对齐到 dispersion 语言，检验：

> 1E 的 shrink / neutral / expand，其 dispersion 变化**究竟来自 layer2、layer3，还是两层交互？**

## 2. 方法（全离线，零训练）

- 输入：`results/experiment_1h/analysis/raw_results.csv`（1125 行 = 25 defect × 3 location × 3 seed × 5 α），逐 seed 取 `std_defect`。
- Δstd = std_defect(α=1) − std_defect(α=0)，**逐 seed 保留**，再算 3-seed mean / sd / sign-consistency / bootstrap 95% CI。
- **LSI_std** = (|Δstd_L3| − |Δstd_L2|) / (|Δstd_L3| + |Δstd_L2| + ε)，−1=layer2 主导，+1=layer3 主导；**同时保留 Δstd_L2 / Δstd_L3 的符号**（避免 LSI 丢失 shrink/expand 方向）。
- 加和性 E_add = |Δstd_concat − (Δstd_L2+Δstd_L3)| / |Δstd_concat|，**25 个 defect 全算**。
- family alignment：用 1G 冻结的 9 个 primary 标签（shrink 3 / neutral 3 / expand 3），**不重新分组**，只报 LSI + 组内方向一致率 + bootstrap CI + effect size（不做花哨显著性）。

## 3. 结果

### 3.1 核心对比：mean 口径 vs std 口径（推翻了 1H 的层偏好）

| defect | 家族 | Δmean L2 / L3（1H） | Δstd L2 / L3（1H-S） |
|---|---|---|---|
| cable/bent_wire | shrink | **−8.26 / +0.58**（反向） | −2.15 / −2.05（**同号**） |
| bottle/contamination | shrink | **−2.73 / +2.48**（反向） | −2.24 / −0.86（**同号**） |
| hazelnut/print | shrink | −1.92 / −1.87 | −1.77 / −1.97 |
| grid/thread | expand | +12.3 / +24.8 | +4.12 / +8.07 |

**关键发现**：1H 用 mean 口径看到的「bottle/cable 是 layer2 主导、layer3 反向」，在 dispersion 口径下**消失**——shrink 家族的两层是**同号 shrink**（dispersion 同时收缩），bent_wire 的 mean-LSI=−0.868 变成 std-LSI=−0.023，contamination 从 −0.048 变成 −0.447。即：**1H 的层偏好主要是 anomaly-score 的 level-shift 现象，不是 dispersion 的来源。**

### 3.2 family alignment（冻结标签，bootstrap CI）

| 家族 | n | LSI_std | 95% CI | effect size | Δstd_L2 | Δstd_L3 | 符号一致率 L2/L3 |
|---|---|---|---|---|---|---|---|
| shrink | 3 | −0.138 | [−0.447, +0.056] | 0.51 | −2.05 | −1.63 | 1.00 / 1.00 |
| neutral | 3 | −0.275 | [−0.513, +0.159] | 0.73 | −0.11 | −0.09 | 0.67 / 1.00 |
| **expand** | 3 | **+0.448** | **[+0.324, +0.627]** | **2.81** | +2.76 | **+6.68** | 1.00 / 1.00 |

- **expand 对齐漂亮**：LSI CI 完全在正区间、与 shrink CI 不重叠，effect size 2.81 → grid 是真正的 **layer3-dominant dispersion expansion**（ΔL3 +6.68 ≫ ΔL2 +2.76）。
- **shrink 无层选择性**：LSI CI 跨零，两层同号 shrink（−2.05 / −1.63），收缩是「双层并行」而非「layer2 主导」。
- **neutral 信号太弱**：Δstd 幅度 ~0.1，LSI 无意义（这是 1E 里 neutral 本来就「响应近零」的反映，一致）。

### 3.3 加和性被推翻

mean 口径下 grid 5/5 加和（误差 2.6~9.1%）；std 口径下 **grid 只有 4/5**（E_add<0.15），且 screw/bottle/hazelnut **全部 0** 加和。7 个近似加和的 defect 是散点（grid 4 + cable 3），不是 family 特征。**「grid 是特殊 additive family」是 mean 口径 + 大数值的假象，dispersion 口径下不成立。**

### 3.4 四象限图（Δstd_L2 × Δstd_L3）

- **shrink 家族全落在第三象限**（两层同号 shrink）：bent_wire / print / contamination。
- **expand 家族全落在第一象限且偏上**（layer3 主导 expand）：grid 三个。
- 无「L2 shrink + L3 expand」的竞争象限 —— 1H mean 口径暗示的「两层方向相反」在 dispersion 上不存在。

## 4. 判定：CASE B — Partial alignment

依据冻结判据：

- ~~CASE A~~（strong alignment）：不满足——shrink 家族无「稳定的 layer-specific shrink」（两层同号，无层选择性），只有 expand 家族对齐。
- **CASE B**（partial alignment）：**命中**——expand 家族（grid）稳定 layer3-dominant expansion（CI 不重叠、effect 2.81）；shrink 家族 dispersion 收缩是双层同号、无层选择性；neutral 信号弱；加和性非家族特征。
- CASE C（mean-only）：不适用——std 口径并非「完全乱」，expand 对齐是真实的。
- CASE D：不适用——expand 的 3-seed 方向一致率 1.00、CI 不跨零，结构稳健。

**结论**：layer selectivity **存在但不足以完整解释 response-family 结构**。可以更精确地说：

> dispersion 的 **expansion 有 layer3-selective 的来源**（grid）；dispersion 的 **shrink 是 layer 无关的双层并行收缩**（bottle/cable/hazelnut）。1H 声称的「bottle/cable = layer2 主导」在 dispersion 口径下不成立，那是 score-level shift 而非 dispersion。

## 5. 对证据链的修正

```
1E：dispersion shrink / neutral / expand 的异质性
1G：1E ↔ NN-dispersion ↔ feature geometry（CASE_A）
1H（mean）：layer2/layer3 层选择性 —— 但口径是 score level shift
1H-S（std）：expand = layer3 来源；shrink = 双层并行，无层选择性  ← 修正
```

**证据链的「缝」被补上了，但补的方式是「发现 1H 的层选择性部分不适用于 dispersion」**。1E→1G 的 dispersion 语言是自洽的（1G 已用 frozen ρ=0.950 确认）；1H 的层选择性在 dispersion 上只有 expand 家族成立。

## 6. 诚实记录与边界

1. **只 9 个 primary defect 有冻结家族标签**；其余 16 个（exploratory）的 LSI 见 `lsi_std.csv`，未纳入 family 统计。
2. **bootstrap CI 用 n=3**（3 seed），CI 宽是样本量的真实反映，不夸大。
3. **layer3 空间粒度混杂**（16×16=256 vs layer2 32×32=1024）未排除——这正是下一步 1I（Spatial-Statistics Control）要控制的。按预注册顺序，**1H-S 若失败则 1I 无意义；现 1H-S 为 CASE B（部分成立）**，1I 有针对性价值：检验 expand 家族的 layer3 主导是否来自「256 像素 IN 统计噪声更大」。
4. mean 与 std 口径的分歧是本次最重要的发现之一，已明确记录，避免后续把两个口径混用。

## 7. 产物

- `results/experiment_1hs/analysis/`：`dispersion_layer_summary.csv`（75 行，逐 seed 统计）、`lsi_std.csv`（25 行，带符号 LSI）、`additivity.csv`（25 行，E_add）、`family_alignment.csv`（3 行）
- `results/experiment_1hs/figures/`：figure1 Δstd 热力图、figure2 四象限图、figure3 加和性散点
- 脚本：`scripts/experiment1hs_analysis.py`（纯 CPU，秒级）

## 8. 下一步（按实验树，不擅自跑）

1. **Experiment 1I — Spatial-Statistics Control**：控制 layer2/layer3 的 IN 统计粒度（1024 vs 256 像素），检验 expand 家族的 layer3 主导是否为统计噪声假象。需重新训练（少量），等用户确认。
2. 或先把 16 个无标签 defect 做 exploratory family 扩展（用 1E 的 Δdefect_std 方向补标签），再复核 1H-S 的 family alignment。
