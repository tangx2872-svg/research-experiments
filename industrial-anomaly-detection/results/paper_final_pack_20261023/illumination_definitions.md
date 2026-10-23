# 光照相关证据：定义、覆盖范围与使用边界（2026-10-23）

本文件只做**整理与定义**，不产生新指标。所有数值来自冻结产物；`illumination_evidence.csv` 是机器可读版本。

**三类证据必须分开报告，不得互相替代：**

| 类别 | 扰动来源 | 考察对象 | 可否外推 |
|---|---|---|---|
| **S. 合成光度扰动** | brightness / contrast / gamma × {0.85, 1.15}（6 个变体，R1 训练用的同一组） | 表征与分数对受控光度变化的稳定性 | 只说明"对这类人工变换稳定"，**不等于真实光照** |
| **R. 真实光照分组（M²AD）** | 数据集自带的 10 个 illumination 条件（01–10） | 正常得分跨真实光照组的 spread、组间特征质心距离、组内 AUROC | 说明"该数据集采集条件下的表现"，不含跨采集设备外推 |
| **X. 外部真实域偏移（RobustAD）** | 数据集原生 split 的 condition group（illumination / shadow / white_balance / pose / scale / background） | ΔAUROC 的方向与量级 | 只到这些条件组为止；本方法未在这些域上训练 |

**多视角（E21，view 090/240/330）不属于光照**，单独列在 `viewpoint_M2AD` 组，不得并入光照结论。

---

## 1. 合成光度扰动（S）

| 量 | 定义 | 单位/方向 | 覆盖 | 来源 |
|---|---|---|---|---|
| `robustness_feat` | 对 bank 内每张正常图 `x_i` 与 6 个变体 `T_k`，逐 patch 计算 `1 − cos(F(x_i), F(T_k(x_i)))`，再对全部 (i,k,patch) 取均值 | [0,2]，**越小越稳定** | 10 类 × 3 seed × {F0,R1} | `scripts/e16_qualify.py:99-110` |
| `robustness_score` | 变体图相对**同一 bank** 的异常分数（`mean_top1p` over patches），除以 F0 分数的标准差（分母固定，F0/R1 共用） | 无量纲，**越小越稳定** | 同上 | `scripts/e16_qualify.py:113-131` |
| `D_illum_shift` | patch 级 `‖F(T_k(x)) − F(x)‖ / ‖F(x)‖`，对全部 (i,k,patch) 取均值 | 无量纲，**越小越稳定** | 10 类 × 3 seed × {F0,R1} | `scripts/e16_qualify.py:191-201` |

**结果（10 类 × 3 seed 均值，见 `illumination_synthetic_summary.csv`）**：`robustness_feat` 在 **10/10 类**下降（如 Bird 0.0763→0.0121，Doll 0.0513→0.0103）；`robustness_score` 同样 10/10 类下降（Bird 1.350→0.319，Doll 1.655→0.476）。

**边界**：这三个量正是 R1 训练目标（`L_cons`）的直接体现，**部分由构造保证**，只能说明"训练目标在训练分布的光度变换上生效"，不能单独作为"真实光照鲁棒"的证据。

---

## 2. 真实光照分组（R，M²AD）

| 量 | 定义 | 覆盖 | 来源 |
|---|---|---|---|
| `B_illum_group_centroid_dist` | 测试图按 illumination 分组，各组图像均值特征归一化后，`1 − mean pairwise cos`（组间质心距离） | 10 类 × 3 seed × {F0,R1} | `scripts/e16_qualify.py:162-179` |
| `good_score_spread_across_illum_groups` | 每个光照组内 good 图的平均分数 `good_mean`，取 10 组 `max − min` | 10 类 × 3 seed × {F0,R1} | `results/e22/illumination_summary.csv` |
| 组内 AUROC | 在单个 illumination 组内计算 AUROC（good vs anomaly） | 10 类 × 3 seed × {F0,R1}，10 组 | `results/e22/illumination_breakdown.csv` |
| `A_disp_pairwise_cos` / `A_disp_patch_std` | bank 正常特征的平均成对余弦 / patch token 标准差 | 10 类 × 3 seed × {F0,R1} | `scripts/e16_qualify.py:157-161` |

**结果（3 seed 均值）**：good 分数跨光照组 spread 在 **9/10 类下降**（Bird 0.0381→0.0146，Motor 0.0550→0.0179，Cube 0.0278→0.0161），**Dice 反向上升**（0.0154→0.0216，ratio 1.41，`spread_reduced=0`）。必须如实并列，不得只报 9/10。

**两条措辞禁令（来自 E16/E22 的既有结论）**：
1. **不得写 "normal dispersion reduced"**：表征离散度 `A_disp_pairwise_cos` 与 `A_disp_patch_std` 是**上升**的（Bird 0.889→0.985，patch std 1.354→1.920）。下降的是**分数层面的跨光照 spread**，两者不是同一件事。
2. **不得写成因果**："illumination shift 下降 ↔ 增益" 在 10 类上只是**弱正关联**（Spearman ≈ +0.58/+0.52，n=10，p≈0.08/0.13，`results/e22/shift_gain_association.json`）；只能写 "consistent with"。

---

## 3. 外部真实域偏移（X，RobustAD，E25）

定义：E25 每个 cell = 一个 (domain, split, seed) 评估；`d_auroc = AUROC(R1) − AUROC(F0)`；按 `condition_group` 对 seed 取均值（`valid=0` 的 cell 不参与，MetalParts `test0` 因源域重叠按原决定排除）。

| domain | condition_group | 平均 ΔAUROC（seed 数） |
|---|---|---|
| PCB | illumination | **+0.0266**（3） |
| PCB | illumination_related_shadow | **+0.0097**（3） |
| PCB | illumination_related_white_balance | **+0.0173**（3） |
| PCB | pose | +0.0191（6） |
| PCB | source_reference | −0.0081（3） |
| MetalParts | illumination | +0.0014（3） |
| MetalParts | pose | +0.0002（6） |
| MetalParts | background | **−0.0394**（6） |
| MetalParts | scale | **−0.0265**（3） |

**解读边界**：三个 **illumination 相关组**（PCB illumination / shadow / white-balance）均为正或持平；但 **背景与尺度组为负**，因此整体是"**条件性**"而非"普遍鲁棒"。`RobustAD PiledBags` 未获取，不得声称覆盖全部子集。E25 使用 E17 冻结配置**重训**（非迁移），3 seed。

---

## 4. 多视角（非光照，单列）

E21：view 090/120/240/330 × 3 类 × seed0。view120 与主表一致（bit-identical 复现）；pooled ΔAUROC = +0.00142（CI 含 0），Car v330 为 −0.0260。**该证据回答的是视角变化，不是光照**，不得用于支持光照结论。

---

## 5. 覆盖与缺口

| 项 | 状态 |
|---|---|
| 合成光度扰动（训练同款 6 变体） | 10 类 × 3 seed，**齐备** |
| M²AD 真实光照分组（10 组） | 10 类 × 3 seed，**齐备**（Dice spread 反升已在表中保留） |
| RobustAD 真实条件组 | PCB 5 组 / MetalParts 5 组，**齐备**；PiledBags **缺失**（未下载） |
| 真实采集设备/曝光元数据 | M²AD 只给 illumination 编号，**无曝光/色温等物理量**，不能做物理归因 |
| 跨数据集光照强度标定 | **缺**，故不得跨数据集比较"光照鲁棒性"强弱 |
