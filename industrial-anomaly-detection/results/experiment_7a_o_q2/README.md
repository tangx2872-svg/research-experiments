# Overnight Queue Q2 — Layer × Representation Composition

- **状态**：DONE（12 GPU units；freeze SHA256 `c76ee8f358e6d159e78df7fff8f9dc30e295df175167141c44fd9de3ee3c0d7c`）
- **判定**：**无晋级（promoted = ∅）→ Q2 STOP**（未扩展 5×3）
- **EXPLORATORY**：selection-set 结果，不得作为 confirmatory。

## 1. 论文导航

- 📍 阶段：Stage 7 — Method Candidate Screening（Q2 分支：**layer × representation composition**）
- 🎯 服务论断：1H/1H-S 显示 normalization 对 layer2/layer3 的 defect response 不同；Q0 测的是
  representation-level intervention。Q2 检验 **"Original / invariant representation 的组合是否应该发生在
  特定 layer"**，即 layer-selective composition 能否优于"全层统一处理"。
- 🧪 为什么必须做：Q0 的 Family C 只在 **α 强度** 上做 layer 分配（差异 ≤0.022 d′）；Q2 把
  **representation 类型本身**（Original vs invarient / canonical residual）分配到 layer，是不同问题。
- 🚦 出口：任一 config 达 Q0 的 Tier S/A/B → 扩展 5×3；否则 STOP（addendum 明文规定）。

## 2. 冻结配置（运行前，`reference/q2_protocol_freeze.json`）

| 名称 | layer2 | layer3 |
|---|---|---|
| `Q2L0_l2org_l3uni` | Original (α=0) | Uniform-Norm (α=0.40091275) |
| `Q2L1_l2uni_l3org` | Uniform-Norm | Original |
| `Q2L2_l2res50_l3uni` | **canonical residual**（A3 energy-preserving, λ=0.5） | Uniform-Norm |
| `Q2L3_l2uni_l3res50` | Uniform-Norm | canonical residual |

- canonical residual = Q0 PRE-RUN grid 中的 A3（**不**依据 Q0/Q2 target 重新设计公式）。
- 数据：`bottle / grid / hazelnut` × seed0（与 Q0 Round 1 同子集，可直接比较）；
  Uniform 基线在该子集：**d′ = 5.4467，\|Δz\| = 0.1660**。
- 单元写入 `results/experiment_7a_o_q2/raw/`（与 Q0 隔离，不污染 Q0 的 sanity 作用域）。

## 3. 结果（vs Uniform，3 categories × seed0）

| rank | config | d′ ↑ | \|Δz\| ↓ | Δd′ | Δ\|Δz\| | 兑换率 Δd′/Δ\|Δz\| | worst cat Δ | negTr | tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Q2L2_l2res50_l3uni** | 6.0260 | 0.2089 | **+0.5793** | **+0.0429** | **13.5** | −0.4065 | 1 | — |
| 2 | Q2L1_l2uni_l3org | 5.6712 | 0.2725 | +0.2245 | +0.1065 | 2.11 | **−0.0274** | 0 | — |
| 3 | Q2L0_l2org_l3uni | 5.5686 | 0.2132 | +0.1218 | +0.0472 | 2.58 | −0.2408 | 2 | — |
| 4 | Q2L3_l2uni_l3res50 | 5.3979 | 0.2844 | −0.0489 | +0.1184 | — | −0.2526 | 2 | — |
| — | B0_original（ref） | 5.7809 | 0.3122 | +0.3342 | +0.1462 | 2.29 | −0.3012 | 1 | — |
| — | B2_uniform（基线） | 5.4467 | 0.1660 | 0 | 0 | — | 0 | 0 | — |

## 4. 回答 Q2 的五个问题

**1) L2 与 L3 是否存在明显非对称性？→ 存在，而且很强。**
- Original 放在 **L2**（L0）：Δd′ +0.1218 / Δ\|Δz\| +0.0472；放在 **L3**（L1）：+0.2245 / +0.1065。
- canonical residual 放在 **L2**（L2）：+0.5793 / +0.0429；放在 **L3**（L3）：**−0.0489 / +0.1184**
  （两轴都不如 Uniform）。
- → **L2 槽位对 representation 干预高度宽容、且兑换率好；L3 槽位是"高风险"槽位**
  （同样的干预放在 L3 上会掉 d′ 并显著恶化 robustness）。

**2) 哪个 layer 更适合保留 Original representation？**
- 按"兑换率"：Original 放 L2（2.58）优于放 L3（2.11）；
- 按"worst-category 安全"：Original 放 L3 更好（worst Δ = −0.0274，ε 内；L0 为 −0.2408）；
- → 没有单一支配答案；但 **L2 是更有效率的干预槽位**，L3 是更"敏感但不划算"的槽位。

**3) 是否比 Uniform α≈0.40 更好？→ 没有。** 4/4 config 的 Δ\|Δz\| > 0，**无一达到 Tier S/A/B**，
没有任何 Pareto improvement；promoted = ∅。

**4) 是否与历史 1H/1H-S 方向一致？→ 定性一致，幅度不足。**
- 5A-H 的几何最优对 G2 = (L2=0.4527, L3=0.75) 即 **L3 施加更强 normalization、L2 更弱** ⇒ 与 Q2 中
  "把 Original（弱化）放在 L2"（L0）方向相同，而 L0 恰是本队列中**兑换率更好的那一侧**。
- 但 Q2 的效应远小于 Tier A 阈值，**不构成"layer 选择有独立价值"的证据**。

**5) 是否值得成为独立方法候选？→ 不值得（Q2 STOP）。**
唯一值得记录的是：**`Q2L2`（L2 用 canonical residual + L3 用 Uniform）是本项目至今
"preservation per robustness" 兑换率最高的 representation composition（13.5，Q0 同子集 A3_lam050 为 9.9，
Soft-GC 为 3.6）** —— 但它仍未突破 Uniform 的 frontier（Δ\|Δz\| > +0.02）。

## 5. 异常 / 限制

- 仅 3 categories × seed0（按协议 STOP，未扩展 seeds）。
- `Q2L2` 的 worst-category Δ = −0.4065（grid 之外的某一类被拉低）→ 即使未来要推进 L2-only residual，
  必须先解决类别安全性（这是 Tier 规则之外的额外风险信号）。
- 未做 pixel-level 指标（Q2 目标是层级比较，沿用 Q0 的口径与主指标）。

## 6. 出口条件判断

| 条件 | 结果 |
|---|---|
| ≥1 config 达 Tier S/A/B | ❌ 0/4 → **Q2 STOP**（不做 5×3） |
| 组合被禁止（addendum：0 个 family winner） | ✅ 未运行 Round-3 类组合 |

## 7. 对论文证据链的贡献

推进一格：**"layer-selective composition"（把 representation 类型分配到特定 layer）与
"layer-selective strength"（Q0 Family C）一样，不能突破 Uniform frontier**。
新增一个可复用的事实：**L2 槽位比 L3 槽位更有效率、更宽容；L3 干预风险高**。

## 8. 下一步

不调参、不扩展。**Q2 = STOP**；由 Q3（illumination stress-test）与 Q4（baseline 补齐）继续，最后由 Q5 汇总。
