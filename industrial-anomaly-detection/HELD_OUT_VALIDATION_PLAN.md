# HELD_OUT VALIDATION PLAN（E4）— 计划文件，**不下载、不运行**

**状态**：PLAN ONLY（Exp16 明确禁止临时下载数据集）
**目的**：为论文的 **E4（held-out external validation）** 选定并预注册一个外部数据集，用于检验
**X6c 的 tail-safety 优势是否跨数据集成立**（当前证据仅限 MVTec 5 类，且这 5 类正是方法开发所用类别）。
**触发条件**：人工批准 + 网络/许可确认后执行；执行时必须**冻结本文件中的指标与判据**（与 5A-H/Exp10–16 完全一致：
`EPS_RZ=0.02`、`EPS_DP=0.10`、catastrophic `Δd′≤−0.25`，PASS 相对同 (category, seed) 的 Original）。

---

## 1. 候选数据集评估

| 数据集 | illumination relevance | anomaly type | train/test 结构 | license | 规模 | 下载难度 | PatchCore 兼容 | 是否真正 held-out | 预计 GPU |
|---|---|---|---|---|---|---|---|---|---|
| **MPDD** (Metal Parts Defect Detection, 2019) | **高** —— 数据集设计即包含**光照/油污变化**（同一零件的不同照明条件） | 表面缺陷（6 类金属件） | 每类 train/good + test 缺陷掩码，标准 AD 结构 | 研究用途免费（需引用原文） | 小（~1,064 图，数百 MB） | 低（GitHub/官方；国内可通过镜像/代理） | 是（anomalib 原生支持 MPDD） | **是**（从未用于 5A-H/Exp10–16 的方法开发） | ~30 min（6 类 × 3 seeds × 2 方法 ≈ 36 units @3 workers） |
| **VisA** (Visual Anomaly, 2022) | 中 —— 部分类别含照明/角度变化，但非核心设计变量 | 表面缺陷 + 复杂背景 | 12 类（split CSV 给出 train/test） | CC BY-NC-SA 4.0（**非商业**） | 中（10,821 图，~1.9 GB） | 中（官方 Google Drive/百度网盘） | 是（anomalib 原生支持 VisA） | **是** | ~60 min（12 类 × 3 seeds × 2 ≈ 72 units） |
| **BTAD** (BeanTech AD, 2019) | 中 —— 真实产线图像，含自然光照波动 | 表面缺陷（3 类产品） | 每类 train/good + test | 研究用途免费 | 小（~2,830 图，数百 MB） | 低 | 是（需自定义 split） | **是** | ~20 min（3 类 × 3 seeds × 2 = 18 units） |
| **MVTec LOCO AD** (2022) | 低（逻辑异常为主） | 逻辑/结构异常（计数、位置、组合） | 5 类 | CC BY-NC-SA 4.0 | 中（~1 GB） | 中 | 部分（PatchCore 在逻辑异常上本身失效） | 是 | 不推荐（会引入与课题无关的失效模式） |
| Real-IAD / 大规模数据集 | 中高（多视角+多光照） | 多类工业缺陷 | 复杂 | 多为研究许可 | **很大（数十–上百 GB）** | 高 | 需适配 | 是 | 过大，超出单卡预算 |

## 2. 推荐

> ### Recommended E4 Dataset: **MPDD**
> **理由**：① **illumination relevance 最高**（数据集本身就是为「不同光照/油污条件下的金属件缺陷检测」设计，
> 与本文的 illumination-nuisance 课题**直接对齐**，是检验 X6c tail-safety 是否跨域成立的最强 test）；
> ② 规模最小、下载与适配成本最低（anomalib 原生支持）；
> ③ 结构标准（train/good + test 掩码），判据可**逐字复用**现有冻结协议；
> ④ 是真正的 held-out（从未参与任何方法开发或调参）。
>
> **次要（若需要第二个外部证据）**：**VisA**（更主流、类别更多，但 illumination relevance 弱于 MPDD，且 license 为非商业）。

## 3. 预注册执行协议（冻结；执行时不得修改）

| 项 | 值 |
|---|---|
| 类别 | MPDD 全部 6 类（不筛选、不删类） |
| seeds | 0, 1, 2（先 3 seeds；若时间允许扩到 0–4） |
| 方法 | **仅 4 个**：Original、Adaptive B2、C2、**X6c**（C6 作为可选第 5 个） |
| 协议 | corrected-189 式的同构协议（train/good 过滤 + strict-V2 RNG replay）；指标 `experiment5a_h_analysis.unit_metrics` |
| 判据 | `PASS = ΔR ≤ −0.02 且 Δd′ ≥ −0.10`；`catastrophic = Δd′ ≤ −0.25`（**与主实验完全一致**） |
| 主指标 | PASS 率、catastrophic 率、worst Δd′、mean Δd′、ΔR |
| 成功条件（E4 通过） | X6c 的 **catastrophic 率 ≤ B2** 且 **worst Δd′ 不差于 B2** 且 PASS 率不低于 B2 超过 1 个 EPS 带 |
| 失败条件 | X6c 在新数据集上 catastrophic 率高于 B2，或 worst Δd′ 崩溃 → 论文必须报告为**域外失效**，并把方法结论限定为 MVTec-specific |
| 预算 | ≤ 90 min（含审计/分析/README） |
| 禁止 | 临时改阈值、按结果挑选类别、在看到 E4 结果后重新调权 |

## 4. 执行前检查清单（人工）

- [ ] 用户明确批准下载（本文件不触发任何下载）
- [ ] 许可条款确认（MPDD 研究用途；VisA CC BY-NC-SA 4.0 非商业）
- [ ] 磁盘/网络可用性确认
- [ ] 冻结 `results/experiment_17_heldout_mpdd/PRE_RUN_PROTOCOL.md` 并给出 SHA256
- [ ] 先跑 1 类 smoke 验证数据加载与掩码解析，再全量

---

## 5. 状态更新（2026-10-08 追加；**不改写以上任何历史记录**）

```text
Historical candidate:
MPDD was the initial held-out candidate.

Superseding protocol decision:
E4-0C protocol-first screening selected M²AD as the primary E4 dataset
because it satisfied 8/8 frozen MUST criteria and provides explicit
specimen/view/illumination metadata with local GT.

MPDD remains an optional generic external-generalization dataset,
not the primary illumination-specific E4.
```

**附：本轮（E4-0C / E4-D0）相关的正式文档**

| 文档 | 作用 |
|---|---|
| `docs/E4_DATASET_SELECTION_CRITERIA.md` | E4 数据集选择判据（预注册，先于任何数据集检视写定） |
| `docs/E4_LL_IAD_M2AD_PROTOCOL_AUDIT.md` | E4-0C 协议审计（LL-IAD vs M²AD，verdict = M²AD WIN） |
| `docs/E4_M2AD_INTEGRITY_AUDIT.md` | E4-D0 真实数据完整性审计（本轮） |

**E4 数据集状态一览（只标状态，不删除历史）**：
- `M²AD` —— **PRIMARY E4**（E4-0C 选出；E4-D0 做完整性审计）
- `MPDD` —— **optional generic external-generalization dataset**（不再是 illumination-specific 主 E4）
- `MVTec AD 2` —— **pending / 暂不作主 E4**（public test 无 illumination condition label；private/mixed 无本地 GT → Exp16 ΔR 无法原义计算）
- `LL-IAD` —— **UNKNOWN，暂不使用**（无官方数据分发入口 / 正文不可达；非证伪，若官方发布数据应重评）

> 本节 §3「预注册执行协议（冻结）」针对的是 **MPDD**。若将来执行 MPDD，仍以 §3 原文为准。
> **主 E4（M²AD）的协议必须另行预注册，不得沿用 §3。**

**重要提醒**：本文件的 §1 表格与 §2 推荐仍是 MPDD 视角，属历史记录，保留不改。
未来任何会话在选定 E4 数据集时，**必须先读本节 §5**。
