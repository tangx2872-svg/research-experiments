# HANDOFF_FOR_PAPER — 本地核查与补实验交接（2026-10-23）

**写给**：论文写作侧 AI / 下一位接手者。
**本轮范围**：0 GPU 新训练；仅 CPU 独立复算 + 文献核对。
**一句话结论**：冻结结果全部**位级复算通过**，无任何影响论文数字的错误；B/C 两类 GPU 任务因本地缺数据缺权重 **NOT_RUN**。

---

## 1. 实际运行 vs NOT_RUN

| 项 | 状态 | 说明 |
|---|---|---|
| A1 PCB 标签/评分审计 | **已运行（PASS）** | 39/39 cell 逐图复算位级一致 |
| A2 聚合审计 | **已运行（PASS）** | 33/3/3 账目 + 各口径复算一致 |
| A3 主表审计 | **已运行（PASS）** | 10 类统计 + Doll std=0.004638 精确匹配 |
| A3 补充 specimen bootstrap | **已运行** | 新补充分析，见下 |
| A2 的 145 字节级复核 | **NOT_RUN** | 需 RobustAD 原图（本地无） |
| B 原图示例 | **NOT_RUN** | 需 M²AD 原图（本地无） |
| C heldout light 训练 | **NOT_RUN** | 需 M²AD 原图 + DINOv2 权重 + 训练服务器（本地均无） |

**GPU 训练次数**：0。**GPU 时间**：0。**CPU 时间**：约 5 分钟（复算）+ 3m41s（specimen bootstrap）。

---

## 2. 冻结原文件校验 / 工作区变化

- **冻结产物**（`results/e22/**`、`results/e25/**`、`results/e17/final_method.json`、`results/paper_ready/**`）：**零修改**。
- **新增**（全部在独立目录，不触碰冻结产物）：
  - `scripts/paper_supplement_20261023/`（4 个独立审计脚本）
  - `results/paper_supplement_20261023/`（6 个审计产物，见 §5）
- **工作区 git 起止变化**：起 `d6336cf`（HEAD 未变）。本轮**未 commit/push**（指令要求）。
  - 注意：工作区原本就有未提交项 `M PROJECT_STATE.md` 和 `?? docs/paper_writing/`（历史遗留，非本轮产生，未触碰）。

---

## 3. 审计发现是否影响现有论文数字

**结论：不影响。** 所有冻结数字经独立复算**位级一致**（max_absdiff = 0.0），未发现：

1. 标签/评分对应错误（A1：0 错位、0 重复、0 漏样本、0 NaN/Inf）
2. 聚合符号错误（A2：PCB +0.018351852、MP −0.017251157、11-split −0.001067971、子数据集等权 +0.000550347，全部与冻结值位级一致）
3. 主表统计错误（A3：macro 0.007596、median 0.010796、Doll std 0.004638 精确匹配）

**唯一"新增信息"**：specimen 级 cluster bootstrap（§4）揭示了一个**之前未被量化的诚实边界**——在"工件级独立重采样"这一更严格假设下，几乎所有类别的 ΔAUROC 的 CI 都跨 0。这**不是错误**，而是补充的第三种不确定性估计对象，强化而非改变 limitation 叙事。

---

## 4. 新协议与旧主表的差异（重要：不得混用）

| 分析 | 估计对象 | 与旧主表关系 |
|---|---|---|
| 旧 image-level paired bootstrap（`bootstrap_e22.json`） | 逐图抽样不确定性 | 已有，主表 CI 来源 |
| 旧 cross-seed std（`main_table_10cat_3seed.csv`） | 3 bank seed 间变异 | 已有 |
| **新 specimen-level cluster bootstrap**（本轮） | **工件级**抽样不确定性 | **新补充，不进入主表、不替代旧 CI** |

**三者是三个不同的估计对象，绝不可互相替代或合并。** 论文若引用 specimen 级结果，必须标注"工件级 cluster 重采样下的 ΔAUROC 分布，n=70 工件/类，与 image 级 CI 及 cross-seed std 是不同不确定性"。

---

## 5. 可入论文的表/图 + 数据来源 + 不夸大中文表述

所有产物在 `results/paper_supplement_20261023/`（绝对路径 `D:\Research\research-experiments\industrial-anomaly-detection\results\paper_supplement_20261023\`）：

| 文件 | 内容 | 可否入论文 | 谨慎表述 |
|---|---|---|---|
| `audit_findings.md` | 完整审计发现 | 否（内部审计文档） | — |
| `metric_reconciliation.csv` | 39 cell AUROC/AP 复算对账 | 可作为复现附录 | "39/39 cell 逐图分数独立复算 AUROC/AP 与冻结值位级一致（max|Δ|=0）" |
| `sample_alignment_audit.json` | 逐图对齐 + 方向检查 | 复现附录 | "无重复/漏样本/长度错配/NaN" |
| `audit_a2_aggregation_reconciliation.json` | A2 聚合复算 | 复现附录 | "33+3+3 账目与各加权口径复算一致" |
| `audit_a3_main_table_reconciliation.json` | A3 主表复算 | 复现附录 | "10 类均值/标准差/macro/median/Doll std 复算一致" |
| `audit_a3_specimen_cluster_bootstrap.json` | **specimen 级 bootstrap（新）** | **可入正文/附录（需标注估计对象）** | "在工件级 cluster 重采样下，绝大多数类别的 ΔAUROC 区间跨 0，说明单类增益在更严格的不确定性下不显著" |
| `closest_method_comparison.md` | 近似方法对照 | related work 素材 | "与 PIAD/AnomalyDINO 的差异"（不写"实验领先"） |

**数据来源声明**：所有复算的**唯一输入**是冻结 cell 内的 `scores_f0`/`scores_r1`/`labels`/`files`/`specimen`（`results/e22/cells/*.json`、`results/e25/cells/*.json`），**未使用任何原图、未重新提取特征、未重新训练**。

---

## 6. 负结果 / 失败边界 / 未解决的限制

1. **specimen 级 CI 几乎全跨 0**（仅 Ring 下界 −0.00512 接近 0）——这是"工件级独立"假设下的诚实不确定性，说明**单类的正收益不稳健**（与既有"仅 Bird/Ring/Teapot 的 image CI 排除零"形成呼应，且更保守）。
2. **Doll 的 specimen 级 P(>0)=0.101**（远高于 image 级 0.001）——工件级重采样降低有效样本量，Doll 的负向在工件级口径下不再"极端显著"，但方向仍稳定为负（mean −0.03549）。
3. **MP test0 的 145 重叠无法本地独立复核**（缺原图），字节级审计仍是"0/145 字节相同、近重复未检验"的保守排除，理由被削弱但决定不变。
4. **PCB 低 baseline 是真实 headroom 信号**（异常类平均分低于正常类，AUROC<0.5），不是评分方向错误——这是 R1-ILA 增益的前提，也是"条件性增益"叙事的逐图证据。
5. **未解决的采集独立性限制**：specimen 级 bootstrap 仍假设 specimen 间独立；若同一物理工件的多光照采集存在时间/空间相关性，该假设可能高估独立性（未验证）。

---

## 7. 下一位 AI 可直接复核的命令与输入清单

**环境**：`D:/miniconda/envs/industrial-ad/python.exe`（Python 3.11.16，sklearn 1.9.1，numpy 2.4.6）。

**输入清单**（只读，勿改）：
- `results/e25/cells/*.json`（39 cell，逐图分数）
- `results/e22/cells/*.json`（30 cell，逐图分数 + specimen）
- `results/e17/final_method.json`（冻结超参）
- `results/e22/main_table_10cat_3seed.csv`（全精度主表）
- `results/e25/multiseed_analysis.json`（E25 聚合）
- `results/e25/overlap_audit.json`（145 重叠审计）

**复核命令**（在仓库根目录执行）：
```bash
cd "D:/Research/research-experiments/industrial-anomaly-detection"
# A1：逐图复算 + 对齐审计
/d/miniconda/envs/industrial-ad/python.exe scripts/paper_supplement_20261023/audit_a1_recompute_e25.py
# A2：聚合复算
/d/miniconda/envs/industrial-ad/python.exe scripts/paper_supplement_20261023/audit_a2_aggregation.py
# A3：主表复算
/d/miniconda/envs/industrial-ad/python.exe scripts/paper_supplement_20261023/audit_a3_main_table.py
# A3 补充：specimen 级 cluster bootstrap（约 3.5 分钟）
/d/miniconda/envs/industrial-ad/python.exe scripts/paper_supplement_20261023/audit_a3_specimen_bootstrap.py
```

**缺失清单（若要做 B/C 或补 AnomalyDINO 对比）**：
| 资源 | 缺失 | 预计大小 | 位置 |
|---|---|---|---|
| M²AD 原图 | 是 | ~6.8GB tar.gz | 远程服务器（本地 `data/m2ad` 不存在） |
| DINOv2-B 权重 | 是 | ~330MB（vitb14） | 远程 `/root/.cache/torch/hub/...`（本地无） |
| RobustAD 原图 | 是 | PCB 6.18GB + MP 0.68GB + PiledBags 9.41GB = 16.27GB | 远程 |
| 训练 GPU | 本地 RTX 5060 8GB（非训练用） | 训练在远程 RTX 3090 | — |

**本轮未**：调用服务器、购买算力、批量下载数据、修改 Word/revise.py/refine_manuscript.py/其他 AI 产物、修改冻结结果/标签/历史分数/E17 配置/paper_ready 生成器。
