# 本地核查审计发现（audit_findings）

**日期**：2026-10-23
**性质**：0 GPU、0 新训练、0 修改冻结产物。仅从冻结逐图分数独立复算 + 权威口径核对。
**结论总览**：A1 PASS、A2 PASS、A3 PASS；B NOT_RUN、C NOT_RUN（原因见下）。

---

## 0. 环境与数据事实（先读）

| 项 | 当前本机（Windows / RTX 5060 Laptop 8GB） | 冻结实验实际运行环境 |
|---|---|---|
| GPU | RTX 5060 Laptop，8151 MiB，已用 4115 MiB | RTX 3090 24GB（E25 run_manifest 记录） |
| Python | conda `industrial-ad`（Python 3.11.16） | 3.12.3（E25 记录） |
| M²AD 原图 `data/m2ad` | **不存在** | 远程服务器 |
| RobustAD 原图 `data/robustad` | **不存在** | 远程服务器 |
| DINOv2 权重（`~/.cache/torch/hub/facebookresearch_dinov2_main`） | **不存在**（本地无 `~/.cache/torch/hub`） | 远程服务器 `/root/.cache/...` |
| 逐图分数缓存（`results/e22/cells/*.json`、`results/e25/cells/*.json`） | **存在且完整**（39 + 30 cell） | 已随 git 同步 |

**关键事实**：`scripts/e13a_screen.py:72` 的 `_HUB_REPO = "/root/.cache/torch/hub/facebookresearch_dinov2_main"` 是 Linux 绝对路径，证明 E13–E25 的 GPU 训练/推理全部在**远程 Linux 服务器**完成，本地 Windows 机器只有**冻结的逐图分数**（cell JSON），无原图、无权重、无特征缓存。因此所有"复算"只能针对逐图分数做，不能重新提取特征。

---

## 1. A1 — PCB 低于 0.5 的核查（PASS）

### 1.1 标签语义（PASS）

- 标签**只读 `metadata.jsonl` 的 `label` 字段**，0=normal、1=anomaly（`scripts/e25_robustad_eval.py:236-262`）。
- 代码内硬断言：`label` 必须等于 `file_name.startswith("anomaly/")` 的布尔值，否则报 `label/folder mismatches`。RobustAD 官方 15 个 metadata 文件 **0 不一致**（ROBUSTAD_PROTOCOL_AUDIT.md §0）。
- 异常正类 = 1，训练 normal 筛选 = `label==0`，均正确。

### 1.2 逐图对应（PASS）

- 39/39 cell：`scores_f0`/`scores_r1`/`labels`/`files` 四者长度一致（`sample_alignment_audit.json`）。
- 无重复 `files`、无 NaN/Inf 分数、labels 全二进制、每 cell 含两类（two_classes=True）。
- `files` 与 `labels` 一一对应（`load_split` 按 `file_name` 排序，逐图记录顺序确定性）。

### 1.3 独立复算 AUROC/AP（PASS，位级一致）

- 用 sklearn `roc_auc_score`/`average_precision_score` 独立复算 39 cell 的 F0/R1 AUROC/AP。
- **39/39 与冻结 cell 位级一致（max_absdiff = 0.0，容差 1e-9）**。见 `metric_reconciliation.csv`。

### 1.4 分数方向（PASS）

- score = `1 - max cosine similarity`（`scripts/e11_p2_runner.py:116` 的 `nn_image_scores`），**分数越大 = 越异常**。
- `roc_auc_score(y, s)` 正确使用该方向，无需反转。

### 1.5 PCB 低 baseline 是真实现象，不是方向错误（重要发现）

PCB 全部 5 个 target split 的 F0 baseline，**异常类平均分数低于正常类**（AUROC 0.44 < 0.5）：

| cell | F0 正常类均分 | F0 异常类均分 |
|---|---|---|
| PCB test1 | 0.30564 | 0.30431 |
| PCB test2 | 0.31955 | 0.30953 |
| PCB test3 | 0.35810 | 0.32655 |
| PCB test4 | 0.33957 | 0.31406 |

这**不是评分方向错误**，而是**强域偏移下 baseline 接近失效**的真实反映（EVIDENCE_MAP §8 已记录 target 平均 AUROC 0.4447、test3 rotation 仅 0.349）。方向检查"异常"恰好是 PCB 有 headroom 的逐图证据。**结论：方向定义正确，PCB 低 baseline 是真实信号。**

### 1.6 训练随机性正确（PASS）

- 同 (domain, split) 的 F0 分数跨 3 seed **完全一致**（F0 不经过 adapter）。
- R1 分数跨 seed **不同**，且 3 seed 的 `adapter_hash` 互不相同（adapter 训练随机性正确生效）。

---

## 2. A2 — MetalParts test0 与聚合（PASS）

### 2.1 cell 账目（PASS）

独立核对 `39 = 33 + 3 + 3`：
- 33 = 11 个有效 target split × 3 seed（PCB 5 + MP 6）
- 3 = PCB test0（source reference，valid=1）
- 3 = MP test0（valid=0，排除）
- **账目精确成立**（`audit_a2_aggregation_reconciliation.json:accounting`）。

### 2.2 聚合复算（PASS，位级一致）

| 口径 | 复算值（全精度） | 冻结值 | 匹配 |
|---|---|---|---|
| PCB 5-split 等权 mean ΔAUROC | +0.018351852 | +0.01835185 | ✅ |
| MP 6-split 等权 mean ΔAUROC | −0.017251157 | −0.01725116 | ✅ |
| 11-split 等权宏平均（主值） | −0.001067971 | −0.00106797 | ✅ |
| 2 子数据集等权 | +0.000550347 | +0.000550（EVIDENCE_MAP 声称） | ✅ |

- 符号敏感的事实**保留**：split 等权 −0.00107 为负、子数据集等权 +0.00055 为正，两者并存，不选择性突出。
- per-seed 宏平均：0: −0.001485 / 1: −0.001695 / 2: −0.000024（t-interval 的 n=3 统计样本，非 11/39）。

### 2.3 MP test0 145 重叠（NOT_RUN — 本地无法独立复核）

- 冻结 `overlap_audit.json` 声称：145 对同名、**0 对字节相同**、抽样 20 对 64×64 灰度 mean|diff| 10.94–32.46/255。
- **本地无 RobustAD 原图与 metadata.jsonl**，无法独立重跑字节级 SHA-256 对比。只能核对 `overlap_audit.json` 的**内部自洽**：
  - `split_inventory.csv`：MP train 231（180 normal + 51 anomaly）、test0 171（120 normal + 51 anomaly）。
  - 145 重叠 = 104 normal + 41 anomaly（overlap_audit 记录），与 train/test0 的 normal/anomaly 数量**兼容**（test0 有 51 anomaly，其中 41 个同名是合理的子集关系）。
  - 但这是**间接一致性核对，不是独立字节级验证**。**如实标注 NOT_RUN**。
- 结论**不变**：MP test0 按原协议 valid=0 保守排除，理由已从"证明污染"降级为"保守排除、理由被削弱"（overlap_audit.json:rationale_status）。**不自动改 valid=0，不改主汇总。**

---

## 3. A3 — 主表一致性（PASS）

### 3.1 独立复算（PASS，全精度位级一致）

| 统计量 | 复算值（全精度） | 声称（5 位舍入） | 匹配 |
|---|---|---|---|
| macro ΔAUROC | 0.007596 | 0.00760 | ✅ |
| median ΔAUROC | 0.010796 | 0.01080 | ✅ |
| macro ΔAP | 0.007773 | 0.00777 | ✅ |
| 正收益类别（ΔAUROC） | 9/10 | 9/10 | ✅ |
| 三 seed 全正（ΔAUROC） | 7/10 | 7/10 | ✅ |
| Doll ΔAUROC mean | −0.038497 | −0.03850 | ✅ |
| **Doll ΔAUROC std（ddof=1，未舍入）** | **0.004638** | **0.004638** | ✅ |

- 每类 3-seed 的 d_auroc/d_ap 均值与 ddof=1 样本标准差，全部与冻结全精度主表 `results/e22/main_table_10cat_3seed.csv` **位级一致**（`audit_a3_main_table_reconciliation.json`）。
- `paper_main_table.csv`（5 位舍入）与全精度值在 5e-5 容差内一致（舍入所致，非错误）。
- **区分"平均指标"与"先平均分数再算指标"**：本复算与冻结主表一致，均采用"每类先做 3-seed delta 均值，再 10 类等权"，**未**混用 image bootstrap 区间（后者是另一个估计对象）。

### 3.2 新增补充：specimen 级配对 cluster bootstrap（新分析，非旧结论）

以 specimen（工件）为 cluster 的有放回重采样，估计对象 = **工件级抽样不确定性下的 ΔAUROC**（区别于既有 image-level bootstrap 的逐图抽样不确定性）。固定 `rng_seed=20261023`，3000 次，F0/R1 使用相同重采样索引（paired），单类退化记 NaN 并计数（本轮全部 degen=0，因 70 个 specimen 两类均充足）。

| 类别 | n_specimen | mean ΔAUROC | CI [2.5, 97.5] | P(>0) |
|---|---|---|---|---|
| Bird | 70 | +0.02892 | [−0.00970, +0.06970] | 0.898 |
| Car | 70 | +0.01519 | [−0.01335, +0.04389] | 0.830 |
| Cube | 70 | −0.00393 | [−0.02940, +0.02138] | 0.445 |
| Dice | 70 | +0.00174 | [−0.01960, +0.02337] | 0.569 |
| **Doll** | 70 | **−0.03549** | [−0.09135, +0.01686] | 0.101 |
| Holder | 69 | +0.01020 | [−0.00924, +0.03136] | 0.831 |
| Motor | 70 | +0.00658 | [−0.02016, +0.03102] | 0.667 |
| Ring | 70 | +0.01638 | [−0.00512, +0.03854] | 0.928 |
| Teapot | 70 | +0.01391 | [−0.01110, +0.03982] | 0.843 |
| Tube | 70 | +0.01718 | [−0.00775, +0.04368] | 0.897 |

**关键观察（必须诚实标注）**：
1. specimen 级 CI **普遍宽于** image 级 CI，且**几乎全部跨 0**（仅 Ring 下界 −0.00512 接近 0）——这是 cluster 重采样保留 specimen 内相关性的**预期结果**，说明在"工件级独立"这一更严格假设下，单类的 ΔAUROC 不确定性更大。
2. Doll 的 specimen 级 P(>0)=0.101，**远高于** image 级的 0.001——因为有效样本量从 ~653 图降到 70 工件。**两者估计对象不同，不得互相替代或混用**；这与 L5（"image bootstrap CI 不得当作 cross-seed 不确定性"）一致，specimen 级是第三种估计对象，仅作补充、不进入主表。
3. **结论不变**：specimen 级重采样进一步强化"绝大多数类别的正收益在更严格不确定性下不显著"这一诚实边界，**不构成任何新主张**。

---

## 4. B — 成功/失败原图示例（NOT_RUN）

**原因**：M²AD 原图（`data/m2ad`）不在本地，无法绘制原图。逐图分数存在，但"原图"是任务 B 的硬前提（"仅在原图与冻结分数均可用时执行"）。

**缺失清单**：M²AD 原图（`data/m2ad/extracted` + `jsons_extract`，官方 tar.gz 约 6.8GB，见 memory 记录）。未确认本地磁盘是否存在任何离线副本（C/D/E 盘浅层均未找到）。

---

## 5. C — 训练未覆盖光照条件的受控验证（NOT_RUN）

**原因（多重，均满足则不可能）**：
1. M²AD 原图/缓存不在本地（`data/m2ad` 不存在，无 `e7_cache`/`l0_448` 缓存）。
2. DINOv2 权重不在本地（无 `~/.cache/torch/hub/facebookresearch_dinov2_main`）。
3. 训练实际在远程 RTX 3090 服务器执行，本地为 RTX 5060 Laptop 8GB（E25 peak_gpu_mb 记录 5011MB，接近本机 8GB 上限，且本机 GPU 已被图形应用占用 ~4GB）。

按指令 §5 开头门槛（"只有 M²AD 原图/缓存、权重与可用 CUDA 齐备且预算合适时执行"），**前置条件不满足**。

**因此 C1（冻结协议 JSON）也未执行**：在没有数据、没有权重、没有可运行环境的前提下"冻结一份协议"并生成 SHA-256 会误导为"已准备可运行"，违反指令第 0 条"不能把缺少数据、GPU 或依赖的项目写成已运行"。**正确做法是如实记录 NOT_RUN + 缺失清单 + 恢复路径**，而非伪造协议。

**预计缺失规模**（供恢复参考）：
- M²AD 原图：约 6.8GB（官方 tar.gz）
- DINOv2-B 权重：约 330MB（vitb14，来自 dinov2 hub）
- RobustAD（若需）：PCB 6.18GB + MetalParts 0.68GB + PiledBags 9.41GB = 16.27GB
- 实际 GPU 训练环境：需远程服务器（RTX 3090，路径 `/root/.cache/...`）

**未调用服务器、未购买算力、未批量下载数据**（指令要求）。

---

## 6. 结论汇总

| 任务 | 结论 | 产物 |
|---|---|---|
| A1 PCB 核查 | **PASS** | `metric_reconciliation.csv`、`sample_alignment_audit.json` |
| A2 聚合核查 | **PASS**（145 字节级复核 NOT_RUN） | `audit_a2_aggregation_reconciliation.json` |
| A3 主表核查 | **PASS** | `audit_a3_main_table_reconciliation.json` |
| A3 补充 bootstrap | 完成（新补充分析） | `audit_a3_specimen_cluster_bootstrap.json` |
| B 原图示例 | **NOT_RUN**（无原图） | — |
| C heldout light | **NOT_RUN**（无数据/权重/训练环境） | — |

**对现有论文数字的影响：无。** 所有冻结数字经独立复算均**位级一致**，未发现任何标签/评分对应错误、聚合符号错误或主表统计错误。唯一"新增"是 specimen 级 cluster bootstrap，它是**补充分析**，强化而非改变现有 limitation 叙事。

**未解决项**：
1. MP test0 的 145 重叠无法本地独立做字节级复核（需 RobustAD 原图）。
2. specimen 级 CI 几乎全跨 0 这一现象，是"工件级独立性"假设下的诚实不确定性，论文若引用需明确估计对象。
