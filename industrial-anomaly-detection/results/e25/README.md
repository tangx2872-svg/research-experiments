# E25 — RobustAD 外部真实域偏移验证（冻结 R1-ILA）

**日期**：2026-10-18
**实验编号**：E25
**方法**：R1-ILA（`z' = z + W z`，DINOv2-B final-layer patch tokens + L2 + 1-NN-max）
**唯一合法配置来源**：`results/e17/final_method.json`
**协议**：`results/e25/ROBUSTAD_PROTOCOL_AUDIT.md`（§8 执行协议 + §10 冻结配置来源修订）
**状态**：已完成（PCB + MetalParts），PiledBags 未运行（磁盘不足）
**多 seed 扩展（2026-10-19 夜间）**：seed 0/1/2 全部完成，结论结构不变 → 见 **§15**（本文档 §7–§12 的 seed-0 叙述保留原样，§15 为三 seed 汇总，冲突时以 §15 为准）

---

## 1. 实验题目

在 **RobustAD**（AmazonScience，CC-BY-4.0）真实域偏移数据上，对已冻结的 R1-ILA 方法做**独立外部泛化验证**，比较：

- **B0** = 冻结 DINOv2-B + L2 + 1-NN-max（无 adapter）
- **B1** = 冻结 R1-ILA（同一 bank、同一输入、同一 scoring）

---

## 2. 论文阶段

**Stage ⑬ / 外部有效性（external validity）**。E1–E23 的全部证据建立在 **M²AD** 内部（3 类 → 10 类，view120，合成光照增强）。E25 是**方法冻结后**的第一次真实域外部检验，服务于论文中「方法不止在 M²AD 上有效」这一断言的外部有效性证据格。

---

## 3. 本实验要验证的核心论断

> 「在 M²AD 上冻结的 R1-ILA 超参（BETA=0.1，mild photometric augmentation，normal-only 训练），在 **RobustAD 每个 sub-dataset 自身的 source-train normal 上重新训练 adapter** 后，能否在未见过的真实工业域偏移（光照/白平衡/阴影/位姿/尺度/背景）上，相对同 bank 的冻结 DINOv2-B 1-NN-max 仍保持检测性能提升。」

该论断的**否定形式**同样重要：若在部分域上退化，则「外部泛化」只能以**有条件**的方式陈述。

**训练方式口径（以代码为准，`scripts/e25_robustad_eval.py` 的 `run_domain`）**：每个 domain、每个 seed 都从零初始化 `LinearAdapter`，在该 domain 的 180 张 source-train normal 上重新训练（前 120 训练 / 后 60 验证）；**从未加载 M²AD 上训练得到的 adapter**（6 次运行的 `adapter_hash` 互不相同）。因此本实验只能称为 **external dataset validation（外部数据集验证）**，**不是 direct cross-dataset transfer（直接跨数据集迁移）**，论文中不得使用后者。

---

## 4. 实验目的

1. 排除「R1-ILA 的增益是 M²AD 合成光照增强 / view120 采样 / 特定 bank 的产物」这一替代解释。
2. 给出**配对**（同图、同标签、同 bank）的 B1−B0 差值，而不是跨数据集比较绝对水平。
3. 为论文提供**可写的外部证据**（含负面结果），而不是只报有利结果。

---

## 5. 数据与参数

### 数据

| 项 | 内容 |
|---|---|
| 数据集 | RobustAD（HF `AmazonScience/RobustAD`，经 `hf-mirror.com` 下载），PCB + MetalParts |
| 规模 | 2677 文件 / 6.4 GB（校验 `download_verification.json`：missing=0、bad_size=0、标签与目录 0 冲突） |
| 结构 | `<Category>/<category>_data_dir_{train,test0..N}/{normal,anomaly}/` + `metadata.jsonl` |
| 标签 | **只读 `metadata.jsonl` 的 `label`**（0/1）；mask 一律忽略（image-level） |
| source bank | 每类 source-train 的 **180 张 normal**（排序后 train 子集=前 120，val=后 60） |
| 未使用 | source-train 的 75 (PCB) / 51 (MP) 张 anomaly **完全未参与训练/选择** |
| PiledBags | **未运行**：9.41 GB 未下载，卷剩余 6.1 GB（`df` 实测），不推测、不补值 |

### 冻结参数（`results/e17/final_method.json`，运行前用 `assert_frozen_config()` 硬断言）

| 参数 | 值 |
|---|---|
| 方法 | R1-ILA，768×768 zero-init 线性残差，**589,824** 可训练参数 |
| BETA | **0.1**（`L = L_cons + 0.1·L_preserve`） |
| augmentations | **brightness + contrast + gamma** |
| strength | **{0.85, 1.15}** → 6 个光度变体 |
| lr / optimizer | **1e-3** / AdamW |
| max_steps / patience | **≤300** / **30**（每 5 步验证） |
| batch_size | **8** |
| seed | **0**（`torch.manual_seed(0)` + `np.random.default_rng(0)`） |
| 与 E17 的唯一偏差 | `TRAIN_N=240` 不可满足 → **120**（bank 只有 180 张 normal；协议 §8.2 已定，属 bank 规模约束，**非调参**） |

### 指标口径（必须随结果一起引用）

- **plain image-level AUROC / AP**，在 split **全部**图像上计算，标签仅来自 `metadata.jsonl`。
- RobustAD **没有** `detectable` 字段 → 与 M²AD 的 `official` 口径（`image_anomaly` + `detectable != False`）**不可数值比较**。

### 输入几何（协议 §10 补充，早于任何 MetalParts 结果写定）

- 协议 §8.3 写「448×448 bicubic resize」，执行时按**字面方形** 448×448 处理。
- PCB 原图 2048×2048 → 与 E13-A 的短边 resize **逐位等价**（实测 20 张：tensor 与 DINOv2 特征 `max|Δ| = 0.0`）。
- MetalParts 原图 2681×1500（16:9）→ 短边 resize 会得到 448×798 = 32×57 = **1824 token**，与冻结的 1024-token 模型/打分器不兼容；方形规则保持 1024 token。
- **代价必须披露**：MetalParts 输入被各向异性压缩，因此其**绝对水平不可与 PCB 比较**；但 B0↔B1 是同张量的配对比较，差值仍然内部公平。

---

## 6. 实验过程

1. `python scripts/e25_robustad_eval.py --smoke`（PCB）→ **18/18 PASS**。
2. `python scripts/e25_robustad_eval.py --smoke --smoke-domain MetalParts` → **18/18 PASS**（验证非方形图像路径）。
3. `setsid nohup bash scripts/e25_run_all.sh`（脱离 SSH；分 domain 独立进程、失败重试 ≤3、每 split 落盘、支持 resume）。
4. 每个 domain 内：加载 180 张 source normal → 冻结 DINOv2-B 提特征（含 6 个增广变体）→ 训练 adapter（120/60）→ 逐 split 提测试特征 → B0 / B1 打分。
5. `python scripts/e25_analysis.py` → 逐图分数**复算校验** + 配对 bootstrap（n_boot=3000, rng=20261018）→ 汇总表 + `final_verdict.json`。

**实际执行命令**

```bash
python scripts/e25_robustad_eval.py --smoke
python scripts/e25_robustad_eval.py --smoke --smoke-domain MetalParts
setsid nohup bash scripts/e25_run_all.sh >> results/e25/logs/driver.log 2>&1 &
python scripts/e25_robustad_eval.py --domain MetalParts   # 收尾（补 partial/geometry）
python scripts/e25_analysis.py
```

**运行成本**：PCB 361 s、MetalParts ≈240 s（含一次重算），合计 **≈6–7 min GPU**；峰值显存 **≈5.0 GB**（1-NN 打分矩阵主导）；`results/e25/` 共 **496 KB**（逐图分数以 JSON 存档）。

**复现信息**：git HEAD `296ab33`（+ 本轮未提交改动，见 §13）、Python 3.12.3、PyTorch 2.8.0+cu128、RTX 3090 24 GB、conda base。

---

## 7. 结果（**事实**）

### 7.1 逐 split（AUROC / AP；`valid=0` 不进入任何聚合）

| domain | split | 条件组 | n(ok/ng) | AUROC B0 → B1 | ΔAUROC | AP B0 → B1 | ΔAP | 95% CI(ΔAUROC) 排除 0 |
|---|---|---|---|---|---|---|---|---|
| PCB | test0 | source reference | 120/75 | 0.92678 → 0.91700 | **−0.00978** | 0.87285 → 0.86311 | −0.00974 | 否 |
| PCB | test1 | illumination | 120/90 | 0.52722 → 0.55537 | **+0.02815** | 0.41804 → 0.43726 | +0.01923 | **是** |
| PCB | test2 | white balance/AWB | 120/60 | 0.44139 → 0.46431 | **+0.02292** | 0.30551 → 0.33059 | +0.02508 | **是** |
| PCB | test3 | pose (rotation) | 80/60 | 0.34917 → 0.38604 | **+0.03688** | 0.33505 → 0.35362 | +0.01857 | **是** |
| PCB | test4 | pose (position) | 80/60 | 0.39792 → 0.40625 | **+0.00833** | 0.36528 → 0.38417 | +0.01889 | 否 |
| PCB | test5 | shadow | 80/60 | 0.50792 → 0.52021 | **+0.01229** | 0.42114 → 0.42332 | +0.00218 | 否 |
| MetalParts | test0 | source ref **(INVALID)** | 120/51 | 0.93627 → 0.95703 | +0.02075 | 0.86555 → 0.89891 | +0.03336 | — |
| MetalParts | test1 | illumination (ambient) | 120/80 | 0.98865 → 0.99021 | **+0.00156** | 0.98306 → 0.98540 | +0.00234 | 否 |
| MetalParts | test2 | pose (position) | 120/80 | 0.95760 → 0.95073 | **−0.00688** | 0.89653 → 0.90414 | +0.00761 | 否 |
| MetalParts | test3 | pose (rotation) | 120/80 | 0.94292 → 0.94781 | **+0.00490** | 0.90340 → 0.91685 | +0.01346 | 否 |
| MetalParts | test4 | scale | 120/80 | 0.92833 → 0.89510 | **−0.03323** | 0.87253 → 0.83908 | −0.03345 | **是** |
| MetalParts | test5 | background #1 | 60/40 | 0.80208 → 0.74000 | **−0.06208** | 0.77126 → 0.66265 | −0.10861 | **是** |
| MetalParts | test6 | background #2 | 60/40 | 0.93708 → 0.90792 | **−0.02917** | 0.92958 → 0.89372 | −0.03587 | 否 |

### 7.2 domain 聚合（仅 valid 非 source split）

| domain | n splits | ΔAUROC mean / median | ΔAP mean / median | 正向 splits | 最差 / 最好 |
|---|---|---|---|---|---|
| PCB | 5 | **+0.021713** / +0.022917 | **+0.016790** / +0.018891 | **5/5（AUROC）/ 5/5（AP）** | test4 +0.00833 / test3 +0.03688 |
| MetalParts | 6 | **−0.020816** / −0.018021 | **−0.025753** / −0.015556 | **2/6 / 3/6** | test5 −0.06208 / test3 +0.00490 |

### 7.3 条件组聚合

| 条件组 | splits | ΔAUROC mean | ΔAP mean | 正向 |
|---|---|---|---|---|
| illumination | PCB:test1, MP:test1 | +0.014855 | +0.010784 | 2/2 |
| illumination-related (white balance) | PCB:test2 | +0.022917 | +0.025083 | 1/1 |
| illumination-related (shadow) | PCB:test5 | +0.012292 | +0.002184 | 1/1 |
| pose | PCB:test3/4, MP:test2/3 | +0.010807 | +0.014631 | 3/4 |
| scale | MP:test4 | −0.033229 | −0.033453 | 0/1 |
| background | MP:test5/6 | −0.045625 | −0.072237 | 0/2 |
| source reference | PCB:test0 | −0.009778 | −0.009739 | 0/1 |

### 7.4 总体（11 个 valid 非 source split）

- **正向 7/11（AUROC）、8/11（AP）**
- **macro ΔAUROC = −0.001485**，**macro ΔAP = −0.006415**
- 配对 bootstrap CI 排除 0：**5/11（AUROC）**、**6/11（AP）**
- `descriptive_pattern` = **MIXED**

### 7.5 校验

`scripts/e25_analysis.py`：13/13 cell 由逐图分数**独立复算** AUROC/AP，`max|Δ| = 0.0`（容差 1e-12）→ PASS。

---

## 8. 异常情况（全部保留，未删除）

1. **MetalParts 首次运行崩溃**：`ValueError: could not broadcast (8,1824,768) into (8,1024,768)`。原因：2681×1500 原图经短边 resize 产生 1824 token。已按协议 §10 的方形几何规则修复；**PCB 在该规则下逐位不变**（见 §5），故 PCB 结果无需重跑。
2. **同一轮中还出现两次工程性失败**（与科学结果无关）：
   - `NameError: AUG_BS`（常量编辑时两行被合并进注释）；
   - `NameError: geom`（domain 收尾元数据路径的变量注入未生效，**发生在全部 cell 已落盘、CSV 已 COMPLETE 之后**）。
   两者均只影响运行脚本的收尾，不影响已写出的逐图分数；修复后重跑仅执行 SKIP 路径，确认 cell 未变化（`SKIP ... (already complete)` ×7）。
3. **MetalParts test0 泄漏**：与 source train 有 **145 个同名文件**（104 normal / 41 anomaly）→ `valid=0`，**不进入任何聚合或退化指标**，仅以 `valid=0` 记录。
4. **PiledBags 不可运行**：9.41 GB 未下载、磁盘剩 6.1 GB，未做任何推测或替代。
5. **PCB 基线低于随机**（0.349–0.527）：这是**数据集/域偏移的真实难度**，不是实现错误——同 pipeline 在 PCB test0 上为 0.927、在 MetalParts 上为 0.80–0.99。必须作为限制随结果一起陈述。

---

## 9. 分析

### 事实

- PCB 5/5 target split 的 ΔAUROC、ΔAP **全为正**；MetalParts 只有 2/6（AUROC）为正，`background` 与 `scale` 组一致为负（−0.046 / −0.033），最差 MP test5 `ΔAP = −0.109`。
- 两域的 **B0 水平差异极大**：PCB target 平均 AUROC **0.4447**，MetalParts target 平均 **0.9261**。
- source split（PCB test0）上 R1 **略降**（−0.0098，CI 含 0）。
- 11 个 split 上 **B0 AUROC 与 ΔAUROC 的 Spearman ρ = −0.627（p = 0.039）**；AP 同值（ρ = −0.627，p = 0.039）。→ baseline 越低，增益越大。

### 解释（基于上述事实的推断，不作因果声明）

- E25 **不支持**「R1-ILA 在 RobustAD 上取得整体提升」这一强陈述：11 个 target split 的 macro Δ 分别为 **−0.0015（AUROC）/ −0.0064（AP）**，方向上与 0 不可区分；AUROC 上 CI 排除 0 的 5 个 split 中 **3 正 2 负**（正向 PCB test1/2/3，负向 MP test4/test5），即效应方向随域翻转而非一致。
- E25 **支持**一个**有条件**的陈述：在 baseline 明显失效（PCB，AUROC 0.35–0.53）的强偏移下，冻结的 adapter 仍带来 **+0.008…+0.037** 的一致改善（5/5）；在 baseline 已接近饱和（MetalParts，0.80–0.99）的偏移下，adapter 多为**小幅退化**。
- 这与 E13-B2（Cube：ΔAUROC 与 frozen R0 的 Pearson −0.84）、E22（Doll 例外）**同一模式**：R1-ILA 是 **baseline/headroom 依赖**的方法，不是普遍增益方法。
- PCB 上「source split 略降、shifted split 全升」与「illumination-consistency 在分布偏移下起作用」**相容**，但**不足以**作为机制证据（无 single-variable 控制，且 PCB 全部 split 的绝对水平过低）。

### 假设（**尚未验证**，不得写成结论）

- H1：MetalParts 的退化来自**近饱和基线的方差代价**（adapter 改写了特征尺度/邻域结构，在已经很高的 AUROC 上净损失）。→ 需要 per-image 分数分布与 bank 邻域统计来检验，本轮未做。
- H2：MetalParts 的负向集中在该域特有的 `background`/`scale` 偏移，而非「域偏移强度」本身。→ n=2 与 n=1，无法区分。
- H3：方形压缩（16:9 → 448×448）对 MetalParts 的绝对水平有影响；**对配对差值的影响未量化**。→ 需要「letterbox / 保持长宽比 + 可变 patch 数」的对照，属新协议。

---

## 10. 结论

1. **冻结方法在 RobustAD 上不是整体增益**（macro ΔAUROC −0.0015、ΔAP −0.0064，7/11 正、5/11 CI 排除 0）。论文**不得**写「R1-ILA 在外部数据集上提升检测」。
2. **可写的是**：在 baseline 失效的强偏移（PCB）上，冻结 adapter 带来一致的小幅改善（5/5，+0.008…+0.037）；在 baseline 近饱和的域（MetalParts）上，改善消失并部分转为小幅退化（`scale` −0.033、`background` −0.046…−0.062）。
3. **因此**，R1-ILA 的外部有效性只能以 **「headroom-dependent / shift-dependent」** 的方式陈述，且必须同时报告 MetalParts 的负向结果与 PCB 的低基线事实。
4. 本实验**没有**触发任何方法修改：BETA / augmentation 类型 / strength **全程未动**（见 §11 出口条件）。

---

## 11. 出口条件判断

**E25 没有预注册的 GO/STOP 阈值**（协议 §10 已声明），因此这里只做**描述性判定**，不假装成假设检验：

| 若观察到的形态 | 本应支持的结论 | 实际是否出现 |
|---|---|---|
| ≥70% valid target split 为正 **且** 每域均值为正 | 外部泛化成立 | **否**（7/11 正，MetalParts 均值为负） |
| 某域一致为正、另一域一致为负 | 条件性泛化（按域） | **是**（PCB 5/5 正；MP 4/6 负） |
| 两域均一致为负 | 外部泛化失败 | 否 |

- 判定：**MIXED（条件性）**，`descriptive_pattern = "MIXED"`。
- **不触发**：方法修改、参数重选、seed 挑选、结果过滤。
- **触发**：论文中「外部泛化」相关表述必须降级为条件陈述（§10）。

---

## 12. 对论文证据链的贡献

- **推进的证据格**：外部有效性（Stage ⑬）。E25 提供 2 个真实数据集/13 个条件/11 个有效 target split 的**配对**外部证据，并给出跨域异质性的量化（PCB +0.022 vs MP −0.021）。
- **闭合/排除的解释**：「R1-ILA 的增益只是 M²AD 合成增广的产物」**未被排除**，但被**限定**：在真实强偏移（PCB）上增益仍出现，说明不完全是合成增广产物；在真实弱偏移/近饱和域（MP）上增益消失。
- **产生的约束**：与 E13-B2/E22 合并后可写「baseline-dependent」的一般性陈述（3 条独立证据线：Cube 8 seeds、10 类 3 seeds 中的 Doll、RobustAD 2 域）。
- **下一格**：`seed 稳健性`（E25 只有 seed 0）→ 或 `E24 参数敏感性`（BETA / strength），二者都需**另行批准**。

---

## 13. 产物清单与 Git 状态

| 文件 | 说明 |
|---|---|
| `results/e25/ROBUSTAD_PROTOCOL_AUDIT.md` | 协议审计（§8 执行协议，§10 冻结配置/几何/口径修订） |
| `results/e25/run_manifest.json` | git HEAD、版本、GPU、冻结参数、seed、bank 规则 |
| `results/e25/smoke_check.json` | PCB smoke 18 项检查 |
| `results/e25/geometry_PCB.json` / `geometry_MetalParts.json` | 原始尺寸审计 + 与 E13-A prep 的等价性证据 |
| `results/e25/cells/*.json` | 13 个 cell：逐图分数、标签、bank/adapter hash、steps、runtime |
| `results/e25/e25_results_{PCB,MetalParts}.csv` | 逐 split 结果（每 split 后即时落盘） |
| `results/e25/e25_summary.csv` | 逐 split + bootstrap CI |
| `results/e25/e25_domain_summary.csv` / `e25_group_summary.csv` | 域 / 条件组聚合 |
| `results/e25/final_verdict.json` | 描述性裁决 + 限制 + 探索性 headroom 统计 |
| `results/e25/analysis.json` | 机器可读完整汇总 + 复算校验 |
| `results/e25/logs/` | driver 与各 domain 运行日志（含 5 次失败尝试，未删除） |
| `scripts/e25_robustad_eval.py` / `e25_analysis.py` / `e25_run_all.sh` | 运行与分析脚本（复用 `e13a_screen` / `e13b_screen` / `e11_p2_runner`，**未修改任何冻结文件**） |

**Git**：`results/e24/` **未创建、未引用**。本轮所有产物**尚未 commit**；拟提交清单见交付说明（数据集 `data/robustad/` 不入库）。

---

## 14. 下一步

1. **（建议，需批准）seed 稳健性**：同一冻结配置跑 seed 1/2，成本 ≈6–7 min/seed，用于区分「域间差异」与「adapter 训练随机性」；单 seed 是本轮最主要的方法学限制。
2. **（需批准，独立协议）E24 参数敏感性**：BETA / augmentation strength 的敏感性研究，**不得**在 E25 结果上反向选参。
3. **（需批准）MetalParts 几何对照**：letterbox 或保持长宽比的可变 patch 打分，用于量化 §9-H3。
4. **（需磁盘）PiledBags**：释放 ≥10 GB 后再下载；否则在论文中标注为未覆盖域。
5. 论文表述按 §10 降级；不要新增「外部验证成功」类措辞。

---

## 15. 多 seed 扩展（seeds 1/2，2026-10-19 夜间补齐）

**目的**：seed 0 是单点观测；本条回答「§7 的 PCB 正 / MP 负结构是否只是 seed 0 的运气」。

**配置**：与 seed 0 **完全相同**（仍只读 `results/e17/final_method.json`；BETA=0.1、brightness+contrast+gamma × {0.85,1.15}、lr 1e-3、≤300 step、patience 30、batch 8、DINOv2-B frozen、1-NN-max）；**只有 adapter 训练的随机种子变化**。bank 构造与测试图完全相同。**未依据任何 RobustAD 结果调参**。

### 15.1 逐 target split（3 seeds；mean ± std）

| domain | split | 条件组 | n(ok/ng) | AUROC F0 → R1 | ΔAUROC | ΔAP | seed 同号 |
|---|---|---|---|---|---|---|---|
| PCB | test0 | source ref | 120/75 | 0.92678 → 0.91870 | −0.00807 ± 0.00192 | −0.00759 ± 0.00261 | 0/3（全负） |
| PCB | test1 | illumination | 120/90 | 0.52722 → 0.55380 | **+0.02657 ± 0.00604** | +0.01691 ± 0.00533 | **3/3 正** |
| PCB | test2 | white balance | 120/60 | 0.44139 → 0.45866 | **+0.01727 ± 0.00489** | +0.01332 ± 0.01161 | **3/3 正** |
| PCB | test3 | pose (rot) | 80/60 | 0.34917 → 0.38125 | **+0.03208 ± 0.00922** | +0.01585 ± 0.00496 | **3/3 正** |
| PCB | test4 | pose (pos) | 80/60 | 0.39792 → 0.40403 | **+0.00611 ± 0.00459** | +0.01505 ± 0.00346 | **3/3 正** |
| PCB | test5 | shadow | 80/60 | 0.50792 → 0.51764 | **+0.00972 ± 0.00241** | +0.00066 ± 0.00198 | **3/3 正** |
| MP | test0 | **(INVALID)** | 120/51 | 0.93627 → 0.95523 | +0.01895 ± 0.00158 | +0.02847 ± 0.00424 | 3/3（不入聚合） |
| MP | test1 | illumination | 120/80 | 0.98865 → 0.99003 | +0.00139 ± 0.00022 | +0.00221 ± 0.00062 | 3/3 正 |
| MP | test2 | pose (pos) | 120/80 | 0.95760 → 0.95438 | −0.00323 ± 0.00400 | +0.01001 ± 0.00346 | 1/3 |
| MP | test3 | pose (rot) | 120/80 | 0.94292 → 0.94649 | +0.00358 ± 0.00118 | +0.00993 ± 0.00308 | 3/3 正 |
| MP | test4 | scale | 120/80 | 0.92833 → 0.90184 | **−0.02649 ± 0.00662** | −0.02614 ± 0.00828 | 0/3（全负） |
| MP | test5 | background #1 | 60/40 | 0.80208 → 0.74653 | **−0.05556 ± 0.01240** | −0.08985 ± 0.01966 | 0/3（全负） |
| MP | test6 | background #2 | 60/40 | 0.93708 → 0.91389 | **−0.02319 ± 0.00550** | −0.02783 ± 0.00799 | 0/3（全负） |

### 15.2 域聚合（11 个 valid 非 source split）

| domain | ΔAUROC mean | 逐 seed 域均值 | seed 间 std | ΔAP mean | 正向 split（均值） | 全 seed 同向的 split |
|---|---|---|---|---|---|---|
| PCB | **+0.01835** | 0: +0.02171 / 1: +0.01850 / 2: +0.01484 | 0.00344 | +0.01236 | 5/5 | **5/5 全 3 seed 正** |
| MetalParts | **−0.01725** | 0: −0.02082 / 1: −0.01852 / 2: −0.01241 | 0.00434 | −0.02028 | 2/6 | 3/6 全 3 seed 负、2/6 全 3 seed 正 |

### 15.3 总体与结构

- **总体 macro ΔAUROC = −0.00107**（逐 seed：−0.00148 / −0.00170 / −0.00002；seed 间 std 0.00091；描述性 t-区间 95% **[−0.00333, +0.00119]**）；**macro ΔAP = −0.00544**。
- 正向 split 比：**7/11（0.636）**；AP 侧 8/11。
- 最差 target：**MP test5 background（−0.0556）**；最好：**PCB test3 rotation（+0.0321）**。
- **结构保持判定**：`pcb_positive=True`、`mp_negative=True`、`pcb_positive_all_seeds=True`、`mp_negative_all_seeds=True` → **「PCB 正、MP 负」不是 seed 0 的偶然，而是跨 3 个独立 adapter 训练种子的稳定结构**（PCB 每个 seed 域均值 +0.0148…+0.0217；MP 每个 seed −0.0124…−0.0208，符号不翻转）。
- 校验：39/39 cell 逐图复算 AUROC/AP，`max|Δ| = 0.0`（容差 1e-12）。

### 15.4 解释（不得超出）

- seed 只改变 adapter 训练轨迹，不改变 bank / 测试图 / 标签，因此 3 seed 的均值差反映**训练随机性**，而域间差异（+0.0184 vs −0.0173）**远大于** seed 噪声（域内 std 0.003–0.004）→ 域效应真实存在，不是噪声。
- 但整体 macro 仍≈0（t-区间跨 0），因此**「R1-ILA 在 RobustAD 上整体提升」依然不成立**；成立的只是**条件性结构**：强偏移/低基线（PCB）一致改善，近饱和（MP）一致退化（background/scale 三 seed 全负）。
- **不得**把 PCB 的 +0.018 当作"外部验证成功"：其 F0 基线平均仅 0.4447（test3 0.349 低于随机），改善发生在**失效的基线**上；也不得把 MP 的负向隐藏。

### 15.5 产物

`multiseed_summary.csv`（逐 split）、`multiseed_analysis.json`（含逐 seed bootstrap CI 与 t-区间）、`figures/e25_multiseed_domain_delta.{png,pdf}`（PCB vs MP × ΔAUROC/ΔAP，含逐 seed 点）、`figures/e25_multiseed_per_target.{png,pdf}`（逐 target mean±std）。脚本：`scripts/e25_multiseed_analysis.py`、`scripts/e25_figures.py`；运行时 `scripts/e25_robustad_eval.py --domain PCB,MetalParts --seeds 1,2`（seed 0 文件命名不变）。
