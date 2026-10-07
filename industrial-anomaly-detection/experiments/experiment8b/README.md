# Mini Experiment 8B — Defect–Illumination Feature Separability Probe

## 0. 论文导航

**📍论文阶段（论文导航四项中的第 1 项）**

Phase 5 — 新方法选型 / Solution Selection。

此前已完成：illumination robustness 问题确认 → defect-specific normalization response →
多类别多 seed 复现 → layer-specific sensitivity → illumination severity response →
α-response curve → 简单 representation intervention 系统筛选（7A-O: **CASE_C**，
simple representation intervention cannot push the preservation–robustness frontier）。

因此本实验**禁止**继续 λ / γ / α / layer allocation / norm type 的无目的调参。

**🎯 论文要证明的话（第 2 项）—— 本实验只检验这一句 hypothesis**

> Defect sensitivity and illumination sensitivity may be heterogeneously distributed across
> pretrained feature dimensions, leaving a potentially selectable subspace that is
> simultaneously defect-sensitive and illumination-stable.

这是 **hypothesis，不是事实**；8B 的任务是证伪或支持它。在看到结果前不得写成结论。

**🧪 当前实验（第 3 项）**：上文 hypothesis 是 Plan C
（Illumination-Stable Anomaly Feature Selection）的唯一科学前提。若不存在这样的子空间，
Plan C 无立足点，应转向 Plan A（Defect-Preserving Illumination Consistency）。

**🚦出口条件（第 4 项）**：见 §4。所有阈值在读取 8B target results 前冻结。

---

## 1. PRE-RUN PLAN（冻结于 2026-10-07，读取任何 8B target result 之前）

### 1.1 日期 / 环境 / Git

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-07 |
| 环境 | AutoDL conda base, Python 3.12.3, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, Anomalib 2.6.2 |
| GPU | NVIDIA GeForce RTX 3090 24GB（P0 时 1 MiB 占用，空闲） |
| branch | `main` |
| HEAD | `5951b755621bec3bfaef5453b1b67d1c7cc0823b` |
| origin/main | `5951b755621bec3bfaef5453b1b67d1c7cc0823b`（与 HEAD 一致，worktree clean） |
| 数据 | MVTec AD 五类（bottle / cable / grid / hazelnut / screw）完整 |

### 1.2 P0 资产审计结论（实测，非根据 README 假设）

审计脚本：`scripts/experiment8b_audit.py` → `results/experiment8b/audit/asset_audit.{json,md}`。

**A. Normal feature assets — 存在，但不是全部可复用**
- `results/experiment_1j/banks/{cat}_seed{seed}_{layer}[23].npy`：**30/30 齐备**
  （5 category × 3 seed × {layer2(512), layer3(1024)}）。
- 但它们是 **coreset 10% 子样本**（fractional k-center），不是完整 normal 分布：
  per-channel σ 中位数 0.82，p01=0.008，**43/23040 channel σ < 1e-6，62/23040 σ < 1e-3**。
  → 用 coreset 估 μ/σ 有偏；8B 的 σ 必须重新从 **完整 train/good 特征** 估计。
- 结论：bank 可作 sanity / 交叉参考，**不可作为 primary normal reference**。

**B. Defect-side assets — 存在，但只覆盖 frozen 9 primary defects**
- `results/experiment_1j/raw/{cat}/{defect}/seed{s}/{layer}/{stem}_a{0,1}.npz`：共 **1848 个 npz**
  （a0=924, a1=924），float16，layer2=(512,32,32)、layer3=(1024,16,16)，与 1J 口径一致。
- 覆盖 = 9 frozen primary defects × 3 seeds × 2 layers × 2 alphas × 全部该 defect 图像。
- **缺口**：9 个 primary defect 只覆盖 bottle×1 / cable×1 / grid×3 / hazelnut×1 / screw×3 个 defect type；
  §15 要求 per-defect-type 报告（bottle 有 3、cable 有 8、grid 有 5、hazelnut 有 4、screw 有 5 类），
  仅靠 1J 无法完成 defect-type control。

**C. Illumination-side assets — 不存在 feature-level 资产（关键缺口）**
- `results/` 下**除 1J 之外不存在任何 .npy/.npz feature 文件**（脚本清点，实测为空列表）。
- 7A-O Q3（severity ±10/30/50%）与 Q4（15/15 uniform-α 点）只保存
  `per_image.csv`（per-image anomaly **score**）+ `info.json`，**没有 feature representation**。
- 因此：**score-level illumination response ≠ feature-level illumination sensitivity**，
  §7 要求的 paired feature 表示 **必须补最少 GPU 提取**。

**D. 数据 / mask 覆盖**
- test 图像：bottle 20 good + 63 defect（3 类）；cable 58 + 92（8 类）；grid 21 + 57（5 类）；
  hazelnut 40 + 70（4 类）；screw 41 + 119（5 类）。全部 defect 图均有非空 GT mask。
- mask→feature-grid（overlap > 0.5，沿用 1G 冻结约定）：
  layer2 上仅 7/167 张 frozen-9 图像无 defect patch，**layer3 上 63/167 张无 defect patch**
  （screw 三个 defect type 尤其严重：17/24、19/23、11/23）。
  → 冻结 fallback 规则（见 §2.4）。

### 1.3 资产是否足够 / 工作量

| 问题 | 结论 |
| --- | --- |
| 已有资产是否足够回答 8B？ | **不足**。D 侧可（部分）复用 1J，I 侧 **完全缺失**。 |
| 是否需要 GPU？ | **需要，且为“最少必要补测”**：只做 frozen backbone forward + 统计聚合，**不重训 PatchCore、不改任何已冻结模型/协议**。 |
| 是否可 Round-1 CPU-only 完成？ | **不能**（I 侧 feature 不存在）。CPU-only 部分：D 侧的资产审计与预检；其余全部依赖 Phase A 输出。 |

**GPU 工作量估算（冻结预算）**

| Phase | 内容 | units | 单元成本 | wall-clock | VRAM | storage |
| --- | --- | --- | --- | --- | --- | --- |
| A | illumination/defect 特征提取（8B-A，无 bank、无 fit） | 15 (cat,seed) | 每个 ~750 次 forward（train+test+shifted），批 16 | **~10–20 min**（单 worker） | < 3 GB | ~60 MB（只存 channel 级摘要） |
| B | score-level selection probe（复用 5A-H 完整评估路径 + channel mask） | 15 × 28 masks = **420** | 46–111 s / unit（历史实测口径） | 单 worker ~5.8 h；**3 workers ~2 h** | ~2.6 GB / worker（mask ≤ full 1536 dim，不会触发 Family-B 的 OOM 情形） | ~120 MB（per_image.csv + info.json） |
| Smoke | bottle:seed0 全路径等价性 + mask alignment + extraction 复现 | 3–5 | — | ~5 min | < 3 GB | — |

执行方式：`tmux` 后台 + 日志 + PID 文件 + resume（unit 级 `info.json status==OK` 跳过）。

---

## 2. 冻结的 primary metrics（在读取 target results 前冻结，禁止事后修改）

### 2.1 分析粒度
- **Priority 1 channel-level**：layer2（C=512）+ layer3（C=1024），共 1536 channel/cell，15 cell。
- 不使用 grouped-channel / PCA（资产足够，无需降级）。
- layer-level 只作 sanity/reference；channel 级结论必须能在 layer 内成立（§3.7）。

### 2.2 表示与数据
- 表示 = **α=0 的 PatchCore pretrained 表示**（wide_resnet50_2, layers=[layer2, layer3],
  ImageNet pretrained, 无 fit、无 α-IN）。layer3 保持原 16×16，不做 upsample（与 1J hook 一致）。
- 预处理与历史严格一致：`e1b.load_image_as_tensor` + `e1b.preprocess_for_model`
  （resize 256 antialias + ImageNet normalize）。
- split 与历史严格一致：`e1b.make_validation_split(cat, seed)` → val 20 / train 其余。
- **normal reference**（D 的分母与 I 的 scale normalization）= `train/good` 的**全部** patch 特征，
  per (cat, seed, layer) 计 μ_c、σ_c、n_patches。
- **paired illumination 图像** = `test/good`（与该实验历史的 robustness 轴同一集合）。

### 2.3 Defect sensitivity `D_c`（primary 定义）
- mask：GT mask → resize 256（BILINEAR）→ `>0.5` 二值 → grid overlap 比例（1G 约定）。
- defect patch = overlap **> 0.5** 的格点。
- 每张 defect 图 i：`m_{i,c}` = 该图 defect patch 在 channel c 上的**均值**（area 只影响 patch 数，
  不影响权重）。
- **per-image equal weight**（禁止把所有 defect pixel/patch 直接堆起来让大缺陷支配）：
  `D_c^signed = ( mean_i m_{i,c} − μ_c ) / σ_eff,c`，`D_c^abs = |D_c^signed|`。
- ⚠ 实现顺序更正：本实验 primary D 使用 **所有该 category 的 defect type**（§15 要求 per-type 报告，
  frozen 9 无法覆盖）；frozen 9 primary defect 作为 secondary 视图单独报告。
- 同时保存 signed 与 absolute；**ranking 用 `D^abs`**（kNN 距离只关心偏离幅度，不关心方向）。
- 报告的 secondary 变体（各一个，不构成调参）：pooled-SD 版本
  `sqrt((σ_c² + sd_between-image(m_{i,c})²)/2)`。

### 2.4 layer3 空 defect patch 的 fallback（冻结）
若某图在该 layer 上 overlap>0.5 的格点为 0（实测 layer3 上 63/167 张 frozen-9 图会触发），
则该图使用 **overlap 最大的单个格点**（top-1）。
- 该规则与任何 D/I 数值无关，仅由 §1.2D 的覆盖率事实驱动，**在读取 target results 前冻结**。
- per-image `fallback_flag` 必须落盘并在 `defect_type_summary.csv` 中报告（不得隐藏）。

### 2.5 Illumination sensitivity `I_c`（primary 定义）
- 复现历史唯一 illumination 实现：`experiment1_illumination_tradeoff.apply_photometric`
  （brightness: img*level；gamma: img^(1/level)；clamp [0,1]），不新增 family、不新增实现。
- 条件集（12 个，冻结）：family ∈ {brightness, gamma} × level ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}；
  severity = |1−level|：mild {0.9, 1.1}、medium {0.7, 1.3}（= 历史值）、strong {0.5, 1.5}。
  另加 identity（level=1.0）仅用于 S6 校验，**不计入 I**。
- 同图、同 spatial location、同 layer/channel：
  `Δ_c(x, T) = mean_patches | f_c(T(x)) − f_c(x) |`（per-image），
  `I_c(T) = mean_x Δ_c(x,T) / σ_eff,c`（**per-image equal weight**）。
- **scale normalization（防“大数值 channel 看起来更敏感”）**：
  `σ_eff,c = sqrt(σ_c² + ε²)`，`ε = 1e-3 × median_c(σ_c)`（per cat-seed-layer 计算，冻结）。
  ε 由 §1.2A 的实测退化 channel 数量驱动（43 个 σ<1e-6），在读取 target results 前冻结。
- pooled：`I_c^mean` = 12 个条件的**无权平均**（primary）；
  `I_c^brightness` / `I_c^gamma` = 各 6 个条件均值；severity 级 = 该 severity 4 个条件均值。
  全部落盘（brightness 与 gamma、severity **分开报告**）。
- secondary 聚合口径（各一个）：patch-pooled 加权版。

### 2.6 分析前的预注册 robustness 变体（仅此一个，不构成网格搜索）
`σ_c < 1e-6` 的 degenerate channel（43/23040 = 0.19%）：**不删除**（§8 禁止删 bad channel），
但在 channel_metrics.csv 中打 `degenerate_flag`，并**额外**报告一次“剔除 degenerate channel 后”
的 correlation / Pareto / selection 结论（`*_nodegenerate.csv`）。
核心结论必须在两种口径下同向，否则按 §4 降级并写入 notes。

---

## 3. 冻结的统计量、Pareto 与 selection probe

### 3.1 二维图与 correlation
- Fig 1：全合法 channel 的 `(I_c, D_c^abs)` scatter（X = illumination sensitivity ↓，Y = defect sensitivity ↑；
  理想 feature 在左上角），layer2/layer3 可区分，不删 channel、不只画 Top-K。
- Spearman(|D|, I) 为 primary correlation；Pearson 仅 secondary。按 category / seed / layer / pooled 分别报告。
- 解释口径（冻结，仅作解释，**不作 CASE_A 判据**）：
  ρ ≥ 0.5 强正耦合 → Plan C 困难；0 < ρ < 0.5 弱正耦合；ρ ≈ 0 无关系；ρ < 0 有选择空间。

### 3.2 Pareto
Feature A 支配 B ⟺ `D_A ≥ D_B` 且 `I_A ≤ I_B` 且至少一个严格不等。
在 **layer 内**计算 Pareto frontier（避免把 L2/L3 尺度差异误读为 feature 级结构）。
报告：frontier 内 feature 数量、layer composition、category consistency、seed consistency。

### 3.3 Selection ranking（冻结，禁止网格搜索）
- 在 **layer 内** 计算 z 分数：`z_D,c = z(D_c^abs)`、`z_I,c = z(I_c)`（within cat-seed-layer）。
- **PROPOSED**（唯一 scalar 版本）：`s_c = z_D,c − z_I,c`。
- **D-ONLY**：按 `D_c^abs` 降序；**I-ONLY**：按 `I_c` 升序。
- **PARETO**：layer 内非支配集。
- **FULL**：全部 channel。
- **RANDOM-k**：layer 内均匀随机 k%（与 proposed 同 layer 结构）。
- 禁止：D/I 比值、D−0.5I、2D−I、log、nonlinear、多组 λ 等任何其他 scalar（§11/§22）。

### 3.4 Selection ratio（冻结，仅两个）
Top **25%** 与 Top **50%**（**在每个 layer 内**各自取 k%，保证 layer-balanced；
这也是 §14 layer-control 的天然实现：结果不能仅由 “L3 差、L2 好” 解释）。

### 3.5 Metric-level selection 统计（CPU）
- 子集聚合：`D̄(S) = mean_{c∈S} D_c^abs`，`Ī(S) = mean_{c∈S} I_c`。
- Null：**1000 次** layer 内随机抽取（与 S 同大小，冻结随机种子 8B0001），报告 mean / std / 95% interval。
- 每 cell 的两轴 z 值：`z_D(S) = (D̄(S) − μ_null,D)/σ_null,D`，`z_I(S) = (Ī(S) − μ_null,I)/σ_null,I`。
- **cell win（metric-level，冻结）**：`z_D ≥ +2` 且 `z_I ≤ −2`。
- **cell 方向改善（冻结）**：`z_D > 0` 且 `z_I < 0`。

### 3.6 Score-level selection probe（Phase B，GPU；复用 5A-H 完整评估路径）
- 用与历史**同一代码路径**（`experiment5a_h_runner.run_config` + 7A-O 的 monkey-patch 方式）
  只把 embedding 换成 **channel-masked embedding**（layer 内选中的 channel 子集，α=0）。
- mask 集（冻结，28 个/cell）：FULL；PROPOSED25/50；D-ONLY25/50；I-ONLY25/50；PARETO；
  RANDOM25 × 10 draws；RANDOM50 × 10 draws（draws 的随机种子 8B0002…，冻结）。
  （metric-level null 用 1000 次；score-level 因 GPU 预算冻结为 10 次/ratio —— 在 §1.3 中已预先申报。）
- 指标沿用历史定义（`experiment5a_h_analysis`）：`mu_g, sd_g` 取自 clean_good（test/good），
  `d′ = (mean_defect − mu_g)/sqrt((sd_defect² + sd_g²)/2)`，per-defect 后取 defect type 平均；
  `|Δz|` = 4 个 medium shift（brightness/gamma × 0.7/1.3）的 `mean|E[(s_shift−mu_g)/sd_g]|`。
- **probe win（cell 级，ratio 25%）**：`d′_prop ≥ d′_ref` 且 `|Δz|_prop ≤ |Δz|_ref`，且至少一个严格不等；
  ref ∈ {FULL, RANDOM-mean}。

### 3.7 Layer-control（§14，冻结）
- Test 1：L2 内部 / L3 内部各自的 D/I separability（within-layer ranking + within-layer null）。
- Test 2：控制 layer identity —— 因为 selection 本身就是 layer 内进行，如果 separability 只来自
  “L3 整体差”，则 L2-only 与 L3-only 子集不可能同时成立。因此：
  **C7 要求 L2 与 L3 各自 independently win（z_D≥2 且 z_I≤−2）于 ≥9/15 cell**。
  只有一个 layer 成立 → 至多 CASE_B，不得 CASE_A。

### 3.8 Defect-type control（§15，冻结）
按 category × defect type 报告：`D̄` 全 channel vs `D̄` PROPOSED-selected channel，
以及 per-defect-type 的 “selection 是否仍优于 random-null”。禁止只报告 pooled mean。
若某 feature 对 crack 好但 contamination 极差，必须如实记录。

### 3.9 Leakage policy（§13，冻结）
- 8B-A 使用 defect mask，属 **oracle / diagnostic**：其结果只能作为 diagnostic evidence，
  **不得**当作最终 method performance。
- ranking 由 **同一 (cat, seed) 的 target-side D/I 自身** 计算（最大 oracle），因此 score-level probe 的
  d′/|Δz| 一律标注为 **oracle-diagnostic**，不进入任何“方法有效”的论文论断。
- 8B-B（normal-only proxy，如 synthetic anomaly / SimpleNet-style perturbation）**本轮禁止开发**；
  只有在 8B-A 支持 separability 时才在下一轮单独预注册。

---

## 4. 冻结的 CASE 判定规则（机械执行，禁止人工“看起来不错”）

**统计口径**：cells = 15（5 categories × 3 seeds）；primary ratio = **25%**（50% 为 secondary 报告）。

### 4.1 布尔条件

| ID | 条件（全部在读取 target 结果前冻结） |
| --- | --- |
| **C1** | defect↑：`z_D(PROPOSED25) ≥ +2` 于 **≥12/15** cell，且 `D̄(sel) > D̄(FULL)` |
| **C2** | illumination↓：`z_I(PROPOSED25) ≤ −2` 于 **≥12/15** cell，且 `Ī(sel) < Ī(FULL)` |
| **C3** | category consistency：**≥4/5** category 的 ≥2/3 seed 同时满足 cell win |
| **C4** | seed consistency：**≥2/3** seed 的 ≥3/5 category 满足 cell win；**且** 3-seed pooled ranking（在每 category 内按 3 seed 合并的 metric 计算）在 ≥4/5 category 满足 cell win |
| **C5** | negative transfer：**0** category 出现 `D̄(sel) < D̄(FULL)` 或 `Ī(sel) > Ī(FULL)`；容差（worst-category degradation）冻结为两轴各 **2% 相对松弛** |
| **C6** | null：**6a** 15 个 cell 的平均 z 满足 `mean(z_D) ≥ +3` 且 `mean(z_I) ≤ −3`；**6b** score-level probe（§3.6）四项全成立：<br>(i) probe win vs FULL ≥ 10/15；(ii) probe win vs RANDOM-mean ≥ 12/15；(iii) `mean(d′_prop − d′_FULL) ≥ 0` 且 `mean(|Δz|_prop − |Δz|_FULL) ≤ 0`；(iv) `mean(d′_prop − μ_d′(RAND)) > 0` 且 `mean(|Δz|_prop − μ_|Δz|(RAND)) < 0` |
| **C7** | 不是 layer identity：L2-only 与 L3-only 各自 `z_D≥2 且 z_I≤−2` 于 **≥9/15** cell |

### 4.2 判定顺序（冻结，机械）

1. 任一 **primary sanity（S1–S13）FAIL** → **CASE_D**，`plan_c = INVALID`（不判 Plan C 成败）。
2. `C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6(a) ∧ C6(b) ∧ C7` 全真 → **CASE_A**，`plan_c = GO`。
3. `C1 ∧ C2` 为真但 C3–C7 有失败 → **CASE_B**，`plan_c = HOLD`
   （**特别地：C6(b) score-level probe 失败时只能是 CASE_B**，因为“selectable subspace 有用”
   必须至少在同口径 score 上体现）。
4. 否则若 directional cell（`z_D>0 且 z_I<0`）≥ **8/15** → **CASE_B**，`plan_c = HOLD`。
5. 否则 → **CASE_C**，`plan_c = STOP`；若此时 pooled `ρ_Spearman(|D|,I) ≥ 0.5`，
   在 verdict 中记录机制为 “defect / illumination sensitivity coupled”。

`verdict.json` 必须携带每个 boolean 的原始数值（不得只写结论）。

---

## 5. Sanity checks（S1–S13，任一 primary FAIL → 停止 verdict）

| ID | 检查 | 判据 |
| --- | --- | --- |
| S1 | feature shape consistency | layer2=(512,32,32)、layer3=(1024,16,16)，全部 unit 一致 |
| S2 | channel ordering consistency | 与 1J 资产逐 channel 对齐（同 index 同 channel） |
| S3 | layer identity consistency | layer3 不 upsample、命名与 1J 一致；layer2/layer3 不混用 |
| S4 | original score equivalence | FULL mask / α=0 的 `per_image.csv` 与历史 α=0（7A-O SMOKE_ORIG / Q4a0 等）逐图对比，容差 1e-4 相对 |
| S5 | image ↔ mask alignment | mask stem 与图像一一对应；defect patch 非空（含 fallback）；样本上 top-5% anomaly patch 与 mask 的重叠显著高于随机 |
| S6 | paired illumination spatial alignment | identity 条件 `level=1.0` 时 `Δ_c ≡ 0`（bit-level）；shape/grid 不随扰动改变 |
| S7 | no test leakage into final method claims | ranking 为 oracle-diagnostic；verdict 中所有 score-level 结果标记 `oracle_probe_only=true` |
| S8 | D metric finite | 无 NaN/Inf；degenerate channel 计数已记录 |
| S9 | I metric finite | 同上 |
| S10 | scale normalization valid | `σ_eff = sqrt(σ²+ε²)`，ε 已冻结并落盘；degenerate 变体报告存在 |
| S11 | category × seed coverage complete | 15/15 cell，每 cell 1536 channel 无缺 |
| S12 | random baseline reproducible | 同 seed 重跑 null 的 mean/std 逐位一致 |
| S13 | source consistency | 新提取 feature 与 1J npz 在同一图像上 `corr ≥ 0.9999`、`median\|Δ\| ≤ 1e-3`、`p99\|Δ\| ≤ 5e-3`、`max\|Δ\| ≤ 0.05`（**PRE-RUN AMENDMENT A1**，见 §9；原“bit-exact ≤1e-6”在 GPU 上不可达） |

---

## 6. 输出清单

`results/experiment8b/` 下：
`audit/asset_audit.{json,md}`；`reference/protocol_freeze.json`(+sha256)；
`smoke/`；`extract/`（channel 级摘要 + sanity_extract.json）；
`analysis/channel_metrics.csv`、`correlation_summary.csv`、`pareto_summary.csv`、
`selection_probe_metric.csv`、`category_stability.csv`、`seed_stability.csv`、
`defect_type_summary.csv`、`random_baseline.csv`、`layer_control.csv`、
`*_nodegenerate.csv`；`probe/`（Phase B per_unit per_image.csv）；`figures/fig1..fig7`；
`sanity/sanity_report.md`；`verdict.json`；`final_report.md`。

## 7. 禁止事项（§22 原文，全部适用）
禁止调 α / λ / γ / 换 normalization / 大规模 architecture search / selection ratio grid search /
ranking formula grid search / 看到结果改 primary metric / 用 test mask 选 feature 后把同一 test
performance 当正式方法结果 / 为得到 CASE_A 改阈值 / 隐藏 negative category / 删除 failed run /
自动进入下一大型实验。

## 8. POST-RUN（实验结束后填写）
见文末 `

## POST-RUN REPORT（生成于 2026-10-07，脚本 experiment8b_report.py）

- 冻结 sha256：`5f2bbd54cbc9964d…`（未事后修改阈值/ranking/ratio）
- Phase A：15/15 cells，524 s；Phase B：420/420 units，19.67 GPU·hour
- S13：240/240 PASS；S4：PASS
- metric level：z_D≥2 15/15、z_I≤−2 15/15、mean z_D 18.64、mean z_I -9.94、pooled ρ 0.0666
- score level (C6b)：win vs FULL 5/15、win vs RANDOM 12/15、mean Δd′ -0.2285、mean Δ|Δz| -0.1545
- **判定：CASE_B，Plan C = HOLD**
- secondary ratio（50%）正向信号（不参与判定）：PROPOSED50 Δd′ = +0.1892、Δ|Δz| = −0.1610，逐 cell 胜 FULL 11/15、胜 matched-size random 15/15；PROPOSED25 对 random 12/15。
- negative results：Pareto ranking 作 mask 失败（d′ 1.1712 / |Δz| 3.9865，0/15 胜）；D-ONLY 无稳健性收益；I-ONLY50 d′ −0.1363；cable 为唯一系统性负迁移 category。
- 事实/解释/假设分离与完整 negative results 见 `results/experiment8b/final_report.md`
- 下一步仅建议、未执行；8B-B（normal-only proxy）本轮禁止开发。
