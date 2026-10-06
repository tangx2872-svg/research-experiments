# Experiment 5A-H — Extended Cross-Category Validation of Frozen G2

> 状态：协议冻结于运行前。运行后仅追加 Results / Verdict / Limitations / Next。
> 前序：5A（bottle/seed0，CASE A 有条件）已冻结于 commit `ba11038`。
> 本实验**不开发新方法**，只验证 frozen G2 的 cross-category / multi-seed 泛化性。

---

## 0. Paper Navigation

📍 Stage ⑤ Method Design → 5A Mini Probe ✅ → **5A-H Extended Validation（当前）**

### ① 我怀疑什么？

5A 的 G2 优势（bottle 上 4/4 shifts 优于 equal-budget control C2 且逃出 fixed-α
frontier）可能是真实的 layer-allocation effect，也可能只是 bottle / seed0 的偶然
sweet spot（bottle 上 AUROC 饱和、FPR 退化，primary 口径失效，结论由 d′/Δz 承载）。

### ② 准备干什么？

冻结 5A 的全部方法参数（G2/C2/C3/B0/B2，零 α search），扩展到全部 5 个 MVTec AD
category × 3 seeds = 15 units × 5 configs = 75 config runs。

### ③ 看什么？

- illumination robustness（primary：mean |ΔNormalScore_z|）
- defect preservation（primary：mean defect d′）
- **G2 vs C2 equal-budget pairwise（核心）**：PAIR WIN = ΔRobustness < 0 AND ΔPreservation ≥ −ε
- Pareto dominance（G2 vs fixed set {B0, B2, C2, C3}）
- cross-seed / cross-category consistency
- defect-level：是否存在被 G2 系统性伤害的 defect family

### ④ 什么结果意味着继续？

只有 frozen G2 在多个类别、多个 seed 上稳定优于 C2 才进入正式 Method v1
（CASE A）。若优势只存在于 bottle → CASE C，停止把 G2 当通用方法。

---

## 1. Frozen Method Configuration（锁死，禁改）

来源：5A geometry_rule.json（ratio = s_L3/s_L2 = 0.603651，md5 `dd08ed5e...`）。
**精确值** = 5A 实际运行值（连续性优先；协议中 .453/.601/.802 为其 3 位小数显示）：

| config | α_L2（exact） | α_L3 | 说明 |
|---|---|---|---|
| B0 | 0.0 | 0.0 | original PatchCore |
| B2 | 0.5 | 0.5 | 中等 uniform |
| **G2** | **0.75×0.603651 = 0.45273825**（≈.453） | 0.75 | frozen geometry-guided |
| **C2** | **0.601369125**（=mean(G2)，≈.601） | 同左 | equal-budget control |
| C3 | 0.8018255（≈.802） | 同左 | strong uniform control |

禁止添加任何新 α。本实验零超参搜索。

## 2. PAIR WIN 与 ε（运行前冻结）

- ΔRobustness = |Δz|(G2) − |Δz|(C2)（< 0 → G2 wins）
- ΔPreservation = d′(G2) − d′(C2)
- **ε = 0.10（d′ 绝对单位）**
  理由：d′ 量级 3.7–13（5A），0.10 ≈ 典型 d′ 的 1–2%，高于数值精度与 float 噪声，
  低于任何有实际意义的效应；defect-level neutral band 同用 |Δd′| ≤ 0.10。
- **PAIR WIN：ΔRobustness < 0 AND ΔPreservation ≥ −0.10**

## 3. 数据与规模

- 5 categories（bottle 209 / cable 224 / grid 264 / hazelnut 391 / screw 320 train good）
  × seeds {0,1,2}；每 unit 5 configs；**75 config runs**
- Split / preprocessing / bank / coreset / illumination：与 5A 完全同一代码路径
  （`make_validation_split(cat, seed)`、`FAlphaDualLayerPatchcore`、coreset 0.1、k=9、
  `apply_photometric` brightness/gamma 0.7/1.3）
- **bottle:0 重跑不复制 5A 数据**：5A raw 无 pixel AUROC/AUPRO/clipped ratio；
  重跑保证 75 units 指标口径统一，并顺带做 5A 确定性复现检查（S14）
- 论文口径：**cross-category frozen-rule validation**（不是 independent external
  validation——Stage ④ geometry statistics 曾使用这 5 类）

## 4. 指标（冻结）

- **Primary robustness：mean |ΔNormalScore_z|**（4 shifts，越低越好）
- **Primary preservation：mean defect d′**（clean，越高越好）
- Secondary（仅当有效时解释）：image AUROC、pixel AUROC（复用 1H 实现）、
  AUPRO（anomalib `AUPRO(fields=["anomaly_map","gt_mask"])`，clean test 全集）、
  FPR@τ_val、clipped_pixel_ratio（每 shifted 图像素 ≥1−1e-6 的比例，纯图像空间统计）

## 5. 统计

- 15 units（及 excluding-bottle 12 units）G2 vs C2：paired 差值、win 计数、
  **Wilcoxon signed-rank**（scipy，two-sided）+ median paired difference
- per-category n=3 不做检验（最小 p≈0.25 无意义，只报 mean±std 与 wins/3）
- S11（seeds 确实不同）：同一 config 跨 seed 的 good 分数应存在差异（bank 随 seed 变）

## 6. 执行计划（并行安全策略）

- 3 个独立 worker 进程（unit 列表互斥，输出路径互斥，无共享 CSV）：
  - W1: hazelnut × 3 seeds
  - W2: screw × 3 + bottle × 3
  - W3: cable × 3 + grid × 3
- 每 worker 串行执行自己 unit 内 5 configs；预计 VRAM 3×2.6≈8 GB（<20 GB），
  RAM/14 核充裕；单 worker 失败不影响其他 worker 的已持久化结果
- Resume：config 级（info.json status==OK 且行数正确即跳过）
- 串行总量 ≈170 min；3 workers 预计 wall ≈60 min（hazelnut 最重）

## 7. Sanity（全部 PASS 才分析）

S1 α=(0,0)==original（<1e-6）；S2 G2 exact=(0.45273825, 0.75)；S3 C2 exact=mean(G2)；
S4 rule hash 不变；S5 同 illumination 协议；S6 同 bank/coreset 协议（unit 内 hash/size 一致）；
S7 同 preprocessing（同一 e1b 函数）；S8 无 defect label 进入 α 决策（常量 α）；
S9 无 NaN/Inf；S10 行数 = 20+n_good+n_defect+4×n_good；S11 seeds 确实不同；
S12 无并发写冲突（路径互斥）；S13 Stage ④ 文件 md5 前后不变；
S14（补充）bottle:0 与 5A raw 的分数复现一致性（max|Δscore|，阈值 1.0，
供参考不 gate）。

## 8. Exit Criteria（预注册）

- **CASE A — STRONG GENERALIZATION**：excl-bottle 12 units 多数 robustness win
  AND preservation non-inferior AND 多 category non-dominated/beyond AND 无系统性
  family 损伤 → Method v1 Design
- **CASE B — PARTIAL**：仅部分 category/seed 成立 → 不调 α，分析什么属性决定有效性
- **CASE C — BOTTLE-SPECIFIC**：excl-bottle G2≈C2 或更差 → STOP frozen G2
- **CASE D — TRADEOFF**：robustness↑ 但 d′ 系统↓ → STOP 当前 rule

## 9. 输出

`results/experiment_5a_h/{raw,summary,figures,logs}/`；
summary：per_unit / per_category / per_defect / g2_vs_c2 / cross_seed_summary /
cross_category_summary / pareto_summary / statistical_tests / sanity_checks；
figures：g2_vs_c2_robustness / g2_vs_c2_preservation / category_consistency /
seed_consistency / pareto_by_category / defect_level_effects。

## 10. Results（运行后追加，2026-10-06）

### 10.0 执行摘要

- Git：5A 冻结于 `ba11038`；5A-H 运行时 HEAD 不变（S13：Stage ④ 5 个文件 md5 前后一致）
- 75/75 config 完成，0 失败，0 重试；3 workers（互斥 unit），wall ≈ 110 min
- Runtime：总计 192 min CPU-GPU 时间，mean 154s / config（max 250s，hazelnut）
- Peak VRAM：单 config 4.8 GB（3 并发时观测 ~12.7 GB < 20 GB）
- **Sanity 23/23 PASS**，关键项：
  - S1 α=0 等价 max_abs_diff = 0.0
  - **S14：bottle:0 与 5A raw 915 行分数复现 max|Δscore| = 0.000000**（完整确定性复现）
  - S11 全部 config 跨 seed 分数确实不同；S10 行数全部精确
- **FPR@τ_val 在全部 5 类退化**（fpr_clean = 1.000 × 5 类）：train/good val 分数系统性低于
  test/good —— 5A 发现的 τ_val 退化是全局现象，不是 bottle 特有；Axis A 一律由 Δz 承载
- clipped_pixel_ratio：mean 6.1%，max 27.0%（brightness/gamma 1.3 高光裁剪，已记录，协议未改）

### 10.1 G2 vs C2（核心 pairwise，PAIR WIN: ΔRob<0 AND ΔPres≥−0.10）

| scope | n | robustness wins | preservation non-inf | **pair wins** | beyond frontier | dominates C2 |
|---|---|---|---|---|---|---|
| all | 15 | 10 | 7 | **4** | 4 | 2 |
| **excluding bottle** | **12** | **8** | **4** | **2** | **1** | **0** |

逐 unit（Δrob / Δpres）：bottle −.045/+.001, +.016/+.320, −.013/+.215；
cable −.009/−.077, +.032/−.061, −.035/−.060；grid −.023/−.184, −.004/−.145, −.023/−.132；
hazelnut −.050/−.258, +.000/−.251, +.033/+.065；screw +.010/−.107, −.022/−.255, −.002/−.284。

### 10.2 统计检验（Wilcoxon signed-rank，two-sided）

| scope | metric | median Δ | mean Δ | p |
|---|---|---|---|---|
| all 15 | robustness | −0.0089 | −0.009 | 0.173 |
| all 15 | preservation | −0.107 | −0.081 | 0.095 |
| **excl bottle 12** | robustness | −0.0066 | −0.008 | **0.292（ns）** |
| **excl bottle 12** | preservation | **−0.138** | −0.146 | **0.0024（显著）** |

- **robustness 优势在 bottle 之外不显著且量级微小**（median −0.007 |Δz|）
- **preservation 损失在 bottle 之外显著**：G2 系统性地比同预算 C2 更伤 defect evidence

### 10.3 Per-category（seed-mean；|Δz|↓ / d′↑）

| category | B0 | B2 | C2 | G2 | C3 | Δpres(G2−C2) | pair wins | beyond |
|---|---|---|---|---|---|---|---|---|
| bottle | .442/8.18 | .286/7.30 | .249/6.97 | **.235/7.15** | .234/6.55 | **+0.179** | 2/3 | 3/3 |
| cable | .331/5.05 | .271/5.01 | .238/4.81 | .234/4.74 | .254/4.62 | −0.066 | 2/3 | 1/3 |
| grid | **.224/3.14** | .100/2.36 | .087/2.13 | .070/1.98 | .105/1.80 | −0.154 | 0/3 | 0/3 |
| hazelnut | .214/5.96 | .151/6.17 | .143/6.15 | .138/6.00 | .138/5.96 | −0.148 | 0/3 | 0/3 |
| screw | .341/2.59 | .232/2.49 | .230/2.48 | .225/2.26 | .221/2.20 | −0.216 | 0/3 | 0/3 |

### 10.4 Defect-level（Δd′ seed-mean；improved 1 / neutral 12 / degraded 12，共 25 defect types）

- **系统性损伤**：screw 全部 5 个 defect type 下降（−0.14~−0.33）；grid 结构类
  bent/broken/glue −0.21~−0.29；hazelnut crack/cut −0.23/−0.37；cable swap/combined −0.23/−0.29
- 唯一 improved：bottle/broken_large +0.52（5A 优势的来源）
- 不存在被 G2 系统性**改善**的类别外 defect family

### 10.5 关键观察（供 Method 重设计讨论，均为事实描述）

1. 5A 的 G2 优势在 bottle 内跨 seed 部分成立（preservation +0.18±0.16，beyond 3/3），
   但在 4 个 validation category 上全面失效
2. **grid（Stage ④ expand 家族所在类）对一切 α-IN 极度敏感**：B0 d′=3.14 → 任何
   uniform/guided α 都降到 1.8–2.4；G2 未能幸免（1.98）
3. hazelnut 的 B2 轻微**改善** d′（5.96→6.17）——normalization 的收益方向存在 category 异质性
4. screw 对 α_L3 分配最敏感（−0.216），与其 Stage ④ neutral 家族（NN 扩张不传导 score）地位不符——
   rule 的 inverse-sensitivity 假设（L3 不敏感→可重压）在 validation 类上不成立

## 11. Verdict（运行后追加）

**CASE C — BOTTLE-SPECIFIC**（伴随 D 型失败模式）

按预注册判据逐条核对（excl-bottle 12 units）：
1. ❌ "多数 unit robustness 优于 C2"：8/12 数量上过半，但 median −0.0066、p=0.292 —— 
   统计上与 0 无差别，量级不足 5A bottle 效应（−0.045）的 1/6，不构成"稳定优于"
2. ❌ "preservation 基本 non-inferior"：仅 4/12，且 Wilcoxon p=0.0024 显著为负
3. ❌ "多个 category non-dominated/beyond"：beyond 仅 1/12（cable:2），dominates C2 = 0/12
4. ❌ "无系统性 family 损伤"：12/25 defect types degraded，screw 全类型受损

同时呈现 CASE D 的失败模式（robustness 未广泛提高但 d′ 系统下降）——两者共同指向：
**5A 的 trade-off 改善是 bottle/seed0 的 sweet spot，frozen G2 不是通用方法**。

**判定依据的强证据**：S14 完整确定性复现（bottle:0 max|Δscore|=0.000000）排除了
"运行差异导致 bottle 内外不一致"的解释——差异是真实的方法属性，不是实现噪声。

按协议：**STOP frozen G2，不救参数，不做 α search，不自动开发 V2。回 Stage ⑤
Method Design 等人工讨论。**

## 12. Limitations（运行后追加）

1. cross-category frozen-rule validation（非 external validation）：rule 统计来自含全部
   5 类的 frozen 9 defects，循环性未排除——但这同时意味着：即使在这种"对自己有利的"
   口径下 G2 仍失败，结论更强而非更弱
2. FPR@τ_val 全类退化（train/test good 系统差），Axis A 只能靠 Δz；AUPRO/pixel AUROC
   为 secondary（数值正常范围 0.92–0.99，但未参与判定）
3. illumination 仅 brightness/gamma 各 2 档；clipped ratio 最高 27% 的极端档仍在协议内
4. Δrobustness 的微小量级（~0.007 |Δz|）接近 seed 噪声尺度，须谨慎解读其符号
5. 单 backbone（wide_resnet50_2）

## 13. Next（运行后追加）

按出口条件：CASE C → STOP，等待人工讨论。讨论输入（不自动执行）：
- geometry rule 的失败原因假设（L3 敏感性排序跨类别不成立 / α_L3 主导本身有害 / 
  category-level 异质性需要 category-adaptive 而非 global rule——均为假设，未验证）
- 候选方向：回到 Candidate B，或基于 10.5 的 category 异质性观察重新审视问题定义
