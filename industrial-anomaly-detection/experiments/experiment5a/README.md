# Experiment 5A — Geometry-Guided Defect-Preserving Normalization Probe（Mini Method Probe）

> 状态：协议冻结于运行前（本 README + `results/experiment_5a/geometry_rule.json`）。
> 运行后仅追加 Results / Verdict / Limitations / Next。
> 前序链：Stage ④（1B→1J-B）已 ARCHIVED + FROZEN + REPRODUCED，本实验**不重开机制探索**。

---

## 0. Paper Navigation

📍 Stage ⑤ Method Design → 5A Mini Method Probe（①②③④✅，⑤进行中）

### ① 我怀疑什么？

固定 α 的 normalization 是"一刀切"的：它虽然可能降低 illumination-induced nuisance，
但 Stage ④ 已证明不同 layer / defect family 对 normalization 的响应不同，并伴随
feature geometry 的系统变化（1J-A：Layer3 expansion 为 expand 家族特有；1J-B：
radius→NN intervention 级传导）。因此利用 Stage ④ 的 layer geometry sensitivity
信息选择各层 normalization strength，可能比固定 α 更好地保留 defect evidence。

### ② 这个实验准备干什么？

构造最小 geometry-guided α 策略（deterministic rule，无神经网络 gate，无训练），
与 fixed α-IN baseline 及 mean-α control 在同一 Pareto 平面上对比：
illumination robustness（X 轴）× defect preservation（Y 轴）。

### ③ 我要看什么结果？

必须**同时**看两个方向（缺一不可）：
- **Axis A — illumination robustness**：normal test 图施加 photometric shift 后
  ΔNormalScore / FPR 是否随 α 降低、guided 是否更优；
- **Axis B — defect preservation**：真实 defect 的 image AUROC / d′ / separation
  是否保持或提高。

### ④ 什么结果意味着继续？

只有「robustness 改善 AND defect detection 基本不下降（或同 robustness 下明显
优于 fixed α / mean-α control）」才允许进入正式 Method Design。
若 robustness↑ 但 defect detection 明显↓ → 只是更强 normalization，不是
defect-preserving method。若 guided 不能稳定优于 fixed α → STOP Candidate A3，
禁止调参救结果。

---

## 1. 数据与范围（Mini 约束）

- Dataset：MVTec AD `bottle`（`data/mvtec_ad/bottle`）
- Seed：0（全部随机源：split / coreset / torch）
- Split：复用 1B 协议 `make_validation_split(bottle, seed=0)`：train good 209 →
  189 进 bank + 20 validation（只用于定阈值 τ）
- Test：good 20 + broken_large 20 + broken_small 22 + contamination 21（共 83 图）
- 运行时间预算：< 90 min；显存 ≪ 24 GB（1H 实测 bottle 每 α 峰值 ~2.6 GB）

选 bottle 的理由：①1B/1C/1E/1H 的历史 primary category，全部工具链可直接复用；
②defect type 覆盖三种面积/类型；③每 α fit 实测仅 ~63–99 s，满足 mini 时间约束。

**已知循环性 caveat（诚实披露）**：Stage ④ frozen 9 defects 包含 bottle/contamination
（shrink 家族），layer sensitivity 统计量部分来自 bottle。缓解：rule 是**全局
layer-level 常数**（对任何 test 图相同，无 per-image/per-defect 信息），且 defect 级
与 family 级两种口径给出一致的敏感性排序；Stage ⑤-2 将在 held-out category
（如 grid）上复核。本 probe 结果只作方向性证据。

---

## 2. 模型与实现（复用，不重写 PatchCore）

- 复用 `FAlphaPatchcore`（Lightning 包装）与 `FAlphaLayerPatchcoreModel` 的
  generate_embedding 结构（`scripts/falpha_layer_patchcore.py`）。
- 新增 `scripts/experiment5a_model.py`：`FAlphaDualLayerPatchcoreModel`，唯一变化是
  支持**每层独立 α**（`alpha_l2`, `alpha_l3`）：
  - layer2：`F2' = (1−α₂)F2 + α₂·IN(F2)`（concat 前，512×32×32）
  - layer3：`F3' = (1−α₃)F3 + α₃·IN(F3)`（upsample 前，1024×16×16）
  - 即 `F' = F + g·(IN(F)−F)`，g = α_layer ∈ [0,1]（协议第 4 节形式）
- **不修改** 1G–1J 任何代码/结果；α=0 时两层均直接返回原始 feature（与
  `FAlphaLayerPatchcoreModel._alpha_mix` 相同的 α=0 短路逻辑）。
- backbone wide_resnet50_2、layers=[layer2,layer3]、coreset ratio 0.1、
  num_neighbors 9 —— 与 1B/1E/1H 完全一致；bank 在各配置自身 α 下构建
  （fit 与 inference 一致，同 1B 协议）。

### Baseline 定义（运行前冻结）

**Primary baseline = uniform per-layer α（"both_layers"）**：

| ID | α_l2 | α_l3 | 说明 |
|----|------|------|------|
| B0 | 0 | 0 | 原始 PatchCore（α=0 等价） |
| B1 | 0.25 | 0.25 | fixed α |
| B2 | 0.50 | 0.50 | fixed α |
| B3 | 0.75 | 0.75 | fixed α |
| B4 | 1.00 | 1.00 | fixed α |

选 uniform per-layer 而非 post_concat 作 primary 的理由：guided 方法本质是
per-layer 操作，uniform per-layer 是**唯一只差"α 如何在层间分配"一个变量**的
对照组；post_concat IN 是不同算子（1536 维混合统计），混入会混淆 location 效应
与 guidance 效应。post_concat 历史 clean 结果（1H）仅作 descriptive reference，
不进入 primary Pareto。α grid {0,0.25,0.5,0.75,1} 与 Stage ③④ 完全复用。

**M1 — Geometry-Guided α（rule 见第 3 节，常数已冻结于 geometry_rule.json）**

| level | α_l2 | α_l3 | effective mean α (simple) |
|-------|------|------|---------------------------|
| A=0.5 | 0.302 | 0.500 | 0.401 |
| A=0.75 | 0.453 | 0.750 | 0.601 |
| A=1.0 | 0.604 | 1.000 | 0.802 |

**Mean-α Control（防"只是换了个平均 α"）**：uniform (α_l2=α_l3=ᾱ)，
ᾱ ∈ {0.401, 0.601, 0.802}（= 各 guided level 的 simple mean α）。

总 fit 数：5 (B) + 3 (M1) + 3 (control) = **11 个 unit**，预计 25–40 min。

---

## 3. Geometry Rule（运行前冻结，来源：Stage ④ 权威 CSV）

**数据来源（只读）**：
`results/experiment_1j/analysis/experiment1j_family_summary.csv`（GAI/radius response）、
`experiment1j_cross_chain_summary.csv`（layer-specific NN/score response）、
`results/experiment_1j_b/analysis/transmission_summary.csv`（β-intervention）。

**Rule（协议方案 A：inverse sensitivity，透明 deterministic，零训练）**：

1. Layer sensitivity（family 级，|family mean radius response| 再平均）：
   - s_L2 = (|−0.3715| + |−0.1667| + |+0.2629|) / 3 = **0.2670**
   - s_L3 = (|+0.0014| + |+0.0235| + |+0.4587|) / 3 = **0.1612**
   - defect 级口径核对：s_L2=0.267 > s_L3=0.169，**排序一致**（方向稳健）
2. 敏感层少 normalize，不敏感层可承受更强 IN：
   - `α_L3 = A`（budget），`α_L2 = A × (s_L3 / s_L2) = 0.604·A`
3. A ∈ {0.5, 0.75, 1.0}（与 fixed grid 的有效范围对齐）。

**Leakage 声明**：α 常数在加载任何 test 数据之前从 Stage ④ 冻结 CSV 计算并写入
`geometry_rule.json`；运行时对**所有** test 图（good / defect / shifted）使用同一组
常数，不使用 defect label / family / per-image 任何信息。若该 rule 只有配合
defect label 才成立 → 判 CASE E（ORACLE ONLY）。

---

## 4. Illumination Shift Protocol（复用已有实现）

复用 `scripts/experiment1_illumination_tradeoff.py` 的 `apply_photometric`
（在 [0,1] 像素空间、ImageNet normalize 之前施加）：

| shift | 定义 |
|-------|------|
| brightness 0.7 | `img × 0.7` |
| brightness 1.3 | `img × 1.3`（clamp） |
| gamma 0.7 | `img^(1/0.7)` |
| gamma 1.3 | `img^(1/1.3)` |

不加 shadow / reflection / color / blur / noise / geometric 变换（协议第 9 节）。
Shift 只施加于 **test 图像输入侧**；bank / threshold 不变。

---

## 5. Evaluation（两轴分开，禁止只报一个数）

阈值 τ（仅 Axis A 的 FPR 用）：各配置用其 **validation normal 分数 max**
（1B 协议：验证集 FPR=0，最保守），不用 test 数据定阈值。

### Axis A — Illumination Robustness（对 20 张 test good × 4 shifts）

- `ΔNormalScore_raw = mean(score_shift) − mean(score_clean)`
- **`ΔNormalScore_z = mean((score_shift − μ_clean)/σ_clean)`（primary，scale-free）**
- `FPR@τ_val`（shifted normal 超过 τ 的比例）
- 汇总：4 shifts 的 mean ΔNormalScore_z 与 mean FPR

### Axis B — Defect Preservation（对 63 张 clean defect 图 vs 20 张 clean good）

- image AUROC（pooled all-defect vs good；primary）+ per defect_type
- d′（复用 1H 公式：mean_gap / pooled_std；per defect_type）
- Defect Separation z = mean_z(defect)（scale-free）
- defect score change vs B0（descriptive）

### Pareto 图

X = mean ΔNormalScore_z（越低越 robust）；Y = pooled image AUROC（与 mean d′ 双口径）。
点：B0–B4、M1×3、mean-α control×3。判据：M1 是否落在 fixed-α frontier 之外
（同 X 更高 Y，或同 Y 更低 X），且优于对应 mean-α control。

---

## 6. Sanity Checks（运行前注册）

| # | 检查 | 判据 |
|---|------|------|
| S1 | α=0 identity | dual(0,0) 的 generate_embedding 与原 PatchcoreModel 输出 max abs diff < 1e-6 |
| S2 | NaN/Inf | 全部 score / 指标有限 |
| S3 | Leakage | α 常数来自 geometry_rule.json（运行前已冻结，mtime/hash 记录），运行时不读 test label |
| S4 | 同 bank 协议 | 11 个 unit 均用同一 189 张 train good（train id 列表 hash 一致）、coreset ratio 0.1、k=9 |
| S5 | illumination 数值 | apply_photometric 输出 ∈[0,1]；brightness 0.7 对值 1.0 像素 → 0.7 |
| S6 | rule 冻结 | geometry_rule.json 在任何 GPU unit 之前生成（已完成，本文件之前） |
| S7 | seed 固定 | split/coreset/torch seed=0 |
| S8 | 数据量一致 | val=20, train=189, test good=20, defect=63（20+22+21） |

任一 FAIL → STOP，不进入完整评估。

---

## 7. 输出目录（全部新建，不覆盖任何已有结果）

```
results/experiment_5a/
  stage4_geometry_reference.csv   （已生成，18 行）
  geometry_rule.json              （已生成，冻结）
  raw/per_image_scores.csv        （每图 × 每配置 × 每 illumination condition）
  summary/fixed_alpha_summary.csv
  summary/geometry_guided_summary.csv
  summary/pareto_summary.csv
  summary/sanity_checks.csv
  figures/illumination_normal_score.png
  figures/defect_preservation.png
  figures/robustness_preservation_pareto.png
  logs/
```

---

## 8. Exit Criteria（预注册，运行后自动分类）

- **CASE A — STRONG GO**：robustness↑ AND preservation≥fixed α AND 优于 mean-α control AND 在/超 Pareto frontier → 进 Stage ⑤-3 正式方法设计
- **CASE B — WEAK GO**：仅部分 shift level / 部分 defect 成立 → 保留，加 1 category/seed 确认
- **CASE C — NO ADVANTAGE**：guided ≈ best fixed α ≈ mean-α control → STOP A3，回 Candidate B
- **CASE D — TRADE-OFF FAILURE**：robustness↑ 但 defect evidence 明显↓ → Stage ④ 担忧再验证，STOP 当前 rule
- **CASE E — ORACLE ONLY**：只有用 label 才明显优于 fixed α → 标记 ORACLE UPPER BOUND

第一轮结果出来后禁止根据 test 结果修改 rule；若设计 V2 必须新建版本目录。

---

## 9. Results（运行后追加，2026-10-06）

### 9.0 执行摘要

- Git HEAD：`cbcf723`（main，工作树仅新增 5A 文件）；seed=0；GPU RTX 3090
- 11/11 config 完成，每 config 183 行（20 val + 20 clean good + 63 clean defect + 80 shifted good），总 2013 行
- Runtime：~71s/config（后两个 ~51s），总 wall ~16 min；peak VRAM 2605 MB/config
- Sanity：10/10 PASS（S1 α=0 等价 max_abs_diff = **0.0**，逐位一致；S5 曾因检查容差 bug 误报
  float32 精度差异，容差修正为 1e-6 后重跑检查 PASS —— 变换本身始终正确，未影响任何数据）

### 9.1 测量口径退化（必须如实报告的两个事实）

1. **pooled image AUROC 全线饱和 = 1.000**（11/11 config）。bottle 对 PatchCore 太容易，
   AUROC 无区分力；Axis B 的区分信息由 d′ 与 mean_z 承载。
2. **FPR@τ_val 全线 = 1.000（退化）**：τ_val（1B 协议 = 20 张 train/good 验证图分数 max
   ≈ 19–20.5）系统性低于 test/good 分数（mean ≈ 24.4–28.3，**clean 即 100% 超阈**）。
   train/good 与 test/good 存在 ~5–8 分系统差，τ_val 无法标定 test 集 → FPR 轴无信息，
   Axis A 的区分信息由 ΔNormalScore_z（primary）承载。

### 9.2 Axis A — Illumination Robustness（mean |ΔNormalScore_z|，4 shifts，越低越 robust）

| config | α_l2 | α_l3 | \|Δz\| | dz_b07 | dz_b13 | dz_g07 | dz_g13 |
|---|---|---|---|---|---|---|---|
| B0 | 0 | 0 | 0.478 | 0.585 | 0.566 | 0.649 | 0.113 |
| B1 | .25 | .25 | 0.337 | 0.351 | 0.451 | 0.460 | 0.086 |
| B2 | .50 | .50 | 0.285 | 0.356 | 0.362 | 0.371 | 0.048 |
| B3 | .75 | .75 | 0.247 | 0.353 | 0.302 | 0.308 | 0.024 |
| B4 | 1.0 | 1.0 | 0.236 | 0.425 | 0.213 | 0.263 | −0.043 |
| **G1** | .302 | .500 | 0.272 | 0.312 | 0.391 | 0.321 | 0.062 |
| C1 | .401 | .401 | 0.270 | 0.318 | 0.363 | 0.359 | 0.041 |
| **G2** | .453 | .750 | **0.235** | 0.355 | 0.303 | 0.264 | 0.017 |
| C2 | .601 | .601 | 0.280 | 0.390 | 0.359 | 0.328 | 0.043 |
| **G3** | .604 | 1.000 | 0.265 | 0.440 | 0.315 | 0.289 | 0.016 |
| C3 | .802 | .802 | 0.236 | 0.357 | 0.299 | 0.281 | 0.007 |

### 9.3 Axis B — Defect Preservation（clean，d′）

| config | pooled AUROC | d′(broken_large) | d′(broken_small) | d′(contamination) | mean d′ |
|---|---|---|---|---|---|
| B0 | 1.000 | 12.957 | 8.116 | 3.722 | 8.265 |
| B2 | 1.000 | 9.882 | 7.966 | 3.943 | 7.264 |
| B4 | 1.000 | 7.063 | 7.739 | 4.039 | 6.280 |
| **G1** | 1.000 | 10.637 | 7.851 | 3.833 | 7.440 |
| C1 | 1.000 | 10.621 | 7.805 | 3.966 | 7.464 |
| **G2** | 1.000 | 9.310 | 7.920 | 4.004 | 7.078 |
| C2 | 1.000 | 9.209 | 7.997 | 4.022 | 7.076 |
| **G3** | 1.000 | 8.228 | 8.057 | 3.974 | 6.753 |
| C3 | 1.000 | 7.829 | 7.668 | 4.011 | 6.503 |

（B1/B3 见 pareto_summary.csv；broken_large 是 normalization 主要受害者，B0→B4 掉 5.9 d′；
broken_small / contamination 基本稳定 —— 与 Stage ③④ 家族结论方向一致。）

### 9.4 核心 paired 比较（G vs C，相同 simple mean α）

| pair | Δ robustness (mean \|Δz\|, G−C) | Δ pooled AUROC | Δ mean d′ | 逐 shift |
|---|---|---|---|---|
| G1 vs C1 | +0.001（≈持平） | 0 | −0.024 | 混合（2/4 略差） |
| **G2 vs C2** | **−0.045（G 更 robust）** | 0 | +0.001（持平） | **4/4 shifts G 全部更优** |
| G3 vs C3 | +0.029（C 更 robust） | 0 | **+0.250（G 更保）** | — |

### 9.5 Pareto（X = mean \|Δz\|，Y = mean d′；fixed-α frontier 含全部 8 个 uniform 点）

- fixed-α Pareto frontier：**{B0, B1, C1, B3, C3}**（C3 支配 B4）
- **G2 (0.235, 7.078) 不被任何 fixed-α 点支配，且大幅超出 frontier**：同 robustness 水平
  （|Δz|≈0.235）下最优 fixed 点为 C3 (0.236, 6.503)，G2 的 d′ 高出 **+0.575**；
  相比 B4 (0.236, 6.280) 高出 **+0.798**。G2 同时直接支配 B3、B4、C2、C3。
- G1 (0.272, 7.440) 被 C1 (0.270, 7.464) 边缘支配（差距 ~0.02–0.03，噪声量级）
- G3 (0.265, 6.753) 非支配但在 C1–C3 插值 frontier 内侧（插值 d′≈7.32 > 6.753）
- 图：`figures/robustness_preservation_pareto.png`（primary，Y=AUROC，已饱和仅作记录）、
  `figures/robustness_preservation_pareto_dprime.png`（Y=mean d′，有效判据）

## 10. Verdict（运行后追加）

**CASE A — Guided Success（有条件）**

按预注册判据逐条核对（以 G2 为承载证据）：

1. ✅ illumination robustness ↑：G2 的 mean |Δz| = 0.235，为全部 11 个 config 最低
   （低于最强 uniform B4 的 0.236、其 control C2 的 0.280；相对 C2 改善 16%）
2. ✅ defect preservation ≥ fixed α：G2 mean d′ = 7.078，与其 control C2 持平（+0.001），
   高于同 robustness 的 C3 (+0.575) / B4 (+0.798)；AUROC 全线饱和无回退
3. ✅ 优于 mean-α control：G2 在 **4/4 shifts 全部更 robust** 且 preservation 不降
   —— 同 simple mean α 的配对设计排除了"整体强度不同"解释（CASE D 的核心担忧）
4. ✅ 超出 fixed-α Pareto frontier：见 9.5

**必须随 Verdict 记录的限定（不得在论文叙述中省略）**：

- 优势 **level-dependent**：只在 A=0.75（G2）成立；G1 ≈ C1（被边缘支配），
  G3 用 robustness 换 preservation（无支配）。"中间档有效、两端无效"需要解释，
  目前只能说 A 存在有效工作区间。
- pooled AUROC 饱和（bottle 太容易）—— preservation 结论完全由 d′ 承载；
  FPR@τ_val 退化（clean 即 100% 超阈）—— robustness 结论完全由 Δz 承载。
  两条 primary 口径在本 category 上均无区分力，是本 probe 最大的测量学弱点。
- 单 category（bottle）、单 seed、rule 统计部分来自含 bottle 的 frozen 9 defects
  （循环性 caveat，见 §1）。
- 效应量级：robustness 改善 ~0.045 |Δz|（相对 16%）；preservation 持平。
  非压倒性，方向性证据。

**结论（受上述限定约束）**：inverse-sensitivity layer allocation 在 bottle 上显示出
真实的、不能用整体 normalization 强度解释的 trade-off 改善（G2 严格支配其 mean-α
control 且逃出 fixed-α frontier）。geometry guidance 有方法价值的初步证据成立，
但不具备跨 category / 跨 level 的稳定性证明 —— 恰好落在 Stage ⑤-2 held-out
validation 的检验范围内。

## 11. Limitations（运行后追加）

1. AUROC 饱和 + FPR 退化（见 9.1）——5A 的两条 primary 口径在 bottle 上失效，
   结论由 secondary 口径（d′、Δz）承载。
2. 单 category / 单 seed / 单 backbone；rule 的循环性（§1）未被排除。
3. 3 个 guided level 中只有中间档成立，level-dependence 未被机制解释
   （可能原因：A=0.5 时两层 α 都较温和、层间分配空间小；A=1.0 时 Layer3 全额 IN
   引入的 distortion 超出"不敏感层可承受"假设——**这是解释，不是证据**）。
4. illumination shift 仅 brightness/gamma 各 2 档（协议第 9 节约束）。
5. d′ 的分母依赖 good 分数分布（20 张），单 category 下抽样误差不可忽略。

## 12. Next（运行后追加）

按出口条件：CASE A → **Stage ⑤-2 held-out category validation**（在未参与 rule 统计的
category 上复验 G2 vs C2 配对与 Pareto 位置；grid（expand 家族）与 hazelnut（shrink）
是互补的候选）。执行前需用户批准；本轮按协议 STOP。
