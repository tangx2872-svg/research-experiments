# Experiment 9B — Second Method Screening（M7–M11）

**日期**：2026-10-07 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / Python 3.12.3 / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**Git HEAD（冻结时）**：`80dabd5` (main)

> **状态：PROTOCOL FROZEN（在读取任何 9B target result 之前冻结）。**
> 冻结内容见 `config.json`（`candidates_frozen` / `normal_only_illumination_sensitivity_frozen` / `budget_estimate_frozen` / `exit_conditions_frozen`）。

---

## 1. 论文导航（实验前四问）

| 项 | 内容 |
|---|---|
| 📍 论文阶段 | **方法发现 / Method Discovery**（继 9A「无方法晋级」之后的第二批候选海选） |
| 🎯 要证明的话 | 「是否存在 **deployable / normal-only / training-free / PatchCore-compatible** 的第二批候选，能改善 robustness–preservation trade-off」 |
| 🧪 为什么做 | 9A 已淘汰 M3/M4、HOLD M1/M2/M5、M6 仅为 oracle；方法侧仍缺一个「可用」的候选 → 本轮测试 5 个**不依赖 oracle、不依赖 test label、无训练参数**的新候选 |
| 🚦 出口条件 | ADVANCE：两类整体改善且无严重跨类别反向恶化；HOLD：有真实改善但轻度 category dependency；STOP：robustness 无改善 / preservation-only / 被 frontier 支配 / 严重跨类别不稳定 / 冗余 / 不稳定 / 需 oracle / 成本不合理。**若全部失败 → 明确输出 "Experiment 9B: no candidate promoted."** |

## 2. 冻结口径

- **Primary robustness** = `mean |ΔNormalScore_z|`（**越低越好**）
- **Primary preservation** = `mean defect d′`（**越高越好**）
- 指标由 `experiment5a_h_analysis.unit_metrics` 从 `per_image.csv` 重算（与 5A-H / 7A-O / 9A 同一实现）
- 判据带：`EPS_DP = 0.10`（5A-H 冻结）、`EPS_RZ = 0.02`（7A-O Tier-A 冻结）
- **Reference**：`B0 (α=0)`（5A-H 冻结 raw）+ **9A 冻结 uniform-α frontier**（`../experiment_9a_screening/summary/uniform_alpha_frontier.csv`）
- 管线：`experiment5a_h_runner.run_config`（bank / coreset 0.1 / kNN 9 / 4 个冻结光照偏移 / scoring）**逐字复用**，仅在 `fit_dual_model` 处注入新模型
- 数据：MVTec AD **bottle / cable × seed 0**（val=20；train 189 / 204；test-good 20 / 58；defect types 3 / 8）

## 3. 候选约束（全部冻结）

所有候选必须且只能：**normal-only**（所有统计量只来自 train/good）· **deployable**（无 oracle、无 test defect mask）· **training-free**（无梯度训练、无 learned fusion）· **不使用 test anomaly labels 做参数选择** · **不针对 bottle/cable target result 临时调参**。

## 4. 候选定义（冻结）

| 候选 | 机制 | 冻结参数 | configs |
|---|---|---|---|
| **M7** Layer-Selective Normalization | 逐层独立 α-IN（复用既有 `FAlphaDualLayerPatchcoreModel`，**无新代码**） | `α_L2 ∈ {0, 0.25, 0.5} × α_L3 ∈ {0, 0.25, 0.5}`，**排除 α_L2 = α_L3** | 6 个（L2-only / L3-only / 非对称） |
| **M8** Normal-Only Channel Gate | 由 **train-only** `s_c` 构造软门：`F'_c = (1−a_c)·F_c + a_c·IN(F)_c` | `a_c = clip(0.25 + β·z_c, 0, 1)`，`β ∈ {0.5, 1.0}` | 2 个 |
| **M9** Soft Channel Weighting | M3 hard projection 的温和替代（不删除方向，只按绝对敏感度重加权） | `w_c = clip((s_c/p90_c)^γ, 0, 1)`，`γ ∈ {1, 2}` | 2 个 |
| **M10** Original + Robust Concat | 固定 concat(original, robust)，比例 γ | γ ∈ {0.25, 0.5, 1.0} | **REUSE：与 7A-O Family B `dual_step(γ)` 定义等价 → 0 GPU** |
| **M11** Layer × Channel Gate | M7（层级强度）× M8（channel 敏感度门） | `(a_bar_L2, a_bar_L3) = (0.5, 0.0)`，`β ∈ {0.5, 1.0}` | 2 个 |

**normal-only illumination sensitivity `s_c`（M8/M9/M11 共用，每类别只估一次）**
`s_c = mean_{train image, 4 shifts} mean_spatial |pooled_feature(T(x))_c − pooled_feature(x)_c|`，在 `feature_pooler` 之后的特征空间、逐层（layer2 C=512 / layer3 C=1024）计算；只使用 **train/good**，落盘 cache 并记录 hash。**禁止**使用 test defect mask / test label / test defect statistics。

**M7 的预注册方向假设**：1H 显示 bottle/cable 的 defect 响应**以 layer2 为主**（`preferred_sensitive_layer=layer2`），Q2 显示 residual 放在 L2 槽位有利（+0.5793）而放在 L3 槽位有害（−0.0489）→ 预测 **α_L2-heavy > α_L3-heavy**。

**M8 的 `a_bar = 0.25` 依据**：来自**历史** 6A/6B 的 knee（α_F=0.25），在读取任何 9B 结果之前确定；`β = 0` 时严格退化为 uniform α=0.25。

## 5. 必需的 sanity（P3，正式前）

S1 `α=0` / `gate=0` 严格退化到原始 PatchCore（bit-exact）· S2 shape 不变 · S3 无 NaN/Inf · S4 gate ∈ [0,1] 且**常数 gate ≡ uniform α** · S5 fit 过程不读 defect GT / test 数据 · S6 M7 的 uniform 设置 ≡ 历史 uniform α · S7 M8/M9/M11 门统计只依赖 train cache（hash 记录）· S8 GPU smoke：`bottle:0 α=0` 与历史 5A-H B0 **逐位一致** · S9 行数 183/402 · S10 无 failed unit 被静默排除。

**任一 candidate 的 baseline equivalence FAIL → 该候选立即停止，不进入正式筛选。**

## 6. 预算（冻结估算）

新增 GPU units = **24**（M7 12 + M8 4 + M9 4 + M11 4；M10 = 0 GPU 复用）+ smoke 5 → 预计 GPU 主实验 15–20 min、总时长 20–35 min。若实测 ETA > 45 min → 只保留核心配置并报告。

---

# 结果（Round-0 完成，2026-10-07）

executed rounds：`smoke`（6 units，含 2 个等价性 smoke）→ `round0`（21 units）→ `diag`（2 units，重复运行判别）。
**GPU：29 unit-runs / 3461 GPU·s（57.7 min 串行；4 workers 墙钟 ≈ 21 min）、failed = 0、峰值显存 2605–2785 MB。**
**Sanity：12/12 PASS**（明细 `summary/sanity_checks.csv`）。

## 7. 最终结论

> **Experiment 9B: no candidate promoted.**
> 第一梯队（ADVANCE）**为空**；M7/M8/M9/M10/M11 五个 family 全部 **HOLD**（均存在真实改善，但 category-dependent 或幅度不足以越过噪声地板）。没有任何候选满足预注册的 ADVANCE 条件（bottle + cable 同时满足 ε 条件且跑出既有 frontier）。

## 8. Candidate Screening Table（M7–M11，bottle/cable seed0）

`beyond B/C` = 该类别相对既有 uniform-α frontier 的 d′ 越界量（正 = 位于 frontier 之上）；
`exceeds NF` = 越界量是否 > 实测噪声地板（NF = 0.054 in d′）。

| Method | Mechanism | Bottle (d′ / \|Δz\|) | Cable (d′ / \|Δz\|) | Robustness (\|Δz\| 2cat) | Preservation (d′ 2cat) | beyond B / C | exceeds NF | Cross-cat stability | Runtime | Memory | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **M7_L2a000_L3a025** | layer-selective α（L2 不归一化，L3 α=0.25） | 8.0216 / 0.3355 | 5.2912 / 0.3154 | 0.3255 | 6.6564 | **+0.047 / +0.019** | **none** | 两类别均非支配 | 98 s/unit | 2605 MB | **HOLD** | 唯一两类别同时非支配；cable 双轴均改善（Δd′ +0.145、Δ\|Δz\| −0.030）；bottle Δd′ −0.243 且越界 +0.047 < NF → 不可判读为真实越界 |
| **M7_L2a000_L3a050** | L2 不归一化，L3 α=0.50 | 7.8422 / 0.2806 | 4.9804 / 0.2930 | 0.2868 | 6.4113 | +0.287 / −0.227 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | bottle 越界、cable 被支配 |
| **M7_L2a025_L3a000** | L2 α=0.25，L3 不归一化 | 7.8876 / 0.3756 | 5.3635 / 0.3594 | 0.3675 | 6.6256 | −0.169 / **+0.091** | cable | unstable | 98 s/unit | 2605 MB | **HOLD** | cable 越界（+0.091 > NF）、bottle 被支配 → **方向与预注册假设相反** |
| **M7_L2a025_L3a050** | L2 0.25 / L3 0.50 | 7.8110 / 0.3128 | 4.8255 / 0.2579 | 0.2853 | 6.3182 | −0.058 / −0.088 | none | 两侧均被支配 | 98 s/unit | 2605 MB | **HOLD** | 唯一「两类别均被支配」的 M7 配置 → 沿 frontier 内部移动 |
| **M7_L2a050_L3a000** | L2 0.50 / L3 0 | 7.5336 / 0.4353 | 5.2740 / 0.2946 | 0.3650 | 6.4038 | −0.644 / +0.055 | cable | unstable | 98 s/unit | 2605 MB | **HOLD** | 同上（L2-heavy） |
| **M7_L2a050_L3a025** | L2 0.50 / L3 0.25 | 7.5159 / 0.3223 | 5.2325 / 0.2882 | 0.3052 | 6.3742 | −0.432 / +0.062 | cable | unstable | 98 s/unit | 2605 MB | **HOLD** | 同上 |
| **M8_gate_ab025_b050** | normal-only channel gate（z-map, a_bar=0.25, β=0.5） | 7.5498 / 0.2487 | 4.6218 / 0.1907 | 0.2197 | 6.0858 | **+0.690** / −0.174 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | bottle 强越界（+0.69 >> NF）；cable 被支配且 Δd′ −0.525 |
| **M8_gate_ab025_b100** | 同上，β=1.0 | 7.4397 / 0.1929 | 4.3830 / 0.2043 | 0.1986 | 5.9113 | **+0.937** / −0.413 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | bottle 越界幅度最大；cable Δd′ −0.763 → 最严重跨类别反向 |
| **M9_sw_gamma1** | soft channel weighting（p90-power, γ=1） | 7.2072 / 0.2251 | 4.8089 / 0.2107 | 0.2179 | 6.0080 | **+0.705** / +0.013 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | bottle 强越界；cable 名义非支配但 +0.013 < NF |
| **M9_sw_gamma2** | 同上，γ=2 | 7.5761 / 0.2596 | 4.7507 / 0.2208 | 0.2402 | 6.1634 | +0.413 / −0.045 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | bottle 越界、cable 被支配 |
| **M11_L2heavy_b050** | layer×channel gate（a_bar_L2=0.5, a_bar_L3=0.0, β=0.5） | 7.5118 / 0.2562 | 4.7732 / 0.2022 | 0.2292 | 6.1425 | +0.442 / −0.022 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | 未显示优于 M8 |
| **M11_L2heavy_b100** | 同上，β=1.0 | 7.3069 / 0.2272 | 4.5688 / 0.1872 | 0.2072 | 5.9378 | +0.804 / −0.227 | bottle | unstable | 98 s/unit | 2605 MB | **HOLD** | 同上 |
| **M10_concat_g100**（REUSE） | fixed concat(original, robust)，γ=1.0（= 7A-O `B1_g100`） | 8.1708 / 0.3379 | 5.1008 / 0.3275 | 0.3327 | 6.6358 | +0.191 / −0.172 | bottle | unstable | **0 GPU（历史复用）** | 5113 MB（7A-O） | **HOLD** | 与 9A M2 / 7A-O 结论一致：bottle 越界、cable 被支配 |
| *M10_concat_g025 / g050*（REUSE） | γ=0.25 / 0.50 | 8.1484 / 0.4018 · 8.0881 / 0.3314 | — | — | — | +0.039 / +0.122 | bottle | n/a | 0 GPU | — | **NOT_ENOUGH_EVIDENCE** | 7A-O 只在 3 类别跑过 g025/g050（无 cable）→ 无法两类别比较 |
| *B0 (α=0)* | baseline | 8.2650 / 0.4782 | 5.1463 / 0.3455 | 0.4119 | 6.7056 | — | — | reference | 0 GPU（5A-H 复用，**逐位一致**） | 2605 MB | BASELINE | `bottle:0` 与 5A-H B0 max\|Δscore\| = **0.0** |
| *B1 (α=0.5)* | baseline | 7.2638 / 0.2846 | 4.9153 / 0.2581 | 0.2714 | 6.0895 | — | — | reference | 0 GPU | — | BASELINE | 历史 best fixed |
| *B2 Uniform (α=0.4009)* | baseline | 7.4638 / 0.2703 | 5.0488 / 0.2641 | 0.2672 | 6.2563 | — | — | reference | 0 GPU（6B 复用） | — | BASELINE | 项目最强 layer-uniform baseline |

## 9. 关键观察（事实 / 解释 / 假设）

**事实**
- F1：**bottle 侧存在真实越界**：M8_b100 +0.937、M11_b100 +0.804、M9_g1 +0.705、M8_b050 +0.690、M7_L2a000_L3a050 +0.287 —— 全部 >> 噪声地板（0.054）。
- F2：**cable 侧无候选越界**；M8/M9/M11 的 cable d′ 大幅下降（Δ −0.34 … −0.76），M8_b100 的 cable 越界为 −0.413（被支配）。
- F3：**M7 出现 layer × category 交互**：L3-only（α_L2=0）配置在 bottle 越界（+0.047 / +0.287），L2-only 配置在 cable 越界（+0.091 / +0.055）→ **bottle 偏好归一化 layer3、cable 偏好归一化 layer2**。
- F4：F3 的方向与**预注册假设相反**（1H 显示 bottle/cable 缺陷响应以 layer2 为主 → 预测 α_L2-heavy 更优）。
- F5：唯一两类别同时非支配的配置是 **M7_L2a000_L3a025**（bottle +0.047 / cable +0.019），但两侧越界量**均 ≤ 噪声地板**（0.054）。
- F6（方法学）：**coreset 轨迹噪声地板 NF(|Δz|) = 0.0202、NF(d′) = 0.0542**，由「embedding 数学完全相同、仅 fit 路径 RNG 状态不同」的对照测得（const-gate 0.25 vs uniform α=0.25，bottle:0）。

**解释**
- E1：F1+F2 表明通道敏感度门在 bottle 上确实能把工作点推到既有 frontier 之外的显著位置，但代价是 cable preservation 崩溃 → 与 8B 记录的「cable 是通道级选择的系统性负迁移类别」一致。
- E2：F6 的机制已定位并验证：本项目冻结的 RNG 约定是「在 `fit_dual_model` 入口 seed 一次、不重播」，而 gate 分支比 α 分支**多构造一个模型**（timm 权重加载消耗全局 torch RNG）→ KCenterGreedy 初始采样点不同（实测 `torch.rand` 0.9509 vs 0.5424、`randint` 78637 vs 3666）→ coreset 轨迹不同 → 在 *embedding 完全一致* 的前提下 score 出现 O(1) 差异。**这是采样轨迹差异，不是方法差异。**
- E3：因此 9B 的判读必须在指标级进行，且任何 < NF 的「改善」不构成证据（F5）。

**假设（未验证）**
- H1：M7 的 bottle/cable 层级偏好反转可能与「缺陷的空间/纹理结构与归一化的空间统计假设是否匹配」有关（cable 细长结构、bottle 大面积均匀区域）。
- H2：M8/M9/M11 在 cable 上的 preservation 崩溃可能来自 s_c 在 cable 上由背景主导（训练图像背景占比高）→ 归一化权重被分配给与缺陷无关的通道。

## 10. Negative results（全部保留）

- **无候选满足 ADVANCE**。
- **M8 / M9 / M11 在 cable 上系统性失效**：cable Δd′ −0.34 … −0.76（β/γ 越大越差）。
- **M7 的预注册方向假设被反驳**（F4）：bottle 偏好 L3 归一化、cable 偏好 L2 归一化。
- **M10（fixed concat original+robust）**：与 9A M2 / 7A-O `B1_g100` 完全一致（bottle 越界、cable 被支配）→ 第三次独立确认只能沿既有 frontier 移动。
- **M10 γ=0.25/0.50 无 cable 覆盖** → NOT_ENOUGH_EVIDENCE，不做推测。

## 11. 出口条件判断

- ADVANCE：**未满足**。HOLD：**M7/M8/M9/M10/M11**（全部单侧）。STOP：本轮无 family 被判 STOP（每个 family 至少有一个配置在 bottle 侧产生超过噪声地板的真实越界）。
- 反例检查：不存在「robustness 无改善」的候选（全部候选两类别 robustness 均改善），也不存在 preservation-only 候选。

## 12. 异常情况与 bug（必须记录）

1. **BUG-1（P3 CPU sanity 捕获，未产生任何 GPU 结果）**：`s_c` 估计误用 `d.sum()`（标量）累加进 per-channel 向量 → 全部 channel 同值（逐 channel std = 0）。若未捕获，M8/M9/M11 会退化为「常数门」（≈ uniform α），9B 全部结论将无效。已修复为向量累加（cache 版本 `9b-v1 → 9b-v2`），退化 cache 隔离在 `cache/_deprecated_bug/`（保留未删除），并新增 **S11「s_c 非退化」硬门槛**。
2. **BUG-2（GPU smoke 捕获）**：gate 配置缺 `alpha_l2/alpha_l3` 键，而 `run_config` 直接读取 → `KeyError`。修复：沿用 7A-O 约定写 `alpha_l2 = alpha_l3 = -1.0` 哨兵（真实门统计记在 `module_meta.json`）；4 个失败 smoke 单元修复后重跑。
3. **BUG-3（显示层）**：`experiment_progress.estimate` 的 throughput 基在「多 unit 同时完成」时算出荒谬速率（曾显示 ETA 00:16）→ 加入 `MIN_RATE_SPAN = 20s` 门槛，过短跨度自动退回 duration/worker 基。不影响任何实验数值。
4. **对 Experiment 9A 归因的更正**：9A 曾把 M4「组合形式 vs 直接 α_eff」的 score 级差异（max 1.773）归因于「fit 路径放大 1e-6 级表示扰动」。9B 的受控实验证明真正机制是 **RNG 状态分歧**（多一次模型构造 → KCenterGreedy 初始点不同）。**9A 结论不受影响**（M4 代数冗余由 embedding 级等价 1.91e-06 确立；score 级差异本来就未被当作证据），**无需重跑任何 9A 单元**，仅归因句子更正（见 `../experiment_9a_screening/README.md` §19）。

## 13. 对论文证据链的贡献

- 完成「第二批候选海选」格：在 deployable / normal-only / training-free 约束下 **M7–M11 无一晋级**。
- 新增可复用的**方法学结论**：PatchCore + KCenterGreedy 管线的 **coreset 轨迹噪声地板 = 0.020（|Δz|）/ 0.054（d′）**，来源是全局 RNG 状态而非 representation —— 为所有「frontier 越界」类结论提供明确判读阈值（此前实验隐含假设该噪声可忽略）。
- 反驳一条预注册方向假设（F4），并首次报告 **layer × category 交互**。
- 为「cable 是通道级选择/门控的系统性弱侧」提供第三次独立证据。

## 14. 下一步（仅建议，不自行启动）

1. **不启动 5 类别 full validation**（本轮无 ADVANCE）。
2. 若继续，优先解决**判读阈值**：(a) 在 `engine.fit` 前显式重播 seed 使各分支 coreset 轨迹可比（代价：与历史 frozen raw 的逐位可比性丢失，需同步重算 baseline/frontier）；或 (b) 引入 matched-RNG 对照。二者均需**新预注册协议**与用户批准。
3. M7 的 layer × category 交互（F3/F4）可作为 H1/H2 的检验对象，但必须先有 (2) 的判读阈值。
4. M8/M9/M11 若继续，必须先在 cable 上解决 preservation 崩溃（需要新的机制性理由，不能靠调参）。

## 15. 目录与复现

```
results/experiment_9b_screening/
  README.md  config.json  progress.json  progress.log
  raw/{smoke,round0,diag}/<cat>/seed_0/config_<name>/{per_image.csv,info.json,module_meta.json}
  summary/{raw_results.csv, screening_summary.csv, screening_summary.json, ranking.csv,
           family_summary.csv, uniform_alpha_frontier.csv, sanity_checks.csv,
           runtime_summary.json, smoke_equivalence.json}
  figures/robustness_preservation_plane.png
  cache/ (s_c npz/json + _deprecated_bug/)   logs/   progress/
```
复现分析：`python -u scripts/experiment9b_analysis.py`（CPU-only，0 GPU）。
复现单元：`python -u scripts/experiment9b_runner.py --round round0 --worker-tag w0 --units "..."`。
