# 受控比较与消融核查表（2026-10-23）

**性质**：只读核查。未重训、未改冻结产物、未调参。
**核查对象**：F0（frozen DINOv2-B + 1-NN-max）与 R1-ILA（同主干 + 线性残差适配器）。
**判定**：**未发现 P0/P1**（无数据泄漏、无评估错误、无标签错误、无受控比较不公平、关键结果可追溯）。P2 建议见文末。

---

## 1. 统一评价口径（冻结协议，先于任何结果）

| 项 | 取值 | 依据 |
|---|---|---|
| 数据集 | M²AD，view120，10 类 | `results/e18/protocol.md` §1 |
| 测试集 | view120 全部 700 图（Holder 690） | 同上；`results/e22/cells/*.json` |
| 标签 | 元数据 `image_anomaly`（非目录名） | `scripts/e13a_screen.py:412-419`（`_rows`） |
| 过滤 | `detectable != False`（official） | `scripts/e11_p2_runner.py:139-148` |
| 正常参考 | 300 张，`E8.bank_for(pool, seed)`，`random.Random(20261009+seed)` | `scripts/e8_common.py:256-261` |
| 特征 | DINOv2-B 最后层 patch token，1024×768 | `scripts/e13a_screen.py:190-200` |
| 归一化 | 逐 patch L2 | `scripts/e16_qualify.py:271-274`（F0）、`:288-289`（R1） |
| 评分 | 1-NN-max：score = 1 − max cosine（越大越异常） | `scripts/e11_p2_runner.py:116-135` |
| 指标 | sklearn `roc_auc_score` / `average_precision_score` | `scripts/e11_p2_runner.py:139-148` |

---

## 2. 七项同一性检查（F0 vs R1）

| # | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 1 | 主干权重相同 | **是** | 单一 `dinov2_vitb14(pretrained=True)` 实例；F0 与 R1 共用同一次前向得到的 `bank_feat/test_feat`（`scripts/e16_qualify.py:249,265-267`） |
| 2 | 输入预处理相同 | **是** | `prep_image` = Resize(448, BICUBIC, antialias) + ToTensor + ImageNet 归一化（`scripts/e13a_screen.py:169-186`）；bank/test tensor 各只构造一次（`:257-258`），F0/R1 复用 |
| 3 | 正常参考图像相同 | **是** | 同一 `bank` 列表；F0 用 frozen 特征，R1 用**同一批图像**经 adapter 后的特征。30/30 cell `bank_hash` 长度 64、`bank_size=300`（`fairness_checks.json`） |
| 4 | 测试样本相同 | **是** | 同一 `test` 列表、同一 `scores` 长度；30/30 cell 长度自洽；10/10 类跨 3 seed 的**图像身份集合一致**（身份键 `(specimen, illumination, kind)`，逐 cell 唯一） |
| 5 | 标签与筛选相同 | **是** | 标签来自 `raw_test_map`（元数据 `image_anomaly`）；`detectable` 掩码在 30/30 cell 中跨 seed **完全一致**（`fairness_checks.json`） |
| 6 | 特征归一化相同 | **是** | F0：`b0/=‖b0‖`、`t0f/=‖t0f‖`（`:271-274`）；R1：`l2norm(torch...)`（`:288-289`）；均为逐 patch L2 |
| 7 | 评分定义相同 | **是** | 两条路径调用同一个 `nn_image_scores`（`:275` 与 `:290`） |

**允许且已声明的差异**：R1 的记忆库特征是 `adapter(bank_feat)`（参考**图像**不变，特征随适配变化）；adapter 训练随机性按 seed 固定（`torch.manual_seed(seed)`，`:283`）。

**同批检查的其它事实**（`fairness_checks.json`）：30/30 cell 分数无 NaN/Inf；`trainable_params=589824`（768×768）在 30/30 cell 一致；`steps_used` 因 early stopping 在 44–124 之间变化（seed 相关，非超参变化）。

---

## 3. 三组结果的解释边界

| 组 | 数值（seed0，AUROC） | 允许的结论 | 不允许的结论 |
|---|---|---|---|
| **F0**（不适配） | Bird 0.74725 / Car 0.78552 / Cube 0.85410 | 冻结主干的基线水平 | — |
| **C0 = 完整 R1**（β=0.1） | 0.75786 / 0.80511 / 0.87010（Δ **+0.0106 / +0.0196 / +0.0160**） | **联合适配**（consistency + preserve）相对 F0 有效 | ❌ 不能把增益单独归因于 consistency 项 |
| **C1 = β=0**（只留 consistency） | 0.73309 / 0.78587 / 0.90904（vs C0：**−0.0248 / −0.0192 / +0.0389**） | 去掉 preserve 项后 **Cube 表征 collapse**（`collapsed=True`），Bird/Car 同时劣于 C0 | ❌ 不能把 Cube 的 0.909 当作"更好的检测器"（collapse 下的名义 AUROC）；❌ 不能据此推断 consistency 项本身有害 |
| **R2 = preserve-only**（MLP，只留 preserve） | 0.74724 / 0.78547 / 0.85416（**Δ vs F0 = −9e-6 / −5e-5 / +6e-5**） | preserve-only 等价于恒等映射，**不需要额外训练** | ❌ 不能把 R2 当作"另一个 baseline 的增益/损失"证据 |

**关于 β=0 的措辞提醒**：`results/e17/README.md` 曾写"BETA=0 三类最差检测器"，按 **AUROC** 对 Cube **不成立**（0.9090 名义最高）。正确表述是：**β=0 使三类表征高度退化（pairwise cos 0.976–0.988），Cube 越过 collapse 阈值（>0.98）**；主表不使用 collapse 组。

判据：`scripts/e13a_screen.py:250-256` —— `std` 降幅 >50%、或 `pairwise_cos > 0.98`、或范数爆炸，任一触发 `collapsed=True`。Bird 0.979 / Car 0.976 未触发但已贴近阈值，不能写成"只有 Cube 有问题"。

---

## 4. preserve-only 在零初始化下恒等：代码 + 数学

- 结构：`z' = z + W z`，`W ∈ R^{768×768}`，**`nn.Parameter(torch.zeros(...))`**（`scripts/e13a_screen.py:126-132`，`adapter_forward` `:154-155`）。
- 损失：`l_preserve = 1 − cos(z', z_frozen)`（`:223-232`）。
- 在 `W=0` 时 `z' = z = z_frozen`，`cos=1` → `l_preserve = 0`，且该点是 **cos 相似度的最大值**，`∂l_preserve/∂W = 0`；AdamW 无梯度 → 参数不更新 → 恒等映射被精确保留。
- 因此 **preserve-only + 零初始化 = 恒等映射，不需要训练**（这是分析结论，不是"少跑了一个 baseline"）。
- R2（MLPAdapter，`W1~N(0,0.02)`、`W2=0`）在初始化时输出为 0，preserve 损失与其梯度同样在 `z'=z` 处为零；实测 R2 与 F0 差异 ≤ 6e-5（上表），与该分析一致。

---

## 5. 外部方法的配置、来源与覆盖（必须随表声明）

| 方法 | 来源 | Scoring | 覆盖 | 与 R1 结构相同？ |
|---|---|---|---|---|
| PatchCore-emb + unified | 本项目 E7 管线 | 1-NN-max（统一） | 3/3 类，seed0 | 否（ResNet 特征） |
| AnomalyDINO | 官方实现（WACV 2025） | 官方 `mean_top1p`（DINOv2-448，L2 kNN k=1） | 3/3，seed0 | 否（training-free，无 adapter） |
| SoftPatch | 官方实现 | 官方 soft-weighted kNN | 3/3，seed0 | 否（WideResNet50） |
| EfficientAD | 官方（PDN-small） | 官方 teacher-student discrepancy | **仅 Bird（1/3）** | 否 |
| PatchCore official | 官方管线（原图） | 官方 | **仅 Bird（1/3）** | 否 |
| F0 / Ours | 本项目 | 1-NN-max | 3/3，**3 seeds** | — |

依据：`results/e18/main_table.md`、`results/e18/protocol.md`；边界：**主表以 seed0 为主比较（唯一所有方法共享的 seed）**，3-seed 仅覆盖 F0/Ours，不构成其他方法 seed 稳定的证据。

---

## 6. P0 / P1 / P2

- **P0（必须修）**：无。
- **P1（必须修）**：无。
- **P2（建议，不在本轮执行）**：
  1. 主表中"3/3 完备"与"1/3"并列时，建议在表注显式写"不同方法的 seed 覆盖不同"，避免读者误读为同样本量比较。
  2. `results/e17/README.md` 的"BETA=0 三类最差"建议按指标限定（AUROC 对 Cube 不成立）——该问题已在 E17 文档修订中记录，此处仅提示正文引用时不要复述旧措辞。
  3. β=0 组的 Cube 名义 AUROC 若要在论文中出现，必须同时给 `collapsed=True` 与 pairwise cos，否则会被误读为性能。
