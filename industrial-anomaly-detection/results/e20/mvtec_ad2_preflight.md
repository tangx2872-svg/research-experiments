# E20-B · MVTec AD 2 独立数据集扩展预检报告

**日期**：2026-10-15
**类型**：只读预检（**未下载、未申请、未接受许可、未训练、未修改任何冻结配置**）
**首选数据集**：MVTec AD 2（MSC-AD 本轮未评估，未切换）
**建议决策**：**HOLD →（满足 §7 两个前置条件后）GO**。判据与逐条对照见 §6。

---

## 0. 执行摘要（五个问题的直接答案）

| 问题 | 答案 |
|---|---|
| 1. AD2 是否具备可行的外部验证条件？ | **结构上具备**：公开测试集（TESTpub）带 good/bad 图像标签与像素 GT，许可 CC BY-NC-SA 4.0，可本地计算 image-level AUROC/AP（非官方指标）。**统计上受限**：单场景 TESTpub 规模小（good 20–35），本轮模拟显示单场景 ΔAUROC 噪声半宽 ≈ 0.04–0.07。 |
| 2. 首轮三候选场景哪些适合？ | **Vial 最合适**（train 291 / TESTpub 35+105，背光+3 聚光灯，光照变化强）；**Sheet Metal 次之**（train 137 偏少，暗场双环光角度变化，4224×1056 极端长宽比）；**Fruit Jelly 最弱**（TESTpub 仅 20 good，半透明高变异）。均可运行，但结论口径需按 §3.3 调整。 |
| 3. 数据获取与存储问题 | 下载需人工在浏览器完成（表单/许可接受，本轮未做）；三场景压缩包合计 ≈ 3.5 GB，解压后估计 2–3.5 GB，加压缩峰值 ≈ 7 GB ≤ 16 GB 可用但余量有限，建议逐个场景分步验证。 |
| 4. Mini Experiment 如何执行 | 冻结 F0 与 R1-ILA（BETA=0.1），AD2 train/good 作 bank 与 adapter 训练集、官方 validation/good 作 val、TESTpub 作测试；image-level AUROC/AP；3 seeds；**必须修改清单 D1–D4（§5.3）待批准**；预算 ≈ 55 min GPU（3 seeds × 3 场景）+ 显存 <4 GB。 |
| 5. 推荐决策 | **HOLD**（数据未获取/许可未接受 + 需先批准统计口径与预处理决策），路径明确可解；无任何 STOP 条件成立。 |

> 与 2026-09-30 历史调查的关系（重要，避免误读）：历史结论 "MODIFY" 针对的是
> **defect-type-level / illumination-separated 的本地敏感性分析**（该问题确实不可本地验证）。
> E20-B 的目标是 **整体 image-level AUROC/AP 的 F0-vs-Ours 外部对照**，不要求光照或缺陷类型标签，
> 因此历史否决不适用于本轮方案。历史记录的公开数据结构限制（无光照标注、bad 混类）在本轮作为**限制**继承。

---

## 1. 环境与磁盘检查（任务 1）

| 项 | 结果 |
|---|---|
| 项目根 | `/root/autodl-tmp/research-experiments/industrial-anomaly-detection` |
| 仓库根 / 分支 / HEAD | `research-experiments` / `research/industrial-anomaly` / `34354aa`（E19，已推送） |
| 工作区 | 干净，仅 6 项已知历史遗留文件（`monitor/exp_monitor.py`、`results/e8/status.json`、`monitor/events.jsonl`、`scripts/_lockprobe.txt`、`scripts/_probe2.py`、`scripts/e11_p4_runner.py`），本轮未触碰 |
| 磁盘 `/root/autodl-tmp` | 50 G 总量，已用 35 G，**可用 16 G** |
| 已有数据 | `data/m2ad/`（27 G，主数据集）、`data/mvtec_ad/`（1.6 G，旧版 bottle）、`data/csem_misd/`；**无任何 MVTec AD 2 数据、缓存或下载残留**（全盘 `*ad2*`/`*ad_2*` 搜索无命中） |
| 可复用数据代码 | ① anomalib 2.6.2 内置 `MVTecAD2` datamodule（已验证可导入）；② 官方 code utils 本地副本：`experiments/2026-09-30_illumination_sensitivity_exploration/mvtec_ad2_illumination_sensitivity/reference/`（`mvtec_ad_2_public_offline.py`、`utils.py`、`check_and_prepare_data_for_upload.py`、`measure_runtime_and_memory.py`）；③ 本项目 DINOv2-B 特征/Adapter/评分管线 `scripts/e13a_screen.py` + `scripts/e16_qualify.py`（§5.2） |
| 历史 AD2 材料 | `experiments/2026-09-30_illumination_sensitivity_exploration/mvtec_ad2_illumination_sensitivity/README.md`（含官方目录结构、样本量表、光照事实的已核验记录） |

未执行任何删除、清理或环境变更。

---

## 2. 官方数据与许可核实（任务 2）

### 2.1 数据集概况与结构（来源：官方论文 arXiv:2503.21622 / IJCV 134(4):2026；官方 code utils；历史已核验记录）

- **8 个场景**：Can, Fabric, Fruit Jelly, Rice, Sheet Metal, Vial, Wall Plugs, Walnuts；共 **8,004 张高分辨率图像**（2.6–5 MP，长宽比不一）；含 RGB 与单通道灰度（**Sheet Metal、Vial、Wall Plugs 为灰度**）。
- 场景刻意覆盖：透明物体（Vial、Fruit Jelly）、**暗场照明（Sheet Metal）**、**背光照明（Vial、Fruit Jelly）**、反射、散料重叠、图像边界缺陷（**明确反对 center cropping**）。
- 目录结构（官方 code utils glob `{split}/[gb][oa][od]*/[0-9][0-9][0-9]*.png` 权威确认）：

```text
mvtec_ad_2/{object}/
├── train/good/*.png                          # 正常，仅常规光照
├── validation/good/*.png                     # 正常，仅常规光照（官方指定用于超参/阈值）
├── test_public/{good,bad}/*.png              # 全部光照条件，有 GT（bad 另有 mask）
├── test_public/ground_truth/bad/*_mask.png
├── test_private/*.png                        # 常规光照，GT 私有（仅评测服务器）
└── test_private_mixed/*.png                  # seen+unseen 光照混合，GT 私有
```

- 测试集三分（论文 Table IV 注）：**TESTpub = 覆盖所有光照条件**的少量正常+异常图（含训练未见光照）；TESTpriv = 与训练同光照；TESTpriv,mix = 混合光照。**私有部分只能通过 https://benchmark.mvtec.com/ 评估**。

### 2.2 下载与申请流程

- 官方入口：https://www.mvtec.com/company/research/datasets/mvtec-ad-2（含 downloads 页）。
- 已确认事实：官方提供**整包（≈30.4 GB）与分类别下载**；历史记录三候选场景为 vial 0.77 GB / sheet_metal 1.53 GB / fruit_jelly 1.2 GB（压缩包）；官方主页文字要求填写表单（"FILL IN THE FORM"）。
- **待人工确认事项**：下载链接是否需要表单/审批、是否需要机构邮箱、实际带宽与文件校验方式。
  本轮未访问下载资源、未提交表单、未接受任何条款。

### 2.3 许可证及使用限制

- **CC BY-NC-SA 4.0**（署名—非商业—相同方式共享）。**非商业**：允许学术研究使用；衍生物须同许可；不得用于商业产品。
- 评测服务器提交需遵守官方提交格式（anomaly maps 为 `.tiff` + 阈值化 `.png`）。本轮不涉及提交。

### 2.4 公开 / 私有标签

| 内容 | TESTpub（公开） | TESTpriv / TESTpriv,mix（私有 GT） |
|---|---|---|
| 图像 | ✅ 全部可下载 | ✅ 图像可下载 |
| 图像级 good/bad 标签 | ✅（目录结构） | ❌（GT 在服务器） |
| 像素级 GT mask | ✅（仅 bad） | ❌ |
| 光照条件标注 | ❌ 图片不标注光照/曝光 | ❌（划分本身按光照，但单图无标注） |
| 缺陷类型标注 | ❌（所有异常混在 `bad/`） | ❌ |

### 2.5 能否用公开测试部分计算 image-level AUROC 和 AP？

**可以（本地，非官方指标）**：TESTpub 的 `good/` 与 `bad/` 目录即图像级二分类标签，任何连续图像分数都能计算 AUROC/AP，无需阈值、无需评测服务器。

**必须明确的口径边界**：
1. **官方指标是像素级 AU-PRO₀.₀₅ 与阈值化 F1（论文 Table VII–XII + 排行榜）；官方从不报告 image-level AUROC/AP**。我们计算的 image-level AUROC/AP 属于"数据集标签支持但非官方口径"的本地评估，与 E18 处理 M²AD 的"统一口径但标注非原生"原则一致。
2. TESTpub 混入**训练未见的额外光源/曝光条件**（§2.1），因此该 AUROC 度量的是"含分布偏移的整体检测能力"，**不能**分解为常规光照 vs 变化光照（无光照标注）。
3. `bad/` 含多类缺陷混合且**单图可含多个缺陷**，图像级指标不受此影响（图像级只看有/无），但不可做缺陷类型分解。

### 2.6 官方预处理 / 划分 / 评估协议要求

- 训练：**仅无异常图像**；验证：仅无异常图像，用于超参与阈值；官方明确反对在测试集上调参。
- 官方 baseline 参考设置：输入统一 resize 到 **256×256**，**未做场景特定调参**；PatchCore 官方评估**关闭 center cropping**（因缺陷出现在边界）。
- 数据集本身不含特殊读取约定（纯图像目录）。
- 官方提供：PyTorch dataloader、runtime/memory 测量代码、提交前数据检查工具（本地已有副本）。

### 2.7 事实分级

| 等级 | 内容 |
|---|---|
| **已从官方资料确认** | 8 场景与规模；TESTpub/priv/mix 定义；目录结构；三候选场景的分辨率/通道/光照设置（Table III–VI）；许可；官方指标与协议；TESTpub 覆盖全部光照 |
| **仍待申请/下载后确认** | 下载表单的具体流程与时效；压缩包完整性；灰度 PNG 实际通道数；解压后精确磁盘占用；TESTpub 各光照条件的实际分布 |
| **本地尚未验证** | 我们在 AD2 上的管线行为（几何、灰度→RGB、bank 规模）——全部需在下载后以"预检跑通"验证（§5.5） |

---

## 3. 首轮候选场景可行性（任务 3）

### 3.1 事实表（来源：论文 Table III / IV / VI，历史已核验记录）

| 项 | Sheet Metal | Vial | Fruit Jelly |
|---|---|---|---|
| 采集设置 | **定向暗场**前光（低入射角环光 R1） | **漫射亮场背光**（面光） | 漫射亮场**背光+前光** |
| 分辨率 W×H | 4224×1056（4:1） | 1400×1900（0.74:1） | 2100×1520（1.38:1） |
| 通道 | **灰度（单通道）** | **灰度（单通道）** | RGB |
| train/good | 137 | 291 | 263 |
| validation/good | 19 | 41 | 37 |
| TESTpub（good/bad） | 114（**24**/90） | 140（**35**/105） | 80（**20**/60） |
| 测试光照变化（Table VI） | R1（常规）→ **R2（第二环光，更陡入射角）**，两环光各 3 档曝光 | F（背光）→ **F+S1（顶部不均匀）/F+S2/F+S3（镜面高光）** | F+S1（背光+前光）→ **F+S1+S2（额外顶部 LED）** |
| 主要难点 | 随机镜面反射、大长宽比、单图多缺陷 | 透明物体与液体、低对比小缺陷、旋转/气泡/QR 高变异 | 半透明、低对比缺陷、正常数据高变异 |

### 3.2 逐问回答（任务 3 要求的 7 项）

| 问题 | Sheet Metal | Vial | Fruit Jelly |
|---|---|---|---|
| 正常训练图像可用？ | ✅ 137 张（仅常规光照） | ✅ 291 张 | ✅ 263 张 |
| 正常验证图像可用？ | ✅ 19 张（官方专用 val） | ✅ 41 张 | ✅ 37 张 |
| 公开测试图像与标签可用？ | ✅ 114 张 + bad 的 mask | ✅ 140 张 + mask | ✅ 80 张 + mask |
| 测试图像确实含相关照明变化？ | ✅ 官方确认（暗场角度变化 R1→R2 + 曝光），三档曝光 | ✅ 官方确认（背光基础上 3 个额外聚光灯 + 曝光） | ✅ 官方确认（额外顶部 LED + 曝光）；变化幅度中等（仅 1 个额外光源） |
| 正常/异常测试样本量足够？ | ⚠️ 24 good 偏少 | ⚠️ 35 good 偏少 | ❌ 20 good 明显不足 |
| 尺寸/通道/预处理适配 | ⚠️ 灰度→RGB；4:1 极端长宽比 | ⚠️ 灰度→RGB；0.74:1 | ⚠️ RGB 直接可用；1.38:1 |
| 能否不用异常标签训练？ | ✅ F0/Ours 均为 normal-only | ✅ | ✅ |

> 注意：**不得**仅凭场景名称推断照明条件——上表全部来自官方 Table III/VI 原文，而非命名直觉。

### 3.3 统计分辨力量化（E20-B 本轮新增分析，CPU-only）

**方法**：以 E16 冻结的 M²AD per-image 分数（F0 与 R1-ILA，view120 seed0，`results/e16/e16_partial.json`）为模板，按 AD2 各场景 TESTpub 规模做 4,000 次配对 bootstrap 重采样，估计 ΔAUROC = AUROC(Ours)−AUROC(F0) 的抽样噪声。
**脚本/数据**：`scripts/e20b_auroc_power_sim.py` → `results/e20/auroc_power_sim.json`。

| 规模（good/bad） | 单方法 AUROC 噪声半宽(95%) | ΔAUROC 噪声半宽(95%) | P(ΔAUROC>0)，若真实效应 = M²AD |
|---|---|---|---|
| Sheet Metal 24/90 | 0.076–0.101 | 0.046–0.058 | 0.66–0.81 |
| Vial 35/105 | 0.065–0.087 | 0.039–0.050 | 0.68–0.86 |
| Fruit Jelly 20/60 | 0.089–0.115 | 0.051–0.068 | 0.63–0.78 |
| **三场景合并 79/255** | 0.042–0.056 | 0.025–0.033 | 0.77–0.95 |
| M²AD 对照（260–320/380–440） | 0.028–0.037 | 0.017–0.019 | 0.87–1.00 |

（区间来自三个 M²AD 模板 category 的跨度；单位 AUROC。）

**解释（与证据分离）**：AD2 TESTpub 单场景的 image-level ΔAUROC 噪声半宽约 **0.04–0.07**，
远大于 M²AD 上观测到的效应（+0.010～+0.024）。**若逐场景单独评估，即使真实效应与 M²AD 相当，
也只有 ~63–86% 概率得到正确符号**。三场景合并后噪声半宽 ≈0.03，仍大于 M²AD 规模（≈0.018）。

**假设（未验证）**：AD2 上 F0/Ours 的实际效应量与分数相关性未知，上述功效数字依赖"M²AD 分数分布可近似模板"这一假设。

---

## 4. 数据获取与存储风险

| 风险 | 等级 | 说明与缓解 |
|---|---|---|
| 下载需人工介入（表单/许可） | 中 | 本轮不申请；需负责人在浏览器完成（CC BY-NC-SA 接受）。缓解：下载页疑似提供分类别直链，人工确认后可先取 1 个场景 |
| 磁盘余量 | 中 | 可用 16 G。三场景压缩 ≈3.5 G + 解压估 2–3.5 G，峰值 ≈7 G，其余留作系统。**全量 30.4 G 不可行（不清理现有数据的前提下）**。缓解：逐场景下载→校验→解压→（经批准后）删除自身 zip |
| 高分辨率解码内存 | 低 | Sheet Metal 4224×1056 单图解码数 MB 级；CPU 侧流式处理即可，GPU 输入固定 448 |
| 许可合规 | 低 | 非商业学术使用合规；衍生物同许可（当前仅本地评估，不发布衍生物） |
| 网络中断/大文件校验 | 低-中 | 分类别下载可重试；保存 md5/sha 校验（官方未公布校验和则本地记录） |
| 私有测试集不可本地评估 | — | 设计如此；本轮只用 TESTpub，不需要评测服务器 |

---

## 5. 固定方法 Mini Experiment 计划（任务 4；**仅方案，未执行**）

### 5.1 方法冻结声明

- **F0**：冻结 DINOv2-B final-layer patch tokens + L2 + 1-NN-max（top-1% mean）。零训练。
- **Ours**：R1-ILA，**BETA=0.1**（`results/e17/final_method.json`，唯一权威来源），线性残差 `z' = z + Wz`（768×768 zero-init），normal-only 训练，`L = L_cons + 0.1·L_preserve`，AdamW lr=1e-3，≤300 step。
- **不重新调参、不改结构**；超参解析一律读冻结 JSON（禁止手工复制）。

### 5.2 现有代码复用性核查（已完成）

| 组件 | 位置 | 复用判定 |
|---|---|---|
| 预处理 `prep_image`（短边 resize 448 + ImageNet norm + 裁到 14 倍数） | `scripts/e13a_screen.py` | ✅ 逻辑可复用（需新增几何决策，见 D1） |
| DINOv2-B 前向 `forward_frozen_batch` | 同上 | ✅ 可复用（**硬假设 448×448 → `N_PATCH=1024`**） |
| R1 Adapter + 训练循环 `LinearAdapter`/`train_adapter` | 同上 | ✅ 可复用（BETA 参数化已存在） |
| 图像级评分（patch 1-NN → `mean_top1p_local`） | `scripts/e16_qualify.py` | ✅ 可复用 |
| 指标（AUROC/AP/d′/bootstrap CI） | `scripts/e16_qualify.py` | ✅ 可复用 |
| M²AD 专用：`load_records` / 特征缓存路由 / `E8.bank_for` | `scripts/e4d1_m2ad_loader.py` 等 | ❌ 不可直接复用（M²AD 文件名/缓存结构），需为 AD2 写 ~150 行加载层（结构简单：目录即标签） |
| AD2 数据加载 | anomalib `MVTecAD2` + 官方 `mvtec_ad_2_public_offline.py`（本地副本） | ✅ 可复用/参考（目录 glob 已权威确认） |

### 5.3 必须修改的差异清单（**待研究负责人批准；未批准前不得动手**）

| 编号 | 差异 | 方案（建议） | 备选 |
|---|---|---|---|
| **D1** | 现有管线硬假设方形输入（`N_PATCH=1024` = 448×448；M²AD 为 1024×1024 方形）。AD2 三场景分辨率均非方形（4:1 / 0.74:1 / 1.38:1） | **A（推荐）**：直接 resize 到 448×448（接受各向异性形变）。理由：与 M²AD 管线/方法定义零差异、F0/Ours 完全公平、不引入新工程变量；限制如实记录 | B：短边 448 + center crop（丢失边界缺陷，与官方"缺陷在边界"设计冲突）；C：letterbox pad（引入 padding 区域与官方点名的 padding artifact 风险） |
| **D2** | Sheet Metal / Vial 为灰度单通道 | `.convert("RGB")` 复制三通道后走同一 `prep_image`；对 F0/Ours 完全一致 | 改用单通道 DINOv2 输入（需改 backbone 接口，拒绝） |
| **D3** | M²AD 协议 bank=300（从大池中随机采样）；AD2 train 仅 137–291 | **bank = 全部 train/good**（自然规模，≤300；Sheet Metal 137 / Vial 291 / Fruit Jelly 263）；adapter 训练集 = train/good，**val = 官方 validation/good**（不再从 train 划 60） | 统一下采样到 137（对齐 Sheet Metal，但浪费数据且与 M²AD 协议含义不同） |
| **D4** | 多 seed 的含义 | 3 seeds；bank 全量不重复采样 ⇒ seed 仅影响 adapter 训练（aug 采样 + batch 顺序）。**保持与 E16 相同的 3 个 seed 值** | 单 seed（省 2/3 训练时间，但失去方差信息；**不推荐**） |
| **D5** | 指标口径 | 主指标 = image-level AUROC/AP（本地，TESTpub good/bad 标签）；**预注册 pooled-3-scene 为主分析，逐场景为探索性**（依据 §3.3）；每个数字附 bootstrap CI | 逐场景为主（功效不足，不推荐） |
| **D6** | 光照分组分析 | **不可做**（TESTpub 无光照标注）——不复制 E18 式 illumination 分组；改为报告"含未见光照的整体 AUROC/AP" | 提交评测服务器（超出 Mini Experiment 范围，需另行批准） |

### 5.4 协议设计（编号步骤；批准后执行）

1. **数据准备**（无 GPU）：逐个场景下载→校验→解压到 `data/mvtec_ad_2/{object}/`；用官方 loader 或 anomalib `MVTecAD2` 打印每 split 计数，与论文 Table IV 逐项核对（114/137/19 等），不一致即 STOP。
2. **预处理一致性冒烟**（无训练）：随机抽 5 张/场景渲染 448×448 张量快照存档，确认 D1/D2 行为；记录每张图的原始尺寸/通道数（落 CSV）。
3. **F0 评估**：train/good 全量 → bank（L2 patch tokens）→ TESTpub 全部图像打分 → image-level AUROC/AP（逐场景 + pooled）。
4. **Ours 评估**：同 bank；adapter 在 train/good 上训练（aug = 冻结的 photometric 6 变体；val = validation/good 监控 loss）；test 特征经 adapter → 同一 1-NN-max → 同一指标。
5. **各阶段共享同一特征缓存**（同 seed 下 F0 与 Ours 的 test 特征来自同一批前向）；**F0 与 Ours 的输入张量逐比特一致**（同 split、同预处理代码路径）。
6. **统计**：per-image 配对 bootstrap（与 E18 同方法）给出 ΔAUROC/ΔAP 的 95% CI；报告 3 seed 的 mean±std。
7. **防伪提升检查清单**（预注册）：① F0/Ours 输入张量 hash 一致；② 两方法使用完全相同的 test 图像列表；③ bank 图像列表一致；④ 任何预处理差异（几何/通道）必须同时对两方法生效；⑤ 不因结果方向调整口径（预注册 pooled 主分析）。
8. **产出**：`results/e20_1_ad2/`（新目录；CSV/JSON/图/README，遵守项目 README 规范）。**不写回任何历史目录。**

### 5.5 资源预算（估算，需实测校正）

| 项 | 估计 |
|---|---|
| 图像总量（三场景 train+val+TESTpub） | 270 + 472 + 380 = **1,122 张** |
| 特征提取（frozen，1 批含 F0 与 Ours 共享） | ≈ 1.5–3 min（大图解码主导） |
| aug 特征（训练用 bank×6 变体） | ≈ 5–8 min |
| adapter 训练 + 评分（每场景每 seed） | ≈ 2–3 min（对齐 E17 单 config 134 s，AD2 测试集更小） |
| **合计 GPU** | **≈ 45–60 min（3 seeds × 3 场景）**，RTX 3090 |
| 显存 | 峰值 < 4 GB（448² 输入 batch 8；bank ≤ 291 图） |
| 磁盘 | 压缩 ≈3.5 G + 解压 ≈2–3.5 G（峰值 ≈7 G，稳定占用 2–3.5 G，取决于是否清理 zip） |
| 人工 | 下载/许可 ≈ 0.5–1 h（不可自动化） |

---

## 6. GO / HOLD / STOP 判据与当前评估

| 判据（任务书提供） | 当前评估 |
|---|---|
| GO-1 数据获取方式符合官方要求 | ⚠️ **未完成**（尚未申请/下载/接受许可；流程需人工确认） |
| GO-2 公开标签足以计算预定指标 | ✅（结构上成立）/ ⚠️（统计功效受限，须按 D5 用 pooled 主分析） |
| GO-3 协议可复现 | ✅（官方 loader + 冻结管线 + §5.4 预注册步骤） |
| GO-4 磁盘与计算资源充足 | ✅（GPU <1 h；磁盘紧张但可解） |
| GO-5 F0/Ours 可公平对照 | ✅（同一 bank/test/预处理/评分代码路径） |
| HOLD 条件 | **成立**：数据尚未获批/下载；且统计口径（D5）与预处理决策（D1–D3）尚未批准；两者均有明确可解方案 |
| STOP 条件 | 均不成立（可合法获取；公开部分可支撑指标；不存在不可解的公平性障碍） |

**结论：HOLD**（不是 STOP，也不应立即 GO）。两个前置条件：
**(i)** 研究负责人批准数据获取并完成官方表单/许可（人工）；
**(ii)** 批准 §5.3 D1–D5 的协议决策。

以上判据只决定是否进入下一阶段，**不代表任何模型已通过验证**。

---

## 7. 尚需研究负责人决定的事项

1. **是否启动数据获取**（CC BY-NC-SA 4.0 非商业许可；人工完成表单/下载）。
2. **场景集**：仅三候选（Sheet Metal / Vial / Fruit Jelly）还是增加其它场景（如 Can 长宽比更极端 2232×1024；Wall Plugs 灰度+散料）。
3. **D1 输入几何**：推荐 A（统一 resize 448×448）。
4. **D3 bank/训练规模**：推荐 bank = 全部 train/good。
5. **D5 统计口径**：推荐 pooled-3-scene 为主分析 + 逐场景探索性（依据 §3.3 功效分析）。
6. **是否叠加像素级 AU-PRO 扩展**（需额外下载官方评估代码；TESTpub 有 mask，可本地算，但会显著扩大范围）。
7. **是否允许"下载→校验→解压→删除本次 zip"** 以控制磁盘峰值（仅删除本轮自己下载的压缩包）。
8. **MSC-AD 后备候选**：本轮未评估；若 AD2 被否，是否启动 MSC-AD 预检（需另行指令）。

---

## 8. 本轮边界与合规声明

- 未下载/未解压任何数据；未访问下载资源；未提交表单；未接受许可条款。
- 未运行任何 GPU 训练或推理（唯一新计算 = CPU bootstrap 模拟，8.8 s）。
- 未修改 R1-ILA / F0 的任何代码或参数；未改冻结配置。
- 未清理任何历史文件；未使用破坏性 Git 命令。
- 本轮新增文件仅：`results/e20/mvtec_ad2_preflight.md`（本报告）、`scripts/e20b_auroc_power_sim.py`、
  `results/e20/auroc_power_sim.json`，及 `.gitignore` 的 `!results/e20/` 白名单行。

## 附：主要来源

- 官方主页：https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- 官方下载页：https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2/downloads
- 评测服务器：https://benchmark.mvtec.com/
- 论文：Heckler-Kram, Neudeck, Scheler, König, Steger, "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection", IJCV 134(4), 2026；arXiv:2503.21622（本轮核实：Table III 采集设置、Table IV 规模与分辨率、Table VI 光照、§III-B/C 划分与光照设计、§IV-B 指标）
- 官方 code utils 本地副本：`experiments/2026-09-30_illumination_sensitivity_exploration/mvtec_ad2_illumination_sensitivity/reference/`
- 历史调查记录：同目录 `README.md`（2026-09-30，含目录结构/样本量/公开标签限制的已核验事实）
- 本轮模拟：`scripts/e20b_auroc_power_sim.py` → `results/e20/auroc_power_sim.json`
