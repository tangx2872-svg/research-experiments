# E4-0C — LL-IAD vs M²AD Protocol Audit

**日期**：2026-10-08
**起始 HEAD**：`31d4602`（branch `main`；`origin/research/industrial-anomaly` = `31d4602`）
**性质**：**dataset selection / protocol design**，不是实验结果
**GPU 使用**：**0**（未运行 PatchCore / Original / B2 / X6c）
**下载**：**未下载任何图像数据**（仅读取官方 README / 论文 / 官方代码）
**判据**：`docs/E4_DATASET_SELECTION_CRITERIA.md`（本轮**先于**任何数据集检视写定）

**科研状态（恢复确认）**：

```text
CURRENT STAGE = Phase II — Paper Validation
METHOD SEARCH = CLOSED
FROZEN METHOD = X6c
```

---

## 0. 事实 / 解释 / 假设 分离约定

- **FACT**：官方论文、官方仓库、官方代码或官方元数据生成脚本中直接读出。
- **INTERPRETATION**：由 FACT 推出的判断。
- **UNKNOWN**：正式材料层面无法确认。**不猜测**。
- **HYPOTHESIS**：尚未验证。

---

## 1. LL-IAD Audit

### 1.1 来源层级

| Tier | 来源 | 状态 |
|---|---|---|
| Tier 1（论文） | Hoang, Tan, Nguyen, ... Tran. *Unsupervised industrial anomaly detection using paired well-lit and low-light images.* **JCDE 12(5):41–61, 2025**. DOI `10.1093/jcde/qwaf043`（2025-05-01 出版；Gold OA，license **CC BY-NC**） | 摘要可得；**正文全文不可达**（OUP 403 / 无 arXiv 版 / Semantic Scholar 反爬） |
| Tier 2（作者/机构） | Shibaura Institute of Technology — Elsevier Pure 记录页 | 仅元数据 + 摘要 |
| Tier 3（正式 hosting） | **未发现任何官方数据托管**（GitHub / HuggingFace / Kaggle / 项目主页均无） | **NOT FOUND** |
| Tier 4（线索） | ivySCI 页（登录墙）、百度学术条目 | 仅复述摘要，**不构成关键事实依据** |

### 1.2 Basic（§5.1）

| 项 | 值 | 等级 |
|---|---|---|
| full name | **LL-IAD**（low-light industrial anomaly detection） | FACT（摘要） |
| publication | JCDE 12(5):41–61, 2025 | FACT |
| year | 2025 | FACT |
| number of images | **UNKNOWN** | — |
| categories | **UNKNOWN** | — |
| resolution | **UNKNOWN** | — |
| total size | **UNKNOWN** | — |
| license（数据集本身） | **UNKNOWN**（论文/记录页均未声明数据集许可；论文文本许可为 CC BY-NC） | — |
| official download source | **NOT FOUND（Tier 1–3 均无入口）** | — |

### 1.3 Normal / anomaly（§5.2）

| 项 | 值 | 等级 |
|---|---|---|
| normal count | **UNKNOWN**（但摘要明确"仅用无异常图像训练"⇒ 存在 normal 集） | FACT（存在性） |
| anomaly count | **UNKNOWN**（但在其上报告 I-AUROC/AUPRO ⇒ 存在 defect 集） | FACT（存在性） |
| train/test split | **UNKNOWN** | — |
| normal-only training 成立？ | **是**（论文方法即 normal-only 训练） | FACT |

### 1.4 Illumination（§5.3 — 本轮最高优先级之一）

**FACT**：论文摘要与标题明确数据集"featuring **paired well-lit and low-light images**"，并自称"first dataset for LL-IAD"。
论文方法学描述为"从低光特征重建正常光照特征"（low-light → well-lit feature reconstruction on nominal samples）。

**FACT**：摘要仅给出两类条件 —— `well-lit` 与 `low-light`。

**UNKNOWN（关键）**：§5.3 要求区分三种互斥情形：

| 情形 | 能否确认 |
|---|---|
| **A** 同一物体/同一场景真实配对拍摄 | **UNKNOWN** |
| **B** 不同样本但属于两个 illumination domain | **UNKNOWN** |
| **C** well-lit 图像经算法合成 low-light | **UNKNOWN** |

> **本次审计未能确认任何一种。** 依 §5.3「不得仅根据论文中的 `paired` 一词推断」，且正文全文不可达 →
> **不得**把 "paired" 直接解读为情形 A。**也不得**推测为 B 或 C。

**INTERPRETATION（风险）**：论文摘要另有一句"even when well-lit images are **unavailable**, our model maintains high performance using **Retinexformer-enhanced low-light images**"。这**只说明作者考虑了无需 well-lit 的部署场景**，**不能**据此推断数据集内的 well-lit/low-light 是合成关系。

### 1.5 Pair Identity（§6）

```text
PAIRING = UNKNOWN
```

**理由**：无法访问任何一项 pairing 信息来源 —— 无 filename 结构说明、无 metadata 文件说明、无 annotation 说明、无 capture protocol 细节。
按 §6 规定，**写 `PAIRING = UNKNOWN`，不推测**。

**Normal pairing level（§13 口径）**：`UNKNOWN`
**Defect pairing level**：`UNKNOWN`

### 1.6 Ground Truth（§7）

| 项 | 是否存在 | 等级 |
|---|---|---|
| image label | 推断存在（报告 I-AUROC） | FACT（间接） |
| pixel mask | **推断存在**（报告 AUPRO，需 pixel-level 区域重叠） | FACT（间接） |
| defect type | **UNKNOWN** | — |
| anomaly location | 推断存在（AUPRO 需定位） | FACT（间接） |
| illumination label | **UNKNOWN** | — |
| low-light anomaly 是否有自己的 mask / 复用 well-lit mask | **UNKNOWN** | — |
| registration / alignment 是否严格 | **UNKNOWN** | — |

### 1.7 ΔR / Δd′ Feasibility（§8 / §9）

| Metric | 判定 | 依据 |
|---|---|---|
| **ΔR** | **UNKNOWN → 按判据记 FAIL** | 需要「clean normal 分数分布」+「可识别的 shift 分组」。llumination label 是否本地可识别 **UNKNOWN** ⇒ 无法构造 `shift_good` |
| **Original d′** | UNKNOWN | 需要 clean normal + clean defect 的可分离子集，结构 UNKNOWN |
| **B2 d′ / X6c d′** | UNKNOWN | 同上 |
| **Δd′** | **UNKNOWN → 按判据记 FAIL** | 同上 |

**结论**：§8/§9 要求的 Exact / Equivalent / Approximate / Impossible 四选一**无法给出** —— 因为**连数据结构都无法确认**。按判据文件 §C.3「任一关键条件不清楚 ⇒ 宁可 C」，**LL-IAD 不得成为主 E4**。

> **诚实声明**：LL-IAD 是 **UNKNOWN，不是被证伪**。其不可用性来自**正式材料不可达 + 无官方分发入口**，而非发现其结构缺陷。若作者提供官方数据分发与文档，应立即重新评估。

---

## 2. M²AD Audit

### 2.1 来源层级

| Tier | 来源 | 状态 |
|---|---|---|
| Tier 1（论文） | Cao, Cheng, Xu, ... Shen. *Visual Anomaly Detection under Complex View-Illumination Interplay: A Large-Scale Benchmark.* **arXiv:2505.10996**（2025-05）；**Pattern Recognition 2026**（2026-04-01 接收） | 正文可得（ar5iv） |
| Tier 2（作者官方仓库 / 项目页） | `https://github.com/hustCYQ/M2AD` ｜ `https://hustcyq.github.io/M2AD/` | 可得；**含官方元数据生成代码** |
| Tier 3（正式 hosting） | Hugging Face Dataset（主推）＋ Google Drive；百度网盘（提取码 "soon!"） | 存在入口（本机网络超时，未取到卡片） |

### 2.2 Basic（§10）

| 项 | 值 | 等级 |
|---|---|---|
| full name | **M2AD**（Multi-View Multi-Illumination Anomaly Detection） | FACT |
| publication | Pattern Recognition（2026 接收）；arXiv:2505.10996（2025-05） | FACT |
| categories | **10 主类**：Bird、Car、Cube、Dice、Doll、Holder、Motor、Ring、Teapot、Tube（每类 2 子类 → 20 个物理样件） | FACT |
| specimens | **999** | FACT |
| views | **12**（单相机 + 电动转台，30° 等分；**分时采集**，非多相机同步） | FACT |
| illuminations | **10**（4 条线性条形光 + 1 个同轴环形光，PLC 独立/同步开启；编号 **01–10**，spectrally-tuned） | FACT |
| configurations | **120**（12×10） | FACT |
| total images | **119,880**（= 999×120）；normal **69,070** / anomalous **50,810** | FACT |
| resolution | 原生 **3,648×5,472**（≈20 MP） | FACT |
| versions | **M2AD-1024**（1024²）与 **M2AD-256**（256²） | FACT（官方 README） |
| total size | **UNKNOWN**（论文与 README 均未给；用户侧提示约 164 GB；本机 HF 请求超时未取到卡片） | — |
| license | **UNKNOWN**（论文、README、仓库侧栏均未声明） | — |
| official download source | Hugging Face（主）＋ Google Drive ＋ 百度网盘 | FACT |

### 2.3 Normal / anomaly（§10）

| 项 | 值 |
|---|---|
| normal（Good） | 目录 `Good/`；全数据集 69,070 图 |
| anomaly（NG） | 目录 `NG/`；全数据集 50,810 图 |
| GT 目录 | `GT/{object_name}/..._mask.png`（像素掩码）、`..._seg.png`（分割） |
| train/test split | **官方代码明确**（见 2.4）：**specimen 级**，normal_ratio = **0.6**，random_seed = **42** |
| normal-only training | **成立**（train 只含 `Good/` 下的 normal specimens；NG 全部进 test） |
| 每类 Good/NG 细分 | 论文仅以 Fig. 3(a) 柱状图呈现，**正文未给数字** |

**官方 split 逻辑（FACT，逐字来自 `data/gen_metadata/m2ad.py`）**：

```python
normal_ratio = 0.6 ; random_seed = 42
random.shuffle(normal_obj_names)                       # 打乱的是「normal 标本目录名」
train = cls_normal_info[:int(len*0.6)]                 # 60% normal 标本（其全部 120 配置图）
test  = cls_normal_info[int(len*0.6):] + cls_abnormal_info   # 40% normal 标本 + 全部 NG 标本
```

**INTERPRETATION**：划分粒度是 **specimen（物理样件）级**。
⇒ 同一 specimen 不会同时出现在 train 与 test ⇒ **不存在 specimen 泄漏**。
⇒ 同一 specimen 的 10 种光照**同侧**（train 侧或 test 侧）⇒ **不存在「光照 1 在 train、光照 2 在 test」的泄漏**。

### 2.4 Metadata Structure（§11 — 本轮最高优先级之一）

**FACT**：官方元数据由 `gen_meta_data.py` → `data/gen_metadata/m2ad.py::M2ADSolver` 生成，输出
`{DATA_ROOT}/jsons/meta.json` 与 `{DATA_ROOT}/jsons/{category}.json`。

**每条记录的实际字段（逐字来自官方代码）**：

```python
{
  'img_path', 'view', 'illumination', 'object_name',
  'object_anomaly', 'image_anomaly', 'cls_name',
  'mask_path', 'seg_path', 'detectable'
}
```

**字段解析方式（FACT）**：

| 目标信息 | 来源 | 官方代码 |
|---|---|---|
| category | `cls_name`（= 目录名） | `for cls_name in self.CLSNAMES` |
| specimen_id | `object_name`（= `Good/` 或 `NG/` 下的子目录名；排序用 `int(object_name[-3:])`） | `int(x['object_name'][-3:])` |
| normal / anomaly（specimen 级） | `object_anomaly`（`Good/`→0、`NG/`→1） | `object_anomaly=0/1` |
| normal / anomaly（**image 级**） | `image_anomaly`；**仅当 mask 存在且 `np.sum(mask) > 0` 才为 1**，否则 0 | `if np.sum(mask) > 0: image_anomaly = 1` |
| view_id | **filename 第 `[1:4]` 位**（3 字符） | `'view': image_path[1:4]` |
| illumination_id | **filename 第 `[6:8]` 位**（2 字符） | `'illumination': image_path[6:8]` |
| mask | `mask_path` = `GT/{object_name}/{image_path[:-8]}_mask.png` | 见代码 |
| defect type | **无独立字段**；仅 `seg_path` 与 `detectable` 两个辅助字段 | — |

**⇒ §11 的结论**：`category / specimen_id / normal-anomaly / view_id / illumination_id` **五项全部可恢复**（前三项在 metadata 字段，后两项从文件名解码）。`mask` 可恢复。`defect_type` **在官方元数据中不存在独立字段**。

**INTERPRETATION（mask 与 illumination 的关系）**：`mask_path` 由去掉文件名末 8 字符后拼 `_mask.png` 得到，且落在 `GT/{object_name}/` 下。
**若**文件名末尾 8 字符为照明编号相关段（如 `_I01.png`），**则同一 (specimen, view) 的 10 张不同光照图共享同一张 mask** ⇒ 对应 §7 的"复用 mask"情形，且因为**光照切换时物体与相机均不移动**，registration 天然严格。
⚠️ **本次未能取得具体文件名样例，故上述为「强指向」而非 CONFIRMED** → 按 §6 纪律记为 **HYPOTHESIS**，须在下载后由 integrity audit 确认。

### 2.5 Fixed-view Test（§12）

**FACT**：`view` 与 `illumination` 均为显式可恢复字段。

**INTERPRETATION**：可以合法构造

```text
category C ／ specimen S ／ 固定 view V ／ illumination ∈ {01..10}
```

对 **normal**：`Good/{S}/<filenames with view=V, illum=01..10>` ⇒ **成立**。
对 **defect**：`NG/{S}/<同一构造>` ⇒ **成立**（NG 标本同样被 120 配置完整采集）。
⇒ §12 的"同时对 normal 与 defect 都成立"**成立**，这是本数据集相对其他候选的**最强 E4 条件**。

**限定（FACT，来自论文 Limitations）**：论文自陈**尚未完成** viewpoint 与 illumination 贡献的解耦量化（将"Modality Contribution Analysis"列为未来工作），并称某些缺陷只在特定视角-光照组合下可见。
**INTERPRETATION**：固定 view 后，illumination 成为受控自变量 ⇒ §15 的要求可满足。

### 2.6 Pairing（§13）

| 层级 | 判定 | 依据 |
|---|---|---|
| Normal pairing level | **Level 3**（same specimen + same view + different illumination） | `view`/`illumination` 显式可解 |
| Defect pairing level | **Level 3 CONFIRMED；Level 4（same physical defect）为 HYPOTHESIS** | 物理上同一 specimen 的缺陷在各光照下是**同一物理缺陷**；且 `NG` 标本被全部 10 种光照完整采集。但"mask 为同一张"取决于 2.4 的 `[:-8]` 推断 ⇒ 未确认 |

### 2.7 ΔR / Δd′（§14）

**先重申判据文件 §E.5 的 FACT**：冻结实现中 `shift_good` 是**项目自己对 clean normal 施加的 synthetic photometric 变换**
（`SHIFTS = brightness_0.7 / brightness_1.3 / gamma_0.7 / gamma_1.3`），**不是数据集自带的光照条件**。
⇒ 数据集的真实光照轴**不是 ΔR 可计算性的必要条件**。

| Metric | 判定 | 原因 |
|---|---|---|
| **ΔR（冻结原义）** | **Exact** | 只需 test 集 normal 图像。M2AD 有 `Good/`（normal-only 可辨识）。项目把 4 个冻结 synthetic 变换施加于 M2AD 的 test normal ⇒ 公式逐字复用，`SHIFTS`/`μ_g`/`σ_g`/`mean_abs_delta_z` 均不需改 |
| **d′** | **Equivalent** | 需 clean normal（μ_g/σ_g）+ clean defect。M2AD 有 `Good/` + `NG/` + `mask_path` + `image_anomaly` ⇒ 可构造。**差异**：Exp16 的 `mean_dprime` 按 **defect_type** 取平均，而 M2AD 官方元数据**无 defect_type 字段** ⇒ 该平均**退化为单一 type（全 defect 池化）**。d′ 的数学定义不变 |
| **Δd′** | **Equivalent（同 d′ 的退化）** | `cand − Original`，同 (category, seed) 配对，公式不变 |
| **paired illumination response** | **Exact（但需新预注册）** | M2AD 的真实光照轴（`illumination` 01–10）可构造 Level 3 配对，用于**独立的、真实光照**的鲁棒性分析。⚠️ 这**不是**冻结的 ΔR（`SHIFTS` 是 4 个 synthetic 变换）；把 `SHIFTS` 换成真实光照 = **改变扰动集合定义 = protocol change**，必须**新预注册协议 + 新 SHA256**，不得就地替换 |
| **defect-tail analysis** | **Exact** | 以 `Δd′` 的 worst/尾部统计为基础，结构与 Exp16 一致；缺陷类型层面的分层则受 d′ 的同一退化限制 |

**⇒ MUST-6 / MUST-7：PASS（Equivalent 及以上）**，附两项必须写进未来协议的文字限定：
1. `mean_dprime` 在 M2AD 上**退化为池化 d′**（无 defect-type 分层）；
2. **真实光照轴的分析属于新协议**，与冻结 ΔR 并列报告，**不得**混称为同一个 ΔR。

### 2.8 Viewpoint Confound（§15）

**问题**：能否在不重新定义核心科学问题的情况下，把 viewpoint 固定为控制变量？

**回答：可以。** 具体 subset construction（**只证明可行性，本轮不选最终 V**）：

```text
选 category C（10 选 1 预告：须按数据集结构规则预注册，不得按模型结果挑）
选 specimen S（同 category 内的 1 个 physical specimen）
固定 view V（12 选 1）
illumination I ∈ {01,...,10}      # 受控自变量
→ 每个 (C, S, V) 给出 10 张仅光照不同的图；对 Good 与 NG 标本都成立
```

- illumination 成为**主要变化变量**；specimen 与 view 被固定为控制变量。
- 固定后无需重新定义核心科学问题（研究对象仍是「照明变化下的 normal 漂移与 defect 保持」）。

**INTERPRETATION（诚实保留）**：论文自陈 view×illumination 未解耦。固定 view 后**view 的多样性被牺牲**，
因此该 subset 只能回答「给定视角下 illumination 的影响」，不能回答「跨视角可见性变化」。
这与本论文的 illumination-nuisance 问题**对齐**（视角不是本研究变量），故**不构成**核心科学问题的改变。

---

## 3. ΔR / Δd′ 可行性总表（§19 前置）

| Metric | LL-IAD | M²AD |
|---|---|---|
| ΔR | **UNKNOWN（记 FAIL）** | **Exact** |
| d′ | UNKNOWN | **Equivalent**（defect-type 分层退化） |
| Δd′ | UNKNOWN | **Equivalent** |
| paired illumination response | UNKNOWN | **Exact，但需新预注册协议** |
| defect-tail analysis | UNKNOWN | **Exact**（分层受同一退化限制） |

---

## 4. Leakage Audit（§18）

### 4.1 LL-IAD

| 检查 | 结果 |
|---|---|
| 是否存在官方 train/test | **UNKNOWN** |
| pair 是否跨 split | **UNKNOWN** |
| illumination pair 是否可能泄漏 physical specimen | **UNKNOWN** |

**INTERPRETATION**：无法完成 leakage audit ⇒ 依 §18 与判据文件 §C.3，**不得**成为主 E4。

### 4.2 M²AD

| 检查 | 结果 | 依据 |
|---|---|---|
| specimen-level split 是否必要？ | **官方即为 specimen 级** ⇒ 已满足 | `m2ad.py` shuffle 的是 `normal_obj_names` |
| 同 specimen 不同 illumination 是否可能 train-test 泄漏？ | **不会**（同一 specimen 的全部 120 配置同侧） | 同上 |
| view-level split 是否可能泄漏？ | **无 view 级 split**；view 维度未被切分 ⇒ 不存在 view 泄漏 | 同上 |
| NG 是否进入 train？ | **不会**（train 只取 `Good/`） | 同上 |

**INTERPRETATION**：M²AD 的**官方 split 天然满足**泄漏控制；一个重要的前提是 **必须使用官方生成的 metadata/split，而不是自行按图像随机划分**（后者会破坏 specimen 级隔离，属 §18 明令禁止的泄漏）。

**仍存在的 E4 泄漏风险（方法侧，与数据集无关，须写入未来 `LEAKAGE_GUARD.md`）**：
- 不得用 M2AD 的 target 结果回头调 X6c / C2 / C6 权重或阈值；
- 不得按 M2AD 结果删类别 / 删光照 / 删 seed；
- 不得据 M2AD 结果重定义 PASS / catastrophic。

---

## 5. PatchCore Compatibility（§17，仅静态判断）

| 问题 | LL-IAD | M²AD |
|---|---|---|
| normal-only training 是否自然成立 | UNKNOWN | **成立**（官方 train = normal specimens） |
| category-wise PatchCore 是否可运行 | UNKNOWN | **可以**（10 类；目录 `{cls}/{Good,NG,GT}` 结构清晰） |
| resolution adaptation 是否只是 engineering change | UNKNOWN | **是**（原生 3648×5472 → 256/512 缩放；anomalib 常规 resize）。**不是** method change |
| mask alignment 是否兼容 | UNKNOWN | **推断兼容**（mask 与图同名 + `_mask.png`，按 `GT/{object_name}/` 组织）；须 integrity audit 确认 |
| 是否需要 method-level modification | UNKNOWN | **不需要修改 X6c**。需要的只是 loader / path / resolution 适配 |
| 是否需要修改 X6c | UNKNOWN | **NO** |

**INTERPRETATION**：M²AD 需要一个**自定义 folder-style dataloader**（读取官方 `meta.json` 到 anomalib/项目 runner）。
按本任务与判据文件的分类，这属于 **engineering adaptation**，**不触碰**方法定义、权重、阈值。

---

## 6. Dataset Size / Download Strategy（§16）

### 6.1 LL-IAD

| 项 | 值 |
|---|---|
| full download size | **UNKNOWN** |
| smallest official version | **不存在**（无官方分发） |
| metadata-only size | **UNKNOWN** |
| minimum E4 Mini subset size | **UNKNOWN** |

### 6.2 M²AD

| 项 | 值 |
|---|---|
| full download size | **UNKNOWN**（论文/README 未给；用户侧提示约 **164 GB**；原生 3,648×5,472 ≈ 20 MP × 119,880 图） |
| smallest official version | **M2AD-256**（256×256） |
| metadata-only size | metadata 由官方脚本**从已下载图像本地生成**（`gen_meta_data.py`），**不提供独立 metadata 包** ⇒ metadata-only 下载**不可行** |
| minimum E4 Mini subset size | **可构造**：`M2AD-256` + 少量 category + 固定 view ⇒ 只下载所需子集（**本轮未实际下载**） |

**INTERPRETATION**：
- 因 metadata 由本地脚本从**图像目录名**（`object_name`）与**文件名**（view / illumination）解析，
  故**无法只下载元数据**——但可以下载 **M2AD-256** 的**单 category 单 view 子集**来跑通完整性审计。
- 是否可从 HF 做**按目录选择性下载**（`huggingface_hub` 支持 `allow_patterns`）**未验证**（本机 HF 请求超时）⇒ **UNKNOWN**。
- **原图（M2AD-1024 / 原生）不适合作为 E4 Mini 入口**；应先以 M2AD-256 验证。

---

## 7. 评分表（§19）

### 7.1 MUST

| Criterion | LL-IAD | M²AD |
|---|---|---|
| MUST-1 Normal | **PASS** | **PASS** |
| MUST-2 Defect | **PASS** | **PASS** |
| MUST-3 Illum variation | **PASS**（两类条件明确；但 A/B/C 机制 UNKNOWN） | **PASS**（10 种 PLC 光照；⚠️ 哪一种是 regular 未在正文定义） |
| MUST-4 Illum identifiable | **UNKNOWN → FAIL** | **PASS**（官方 metadata `illumination` 字段） |
| MUST-5 Local GT | **PARTIAL/UNKNOWN → FAIL** | **PASS**（`mask_path` / `image_anomaly` / `object_anomaly` 本地可得） |
| MUST-6 ΔR | **UNKNOWN → FAIL** | **PASS（Exact）** |
| MUST-7 Δd′ | **UNKNOWN → FAIL** | **PASS（Equivalent）** |
| MUST-8 Held-out | **PASS** | **PASS** |
| **MUST PASS COUNT** | **4 / 8**（其余 4 项 UNKNOWN） | **8 / 8** |

### 7.2 SHOULD（0 / 1 / 2）

| Criterion | LL-IAD | M²AD |
|---|---:|---:|
| paired illumination | 1（声称，未证实） | 2 |
| same-object pairing | 1（声称，未证实） | 2 |
| pixel mask | 1（AUPRO 间接） | 2 |
| defect type | 0（UNKNOWN） | 1（论文列 4 类，但**不在官方 metadata 中**） |
| illumination ID | 0（UNKNOWN） | 2 |
| multiple illumination levels | 1（2 档） | 2（10 档） |
| normal-only compatible | 2 | 2 |
| industrial relevance | 2 | 2 |
| multiple categories | 0（UNKNOWN） | 1（10 主类 / 20 物理样件） |
| recognized publication | 1（JCDE，同行评审，Gold OA） | 2（Pattern Recognition 2026） |
| official / public access | 0（**无官方分发入口**） | 2（HF + Google Drive + 官方仓库） |
| manageable size | 0（UNKNOWN） | 1（原生巨大；存在 M2AD-256） |
| PatchCore compatibility | 0（UNKNOWN） | 1（需自定义 folder loader，属工程适配） |
| reproducible local evaluation | 0（UNKNOWN） | 2（官方 metadata 生成器 + 官方 split） |
| fixed-view illumination analysis possible | 1（UNKNOWN） | 2（view/illumination 均为显式字段） |
| **SHOULD 合计（满分 30）** | **10** | **26** |

---

## 8. Verdict（§20）

依 §20 的判定条件：

- **A — LL-IAD WIN**：需要 8 MUST 全 PASS → **不满足（4/8，4 项 UNKNOWN）**。
- **B — M²AD WIN**：需要 8 MUST 全 PASS + ΔR/Δd′ 至少 Equivalent + pairing/GT 足够明确 + 无不可控 leakage
  → **MUST 8/8 PASS**；**ΔR = Exact / Δd′ = Equivalent**；**pairing = Level 3（normal）/ Level 3 confirmed & Level 4 hypothesis（defect）**；**官方 split 为 specimen 级 ⇒ 泄漏可控** ⇒ **满足**。
- **C — NEITHER**：不适用（M²AD 关键条件已由**官方代码**确认，非"不清楚"）。

**⇒ WINNER = `M²AD`（B — M²AD WIN）**

**必须随判定一起声明的限定（不得省略）**：

1. **LL-IAD 是 UNKNOWN，不是被证伪**。其 4 项 MUST 未 PASS 的原因是**正式材料不可达 + 无官方分发入口**。若获得官方数据与文档，应重新评估。
2. M²AD 的 **license 未声明**、**总体积未声明** ⇒ 使用前须人工确认（legal + 存储）。
3. **哪一个 illumination id 对应 regular/well-lit 未被任何正文材料定义**（仅存在于论文 Fig. 10 图内）⇒ 必须人工确认。
4. `mean_dprime` 在 M2AD 上**退化为池化 d′**（官方 metadata 无 defect_type 字段）。
5. **真实光照轴的 ΔR 属于新协议**，必须重新预注册 + 冻结 SHA256，**不得**当作冻结 ΔR 的等价替换。

---

## 9. 对论文证据链的贡献

- **本审计不产生任何方法证据，也不改变 Exp16 的任何数字或结论**。
- 完成/推进的证据链格子：**E4 的「数据集可行性」前置门** —— 从「无可用 external dataset」推进到「**已选出一个在原义口径下可计算冻结指标、且 illumination 轴可本地隔离与配对的候选数据集**」。
- **下一格**：`M²AD lightweight subset download + integrity audit`（验证 2.4 的 mask/文件名推断、确认 illumination reference、核实 license 与体积），之后才是 E4 Mini 协议设计。

## 10. 本轮未做（明确记录）

未下载任何图像；未运行 GPU；未运行 Original / B2 / X6c；未修改 X6c / C2 / C6 / 任何阈值；
未创建 experiment results 目录；未 push Git。
