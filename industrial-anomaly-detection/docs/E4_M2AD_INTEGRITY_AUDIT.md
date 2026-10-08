# E4-D0 — M²AD Lightweight Download & Integrity Audit

**日期**：2026-10-08
**分支 / 起点 HEAD**：`research/industrial-anomaly` / `44ea480`（push 后 `origin/research/industrial-anomaly` = `44ea480`）
**性质**：dataset integrity audit（**不是实验**）
**GPU**：**0**（未运行 PatchCore / Original / B2 / X6c）
**判据**：`docs/E4_DATASET_SELECTION_CRITERIA.md`（E4-0C 预注册）
**上游**：`docs/E4_LL_IAD_M2AD_PROTOCOL_AUDIT.md`（E4-0C，verdict = M²AD WIN）

**科研状态（恢复确认）**：

```text
CURRENT STAGE = Phase II — Paper Validation
METHOD SEARCH = CLOSED
FROZEN METHOD = X6c
PRIMARY E4 DATASET = M²AD
CURRENT TASK = E4-D0 Integrity Audit
```

> **事实 / 解释 / 假设**照 E4-0C 约定分离。所有"FACT"均由**官方 artifact 或真实文件**读出。

---

## 1. 数据来源与版本（§8）

| 项 | 值 | 等级 |
|---|---|---|
| source | HF dataset **`ChengYuQi99/M2AD`**（作者 Yuqi Cheng / HUST；repo tag `arxiv:2505.10996`） | FACT |
| endpoint | `hf-mirror.com`（本服务器 `huggingface.co` **不可达**；官方 M2AD 代码本身即设 `HF_ENDPOINT=https://hf-mirror.com`） | FACT |
| version / resolution | **1024×1024**（⇒ 本 repo 即 GitHub README 所称 **M2AD-1024**） | FACT（实测图像尺寸） |
| **M2AD-256** | **官方 HF repo 不存在**；GitHub README 指向 Google Drive，而 **`drive.google.com` 在本服务器不可达** | FACT |
| 结构 | 10 个 per-category zip：Bird/Car/Cube/Dice/Doll/Holder/Motor/Ring/Teapot/Tube，各 13.9–16.4 GB，**合计 ≈ 153 GB** ＋ `jsons.zip`（1.1 MB） | FACT |
| license | **`apache-2.0`**（HF repo metadata tag） | FACT（标注）；论文 / GitHub README **未声明**数据集许可 |
| 再分发 | **本审计不主张任何再分发权利**；如用于论文发布，须由人工确认许可条款 | — |

**archive 完整性**：

| 文件 | size (bytes) | SHA256 | 校验 |
|---|---:|---|---|
| `jsons.zip` | 1,100,022 | `bbeaa75c…676725980` | — |
| `Bird.zip` | 15,506,344,175 | `b5bb22a976440d8739e84724eb864b290f23eee053511ab47f8a4c200fabace7` | **= 官方 `x-linked-etag`**（逐字相等）✓ |

- 解包：`unzip` → `EXIT=0`；解包后 `Bird.zip` 已删除以恢复磁盘余量（用户批准）。
- 下载日志：`data/m2ad/bird_dl.log`；解包日志：`data/m2ad/unzip.log`。

---

## 2. 下载策略与磁盘（§7 / §9）

| 项 | 值 |
|---|---|
| Priority 1（M2AD-256） | **不可用**（不在官方 HF repo；Google Drive 不可达） |
| Priority 2（单 category） | **可用并已执行** —— 下载 **Bird**（官方顺序第一个） |
| Priority 4（只能下 164 GB 原始数据） | 不适用（单 category 下载可行） |
| 类别选择规则 | **先于下载记录**：官方顺序（HF 列表序 ≡ `meta_unsupervised.json` class 序，字典序）第一个完整提供 Good+NG+GT+metadata 的 category ⇒ **Bird**。不含任何模型结果信息。 |

**磁盘**：

| 时刻 | total | used | available |
|---|---:|---:|---:|
| 下载前 | 50 G | 6.35 G | **43.65 G** |
| 下载完成（zip 在盘） | 50 G | 22 G | 29 G |
| 解包中（峰值） | 50 G | 36 G | **15 G** |
| 解包后（zip 已删） | 50 G | 36 G | **15 G** |

- **安全余量要求 ≥ 10 GB → PASS（15 GB）**。
- **观察（需知悉）**：内容层面 `du /root/autodl-tmp` = **21.6 GB**（`data` 17 G ＋ `results` 3.2 G ＋ `mvtec_ad_5cats.tar.gz` 1.6 G），而 `df` 仍报 **used 36 G / avail 15 G**；差值 ≈ 14.4 GB = `Bird.zip` 的体积。
  `sync` 后未变化，且全进程 `/proc/*/fd` 扫描**未发现任何进程持有该已删除文件**。
  **解释（假设，未验证）**：`/root/autodl-tmp` 以 `prjquota` 挂载的 XFS project-quota 记账滞后。
  **影响**：后续按 `df` 的 15 GB 预算使用（保守），不要假设 43 GB 可用。

---

## 3. 真实目录结构（§13 — 直接读实际文件树，非 README 推断）

```text
data/m2ad/extracted/Bird/
├── Good/<specimen>/A<view>_I<illum>.png        # specimen 为 3 位数字，如 000, 024
├── NG/<defect_specimen>/A<view>_I<illum>.png   # specimen 为缺陷命名，如 scratch_1_000
└── GT/<defect_specimen>/A<view>_mask.png
                         A<view>_seg.png
```

**实测统计（Bird）**：

| 项 | 值 |
|---|---|
| category | 1（Bird；数据集共 10 类） |
| Good specimens | **50** |
| NG specimens | **50** |
| Good images | **6,000**（50 × 120） |
| NG images | **6,000**（50 × 120） |
| GT files | **910** = **455 `*_mask.png`** ＋ **455 `*_seg.png`** |
| total files | **12,910** |
| image extensions | `.png`（100%） |
| image resolution | **1024 × 1024** |
| mask resolution / mode | **1024 × 1024** / `L`（灰度） |
| mask 前景占比 | 0.005% – 0.46%（**缺陷极小**，与论文"256×256 下可仅 4 像素"一致） |

**STRUCTURE MISMATCH？** → **无。** 与 E4-0C 静态协议审计一致。

> **关于 `GT/` 的两类文件**：`*_mask.png` 是像素级 anomaly mask；`*_seg.png` 是缺陷分割。
> 官方 metadata 的 `seg_path` 字段**全为空**，故 `*_seg.png` 没有 metadata 条目——这**不是** mismatch。
> 本审计将它们分别记账（`docs/E4_M2AD_STRUCTURE_MISMATCH.csv` 中标记为 `INFO_*`）。

**文件级一致性（真实树 vs 官方 metadata）**：

| 检查 | 结果 |
|---|---|
| 真实存在但 metadata 无条目 | **0** |
| metadata 有条目但真实不存在 | **0** |
| mask 在盘但 metadata 无条目 | **0** |
| mask 在 metadata 但盘上不存在 | **0** |
| 非预期 GT 命名模式 | **0** |
| **REAL MISMATCH 合计** | **0** ✓ |
| INFO（`*_seg.png`） | 455（预期行为） |

---

## 4. Filename Parser Audit（§14）

E4-0C 的假设来自官方代码 `data/gen_metadata/m2ad.py`：

```python
'view':         image_path[1:4]
'illumination': image_path[6:8]
```

**实测真实文件名样例**（`Bird/Good/000/A000_I01.png`）：

```text
A 0 0 0 _ I 0 1 . p n g
0 1 2 3 4 5 6 7 8 ...
        ^^^ view = name[1:4] = '000'      ✓
                ^^ illumination = name[6:8] = '01'  ✓
```

| 检查 | 结果 |
|---|---|
| 文件名可解析率 | 100%（12,910 / 12,910） |
| `view == name[1:4]` 且 `illumination == name[6:8]`（对官方 metadata 的 119,760 条记录全量核对） | **mismatches = 0 / 119,760** ✓ |
| `view` 取值域 | **12 个**：`000, 030, 060, 090, 120, 150, 180, 210, 240, 270, 300, 330`（即**每 30° 一个视角**，与论文转台描述一致） |
| `illumination` 取值域 | **10 个**：`01 … 10` ✓ |
| mask 命名规则 `image_stem[:-4] + '_mask'`（如 `A060_I01` → `A060_mask`） | **20/20 正确** ✓ |

⇒ **parser 与真实文件名完全一致。无 parser 不匹配，无需"改 parser 以适配"。**

**NOT a STOP condition.**

---

## 5. Illumination Coverage Audit（§15）

`docs/E4_M2AD_ILLUMINATION_COVERAGE.csv`（Bird，来自官方 metadata，与真实文件 0 差异）：

| illumination_id | Good images | NG images | total |
|---|---:|---:|---:|
| 01 | 600 | 600 | 1,200 |
| 02 | 600 | 600 | 1,200 |
| 03 | 600 | 600 | 1,200 |
| 04 | 600 | 600 | 1,200 |
| 05 | 600 | 600 | 1,200 |
| 06 | 600 | 600 | 1,200 |
| 07 | 600 | 600 | 1,200 |
| 08 | 600 | 600 | 1,200 |
| 09 | 600 | 600 | 1,200 |
| 10 | 600 | 600 | 1,200 |

⇒ **illumination 01–10 真实存在且完全均衡**（每档 600 Good + 600 NG = 50 specimens × 12 views）。**PASS。**

---

## 6. View Coverage Audit（§16）

`docs/E4_M2AD_VIEW_COVERAGE.csv`

| 检查 | 结果 |
|---|---|
| specimens（Bird） | 100（50 Good + 50 NG） |
| 每 specimen 的 views | **12 / 12** |
| 每 specimen 的 illuminations | **10 / 10** |
| 每 specimen 的 configs | **120 / 120** |
| `12 × 10 = 120` 完整成立的 specimen 数 | **100 / 100** ✓ |
| 缺失 | **无（0）** |

⇒ 理论结构 `1 specimen × 12 views × 10 illuminations = 120` **在真实数据上完全成立**。**PASS。**

---

## 7. Same-specimen + Same-view Pair Audit（§17 — 本轮最重要检查）

`docs/E4_M2AD_PAIRING_AUDIT.csv`
**选择规则（结果无关，运行前记录）**：固定 `seed=0`，随机抽 5 个 Good specimen ＋ 5 个 NG specimen；对每个 specimen **随机抽一个真实存在的 view**；在该 (specimen, view) 下逐一检查 illumination `01…10` 是否存在。

| 项 | 结果 |
|---|---|
| Good：complete-illumination pairing rate | **1.000（50/50）** ✓ |
| NG：complete-illumination pairing rate | **1.000（50/50）** ✓ |

**抽样示例**（`docs/E4_M2AD_PAIRING_AUDIT.csv`）：

```text
Bird,Good,024,000,01,Bird/Good/024/A000_I01.png,1
Bird,Good,024,000,02,Bird/Good/024/A000_I02.png,1
...
```

⇒ **same specimen + same view + illumination 01–10 对 Good 与 NG 都真实成立**。**PASS。**

---

## 8. Defect Identity Audit（§18）

证据链（按 §18 优先级）：

| 证据 | 内容 | 结论 |
|---|---|---|
| ① 官方 metadata | `object_name` 在同一 (specimen, view) 下，10 个 illumination 共享同一 specimen 目录 | 同一物理样本 |
| ② GT filename mapping | mask = `GT/<specimen>/A<view>_mask.png` —— **只由 (specimen, view) 决定，与 illumination 无关** | **同一 view 的 10 张光照共用同一张 mask** |
| ③ mask identity | 实测 20/20 mask **逐字节同源**（同一路径），且 mask 存在率 = `image_anomaly=1` 比例 | 共用 mask 成立 |
| ④ capture protocol | 论文：单相机 + 电动转台，**切换光照时物体与相机均不移动** | 无跨光照配准问题 |
| ⑤ image inspection | `docs/figures/e4_m2ad_pairing_sanity.png`：同一 specimen、同一 view、10 个光照下的物体位置/姿态**一致**，仅光照不同 | 支持同一物理缺陷 |

⇒ **`DEFECT PAIRING = LEVEL 4`**
（same physical defect + same view + different illumination）

**限定（必须随结论声明）**：Level 4 的"同一物理缺陷"由 GT 的单张共享 mask + 静止采集协议共同支持；**mask 本身是 2D 投影**，缺陷在不同光照下的**可见性**会变化 → 这正是论文所述"平均仅 ~75% 异常图可检出"的来源（见 §10）。

---

## 9. GT / Mask Audit（§19）

`docs/E4_M2AD_MASK_AUDIT.csv`（随机抽 20 张 NG 图，`seed=0`）

```text
matched (mask exists) = 20 / 20
resolution_match      = 20 / 20     (1024x1024 img vs 1024x1024 mask)
nonempty              = 20 / 20     (mask.max() > 0)
name_rule_ok          = 20 / 20     (A<view>_I<illum>.png -> A<view>_mask.png)
```

⇒ **20 / 20 matched。** mask 与图像**分辨率一致，无需 resize / re-registration**。
（即便将来需要，也属 **engineering adaptation**，**不得**改变方法。）

---

## 10. Official Split / Leakage Audit（§21）

**官方 split 来源**：`jsons.zip` → **`meta_unsupervised.json`**（这是官方提供的 split，非本审计自行划分）。

| 项 | 值 |
|---|---|
| 顶层结构 | `{"train": {10 classes}, "test": {10 classes}}` |
| train 记录数 | **35,880**（`object_anomaly = 0` 全部；`image_anomaly = 0` 全部） |
| test 记录数 | **83,880** |
| 合计 | **119,760** |
| train specimen 数 | **299** |
| test specimen 数 | **699** |
| **train specimen ∩ test specimen** | **0** ✓ → **无 specimen 泄漏** |
| NG 是否进入 train | **否**（train 全部 `object_anomaly = 0`） |
| illumination 是否被切成 train/test | **否** → 同一 specimen 的 10 个光照**同侧** ⇒ **无跨光照泄漏** |
| view 是否被切成 train/test | **否** → **无 view 泄漏** |
| NG allocation | **全部 NG 进 test**（官方代码 `train = normal[:60%]`、`test = normal[60%:] + all_NG`） |

**按类记录数（train / test）**：

| class | train | test | total | train spec | test spec |
|---|---:|---:|---:|---:|---:|
| Bird | 3,600 | 8,400 | 12,000 | 30 | 70 |
| Car | 3,600 | 8,400 | 12,000 | 30 | 70 |
| Cube | 3,600 | 8,400 | 12,000 | 30 | 70 |
| **Dice** | **3,480** | 8,400 | **11,880** | **29** | 70 |
| Doll | 3,600 | 8,400 | 12,000 | 30 | 70 |
| **Holder** | 3,600 | **8,280** | **11,880** | 30 | **69** |
| Motor | 3,600 | 8,400 | 12,000 | 30 | 70 |
| Ring | 3,600 | 8,400 | 12,000 | 30 | 70 |
| Teapot | 3,600 | 8,400 | 12,000 | 30 | 70 |
| Tube | 3,600 | 8,400 | 12,000 | 30 | 70 |
| **TOTAL** | **35,880** | **83,880** | **119,760** | **299** | **699** |

### ⚠️ 与论文的两处数字差异（FACT，必须记录）

| 项 | 论文 | 实测 metadata | 差异 |
|---|---|---|---|
| 总图像数 | 119,880（999 × 120） | **119,760**（119,880 − 120） | **−120 图 = −1 个 specimen** |
| specimen 数 | 999 | **998**（8 类 × 100 ＋ Dice 99 ＋ Holder 99） | **−1** |
| 正常图 | 69,070 | 24,000（test）＋ 35,880（train） = **59,880** | **−9,190** |
| 异常图 | 50,810 | **59,880** | **＋9,070** |

**解释（假设，未验证）**：论文的 normal/anomaly 口径可能把 `object_anomaly=1` 但 `image_anomaly=0`（该视角看不到缺陷）的 9,080 张图算作 normal；119,880 − 119,760 = 120 则可能来自某个未随 metadata 发布的 specimen。
**这两个差异不影响 E4 的可用性，但不得在论文中把论文数字当作实测数字引用。** 本审计一律以 **metadata 实测值**为准。

---

## 11. 缺陷类型（修正 E4-0C 的一条结论）

**E4-0C 结论**：官方 metadata **无 `defect_type` 字段** ⇒ `mean_dprime` 退化为池化 d′。
**修正（FACT）**：字段确实不存在，但 **defect type 可从 NG 的 `object_name`（specimen 目录名）恢复**：

```text
Good specimen 目录名 : 3 位数字（000, 001, …）
NG   specimen 目录名 : <type>[_<type>…]_<instance>_<index>
    例: scratch_1_000            -> scratch
        damage_1_hole_5_scratch_1_046 -> damage + hole + scratch   (多缺陷)
```

**全局 defect-type token 分布**（test，按图像数）：

| token | 图像数 |
|---|---:|
| damage | 27,480 |
| hole | 22,200 |
| scratch | 20,400 |
| abrasion | 3,600 |
| bending | 1,800 |

**多缺陷 specimen**：1 类 49,080 图 / 2 类 6,000 图 / 3 类 4,800 图。
**Bird**：`scratch` / `hole` / `damage` 各 2,640 图（50 个 NG specimen）。

⇒ **`mean_dprime` 可以按 defect type 分层**（与 Exp16 口径一致），无需退化。但**多缺陷 specimen** 使分层口径需要**新预注册**（一图属多类时如何归属）。

**`detectable` 字段存在**（修正 E4-0C 的 UNKNOWN）：test 中 `True` 47,855 / `False` 2,945 / 空 33,080（= 24,000 normal ＋ 9,080 无可见缺陷 NG）。

**可见缺陷比例**：Bird NG 中 `image_anomaly=1` = **4,550 / 6,000 = 75.8%** —— 与论文"平均约 75% 异常图可检出"**高度一致** ✓

---

## 12. Frozen External Protocol Feasibility（§22）

**E4-A（冻结协议原义复现，Original / Adaptive B2 / X6c + Exp16 的 4 个合成扰动）**

| 项 | 判定 | 依据 |
|---|---|---|
| test normal 可得（μ_g/σ_g 与 `shift_good` 底图） | OK | Good specimens 有本地 GT（label 0） |
| test defect 可得 | OK | NG ＋ `image_anomaly` ＋ `*_mask.png` |
| 4 个冻结合成扰动可施加 | OK | `brightness_0.7/1.3`、`gamma_0.7/1.3`（项目自身实现） |
| `ΔR` / `d′` / `Δd′` / PASS / catastrophic / tail | OK（Δd′ 需 defect-type 归属口径预注册） | §11 |
| 是否需要改 X6c / C2 / C6 / 阈值 | **NO** | 全部为 score-level；本审计未触碰任何方法定义 |

### **E4-A 是否可运行？→ `YES`**

---

## 13. Real Illumination Protocol Feasibility（§23）

| 项 | 判定 |
|---|---|
| `fixed specimen + fixed view + illumination 01–10` 对 Good 成立 | **YES**（§7：Good pairing 1.000） |
| 同上对 NG 成立 | **YES**（§7：NG pairing 1.000） |
| 官方是否定义哪个 illumination 是 `regular/reference` | **NO** |
| **`REFERENCE ILLUMINATION = UNDEFINED`** | 论文正文未枚举 10 档光照的物理定义（仅 Fig. 10）；contact sheet 目视也**未见明显的"常规良好照明"档**（全部为暗背景定向光） |

⇒ **禁止擅自指定 illumination 01 为 reference。**
后续 **E4-B** 优先考虑（本轮**不冻结**其指标）：
`across-illumination dispersion` / `pairwise score variation` / `worst illumination deviation` / `illumination response curve`。

**E4-B 是否可构造？→ `YES`（数据结构支持；指标待预注册）**

---

## 14. Runtime Sampling（§24，**无 GPU**）

**PatchCore unit 数（估计）**

| 范围 | 组合 | GPU units（Original + Adaptive B2） | X6c |
|---|---|---|---|
| E4-A Mini | 3 categories × 3 seeds | 3×3×2 = **18** | 0（score-level） |
| E4-A Full | 10 categories × 3 seeds | 10×3×2 = **60** | 0 |

**每 unit 图像量（Bird，全部 12 views 全用）**：train/good 特征提取 3,600 ＋ `clean_good` 2,400 ＋ `clean_defect` 6,000 ＋ `shift_good` 4×2,400 = 9,600 → **≈ 21,600 图 / unit**。

**粗估（lower / upper）—— 明确说明：这不是 M²AD 实测，只是按 Exp16 历史速度的换算**

- Exp16 历史：MVTec 单 unit ≈ 46–111 s，图像量 ≈ 370（209 train ＋ ~163 test/scoring）→ **≈ 0.12–0.30 s/image**（256 px 输入）。
- 按同速率线性外推：**M²AD 单 unit ≈ 45–110 min**；E4-A Mini（18 units，3 workers）≈ **4.5–11 GPU-h**；E4-A Full（60 units）≈ **15–37 GPU-h**。
- **若改用固定 view（10 illuminations，图像量降至 ~1/12 ⇒ ≈1,800 图/unit）**：单 unit ≈ **4–9 min**，Mini ≈ 0.4–0.9 GPU-h，Full ≈ 1.2–3 GPU-h。
- **上限风险**：上述假设输入被 resize 到 256 px。若沿用 1024 px 原生输入，显存与时间都会显著上升（PatchCore embedding 与 batch 数随分辨率增长）。
- **本估算不得当作正式 runtime**；**正式 runtime 必须等 E4-D1 的 Original-only GPU smoke 实测**。

**显存策略**：沿用 Exp16 结论（3 workers peak ≈ 14.7 GB；4 workers 曾 OOM）。**E4-D1 smoke 默认 1 worker**。

---

## 15. 最大风险

**最大工程风险**：M²AD 单 category 的图像量（18,000 scoring + 3,600 train）比 MVTec 单类大约 **58 倍** ⇒ 若不加限制，E4-A 的 GPU 成本可能达到 **15–37 GPU-h**（Full）。缓解方向（**均需人工批准 + 新预注册**）：固定 view / 子采样 specimen / 提高到 512 px 以外更低分辨率。

**最大科学风险**：`REFERENCE ILLUMINATION = UNDEFINED`（§13）。M²AD 的 10 档光照是**方向光组合**而非"良好照明 vs 低照度"两档，因此**E4-B 无法用"相对常规照明的退化"来定义**；只能改用 across-illumination 离散度类指标 —— 这与 Exp16 冻结 ΔR 的语义**不同**，须作为**新协议**另行预注册。

---

## 16. 输出产物

| 文件 | 说明 |
|---|---|
| `docs/E4_M2AD_INTEGRITY_AUDIT.md` | 本文件 |
| `docs/E4_M2AD_LOCAL_INVENTORY.csv` | 真实文件树清单（12,910 行，Bird） |
| `docs/E4_M2AD_PAIRING_AUDIT.csv` | same-specimen + same-view × I01–10 存在性 |
| `docs/E4_M2AD_ILLUMINATION_COVERAGE.csv` | 每档光照的 Good/NG 计数 |
| `docs/E4_M2AD_VIEW_COVERAGE.csv` | 每 specimen 的 view × illumination 完整度 |
| `docs/E4_M2AD_MASK_AUDIT.csv` | 20 张 NG → mask 匹配审计 |
| `docs/E4_M2AD_STRUCTURE_MISMATCH.csv` | 真实树 vs metadata 差异（0 real ＋ 455 INFO） |
| `docs/E4_M2AD_AUDIT_SUMMARY.json` | 机器可读汇总 |
| `docs/figures/e4_m2ad_pairing_sanity.png` | §20 contact sheet（**唯一**图片产物） |
| `scripts/e4d0_m2ad_audit.py` | 可复现审计脚本 |
| `scripts/e4d0_contact_sheet.py` | contact sheet 脚本 |
| `data/m2ad/DATASET_LOCAL_INFO.md` | 本地下载记录（`data/` 已 gitignore） |

---

## 17. 对论文证据链的贡献

- 本审计**不产生方法证据，不改变 Exp16 的任何数字或结论，未运行任何模型**。
- 推进的证据链格子：**E4 的「数据集可用性」门** —— 从「静态协议审计通过」推进到
  **「真实下载的数据在结构 / parser / 光照覆盖 / view 覆盖 / same-view 配对 / GT mask / 官方 split 泄漏控制上全部实测通过」**。
- **下一格**：`E4-D1 Original-only GPU smoke`（单 category × 单 seed × Original；同时取得正式 runtime 实测）。

## 18. 本轮未做（明确记录）

未运行 GPU；未运行 PatchCore / Original / B2 / X6c；未修改 X6c / C2 / C6 / 任何阈值；
未创建 experiment results 目录；未下载 164 GB 原始数据；未 push 本轮 commit。
