# E20-A · M²AD 类别与场景扩展预检（追加任务）

**日期**：2026-10-15
**论文阶段**：`FINAL BENCHMARK AND PAPER CONVERGENCE`（E19 已完成，论文资产已冻结；本任务为**追加的路线 A 资源盘点**，不新增 GPU、不下载、不训练）
**性质**：dataset / resource audit（不是实验结果；本轮**未运行任何模型**）
**状态**：完成（0 GPU · 0 下载 · 0 解压）
**范围约束**：执行 AI 不决定新增类别、不改方法、不启动训练；候选选择权在研究负责人
**并行任务**：路线 B 的 MVTec AD 2 资料核实已由并行会话完成（`results/e20/mvtec_ad2_preflight.md`，commit `0664943`，含功效模拟 `auroc_power_sim.json`）；本报告 Task 4 直接引用其结论。

---

## 0. 论文导航（先于执行核对）

```text
📍 PAPER STAGE
   METHOD SEARCH CLOSED → FINAL BENCHMARK → PAPER CONVERGENCE（E19 完成）。
   论文主张 A1/A2/A3 的资产已定稿；本任务回答"路线 A（M²AD 扩类别/扩场景）是否可以
   在低成本、不破坏冻结协议的前提下扩大证据范围"。

🎯 PAPER CLAIM UNDER TEST
   A1 的泛化边界：R1-ILA 的领先是否只对既有的 Bird/Car/Cube 3 类成立？
   A3 的机制边界：illumination-shift 与收益的关联（E16 "consistent with"）能否被新类别检验？
   （注：A1/A2/A3 的原文主张与边界见 results/e19/README.md §2，本轮不修改。）

🧪 WHY THIS TASK
   这是**资源盘点**，不是实验：任何新实验都会消耗 GPU / 网络 / 磁盘并触及冻结协议外围，
   必须先给出"数据是否真实存在、协议是否兼容、成本几何、失败如何解释"的事实清单，
   供研究负责人对路线 A/B 做裁决。

🚦 EXIT CONDITIONS
   支持（允许提交批准）：≥1 个候选同时满足 ①数据真实存在 ②协议兼容 ③成本在磁盘/时间预算内 → 提交人工批准；
   暂停：候选存在但存在未核实的阻断项（如网络可达性、磁盘余量）→ 列为"批准前必须复核"；
   否定：不存在可行候选 → 如实报告，建议改走路线 B。
   本轮不产生"科学成败"结论，只产生资源事实与候选方案。
```

---

## 1. Task 1 — 现有 M²AD 数据资源盘点

> **规则**：以下每条均来自文件、官方 metadata 或历史实验记录，**不由目录名推测**。
> 盘点脚本：`scripts/e20a_inventory.py`、`scripts/e20a_crosscheck.py`、`scripts/e20a_candidates.py`（只读）。

### 1.1 官方 metadata（本地已有：全部 10 类）

来源：`data/m2ad/jsons_extract/meta_unsupervised.json`（官方 `jsons.zip` 解出，51 MB，SHA256 记录于 `data/m2ad/DATASET_LOCAL_INFO.md`）。

| category | train 图 / specimens | test 图 / specimens | test 中 NG(anom=1) | detectable=True | detectable=False | 官方顺序 |
|---|---:|---:|---:|---:|---:|---:|
| Bird | 3600 / 30 | 8400 / 70 | 4550 | 4185 | 365 | 1 |
| Car | 3600 / 30 | 8400 / 70 | 4850 | 4541 | 309 | 2 |
| Cube | 3600 / 30 | 8400 / 70 | 5530 | 5308 | 222 | 3 |
| Dice | 3480 / 29 | 8400 / 70 | 4490 | 4170 | 320 | 4 |
| Doll | 3600 / 30 | 8400 / 70 | 5010 | 4730 | 280 | 5 |
| Holder | 3600 / 30 | 8280 / 69 | 4770 | 4616 | 154 | 6 |
| Motor | 3600 / 30 | 8400 / 70 | 6000 | 5893 | 107 | 7 |
| Ring | 3600 / 30 | 8400 / 70 | 5480 | 5375 | 105 | 8 |
| Teapot | 3600 / 30 | 8400 / 70 | 5070 | 4761 | 309 | 9 |
| Tube | 3600 / 30 | 8400 / 70 | 5050 | 4276 | 774 | 10 |

- 每类 12 views（000–330，30° 步进）× 10 illuminations（01–10）。
- `detectable` 字段三值：`true` / `false` / `""`（JSON 的 bool 与空串）。**E11+ 线的评测过滤 = `detectable != False`**（保留 true 与空串），与 `scripts/e11_p2_runner.py::official_metrics` 一致。
- **10 类 metadata 全部在本地**；本地**无**任何其他 7 类的像素。

### 1.2 本地像素数据（实际下载 / 缓存）

| 资产 | 内容 | 规模 | 性质 |
|---|---|---|---|
| `data/m2ad/extracted/Bird/` | Bird 原图（1024×1024，train 3600 + test 6000 + GT 910 文件） | 15 GB | **唯一本地原图** |
| `data/m2ad/e7_cache/l0_448/` | DINOv2 / F0 / R1-ILA 主输入（448, BICUBIC, WebP-lossless） | 2.8 GB（Bird 1.2 G / Car 791 M / Cube 848 M） | 派生缓存 |
| `data/m2ad/e7_cache/l0_256/` | PatchCore 线输入（256, BICUBIC） | 859 MB | 派生缓存 |
| `data/m2ad/e7_cache/{clahe_mild,clahe_medium,retinex,retinex_clahe_mild}` | 历史预处理缓存 | ~7 GB | 派生缓存（E7/E9 线） |
| `data/m2ad/e9_cache/*` | 历史归一化缓存（gamma08/grayworld/homomorphic/shades6） | 1.1 GB | 派生缓存 |
| `data/m2ad/e8_gt/` | Car/Cube 的 GT masks；Bird mask 在 extracted 内 | 346 png | 官方 GT 副本 |

**缓存覆盖（经逐文件与官方 metadata 交叉核对，`inventory_crosscheck.json`）**：

| 类别 | l0_448 Good | l0_448 NG | 覆盖 view | 覆盖 split |
|---|---:|---:|---|---|
| Bird | 4400 = train 3600（30 specs × 12 views × 10 illum）+ test 800 | 2000（test only） | Good 12 views；**test NG 仅 090/120/240/330** | 官方 train/test |
| Car | 2000 = train 1200（**10 specs** × 12 views × 10 illum）+ test 800 | 2000（test only） | 同上 | 官方 train/test |
| Cube | 2000（同 Car） | 2000（同 Car） | 同上 | 官方 train/test |

- cross-check 结果：**0 unmatched**（每张缓存图都能在官方 metadata 中找到 (specimen, view, illumination) 记录）。
- Car/Cube 的 train → Good 只有 10 个 specimens = **bank pool 规则 `sorted(train_specs)[::3][:10]`**；Bird 是全 30 个（E7 时代自原图生成的历史差异）。
- test 的 4 views = E8 冻结协议 `VIEWS = [120, 090, 240, 330]`。

### 1.3 当前 E16/E18 实际覆盖（评测口径）

- test = official test ∩ view **120** ∩ (`image_anomaly==0` ∪ `detectable!=False`)：
  **Bird 653（320 ok + 333 ng）、Car 656（300 + 356）、Cube 678（260 + 418）**（与 E16/E18 文档完全一致）。
- bank = `random.Random(20261009+seed).sample(pool_1200, 300)`；R1-ILA 训练 train/val = 240/60（bank 内切分）。
- **结论：当前论文主结果的真实数据覆盖 = 3 类 × 1 view（120）× 10 光照 × 约 660 张 test/类。**

### 1.4 未纳入 E16/E18 的数据（扩展空间，含“已缓存但未用”）

1. **同一 3 类的另外 11 个 view**：l0_448 中 train 部分有 12 views；**test 部分已缓存 4 views（090/120/240/330）中仅 120 被使用** → 另 3 个 view 的 test（Bird 2023 / Car 2029 / Cube 2051 张 kept，见 `expansion_candidates.csv`）**已在本地、0 获取成本、从未用于主结果**。
2. **同一 3 类的 8 个未缓存 view**（000/030/060/150/180/210/270/300）：Bird 原图在本地可重建缓存；Car/Cube 原图不在本地（需重新 streaming）。
3. **其余 7 个独立类别**（Dice/Doll/Holder/Motor/Ring/Teapot/Tube）：metadata 齐备、像素零；每类 view120 kept 639–688 张。
4. 光照条件本身（10 档）**无遗漏**——E16 的 view120 test 已含全部 10 档（E18-P2 已做逐档分解）。

### 1.5 是否能复用当前 F0 / R1-ILA 评估代码

| 组件 | 可复用性 | 依据 |
|---|---|---|
| 数据加载 `e4d1_m2ad_loader` | ✅ 类别无关（`load_records(category)`；`require_exists=False` 时仅依赖官方 metadata） | 代码阅读 |
| scope/bank 规则 `e8_common.build_scope` + `bank_for` | ✅ 类别无关；10 类 train specimens ≥ 29 个，`[::3][:10]` 规则对全部类别有效（已对 10 类验证） | `inventory` 复算 |
| 新类别取数 `e8_fetch.py` | ✅ 完全参数化（`--categories X`），streaming selective extraction + 硬 gate A1–A5；支持 resume | 代码阅读 |
| F0 / R1-ILA 评估 `e16_qualify.run_cell` | ⚠️ 主体类别无关（load/test/训练/诊断全通用），但含 **历史复现校验 `hist[(cat, seed)]`（E16 专用）** → 新类别必须写**新脚本变体**（去掉历史断言），**不得修改 e16_qualify.py 本体** | 代码阅读 |
| 主表/统计 `e18p2_*` / `e19_*` | ✅ 仅消费结果 JSON/CSV，与新类别解耦（如需要新表则新增，不覆盖历史） | 代码阅读 |

### 1.6 关键资源事实（决定可行性的硬数字）

| 项 | 值 | 来源 |
|---|---|---|
| 磁盘余量 | **16 GB**（/root/autodl-tmp，50 G 分区，35 G 已用） | `df` 实测 2026-10-15 |
| 单类 zip 体积 | **14.9–17.6 GB**（Car/Cube 历史实测） | `docs/E8_FROZEN_PROTOCOL.md` §2.8 |
| 全量解压单类 | ~15 GB → **磁盘不可行**（剩余 16 GB 装不下 zip+解压峰值） | E18 run_plan §P1 |
| streaming selective extraction 实测 | **Car+Cube 共 ≈45 min（≈22 min/类）**，只落缓存不落原图 | `results/e8/README.md` §11 |
| 新类别缓存磁盘 | ≈0.8 GB/类（仅 l0_448）或 ≈2.7 GB/类（4 kinds + masks） | 本地 Car/Cube 实测外推 |
| hf-mirror 带宽 | ≈16 MB/s（2026-10-08 实测）；**可达性需在批准执行前复核** | memory / E4-D0 |

---

## 2. Task 2 — 三类扩展机会（区分证据边界）

> **反混淆声明（任务明确要求）**：更多图像、更多视角、更多光照条件 **≠ 更多独立类别**。
> 下表把每个候选的"证据类型"单独标注；"type A"才是类别级泛化证据。

### A. 新增独立类别（类别级泛化证据）

- **候选**：Dice / Doll / Holder / Motor / Ring / Teapot / Tube（7 选 N，**由研究负责人决定**）。
- 最有依据的顺序 = E8 协议 §4 Extension A **预注册的机械顺序 `Dice → Doll → Holder`**（按官方字母序取下一个，从不按性能选择）。其余 4 类无预注册依据，仅作补充。
- **能回答**：R1-ILA 的收益是否跨独立类别成立？illumination-shift（E16 D 指标）与收益的关联是否在新类别复现（样本外检验）？
- **证据边界**：解决了"只有 3 类"的泛化质疑；**不解决**"单数据集（M²AD）"的质疑（E19 limitations #5 仍成立）；不解决"单 view"问题（若沿用 view120）。

### B. 同一类别的新增视角 / 照明条件

| 子类 | 现状 | 证据边界 |
|---|---|---|
| B1 另 3 个**已缓存** view（090/240/330） | l0_448 本地 100% 就绪，0 成本 | 同类别几何视角稳健性；**不是类别扩展** |
| B2 另 8 个未缓存 view | Bird 可本地重建（原图在）；Car/Cube 需重新 streaming | 同上；Bird 12-view 是单类别视角上限 |
| B3 新的照明条件 | **不存在"未覆盖的照明"**：10 档全部在缓存中、已被 E16 使用 | 照明轴无新数据可加；只能改**分析口径**（见 C2） |

### C. 新增测试条件 / 数据划分

| 子类 | 内容 | 证据边界 |
|---|---|---|
| C1 4-view 汇测（090/120/240/330） | test 从 ~660 → ~2700 张/类，0 获取成本 | 更大 test（统计功效）+ 跨视角；**不增加类别** |
| C2 per-illumination 分析强化 | 现有 10 档分组的更深分解（worst-illumination AUROC、逐档 CI） | 描述性/稳健性视角；**不是新数据** |
| C3 光照留出划分（训练/测试光照分离） | **会改动 R1-ILA 的训练输入定义**（R1 目前用合成光照增强 + normal-only，无真实光照分组） | 属"改变方法训练协议"——**当前阶段不建议**，需新预注册；仅列为远期选项 |
| C4 新 split（如 specimen 重抽） | 官方 split 为 specimen 级冻结，重抽会破坏与 E16/E18 的可比性 | **不建议** |

---

## 3. Task 3 — 路线 A Mini Experiment 候选（3 个，已排序）

> 以下方案**均未执行**。共同前提：F0 = frozen DINOv2-B + 1-NN-max（复用）；R1-ILA = `results/e17/final_method.json` 冻结参数（禁止调参）；
> 指标 = AUROC / AP / d′ + E16 的诊断（A–E）；统计 = 3 seeds + per-seed 符号 + test-image paired bootstrap。
> 机器可读版：`mini_experiment_proposals.csv`。

### M1（rank 1）— 新类别泛化（Dice/Doll/Holder 候选，类别数由人工批准）

| # | 项 | 内容 |
|---|---|---|
| 1 | 对应论文主张 | **A1 扩展**：R1-ILA 增益不限于 Bird/Car/Cube；**A3**：illumination-shift 剂量-反应在新类别复现 |
| 2 | 为什么现有证据不足 | A1/A2/A3 全部基于 3 类；审稿人可质疑"3 类巧合 / 对 Bird/Car 场景过拟合" |
| 3 | 数据真实存在且协议兼容 | metadata 本地（10 类）；像素经 `e8_fetch.py` streaming（已验证管线 + 硬 gate A1–A5）；bank/test/labels 规则不变 |
| 4 | F0 vs R1-ILA 公平对照 | 同 bank 抽样规则（同 seed）、同 view120 test、同 1-NN-max、同 `official_metrics` 过滤；**唯一改动 = 去掉 E16 历史复现断言（新脚本变体）** |
| 5 | 主要指标与统计 | AUROC/AP/d′；3 seeds；mean Δ + 3/3 符号一致性 + paired bootstrap（test-image 重采样） |
| 6 | 预算 | 数据 ≈22 min/类（网络，CPU）；磁盘 ≈0.8 GB/类（仅 l0_448）；**GPU ≈4.5 min/cell × 3 seeds × 1–2 类 ≈ 14–28 min**（E16 实测 264.9–276.3 s/cell 含 F0+R1+diagnostics） |
| 7 | GO / HOLD / STOP | GO：3/3 seed ΔAUROC>0 且 seed-mean ≥ +0.005，且 D 指标排序与收益方向一致；HOLD：seed-mean ∈ [0,+0.005) 或 2/3 正；STOP：seed-mean < 0 |
| 8 | 失败解释（禁止挽救） | Δ≈0 且 frozen baseline 高 → **与 E13-B2 的 baseline/headroom 依赖一致**（边界条件，不是调参目标）；shift 大但收益负 → 假说受损，如实报告；**禁止**改参数 / 挑 seed / 改标签 |

**风险**：hf-mirror 可达性需复核；streaming/gate 失败即停机报告；类别选择必须人工批准（执行 AI 不得自选）。

### M2（rank 2）— 4-view 测试扩展（Bird/Car/Cube，零数据获取）

| # | 项 | 内容 |
|---|---|---|
| 1 | 对应论文主张 | **A1/A3 稳健性限定**：结论非单一 view120 假象（跨视角） + 更大 test 的统计功效 |
| 2 | 为什么现有证据不足 | E19 limitations #5 明列"单 view"缺口；视角变化是审稿标准关切 |
| 3 | 数据真实存在且协议兼容 | **100% 已在本地**（l0_448 的 4-view test，每类 Good 800 + NG 2000）；specimen 级 split 无泄漏；VIEWS 为 E8 冻结定义 |
| 4 | 公平对照 | 仅把 test 从 view120 扩到冻结 4 views；bank/训练/score 不变 |
| 5 | 主要指标与统计 | 4-view pooled AUROC/AP + per-view ΔAUROC（胜 view 数）+ illumination × view 分层 |
| 6 | 预算 | **数据 0 成本**；GPU：pilot（1 seed × 3 类）≈30–40 min，full（3 seeds）≈1.5–2 h（需先 smoke 实测 test 提取扩展系数） |
| 7 | GO / HOLD / STOP | GO：pooled seed-mean ΔAUROC ≥ view120 值的 50% 且 ≥6/8 类别-view cell 为正（Bird+Car）；HOLD：正但视角异质；STOP：pooled < 0 或系统性反向 |
| 8 | 失败解释 | 若仅 view120 为正 → **如实写成 scope limitation**；**禁止**重选 view 恢复符号 |

**风险**：runtime 上升（需 smoke 校准）；写作时必须明确"这不是新类别"。

### M3（rank 3）— Bird 12-view 全视角（本地原图，零网络）

| # | 项 | 内容 |
|---|---|---|
| 1 | 对应论文主张 | A1/A3 的**最大角度覆盖**稳健性限定（单类 12 views） |
| 2 | 为什么现有证据不足 | 同 M2（单 view）；12-view 是单类别可得的几何泛化上限 |
| 3 | 数据真实存在且协议兼容 | Bird 原图本地；需本地生成 8 个未缓存 test view 的 l0_448（≈5600 图，CPU，必须走 E8 冻结写缓存代码路径） |
| 4 | 公平对照 | 同 F0/R1-ILA、同 bank；test = 12 views × 10 illum |
| 5 | 主要指标与统计 | 12-view pooled + per-view Δ + illumination 分解 |
| 6 | 预算 | 缓存构建 ≈30–60 min（CPU）；GPU ≈45–60 min（3 seeds，需 smoke 实测）；磁盘 ≈2.2 GB |
| 7 | GO / HOLD / STOP | GO：12-view seed-mean 与 view120 同号且 ≥9/12 views 为正；HOLD：整体正但 1–3 views 负；STOP：多数 view 反向 |
| 8 | 失败解释 | 写成"类别特异视角敏感性"；禁止重选 view |

**风险**：仅覆盖 1 类（不是类别扩展）；缓存生成须复用冻结代码路径。

### 排序说明（证据增益 × 成本 × 风险）

| 方案 | 证据增益 | 实现成本 | 风险 | 备注 |
|---|---|---|---|---|
| M1 新类别 | **最高**（类别级泛化，直击 A1 边界） | 中（网络 22 min/类 + GPU 14–28 min） | 中（网络可达性/gate） | E8 预注册顺序为其提供依据 |
| M2 4-view | 中高（跨视角 + 功效） | **最低**（0 获取，纯 GPU） | 低 | 唯一 0 网络选项 |
| M3 Bird 12-view | 中高（最大视角，但单类） | 中低（本地 CPU + GPU） | 低 | 与 M2 同族，覆盖窄 |

**组合建议（供裁决）**：M2（先行，0 成本）→ M1（批准后隔夜执行）→ M3（可选）。三者互不冲突、互不覆盖既有产物。

---

## 4. Task 4 — 路线 A 与路线 B（MVTec AD 2）协同对比

> **路线 B 的本轮资料核实已由并行的 E20-B 任务完成**：`results/e20/mvtec_ad2_preflight.md`（commit `0664943`）
> ＋ `results/e20/auroc_power_sim.json`（功效模拟）。下表路线 B 列以 E20-B 已核实事实为准；E20-B 的裁决 = **HOLD**
> （前置条件：①人工完成官方表单/许可并获取数据；②批准其 §5.3 D1–D5 协议决策）。

| 维度 | 路线 A（M²AD 扩展） | 路线 B（MVTec AD 2，据 E20-B） |
|---|---|---|
| **证据增益** | 类别级 / 视角级泛化（同数据集内）；**不解决**跨数据集质疑 | **跨数据集**（直击最强攻击面）；公开部分可本地算 image-level AUROC/AP（非官方指标） |
| **统计分辨力（新增行）** | **高**：4,000 次 bootstrap 对照——M²AD 现状噪声半宽 ≈0.017–0.019；M2 扩到 4-view（≈2700 图/类）后预计压到 ≈0.009 | **受限**：AD2 单场景 ΔAUROC 半宽 **≈0.04–0.07**（远大于效应本身）；三场景合并 ≈0.03 → 必须 pooled 为主分析 |
| **致命约束** | 无结构性阻断；主要约束 = 网络与磁盘 | 光照条件混合且无标注 → 逐光照/ΔR 分解**本地不可做**；`detectable` 类缺陷类型分层亦不可做；TESTpub good 仅 20–35 |
| **获取成本** | 0（M2/M3）～22 min/类（M1） | 人工浏览器表单/许可（不可自动化）；三场景压缩 ≈3.5 GB，磁盘峰值 ≈7 GB（16 GB 余量可容；全量 30.4 GB 不可行） |
| **资源需求** | GPU 0.5–2 h；磁盘 0–3 GB | GPU ≈45–60 min（3 seeds × 3 场景，E20-B 估算）；显存 <4 GB；**需新加载层 + D1–D6 协议决策** |
| **与照明鲁棒性研究契合度** | 高（M²AD 天然 10 档真实光照；现有 A3 分析可直接扩展） | 中（真实光照变化，但公开条件不可分离；合成光照 robustness 指标可迁移） |
| **对论文说服力** | 中-高（补多类别/跨视角证据；单数据集限制仍在） | 中-高（若协议适配成功；但功效弱 + 协议语义降级需如实写入限制） |
| **推荐动作** | M2 可立即批准执行；M1 待网络复核 + 类别批准 | 待人工下载 + D1–D5 批准；已明确无 STOP 条件 |

**互补建议（供裁决）**：两路线不互斥。执行顺序建议 **A-M2（0 成本、增强主结果统计分辨力）→ B（数据到货后）→ A-M1**。
**共同底线**：两条路线都不改变 R1-ILA 结构/参数（E18–E20 冻结）；路线 B 若实施需要**全新的冻结预注册**（bank 规则、分辨率、评估口径），属于新的协议设计，不是现有协议的平移。

---

## 5. 产物清单与本轮约束执行记录

| 产物 | 说明 |
|---|---|
| `results/e20/m2ad_expansion_review.md` | 本报告 |
| `results/e20/inventory_assets.json` | 全资产盘点（含 10 类 metadata 摘要） |
| `results/e20/inventory_summary.csv` | 缓存 × 类别 × Good/NG 汇总（58 行） |
| `results/e20/inventory_crosscheck.json` | 缓存 vs 官方 split 逐视图核对（0 unmatched） |
| `results/e20/category_view_coverage.csv` | 10 类 × {train,test} × 12 views 官方计数矩阵（240 行） |
| `results/e20/expansion_candidates.csv` | 类别级扩展候选表（10 行，含成本/证据类型） |
| `results/e20/mini_experiment_proposals.csv` | M1/M2/M3 方案元数据（含 GO/HOLD/STOP 判据） |
| `scripts/e20a_inventory.py` / `e20a_crosscheck.py` / `e20a_candidates.py` | 只读盘点脚本（可重跑） |

**约束执行记录（本轮全程）**：
- ✅ 0 GPU 训练 / 0 模型运行；✅ 未下载、未解压任何数据（未发起任何网络请求）；
- ✅ 未修改任何冻结方法/脚本本体（仅**新增** `scripts/e20a_*.py`）；
- ✅ 未覆盖任何历史结果；未自动选择候选或启动实验；
- ✅ 所有判断均注明来源（文件 / 官方 metadata / 历史实验记录）。

---

## 6. 待批准事项（NEXT，不自动执行）

1. **路线排序裁决**：先 M2（0 成本）还是直接批准 M1（含网络）？
2. **M1 类别选择（必须人工指定）**：E8 预注册 `Dice → Doll → Holder` 是既有依据；是否采用、选 1 类还是 2 类，由研究负责人决定。**执行 AI 不自选**。
3. **批准 M1 执行前的复核项**：① `hf-mirror.com` 可达性/带宽复测；② 新类别 streaming 的磁盘峰值预估（基于 §1.6，当前 16 GB 余量下仅落缓存方案可行）；③ 新脚本变体的预注册（去掉 E16 历史断言后的等价保证）。
4. **E20-B（路线 B 资料核实）**：已完成（`results/e20/mvtec_ad2_preflight.md`，commit `0664943`）——MVTec AD 2 结构/许可/三场景/功效模拟 + mini-experiment 设计（D1–D6 决策清单）；其待批准事项见该报告 §7。**若同时批准 M1 与路线 B，执行顺序建议：A-M2 → B → A-M1**。
5. 任何执行都需要预先登记：实验编号、冻结 SHA、输出目录、判据；**不得修改 R1-ILA 参数**。

---

## 附：事实 / 解释 / 假设 分离

- **事实**：§1 全部数字（来自文件与官方 metadata）；§1.6 资源实测值；E8/E16 的历史运行记录数字。
- **解释**：M1/M2/M3 的成本外推与排序；路线 A/B 对比表中的判断（"证据增益中-高"等）。
- **假设**（未验证）：
  - (H1) 其余 7 类的 per-specimen 图完整性 ≥ 官方 metadata（Bird 已实测 0 mismatch；新类别未实测，由 fetch gate A1–A5 覆盖）；
  - (H2) hf-mirror 当前仍可达且带宽不低于 16 MB/s（2026-10-08 实测，需复核）；
  - (H3) 4-view 扩展的 GPU 成本约为 view120 的 2.4–4×（需 smoke 实测校准）。
