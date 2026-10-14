# E18 — Execution Plan (P0 产出，**尚未执行**)

日期：2026-10-14 · HEAD `28d02b3` · **本轮 0 GPU 已消耗**

## 1. 资源状态

| 项 | 值 |
|---|---|
| GPU | RTX 3090 24 GB，空闲（0% / 1 MiB / 0 进程） |
| disk free | 16 GB（≥12 GB ✅） |
| 已完成（0 GPU） | `audit.md`、`compatibility_matrix.csv`、`protocol.md`、PatchCore Bird official 重算 |

## 2. 可直接复用（0 GPU）

| 行 | Bird | Car | Cube | 来源 |
|---|---|---|---|---|
| PatchCore-emb + unified scoring | 0.68454 / 0.72797 | 0.62247 / 0.59298 | 0.60857 / 0.66489 | E11-P2 |
| AnomalyDINO | 0.70214 / 0.72234 | 0.75684 / 0.78277 | — | E14 F1 |
| SoftPatch | 0.64396 / 0.57510 | 0.50219 / 0.55541 | — | E14 F2 |
| EfficientAD | 0.61206 / 0.62196 | — | — | E15 |
| F0 | 0.74725 / 0.78863 | 0.78552 / 0.84207 | 0.85410 / 0.91686 | E16（3 seed） |
| Ours = R1-ILA | 0.75786 / 0.79911 | 0.80511 / 0.85502 | 0.87010 / 0.92856 | E16/E17（3 seed） |
| **PatchCore (official)** | **0.69318 / 0.72304** | — | — | E7-1 P00 **official 重算（本次，0 GPU）** |

## 3. 不可直接比较

| 项 | 原因 |
|---|---|
| E7 线全部 image 指标 | `object_anomaly` 旧标签口径（已对 PatchCore 完成重算修正） |
| EfficientAD Bird-only | 不能称为三类别结果 |
| 各方法 native scoring 不同 | 方法自身定义，禁止为表面统一而改写 |

## 4. 缺失单元与补跑成本（实测外推，非猜测）

| # | 缺失单元 | 数量 | 依据 | ETA |
|---|---|---|---|---|
| A | PatchCore (official) · **Car + Cube** | 2 cells | E7-1 P00 Bird 实测 130.8 s/cell | **≈ 4.4 min** |
| B | AnomalyDINO (F1) · **Cube** | 1 cell | E14 实测 4 cells / 3.4 min | **≈ 0.9 min** |
| C | SoftPatch (F2) · **Cube** | 1 cell | 同上 | **≈ 0.9 min** |
| **小计（最小方案）** | | **4 cells** | | **≈ 6.2 min** |
| D | EfficientAD · **Car + Cube** | 2 cells | E15 Bird 实测 1750.6 s/cell | **≈ 58.4 min** |
| **合计（含 EfficientAD）** | | **6 cells** | | **≈ 64.6 min** |

含 smoke 与分析的保守总预算：**最小方案 ≈ 20 min**；**含 EfficientAD ≈ 80 min**（均在 2.5 h 硬顶内）。

## 5. 最小执行方案（推荐，Plan A）

```
E18-A  补跑 4 个缺失单元（A + B + C），screen 后台
       顺序: PatchCore/Car → PatchCore/Cube → AnomalyDINO/Cube → SoftPatch/Cube
       协议: 严格复用各方法在 E14/E7 中的原始配置（不改 k / coreset / soft weighting / checkpoint）
       标签: official image_anomaly + detectable!=False
       bank : E8.bank_for(pool, 0)（与主表一致）
       ETA : ≈ 6.2 min（含 smoke ≈ 12 min）
E18-B  生成主表 main_table.csv / main_table.md（6 行 × 3 类 × AUROC/AP，全精度）
E18-C  允许的消融：(1) PatchCore official vs unified-scoring 分解；
                    (2) illumination 01–10 分组 breakdown；
                    (3) Ours 3-seed 稳定性（复用 E16）
E18-D  配对 bootstrap（仅 per-image 分数可得者：F0 / Ours / PatchCore-Bird）
E18-E  4 张图 + decision_memo.md + README.md + git commit/push
```

**Plan A 不改变任何方法配置，纯粹补齐主表空单元。**

## 6. EfficientAD 的单独申请（Plan B，需人工批准）

**事实**：E15 在 Bird 上已按预注册 Stage-A 规则 STOP（AUROC 0.61206，ΔAUROC **−0.13519** ≪ −0.010），
Car/Cube 未运行。EfficientAD 是**唯一真正换 paradigm** 的基线（teacher-student），
其余 5 行都是 frozen-feature / memory-based。

**补跑 Car + Cube 的理由（仅供裁决）**：
1. 论文主表若列出 EfficientAD 却只有 1/3 类别，审稿人会要求完整；
2. 补跑**不是重新开启搜索**：配置已在 E15 冻结（PDN-small / 256 / 70000 steps），
   补跑只是把同一冻结配置应用到另外两类，**不做任何调参**；
3. 成本 ≈ 58.4 min GPU（实测外推），在预算内。

**反对理由**：E15 的 STOP 是预注册结论，补跑有"事后扩张"风险；且 EfficientAD 已在 Bird 上
大幅落后 F0，补跑预期不会改变结论。

**建议**：**先执行 Plan A**；EfficientAD 保持 `EARLY_STOP` 并在主表中以
"Bird only (early-stopped per E15 pre-registered rule)" 显式标注。是否补跑由人工裁决。

**不得**将缺失类别补成推测值，**不得**自动重新开启 EfficientAD 搜索。

## 7. 执行时的强制要求

- 正式补跑必须放 `screen` 后台（`screen -dmS e18_bench`），与聊天前端解耦。
- 持续输出 `completed / total`、百分比、`elapsed`、动态 `ETA`、当前单元。
- 单 unit >10 min 提供 heartbeat。
- 断点续跑：优先读 `results/e18/e18_partial.json`，不重复已完成单元。
- **禁止**修改 R1-ILA 或任何基线配置；**禁止**为结果挑选最好设置。

## 8. 本轮的 7 问回答（§8 完成标准）

1. **可直接复用**：F0、Ours（3 seed 全 9cell）、PatchCore-emb+unified、AnomalyDINO Bird/Car、SoftPatch Bird/Car、EfficientAD Bird、以及 PatchCore official **Bird**（本次 0 GPU 重算）。
2. **不具严格可比性**：E7 线 `object_anomaly` 口径（已修正 PatchCore）、EfficientAD 的 Bird-only、单 seed 基线未做 seed 方差。
3. **主表统一协议**：M²AD view120 / seed0 / 300-bank(`33381cd3004bfe68`) / official `image_anomaly`+`detectable` / image AUROC+AP / `official_metrics`。
4. **确实需要补跑**：PatchCore official Car+Cube、AnomalyDINO Cube、SoftPatch Cube（共 4 cells，≈6.2 min）。
5. **补跑成本是否值得**：**值得** —— 6.2 min 换一张完整主表，且不改变任何方法配置。
6. **EfficientAD**：**保留为早停证据**，主表显式标注 Bird-only；Car/Cube 补跑（≈58.4 min）需单独批准。
7. **E18 最小执行方案**：见 §5 Plan A。

---

# E18-P1 执行结果（2026-10-14 更新）

## P1 执行摘要

| # | 计划单元 | 状态 | AUROC | AP | 说明 |
|---|---|---|---|---|---|
| A | PatchCore official / Car | **BLOCKED** | — | — | 原图不可得（见 §P1-阻断） |
| A | PatchCore official / Cube | **BLOCKED** | — | — | 同上 |
| B | AnomalyDINO / Cube | **DONE ✅** | 0.8488866396761134 | 0.911341244132325 | 6/6 校验 PASS |
| C | SoftPatch / Cube | **DONE ✅** | 0.6934302539565698 | 0.7590847027020443 | 6/6 校验 PASS |

- 实际 GPU 耗时：**1.8 min**（2 单元，各 56.4 s / 50.7 s）。
- 产物：`results/e18/filled/{anomalydino_cube.json, softpatch_cube.json}`、`filled/per_image/*.csv`、
  `results/e18/filled_summary.csv`、`results/e18/e18_partial.json`、`results/e18/logs/p1_run2.log`。
- **未覆盖任何历史结果**；全部新产物写入 `results/e18/filled/`。

## P1-阻断：PatchCore official / Car + Cube

**结论**：**无法在本轮以逐字协议执行**（触发 §6 停止条款）。**已停止该 2 个单元，未做任何自动修补。**

根因链（全部有证据）：
1. `e7_pipelines.run_patchcore → e1b.predict_one → preprocess_for_model`
   消费**原图**：`TF.resize(TF.to_tensor(原图 1024), [256,256], antialias=True)`（**BILINEAR**）。
2. `data/m2ad/extracted/` **只有 Bird**；`DATASET_LOCAL_INFO.md` 记录 Car/Cube 从未下载，
   且 `bird.zip`（14.4 GB）解包后即删除以恢复磁盘。**当前磁盘仅余 16 GB。**
3. `l0_256` 缓存是 **BICUBIC + uint8** 产物，与 E7-1 P00 的输入**不等价**
   （Bird 20 图实测 max|Δ| = 0.056775 / mean|Δ| = 0.001245）→ **已拒绝替代方案**。

### 三个选项（需人工裁决，本轮不自动执行）

| 选项 | 动作 | 成本 | 风险 | 需要 |
|---|---|---|---|---|
| **A** | 从 `hf-mirror.com`（`ChengYuQi99/M2AD`）下载并解压 `car.zip` / `cube.zip` | 每个 ~14.4 GB zip + 15 GB 解压；磁盘仅 16 GB → 保留 zip 时峰值 ~29 GB（**不可行**）；流式解压峰值 ~15 GB（余 ~1 GB，**高风险**）；耗时 ~20 min/类 | 磁盘被写满 / 解压中断 | 人工批准（磁盘 + 网络） |
| **B** | 用 `l0_256` 缓存替代原图 | 0 | **已按实测拒绝** | — |
| **C** | 主表将 PatchCore official 的 Car/Cube **留空**（显式标注 BLOCKED） | 0 | 主表该行只有 Bird | 人工接受 |

**推荐**：**选项 C**。理由：论文主表若采用**统一 scoring 协议**，`PatchCore-emb + 1-NN-max`
（E11-P2，**三类别齐全、REUSE 完整**）已经提供了完全可比的 PatchCore 对照；
"PatchCore official scoring" 行属于 native-scoring 附录，其 Car/Cube 缺失可用
"not run (input images unavailable in the frozen harness)" 明确标注，不影响主表结论。

## P1 后的主表完备度

| 行 | Bird | Car | Cube | 完备 |
|---|---|---|---|---|
| PatchCore official scoring | ✅ 0.69318 | ❌ BLOCKED | ❌ BLOCKED | 1/3 |
| PatchCore-emb + unified scoring | ✅ 0.68454 | ✅ 0.62247 | ✅ 0.60857 | **3/3** |
| AnomalyDINO | ✅ 0.70214 | ✅ 0.75684 | ✅ 0.84889 | **3/3** |
| SoftPatch | ✅ 0.64396 | ✅ 0.50219 | ✅ 0.69343 | **3/3** |
| EfficientAD | ✅ 0.61206 | ❌ EARLY_STOP | ❌ EARLY_STOP | 1/3 |
| F0 | ✅ 0.74725 | ✅ 0.78552 | ✅ 0.85410 | **3/3** |
| Ours = R1-ILA | ✅ 0.75786 | ✅ 0.80511 | ✅ 0.87010 | **3/3** |
| **3/3 完备的行** | | | | **5/7** |
