# E26 · AnomalyDINO 官方实现完整基线（M²AD 本协议配置）

**日期**：2026-10-30
**论文阶段**：投稿前最后一轮补充 — 外部基线比较（收官轮，不追加实验）
**要验证的核心论断**：R1 相对一个独立发表的完整外部方法是否仍有竞争力。
**排除的替代解释**："R1 相对 F0 的提升只是因为我们自己的 baseline 太弱"。

---

## 1. 实验目的

用 **AnomalyDINO 官方实现**（WACV 2025, Oral；本地克隆 `third_party/AnomalyDINO` @ `b9d1c26`）
在 M²AD 10 类 × view120 × 3 seed 上运行，与本项目 F0 / R1 在**完全相同的 300-bank、相同测试集、
相同 detectable 过滤**下比较。不修改 R1 方法、超参数、划分与评分规则；不据新结果重选配置。

## 2. 数据与参数

| 项 | 值 |
|---|---|
| 类别 | Bird, Car, Cube, Dice, Doll, Holder, Motor, Ring, Teapot, Tube |
| 视角 / seed | view120 / 0,1,2（共 30 cell） |
| 参考集 | 每 (类, seed) 的 300 张 bank，与 F0/R1 **逐图相同**（`bank_hash` 30/30 一致） |
| 参考增广 | 8 角度旋转（官方 `agnostic` 默认，`src/utils.py:augment_image`） |
| backbone | `dinov2_vits14`（官方默认），448 分辨率 |
| 检索 / 聚合 | `L2_normalized` k=1 精确；图像得分 = 前 1% patch 距离均值（官方 `mean_top1p`） |
| masking | 关闭（官方对**自定义数据集**的 fallback：`src/utils.py:172-196` → `agnostic_no_mask`） |
| 输入 | `e7_cache/l0_448`（见 §6 偏差 A8） |
| 测试集 | 700 图/类（Holder 690），`image_anomaly` + `detectable != False` |

## 3. 实验过程

1. 核验官方仓库/论文/commit（`protocol.md` §3）。
2. 后端等价性验证：缩小设定下官方 `run_anomaly_detection`（faiss-CPU）vs 本脚本 torch 后端
   → **max|Δscore| = 1.82e-07**（`backend_equivalence.json`，10 图：5 正常 + 5 异常）。
3. 30 cell 全量运行（`scripts/run_anomalydino_m2ad.py --all --input cache`，`logs/run_all.log`）。
4. 逐 cell 身份核对：样本集合、标签、detectable、bank 哈希 → **30/30 `identity_ok=True`**，`fails=0`。
5. 分析（`scripts/e26_analysis.py`）：指标、宏平均、工件级配对 bootstrap（3000 次，seed 20261030）。

**实际 GPU 用时**：30 × (银行 40 s + 测试 97 s) ≈ 68 min；加 3 个原图 cell（76+115 s）与两次等价性验证 ≈ **1.4 GPU·h**（预算 4 GPU·h）。

## 4. 结果

### 4.1 10 类宏平均（每类先 3-seed 平均，再类间等权）

| 方法 | AUROC | AP |
|---|---:|---:|
| **AnomalyDINO（官方实现·本协议配置）** | **0.80357** | **0.85918** |
| F0（本文受控基线，DINOv2-B + max） | 0.83881 | 0.89396 |
| **R1（本文）** | **0.84641** | **0.90173** |
| R1 − AnomalyDINO | **+0.04284** [+0.02508, +0.06018] | **+0.04254** [+0.02726, +0.05870] |
| AnomalyDINO − F0 | −0.03525 | −0.03477 |

（区间为工件级配对 bootstrap，3000 次，seed 20261030，3 seed 共用同一组工件抽样。）

### 4.2 逐类别（3-seed 均值）

| 类别 | AD AUROC | F0 | R1 | Δ(R1−AD) | Δ(R1−AD) 95% CI |
|---|---:|---:|---:|---:|---|
| Bird | 0.66253 | 0.75131 | 0.77717 | +0.11464 | [+0.04857, +0.18627] |
| Car | 0.74053 | 0.79979 | 0.81102 | +0.07049 | [+0.01618, +0.12627] |
| Cube | 0.82765 | 0.86380 | 0.86645 | +0.03880 | [−0.00921, +0.09506] |
| Dice | 0.84429 | 0.90458 | 0.90794 | +0.06365 | [+0.02185, +0.11578] |
| Doll | 0.68349 | 0.71250 | 0.67401 | **−0.00949** | [−0.08204, +0.05992] |
| Holder | 0.92058 | 0.89321 | 0.90357 | **−0.01701** | [−0.05960, +0.02602] |
| Motor | 0.86205 | 0.82616 | 0.83423 | **−0.02782** | [−0.08761, +0.02691] |
| Ring | 0.85230 | 0.89326 | 0.91357 | +0.06128 | [+0.00082, +0.13844] |
| Teapot | 0.81345 | 0.87232 | 0.88962 | +0.07617 | [+0.03123, +0.12781] |
| Tube | 0.82878 | 0.87119 | 0.88651 | +0.05773 | [+0.00128, +0.11659] |

**R1 正向 7/10；其中 5 类 CI 排除 0（Bird/Car/Dice/Teapot/Tube），Ring 临界（下界 +0.0008）。
AnomalyDINO 正向 3/10（Doll/Holder/Motor），但这 3 类 CI 均跨零 → 无任何类别上 AnomalyDINO 显著优于 R1。**

### 4.3 开发 3 类 vs 扩展 7 类（仅描述性，不当作独立验证）

| 分层 | AD | F0 | R1 | Δ(R1−AD) | CI |
|---|---:|---:|---:|---:|---|
| 开发 3 类（Bird/Car/Cube） | 0.74357 | 0.80497 | 0.81821 | +0.07464 | [+0.04080, +0.11060] |
| 扩展 7 类 | 0.82928 | 0.85332 | 0.85849 | +0.02922 | [+0.00804, +0.05027] |

扩展 7 类未参与任何配置选择，其均值增益（+0.029）小于开发 3 类（+0.075），**方向一致但幅度衰减**。

## 5. 异常情况

| 事件 | 处理 | 影响 |
|---|---|---|
| 服务器只有 Bird 保留 1024² 原图，其余 9 类仅有 `l0_448` 缓存 | 统一改用缓存输入；Bird 保留原图版做偏差量化 | `ΔAUROC ≤ 0.0009`、`ΔAP ≤ 0.0017`（`input_mode_cache_vs_orig.csv`）→ 不影响结论 |
| 本机 faiss 为 CPU-only，禁止安装 | torch fp32 精确 1-NN 替代，官方 faiss-CPU 逐图校验 | max\|Δscore\| = 1.8e-07 |
| GitHub 网络不稳，`torch.hub` 校验阻塞 130 s 并曾失败 | 改为本地 `source='local'` 载入同一 repo/权重 | 无数值影响 |
| 环境禁止 bulk delete（>500 文件） | 符号链接树改为幂等创建、不删除 | 无 |
| 首次运行的样本顺序与 E22 不同（NG 目录名排序差异） | 改按 `(specimen, illumination)` 唯一键对齐后再比标签 | 集合一致，已修正 |

## 6. 分析（事实 / 解释 / 假设）

**事实**：官方 AnomalyDINO 在本协议下 10 类宏平均 AUROC 0.8036 / AP 0.8592，低于 F0（0.8388 / 0.8940）
与 R1（0.8464 / 0.9017）；R1−AD 的 ΔAUROC 与 ΔAP 区间均不含 0。

**解释（关键，必须写进论文）**：本比较**不能**归因于"R1 的方法更优"。AnomalyDINO 与本文 F0 共享同一技术骨架，
但本实验有三处未对齐的官方组件：backbone（ViT-S vs ViT-B）、聚合（top-1% 均值 vs max）、参考旋转增广。
即使完全不训练，F0 也高出 0.0352，说明差距主要来自**配置**而非适配器。
因此正确表述是"本文方法在与一个独立发表的完整方法同协议比较时未处劣势"，**不是**"显著领先"。

**假设（未验证）**：AnomalyDINO 在 Holder/Motor 上反超（+0.017/+0.028）可能与其 top-1% 聚合对小面积缺陷更稳有关；
若要验证需做"backbone/pooling 匹配"的消融，**本轮不做**。

## 7. 结论

外部基线比较**已补齐**：30/30 cell 通过身份与 bank 哈希核对，官方实现未修改，结果完整保留。
R1 在 7/10 类别上高于该基线，宏平均 +0.0428 [+0.0251, +0.0602]。

## 8. 出口条件判断

- 若 AnomalyDINO 宏平均高于 R1 → 收缩表述。**未发生**。
- 但"R1 > AD"含配置混杂 → **只能写"同协议下未处劣势"**，不得写"SOTA / 显著领先"。
- 不追加任何实验。

## 9. 对论文证据链的贡献

补齐"外部方法比较"这一格：论文不再只依赖 F0 受控比较。
**下一格**：全文修订（统计说明、related work 划界、图表更新），本轮后停止实验。

## 10. 产物清单

| 文件 | 内容 |
|---|---|
| `protocol.md` | 预注册协议（含 8 条协议调整 A1–A8） |
| `scripts/run_anomalydino_m2ad.py` | 官方实现运行脚本（含 `--equiv` 等价性验证） |
| `scripts/e26_analysis.py` | 指标、宏平均、工件级 bootstrap |
| `backend_equivalence.json` | 官方 faiss-CPU vs torch 后端：max\|Δ\| = 1.82e-07 |
| `scores/{Cat}_seed{s}.json` | 30 个 cell：逐图得分 + 标签 + 身份核对 + bank 哈希 |
| `scores_originput/Bird_seed{0,1,2}.json` | Bird 原图输入版（用于偏差量化） |
| `baseline_30cells.csv` | 30 行逐 cell 指标 |
| `baseline_category_summary.csv` | 10 行类别层（3-seed 均值±sd） |
| `baseline_macro.csv` | 3 行宏平均（10 类 / 开发 3 类 / 扩展 7 类） |
| `baseline_bootstrap.csv` | 24 行工件级 bootstrap |
| `input_mode_cache_vs_orig.csv` | 3 行输入模式偏差 |
| `comparison_conditions.csv` | 11 项比较条件对齐表 |
| `logs/run_all.log` | 运行日志（`[END] fails=0`） |
