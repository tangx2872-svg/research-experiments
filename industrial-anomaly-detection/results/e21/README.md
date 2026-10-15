# E21 — M²AD Multi-View Generalization Validation

**日期**：2026-10-15
**状态**：**Round 1 完成 → HOLD（停止，不补 seed1/2，待策划 AI 裁决）**
**协议**：[`protocol.md`](protocol.md)（预注册于运行前）
**方法**：F0（frozen DINOv2-B, 1-NN-max）/ R1-ILA（BETA=0.1，`results/e17/final_method.json` 冻结）
**产物**：`data_audit.csv` · `e21_partial.json` · `round1_seed0.csv` · `analysis_{cells,category_pooled,view_pooled,overall}.csv` · `bootstrap_ci.csv` · `gate_decision.json` · `figures/` · `verification.csv`

---

## 1. 为什么做 E21

E16/E18/E19 的全部主结果（A1/A2/A3）都在 **view120 单一拍摄视角**上评估。这是 E19 已登记的 limitations 之一（单 view）。E21 用**已有的 4 个冻结视角**（090/120/240/330，l0_448 已缓存、零数据获取）把评估扩展到多视角，检验 R1-ILA 收益的跨视角总体正向性。

## 2. E21 服务论文哪句话

> "R1-ILA 的收益并非仅存在于 view120 单一拍摄视角；在多个已有拍摄视角下，其整体异常检测性能仍保持正向收益，并表现出一定的跨视角稳定性。"

**边界**：这不是 viewpoint-invariance / universal-robustness 证明；跨数据集泛化仍属路线 B（E20-B）。

## 3. 数据协议

- 类别：Bird / Car / Cube；视角：**090 / 120 / 240 / 330**；seed：Round 1 = seed0。
- 输入：`data/m2ad/e7_cache/l0_448/`（448 BICUBIC）；标签 = official `image_anomaly`；过滤 = `detectable != False`（继承 E18/E19 `official_metrics`）。
- 审计（`data_audit.csv`，运行前 PASS）：12 个 category×view cell 全部同时含 good 与 anomaly、illum 10 档全覆盖、缓存文件齐备；**view120 = 700 raw / 653·656·678 kept，与 E16/E18/E19 完全一致**。
- **训练数据不绑定 view120（运行前协议审计，实测）**：bank pool = 10 train specimens × 12 views × 10 illum = 1200 图；300-bank 的 view 分布横跨全部 12 views（每 view 18–32 图）；bank ∩ test specimens = 0。因此本实验 = **"冻结 12-view bank 下的多视角评估"**（每 (cat, seed) 只训练 1 个 adapter，4 个 view 分别测试）；**不是** zero-shot 视角迁移、也不是 per-view 训练。

## 4. 为什么选择 090/120/240/330

E8 冻结协议的 4-view 集（`VIEWS = [120, 090, 240, 330]`），且这 4 个 view 的 test 图 **100% 已在本地 l0_448 缓存**（0 下载、0 解压、0 新数据），符合磁盘与资源约束。

## 5. 哪些结果复用了历史 / 哪些是新跑

| 来源 | 内容 |
|---|---|
| **复用（REUSED_VERIFIED）** | view120 的全部报告值（AUROC/AP/per-image scores 以 E16/E18/E19 为准）；`round1_seed0.csv` 中 view120 行标记 `source=REUSED_VERIFIED` |
| **新跑（NEW）** | view 090/240/330 × 3 类 × seed0（9 个新评估单元）；每类 1 次 R1-ILA 训练（与 E16 相同代码路径）+ 4-view 测试提取 |
| **复现校验** | E21 运行中提取了 view120（S1/S2 的必要复现成本），结果与历史 **逐图分数差 0.0（9 个 cell 全部位级一致）**，报告值仍采用历史值 |

## 6. 结果（Round 1, seed0）

### 6.1 12-cell 表（AUROC / AP / Δ）

| category | view | n_good | n_anomaly | F0 AUROC | R1 AUROC | ΔAUROC | F0 AP | R1 AP | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bird | 090 | 330 | 336 | 0.77745 | 0.79885 | **+0.02140** | 0.75655 | 0.77784 | +0.02129 |
| Bird | 120 | 320 | 333 | 0.74725 | 0.75786 | +0.01061 | 0.78863 | 0.79911 | +0.01048 |
| Bird | 240 | 330 | 339 | 0.80553 | 0.81059 | +0.00506 | 0.80907 | 0.82268 | +0.01361 |
| Bird | 330 | 320 | 368 | 0.82985 | 0.83319 | +0.00334 | 0.86019 | 0.84246 | **−0.01773** |
| Car | 090 | 320 | 356 | 0.81415 | 0.81524 | +0.00109 | 0.85694 | 0.85221 | **−0.00473** |
| Car | 120 | 300 | 356 | 0.78552 | 0.80511 | +0.01959 | 0.84207 | 0.85502 | +0.01295 |
| Car | 240 | 270 | 401 | 0.75602 | 0.73970 | **−0.01632** | 0.82288 | 0.80409 | **−0.01879** |
| Car | 330 | 290 | 392 | 0.74303 | 0.71703 | **−0.02600** | 0.78905 | 0.78368 | **−0.00537** |
| Cube | 090 | 260 | 418 | 0.85780 | 0.85751 | **−0.00029** | 0.92187 | 0.91415 | **−0.00772** |
| Cube | 120 | 260 | 418 | 0.85409 | 0.87010 | +0.01600 | 0.91686 | 0.92856 | +0.01170 |
| Cube | 240 | 230 | 458 | 0.86774 | 0.87277 | +0.00503 | 0.94128 | 0.94183 | +0.00056 |
| Cube | 330 | 210 | 475 | 0.87085 | 0.86656 | **−0.00429** | 0.94586 | 0.94076 | **−0.00511** |

### 6.2 Pooled 与 bootstrap（paired image bootstrap, n_boot=3000）

**per-category 4-view pooled**（合并样本）：

| category | n | pooled F0 | pooled R1 | ΔAUROC [95% CI] | ΔAP |
|---|---:|---:|---:|---|---:|
| Bird | 2676 | 0.78980 | 0.79739 | +0.00759 [−0.00048, +0.01539] | +0.00459 |
| Car | 2685 | 0.77334 | 0.76597 | **−0.00738** [−0.01598, +0.00192] | −0.00904 |
| Cube | 2729 | 0.86102 | 0.86506 | +0.00404 [−0.00311, +0.01112] | −0.00051 |

**per-view 3-category mean**：v090 +0.00740 / v120 **+0.01540** / v240 −0.00208 / v330 −0.00899（ΔAUROC）

**overall（主口径 = 3 类 pooled Δ 的 macro 平均）**：ΔAUROC = **+0.00142**（stratified bootstrap 95% CI [−0.00324, +0.00616]）；ΔAP = **−0.00165**；辅助口径：cell-mean +0.00293、样本加权 +0.00142；12 cells 中 8 正 4 负。

> 声明：以上 CI 反映 image sampling uncertainty，不等价于 cross-seed uncertainty（未补 seed1/2，见 §7）。

### 6.3 图

`figures/`：fig1（4 view 的 F0 vs R1 AUROC）· fig2（category×view ΔAUROC heatmap）· fig3（per-view ΔAUROC 点图）· fig4（4-view pooled F0 vs R1）。**Fig5（3-seed 均值）因 HOLD 未生成**（不存在的数据不补）。

## 7. seed0 判据与判定（协议 §7，逐条）

| 条件 | 实测 | 满足 |
|---|---|---|
| overall pooled ΔAUROC > 0 | +0.00142 | ✅ |
| overall pooled ΔAP > 0 | −0.00165 | ❌ |
| ≥2/3 category pooled ΔAUROC > 0 | 2/3（Bird、Cube 正；Car 负） | ✅ |
| 无 catastrophic regression（≥ −0.03） | min cell = −0.02600（Car v330） | ✅ |
| **GO（需全部满足）** | — | **❌** |
| STOP 条件（任一） | ΔAUROC≤0 且 ΔAP≤0？否；≥2 类 pooled≤0？否（1 类）；任一 cell < −0.03？否 | 未触发 |

→ **HOLD**（协议 §11："overall pooled 正" + "出现单个明显负 view 但未达到 −0.03" + "AUROC/AP 明显冲突" 三重命中）。`gate_decision.json` 记录。

## 8. 是否进入 seed1/2

**否。** 协议 §11：HOLD → 停止，不自动补 seed1/2，完整回报给策划 AI 决策。本目录不含 seed1/2 数据。

## 9. 哪些结论可以写 / 不能写

**可以写（事实）**：
- 12 个 category×view cell 中 8 个 ΔAUROC 为正；4 个为负（§6.1 全列出）。
- view120 上三类收益 +0.0106/+0.0196/+0.0160（与历史一致，逐图 0.0 差异）；两个新视野（Bird v090、Car v120 之外的 Car v330）分别显著为正/负。
- 4-view pooled：Bird +0.0076、Cube +0.0040、Car −0.0074；overall macro +0.0014（CI 跨 0）。
- 多视角结果表明收益**并非仅局限于 view120**（存在多处正收益），但**跨视角高度异质**（Car 在 240/330 反转、显著负的 cell 存在）。

**不能写**：viewpoint invariant / universally robust / all views improved / all categories significant / 每 view 都提升 / generalizes to arbitrary datasets / causal proof。

## 10. 已知 failure cases（全列，不隐藏）

1. **Car v330**：ΔAUROC −0.02600（bootstrap 95% CI [−0.04449, −0.00761]，**显著负**）；ΔAP −0.00537。Round 1 的最差 cell。
2. **Car v240**：ΔAUROC −0.01632（CI [−0.03677, +0.00369]）；ΔAP −0.01879。
3. **Car pooled**：ΔAUROC −0.00738、ΔAP −0.00904 —— 唯一 pool 值为负的类别。
4. **Cube v090**：ΔAUROC −0.00029、ΔAP −0.00772（微负）。
5. **Cube v330**：ΔAUROC −0.00429、ΔAP −0.00511。
6. **Bird v330**：ΔAUROC +0.00334 为正，但 **ΔAP −0.01773**（AUROC/AP 符号冲突的 cell）。
7. **Car v090**：ΔAP −0.00473（点估计弱负）。
8. **overall ΔAP −0.00165**：pooled AP 与 pooled AUROC 方向冲突（HOLD 的直接来源之一）。

## 11. 复现证据（S1–S4）

- **S1/S2**：view120 复算 vs E16/E18/E19 历史：9 个 cell 的 per-image 分数 **max|Δ| = 0.0**（位级一致，容差 1e-6）；AUROC/AP 差 0.0。展示精度注记见 `protocol.md` §3。
- **S3**：4-view test 的 (category, specimen, view, illumination) 唯一映射，0 duplicates（运行前检查）。
- **S4**：每 cell 记录 `bank_hash` / `adapter_hash`（`round1_seed0.csv`）；bank 构建路径与 E16 相同（同函数同顺序；逐图 0.0 复现间接证明 bank 一致，`verification.csv` 独立复算 bank hash PASS）。

## 12. 产物清单

`protocol.md` · `data_audit.csv` · `e21_partial.json`（全部 per-image scores）· `round1_seed0.csv`（12 行，含 bank/adapter hash）· `analysis_cells.csv` · `analysis_category_pooled.csv` · `analysis_view_pooled.csv` · `analysis_overall.csv` · `bootstrap_ci.csv`（17 行）· `gate_decision.json` · `figures/`（4 图 × PNG+PDF）· `verification.csv` · `logs/`。
