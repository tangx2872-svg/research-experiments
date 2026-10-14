# E16 — R1 Signal Qualification · P0 READ-ONLY AUDIT

日期：2026-10-14 · 执行前状态快照（未跑任何 GPU）

## 1) git branch / HEAD / push 状态

| 项 | 值 |
|---|---|
| branch | `research/industrial-anomaly` |
| HEAD | `55542e9`（docs: record DINOv2 line E11–E15 CLOSED） |
| remote | `origin = git@github.com:tangx2872-svg/research-experiments.git` |
| push 前 | ahead **29** / behind **0** |
| push 动作 | `git push`（本分支 fast-forward，**未触碰 main**） |
| push 后 | ahead **0** / behind **0**（`1caecfc..55542e9`） |
| main | ahead 13 / behind 4 → **未 push main**（遵守既有约束） |

## 2) git status --porcelain（E16 前既有，均与 E16 无关）

```
 M monitor/exp_monitor.py        (历史遗留修改)
 M results/e8/status.json        (历史遗留修改)
?? monitor/events.jsonl          (运行时日志)
?? scripts/_lockprobe.txt        (探针残留)
?? scripts/_probe2.py            (探针残留)
?? scripts/e11_p4_runner.py      (E11-P4 遗留，从未 commit)
```
→ E16 不动这些文件；E16 只提交 `results/e16/` + `scripts/e16_*.py`。

## 3) GPU / 内存 / 磁盘

| 项 | 值 |
|---|---|
| GPU | RTX 3090 24 GB，util **0%**，占用 **1 MiB**，**0 个 compute 进程**（空闲） |
| RAM | 总 755 GB / 可用 **723 GB** |
| disk `/root/autodl-tmp` | 50 G 总 / 35 G 用 / **16 G 可用**（≥12 GB ✓，无需清理） |

## 4) 数据完整性

| category | l0_448 缓存文件数 | 说明 |
|---|---|---|
| Bird | 6400 | 完整（含 1200 bank pool + view120 test 等） |
| Car | 4000 | 完整 |
| Cube | 4000 | 完整 |

## 5) F0 历史结果可直接复用 ✅

F0 = frozen DINOv2-B final-layer + LayerNorm + 1-NN-max，300-bank，view120。
seed0 复用 E13-A `R0`；seed1/seed2 复用 E13-B `r0_*` 列（每 seed 各自 bank 重算，已按同一协议）。

| category | seed0 | seed1 | seed2 |
|---|---|---|---|
| Bird | 0.74725 / 0.78863 | 0.75093 / 0.79216 | 0.75576 / 0.79174 |
| Car | 0.78552 / 0.84207 | 0.79985 / 0.85393 | 0.81399 / 0.85405 |
| Cube | 0.85410 / 0.91686 | 0.87203 / 0.92662 | 0.86528 / 0.92290 |

## 6) R1 当前准确实现位置与历史结果来源

| 项 | 值 |
|---|---|
| 结构 | `z' = z + W z`，W: 768→768，zero-init（identity 起点）；trainable 589,824 |
| 实现 | `scripts/e13a_screen.py::class LinearAdapter` + `train_adapter`（R1 分支） |
| 复用路径 | `scripts/e13b_screen.py::train_r1`（E13-B/B2 使用，超参完全一致） |
| 损失 | `L = L_cons + 0.1·L_preserve`（`compute_loss` objective=`cons_preserve`） |
| 超参 | AdamW lr=1e-3；batch=8；≤300 step；early-stop 30（每 5 步评估 val consistency）；train 240 / val 60 |
| 增强 | mild illumination：{brightness, contrast, gamma} × {0.85, 1.15} = 6 变体（无几何变换） |
| seed0 来源 | `results/e13a/e13a_screening.csv`（R1 行） |
| seed1/2 来源 | `results/e13b/e13b_results.csv`（r1_* / d_* 列） |
| Cube seed3–7 | `results/e13b2/e13b2_results.csv`（本轮不用，仅备查） |

### R1 历史 9-cell 结果（AUROC/AP，全部可复用）

| category | seed0 | seed1 | seed2 |
|---|---|---|---|
| Bird | 0.75786 / 0.79911 | 0.78142 / 0.82107 | 0.79223 / 0.82349 |
| Car | 0.80511 / 0.85502 | 0.80770 / 0.86315 | 0.82025 / 0.86533 |
| Cube | 0.87010 / 0.92856 | 0.85974 / 0.92342 | 0.86952 / 0.92679 |

## 7) 协议一致性（Bird / Car / Cube）

✅ categories 全部来自 `e4d1_m2ad_loader`（M²AD）；bank = `E8.bank_for(pool, seed)` 取 300 normal；
✅ test = view120（`LD.select(recs, views=["120"])`，split=test）；label = official `image_anomaly` + raw `detectable`；
✅ preprocessing = l0_448 缓存 + `e13a_screen.prep_image`（Resize 448 BICUBIC + ToTensor + ImageNet norm），与 E11–E13 完全一致；
✅ backbone = 本地 `dinov2_vitb14_pretrain.pth`（离线 loader，无网络），全 frozen。

## 8) 历史 seed 可直接复用性

**AUROC / AP 全部 9 个 cell 可直接复用（0 GPU）**。
但协议第 4 节要求的 **d′、illumination robustness、runtime、peak GPU memory 历史上未落盘**（E13 只保存了聚合 AUROC/AP，未保存 per-image scores，也未保存 adapter 权重）。
→ E16 必须做一次**确定性复现运行**以获得 per-image scores 与 d′/robustness；该复现必须与历史 AUROC/AP **逐位一致**（容差 1e-6），以此证明复现 = 历史 cell。

## 9) 等价性检查（R1 vs 历史方法）

E16 只验证 R1 本身，不新增候选。R1 = `z + Wz` 线性残差（zero-init）。
与历史 P4/P5（frozen，无参数）、E12-A（PCA subspace）、E12-B S2/S3/S4（block9+final 融合 gate，非单一表示）、E13 R2/R3/R4（MLP / LoRA / MLP+cons）**均不等价**。

## 10) E14/E15 commits 是否已 push

**push 前未 push**（29 commits 含 E12-B/E13-A/B/B2/E14/E15 全部在本地）。**已按指令执行 `git push` 成功**。

## 11) 审计结论

| 检查 | 结果 |
|---|---|
| 协议一致性 | ✅ PASS |
| F0 可复用 | ✅ PASS（9/9 cell） |
| R1 可复用 | ✅ PASS（9/9 cell AUROC/AP） |
| d′/robustness 缺失 | ⚠️ 必须新增（确定性复现 + 新指标） |
| GPU / disk | ✅ 空闲 / 16 GB |
| 阻断性问题 | **无** → 继续 protocol freeze → smoke → full run |
