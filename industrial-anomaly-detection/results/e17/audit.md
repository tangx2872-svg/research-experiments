# E17 — Lightweight Refinement · P0 READ-ONLY AUDIT + 参数审计

日期：2026-10-14 · 执行前状态快照（未跑任何 GPU，未修改任何代码）

## 1) 环境状态

| 项 | 值 |
|---|---|
| branch | `research/industrial-anomaly` |
| HEAD | `52b2b74`（`exp: qualify R1 illumination adaptation in E16`） |
| ahead/behind | **0 / 0**（E16 已 push，`55542e9..52b2b74`） |
| git status | 6 项历史遗留（`monitor/exp_monitor.py`、`results/e8/status.json`、`monitor/events.jsonl`、`scripts/_lockprobe.txt`、`scripts/_probe2.py`、`scripts/e11_p4_runner.py`）—— 均与 E17 无关，不触碰 |
| GPU | RTX 3090，util **0%**，占用 **1 MiB**，**0 个 compute 进程** |
| disk free | **16 GB** |
| E16 commit / push | `52b2b74` ✅ pushed（`0/0`） |

## 2) R1 当前准确实现位置

| 项 | 位置 |
|---|---|
| 结构 | `scripts/e13a_screen.py :: class LinearAdapter`（`z' = z + Wz`，768→768，**zero-init**） |
| forward | `e13a_screen.py :: adapter_forward(adapter, z) -> z + adapter(z)`（**不做 L2**） |
| 损失 | `e13a_screen.py :: compute_loss(..., objective="cons_preserve")` → `loss = L_cons + BETA * L_preserve` |
| 训练 | `e13b_screen.py :: train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)`（seed-aware；E13-B/B2/E16 均用它） |
| 早停评估 | `e13a_screen.py :: val_adapter_loss`（每 5 步，patience 30） |
| 特征适配 | `e13a_screen.py :: adapted_feat_adapter`（分块，不做 L2；打分前才 L2-norm） |
| collapse 检查 | `e13a_screen.py :: collapse_stats / is_collapsed` |
| E16 资格赛运行器 | `scripts/e16_qualify.py`（含 d′ / robustness / diagnostic，可复用其函数） |

## 3) R1 全部可调参数审计（本次唯一搜索轴的依据）

```
parameter            current value   meaning                                     expected direction        cost
---------------------------------------------------------------------------------------------------------------
BETA (L_preserve)    0.1             L_cons 权重隐式为 1.0；BETA 是 L_preserve      BETA↑ = 锚定更强 =          0
                                     (锚回 frozen) 的权重。它控制                  适配更弱（更靠近 F0）
                                     "光照一致性 ↔ 保持 frozen" 的权衡，            BETA↓ = 适配更强
                                     即 adaptation strength。                     BETA=0 = 纯一致性
LR_ADAPTER           1e-3            AdamW 学习率                                 ↑ 移动更快/更大           0
MAX_STEPS            300             最大优化步数                                  ↑ 训练更久                GPU
PATIENCE_STEPS       30              早停耐心值                                   —                        —
EVAL_EVERY           5               每 5 步评估 val                               —                        —
BATCH_IMGS           8               batch 图像数                                 —                        —
TRAIN_N / VAL_N      240 / 60        bank 内 train/val 切分                       —                        —
AUG_LEVELS           [0.85, 1.15]    mild 光照增强强度（brightness/contrast/gamma） ↑ = 一致性任务更难         0
AUG_FAMILIES         3 族 × 2 档 = 6 K=6 个增强变体                              —                        —
结构                 768×768 线性残差   z' = z + Wz（zero-init，589,824 参数）       —                        —
---------------------------------------------------------------------------------------------------------------
```

**关键事实**：R1 的损失 `L = l_cons + BETA * l_preserve` 中，`l_cons` 的权重是**硬编码 1.0**；
**BETA 是唯一的显式损失级自由参数**，且它单调、干净地控制 illumination-consistency 与 frozen-preservation
的权衡，即"adaptation strength"。→ 满足 §2「最直接控制 illumination consistency / adaptation strength 的单一参数」。

## 4) 唯一搜索轴冻结：**BETA**

- **不改**：结构、LR、MAX_STEPS、PATIENCE、BATCH、TRAIN/VAL 切分、AUG（家族与强度）、scoring、backbone、bank、resolution。
- 实现方式：`compute_loss` 在调用时从模块全局读取 `BETA`，因此通过**运行时设置 `e13a_screen.BETA = <value>`**
  即可改变该参数，**无需修改任何冻结代码**。

### 候选（5 个，含当前 R1；乘法对称、局部、低风险）

| ID | BETA | vs 当前 | 语义 |
|---|---|---|---|
| **C0** | **0.1** | **1.0×** | **当前 R1（已资格认证，全部复用，禁止重跑）** |
| C1 | 0.0 | 0× | 纯光照一致性，无 frozen 锚定（最强适配） |
| C2 | 0.05 | 0.5× | 锚定减半 |
| C3 | 0.2 | 2× | 锚定加倍（适配更弱） |
| C4 | 0.5 | 5× | 强锚定（接近 frozen 的温和适配） |

> 说明：BETA=0 时 `l_preserve` 仍被计算但权重为 0（不参与梯度），语义清晰且无副作用。

## 5) 可复用资产（避免重复 GPU）

| 资产 | 覆盖 | 来源 |
|---|---|---|
| **C0（= R1）9/9 cell 全指标** | AUROC / AP / **d′ / robustness_feat / robustness_score** / runtime / peak | `results/e16/e16_partial.json`（9 cell 全量） |
| C0 的 seed1/2 AUROC/AP（独立记录） | 交叉核对 | `results/e13b/e13b_partial.json` |
| F0 9/9 cell 全指标 | 同上 | `results/e16/e16_partial.json`（`f0` 字段） |
| l0_448 图像缓存 | Bird 6400 / Car 4000 / Cube 4000 | `data/m2ad/e7_cache/l0_448/` |
| DINOv2-B 权重 | 本地 `dinov2_vitb14_pretrain.pth`（离线 loader） | `/root/.cache/torch/hub/` |

→ **Round 1 每条 category-run 只需 1 次特征提取**，然后循环 5 个 BETA 配置（C0 仅作机器正确性复现门，其余为新增）。

## 6) 审计结论

| 检查 | 结果 |
|---|---|
| E16 已 push / worktree 干净（除历史遗留） | ✅ |
| GPU / disk 空闲 | ✅ |
| R1 实现与超参已定位 | ✅ |
| 唯一搜索轴可运行时注入且不改冻结代码 | ✅ |
| C0 / F0 全指标可复用（0 GPU） | ✅ |
| 阻断性问题 | **无** → 继续 protocol freeze → Round 1 |
