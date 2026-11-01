# E26-B · Early stopping / checkpoint 行为审计（静态代码审计）

**日期**：2026-11-01
**实验编号**：E26-B（投稿前 P0 审计）
**性质**：**只读代码审计**。未修改任何训练逻辑、未训练、未重跑、未替换任何结果文件。
**GPU**：0（仅读取 JSON 统计 `steps_used`）

---

## 1. B1 · 真正用于最终实验的实现（追溯链）

| 环节 | 路径 | 证据 |
|---|---|---|
| 训练函数（唯一实现） | `scripts/e13b_screen.py` → `train_r1()`（第 44–90 行） | E16/E17/E21/E22/E24/E25 全部 `from e13b_screen import train_r1` |
| 适配器定义 | `scripts/e13a_screen.py` → `LinearAdapter`（768×768，zero-init，无 bias） | `e22_full_category_validation.py:57-59` 直接 import |
| 冻结超参常量 | `scripts/e13a_screen.py` → `LR_ADAPTER / MAX_STEPS / PATIENCE_STEPS / EVAL_EVERY / BATCH_IMGS / TRAIN_N / VAL_N / AUG_LIST` | 同上 |
| 冻结配置（权威） | `results/e17/final_method.json` | `PATIENCE_STEPS=30, EVAL_EVERY=5, MAX_STEPS=300, BATCH_IMGS=8, TRAIN_N=240, VAL_N=60` |
| 冻结决策记录 | `results/e17/e17_decision_memo.md:31`、`results/e17/README.md:41` | “patience 30” 列入冻结不变量 |
| 审计时 git 状态 | 分支 `research/industrial-anomaly`，HEAD `417d883`（工作区干净，除 2 个历史无关未跟踪探针文件） | `git status` |

**共用该训练函数的最终实验**（因此本节结论覆盖全部已发表数字）：

| 脚本 | 用途 | 行号 |
|---|---|---|
| `scripts/e16_qualify.py` | E16 资格验证 | 285 |
| `scripts/e17_refine.py` | E17 BETA 精炼 / 冻结 | 119 |
| `scripts/e21_multiview_validation.py` | E21 多视角 | 193 |
| `scripts/e22_full_category_validation.py` | **E22 十类 × 3 seed 主结果** | 245 |
| `scripts/e24_runner.py` | E24 光度增广敏感性 | 213 |
| `scripts/e25_robustad_eval.py` | **E25 RobustAD 外部验证** | 374 |

**明确排除的废弃实现**：`scripts/e12b_screen.py`（E12-B gate 方法，非最终方法）确实**有** best-state 快照与恢复（第 326、333–334 行 `best_state` / `gate.load_state_dict(best_state)`）。该文件**不在**最终方法的 import 链上，不得据此判断最终方法行为。这一点是本节的第一个陷阱，已在审计中区分。

---

## 2. B2 · 静态代码检查

搜索关键词：`best_state / best_model / best_checkpoint / deepcopy / state_dict / torch.save / load_state_dict / patience / early_stop / no_improve / validation / current model`。

结果：
- `best_state` / `load_state_dict` / `torch.save` / `deepcopy` 在**最终方法路径（e13a/e13b/e16/e17/e21/e22/e24/e25）中一次都没有出现**；
- 全文仅 `e12b_screen.py:334`（废弃路径）与 `e5_1_runner.py`（E5 的 C06/C07/C08，完全不同的方法族，存权重文件）出现。

### Q1：训练过程中是否记录历史最佳验证指标？

**是（但只记录标量）。**

```52:82:scripts/e13b_screen.py
    best_val = float("inf")
    no_improve = 0
    ...
        if (step + 1) % EVAL_EVERY == 0:
            v = val_adapter_loss(adapter, val_feat, val_aug, "cons_preserve", dev)
            if v < best_val - 1e-6:
                best_val = v
                no_improve = 0
            else:
                no_improve += 1
            if no_improve * EVAL_EVERY >= PATIENCE_STEPS:
                break
```

- `best_val`：记录到目前为止最小 val consistency loss（标量），用于**计数 early-stop**；
- `v` 由 `val_adapter_loss` 在 **bank 中的 60 张留出正常图**（`bank_feat[240:300]`）上计算，见 `e13b_screen.py:135-136`、`e25_robustad_eval.py:374` 前文构造；
- 记录为 `report["best_val"]`（第 89 行），并被 E24（`e24_runner.py:230`）写入 cell JSON。

### Q2：是否保存历史最佳模型参数？

**否。**

- 循环内**没有**任何 `copy.deepcopy(...)` / `{k: p.detach().clone() ...}` / `state_dict()` 快照；
- `adapter` 参数始终是原地更新（`opt.step()`，第 72 行），`train_r1` 只返回一个 report dict（第 90 行），**不返回权重**；
- 因此 `best_val` 是**死变量**：它只影响"何时 break"，不影响"用哪组权重"。

### Q3：触发 early stopping 后，用于 test / bank / test score 的权重是哪一个？

**A. 当前停止时权重（last weights at the stopping iteration）。**

```44:44:scripts/e13b_screen.py
def train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed):
```
```137:141:scripts/e13b_screen.py
    torch.manual_seed(seed)
    adapter = LinearAdapter().to(dev)
    rep = train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)
    bank_af = adapted_feat_adapter(adapter, bank_feat, dev)
    test_af = adapted_feat_adapter(adapter, test_feat, dev)
```

`run_cell()` 在 `train_r1` 返回后**直接使用同一个 `adapter` 对象**（第 140–141 行）重算 bank 与 test 特征，中途没有任何恢复动作。E25 同构（`e25_robustad_eval.py:374` 之后同样就地使用 `adapter`）。

**由此可确认的补充事实**：`best_val` 代表的"历史最佳验证点"**从未被用于打分**；所有已发表数字对应的是 **early-stop 触发那一刻的权重**。

---

## 3. B3 · 与冻结协议的交叉核查

| 来源 | 原文 | 是否规定 checkpoint 策略 |
|---|---|---|
| `results/e16/protocol.md:45` | “batch / steps 8 / ≤300，early-stop 30（每 5 步评估 val consistency）” | **否**（只规定停止规则） |
| `results/e16/audit.md:64` | “≤300 step；early-stop 30（每 5 步评估 val consistency）；train 240 / val 60” | **否** |
| `results/e17/e17_decision_memo.md:31` | 冻结不变量含 “patience 30” | **否** |
| `results/e17/audit.md:25` | “早停评估 = `e13a_screen.py::val_adapter_loss`（每 5 步，patience 30）” | **否** |
| `results/e17/final_method.json` | `PATIENCE_STEPS: 30, EVAL_EVERY: 5, MAX_STEPS: 300` | **否**（无 checkpoint 字段） |
| `results/paper_ready/EVIDENCE_MAP.md:36` | “优化：AdamW，lr = 1e-3，≤300 step，patience 30（每 5 step 验证），batch = 8 张” | **否** |

**仓库内不存在论文正文草稿**（无 `.tex` / `.docx` / `WRITING_BASELINE.md`），因此 "PAPER_DESCRIPTION" 只能以 `results/paper_ready/` 的论文面向材料为准（上表最后一行）。

### 三方对照

| 维度 | 内容 | 是否明确 |
|---|---|---|
| **IMPLEMENTATION** | 使用 early-stop 触发时刻的**当前权重**；`best_val` 仅用于停止计数与报告 | **明确**（代码第 44–90 行，无快照） |
| **FROZEN_PROTOCOL** | 只规定 "≤300 step / patience 30 / 每 5 步在 60 张留出正常图上评估 val consistency"，**未规定**停止后是否回到最佳点 | **沉默** |
| **PAPER_DESCRIPTION** | 仓库内论文面向材料同样是 "patience 30（每 5 step 验证）"，**未出现** "best checkpoint" / "restore" / "model selection" 等表述 | **沉默** |

⇒ **三方之间不存在矛盾**：实现做了一件事，协议与论文材料都没有对它作相反的规定或相反的宣称。

---

## 4. B4 · 裁决

### 结论：**PASS_A**（实现 = 当前 checkpoint；协议与论文面向材料对该点沉默，未要求相反行为）

- 不是 PASS_B：实现**不是** best checkpoint，因此不属于"论文误写"这一情形。
- 不是 ESCALATE：ESCALATE 的定义是"冻结协议**明确要求**一种行为，而最终实验用了另一种行为"。本仓库的冻结协议从未规定 checkpoint 策略，因此**不构成协议违背**。

### 但必须附带一个写作义务（不是重跑义务）

因为 **100% 的 cell 都实际触发了 early stopping**（见下节统计），"用最后一版权重"与"用最佳验证权重"在**每一个**已发表数字上都不是同一组参数。论文必须把这一点写清楚，否则审稿人会按通行惯例默认 best-restore：

> 建议正文（Methods，训练细节一句）：
> "Early stopping monitors a validation consistency loss on 60 held-out normal images (evaluated every 5 steps, patience 30); **the parameters at the stopping iteration are used for evaluation — no best-validation checkpoint is restored.**"

若作者原本的意图是 best-restore，则该情形按任务规则应升级为 **ESCALATE**（协议沉默 + 意图不一致），需人工裁决；本轮不自行改为 best-restore，也不重跑。

---

## 5. 量化：early stopping 的实际触发率（只读 cell JSON）

| 结果集 | cell 数 | steps_used 范围 | 中位数 | 达到 300 步上限 | 实际早停 |
|---|---:|---|---:|---:|---:|
| E22（十类 × 3 seed 主结果） | 30 | 44 – 124 | 59 | 0 | **30 / 30** |
| E25（RobustAD，13 split × 3 seed） | 39 | 49 – 164 | 74 | 0 | **39 / 39** |
| E24（光度增广敏感性） | 51 | 44 – 219 | 59 | 0 | **51 / 51** |

⇒ 没有任何 cell 跑满 300 步；"early stop 之后用哪组权重"对**全部**已发表数字都是实质性问题（虽然不改变任何一个数字的合法性，只改变它的**语义描述**）。

**影响上界（未能量化，且不重跑）**：last-step 与 best-val step 的差距上界是 `patience = 30` 步（AdamW lr=1e-3，768×768）。在没有重训的前提下无法给出两者指标差；本轮**不做**该估计（属于新实验）。

---

## 6. 附带发现（不影响任何数字，仅记录）

1. **`steps_used` 在早停时少记 1 步**：`report["steps_used"] = step + 1`（第 83 行）位于 eval 分支之后；当第 82 行 `break` 时该行被跳过，因此记录值 = 最后一个**已执行**步编号，而不是 `+1`。属于报告层 off-by-one，**与权重、分数、指标无关**。按审计规则**不修改**代码。
2. **`best_val` 是死变量**：被计算、被写入 report、被 E24 写入 cell JSON，但从不影响权重选择。若后续论文要声称 "model selection on validation"，当前实现无法支撑该说法。
3. `e12b_screen.py`（E12-B，废弃）中的 `best_state` 恢复逻辑与最终方法无关，**不得**用来推断最终方法行为，也不得作为"曾经用过 best checkpoint"的证据。

---

## 7. 产物与证据

| 文件 | 内容 |
|---|---|
| 本报告 | 三方对照与裁决 |
| `results/e26/checkpoint_audit/step_stats.json` | 上表统计的机器可读版本（由 `scripts/e26_b_checkpoint_audit.py` 生成） |
| `scripts/e26_b_checkpoint_audit.py` | 只读脚本：扫描最终 import 链 + 汇总 steps/best_val 统计 |

---

## 8. 一句话结论

**实现用的是 early-stop 时刻的当前权重，`best_val` 仅用于决定何时停；冻结协议与论文面向材料都未规定 checkpoint 策略，故无协议冲突（PASS_A），但必须在正文补一句"不恢复最佳验证权重"。**
