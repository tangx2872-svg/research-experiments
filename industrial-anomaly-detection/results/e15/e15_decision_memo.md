# E15 · EfficientAD Teacher-Student Feasibility + Mini-Screen — Decision Memo

## Verdict: **STOP**（EfficientAD 在 Bird 上远低于 F0，teacher-student paradigm 未提供足够优势）

## 1. 实验题目
teacher-student anomaly detection（EfficientAD）是否突破 frozen DINOv2-B + 1-NN-max 的 performance ceiling？

## 2. 日期
2026-10-14

## 3. 论文阶段
detection paradigm change feasibility test。

## 4. 要验证的核心论断
「换成 teacher-student paradigm，是否值得继续投入？」

## 5. 实验目的
E11–E14 已穷尽 frozen/scoring/fusion/adaptation 路线；EfficientAD 是唯一未测的不同 paradigm。

## 6. 数据与参数
Bird/Car/Cube，view120，seed0。F0=DINOv2-B 1-NN-max（复用）。EfficientAD：官方 `nelson1425/EfficientAD`（Apache-2.0，commit fcab514），small 模型 / 70000 steps / batch=1 / 256 输入 / teacher_small.pth（仓库自带）/ `pretrain_penalty=False`（无 ImageNet）。

## 7. 实验过程
P0 审计（torch 2.8 实测可跑）→ 协议冻结 → Bird smoke（1000 步 PASS，工程 OK）→ Bird full（70000 步，29.2 min）。Bird ΔAUROC −0.135 < −0.020 → 按协议 STOP，不跑 Car/Cube。

## 8. 结果（Fact）

| category | F0 AUROC/AP | EfficientAD AUROC/AP | ΔAUROC / ΔAP |
|---|---|---|---|
| Bird | 0.74725 / 0.78863 | 0.61206 / 0.62196 | **−0.13519 / −0.16667** |
| Car | 0.78552 / 0.84207 | —（未跑） | — |
| Cube | 0.85410 / 0.91686 | —（未跑） | — |

- train time = 29.2 min（Bird）；n=653（good 320 / ng 333）。

## 9. 异常情况
无 OOM / NaN / 崩溃。工程修复 2 处（纯工程，不涉方法）：(1) symlink farm 幂等化（避免 bulk delete 触发安全拦截）；(2) 输出目录按 train_steps 区分（官方 `os.makedirs` 无 exist_ok，重跑报 FileExistsError）。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. EfficientAD 在 Bird 上 AUROC 0.612，远低于 F0 的 0.747（−0.135）。

**Interpretation**
1. **teacher-student 蒸馏在 M2AD 真实光照（view120）下表现远弱于 frozen DINOv2-B**：EfficientAD 用 DINO-v1-small teacher + 轻量 PDN student，其在 MVTec（可控光照）达 ~0.99，但在 M2AD 多光照下仅 0.612。光照变化使 student 难以稳定匹配 teacher 特征，异常与光照扰动混淆。
2. 结合 E14（AnomalyDINO/SoftPatch 也弱于 F0），**DINOv2-B 的强表示 + 简单 1-NN-max 在本协议下是难以撼动的强 baseline**。

**Hypothesis（未验证）**
- 猜测 EfficientAD 的 teacher（DINO-v1 small，非 DINOv2）表示能力不足是主因；换更强 teacher（DINOv2）可能改善，但已超出本轮「不 sweep teacher」的约束，且工程收益存疑。

## 11. 结论
EfficientAD teacher-student paradigm 在 M2AD 上未提供足够优势（Bird −0.135）。**STOP。**

## 12. 出口条件判断
Stage A 晋级要求 ΔAUROC ≥ −0.010 且 ΔAP ≥ −0.010；实际 −0.135 / −0.167，且 ΔAUROC < −0.020 触发直接 STOP。→ **STOP**，不跑 Stage B/C。

## 13. 对论文证据链的贡献
补齐「换 detection paradigm（teacher-student）」这一格的否定答案。至此 **major detector-family search = CLOSED**：frozen DINOv2-B + 1-NN-max 是最强 baseline，frozen scoring（P4/P5/E12-A）、fusion（E12-B）、adaptation（E13，ceiling-limited）、anomaly-oriented family（E14）、teacher-student paradigm（E15）均无法超越。

## 14. 下一步（人工裁决，不自动执行）
E15 STOP → **major detector-family search CLOSED**。人工裁决只剩：
- A. 重定义论文问题；
- B. 收敛成有限应用型方案；
- C. 回到已有最稳定正信号（R1 illumination-consistent adaptation，Bird/Car 正收益）做小论文包装。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e15_screen.py --categories Bird --train-steps 70000`（screen `e15_efficientad`）+ `scripts/e15_analysis.py`
- 输出：`results/e15/{e15_audit.md, e15_protocol.json, e15_metrics.csv, e15_summary.json, e15_decision_memo.md, README.md, figures/}`
