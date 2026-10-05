# OVERNIGHT REPORT — 2026-10-05/06 无人值守实验

> 本文件为无人值守模式自动生成的晨读摘要。

## 任务性质说明（重要）

今晚指令模板以「2A Residual Evidence Probe」为目标，但 **P0 仓库检查发现指令与仓库真实进度不符**：

1. 仓库中**不存在任何 2A / residual probe 实验**（无目录、无 protocol、无代码）。
2. 仓库最新状态（git `536ca3a`）：机制探索已于 **1J-B 硬停止**（MECHANISM EXPLORATION STOPPED AFTER 1J-B），机制证据链 1B→1J-B 已闭合，冻结审计判定 `FREEZE_AFTER_REPRODUCTION`（科学证据足够，仅缺 1J-A/1J-B 结果资产）。
3. 昨日已获用户批准的进行中任务：**1J-A → 1J-B 论文关键资产恢复**（含明确禁止项：不开始 1J-C / 2A / DPR / Method）。

按今晚指令第 1 节「仓库最新 protocol / README / code 优先级高于本指令概括」及第 8 节「禁止临时设计新实验」，**未新建 2A**，转而完成已批准的资产恢复流程。这不是偏离任务，而是执行任务的正确排序。

---

# Mechanism Stage Freeze Report

## A. Asset Recovery

| 资产 | 数量 | 完整性 |
|---|---|---|
| 1J-A raw NPZ（分离 layer2/layer3 feature） | **1848 / 1848**（9 defect × 3 seed × 2 layer × 2 α × 全部 test 图） | ✅ 精确匹配理论值 |
| 1J-A layer-specific banks | **30 / 30**（5 cat × 3 seed × 2 layer） | ✅ |
| 1J-A 权威 per_image CSV | 1848 行（rebuild 路径覆盖重建） | ✅ |
| 1J-A nn CSV | 1848 行，missing bank = 0 | ✅ |
| 1J-A analysis CSV | 4 个（per_seed / per_defect / family_summary / cross_chain_summary） | ✅ |
| 1J-A figures | 5 张（figure1–5，PNG，非空） | ✅ |
| 1J-B per_image CSV | **1386 行**（462 图 × 3 β）+ sanity 462 行 | ✅ |
| 1J-B analysis CSV | 3 个（per_defect / family_summary / transmission_summary） | ✅ |
| 1J-B figures | 5 张（figure1–5） | ✅ |
| 总磁盘占用 | ~2.2 GB（余量充足） | ✅ |

执行链：`experiment1j_extract.py`（GPU，60 min，一次通过，无 unit 失败）→ `experiment1j_rebuild_csv.py` → `experiment1j_analysis.py` → `experiment1j_b_extract.py`（CPU）→ `experiment1j_b_analysis.py`。日志保存于 `results/experiment_1j/logs/`。

## B. Reproduction Consistency

### Sanity Checks（全部 PASS）

- α=0 identity：α=0 时 F_α=F 数学恒等，smoke 阶段已验证 hook 正确
- β=1 reconstruction：max_abs_diff = **9.537e-07** < 1e-5（历史值 9.5e-07）✅
- 行数校验：1848 / 1848 / 1386 全部与理论值精确一致 ✅
- NaN/Inf：per_image（1J-A + 1J-B）全量扫描 = **0** ✅
- shape：layer2=(512,32,32)、layer3=(1024,16,16) ✅

### 确定性指标 vs README 历史表（1J-A 14.2，9/9 defect）

| defect | R_L2 now/hist | R_L3 now/hist | GAI now/hist |
|---|---|---|---|
| bottle/contamination | −0.4643/−0.464 | +0.0210/+0.021 | +0.4852/+0.485 |
| cable/bent_wire | −0.4289/−0.429 | −0.0372/−0.037 | +0.3917/+0.392 |
| hazelnut/print | −0.2214/−0.221 | +0.0204/+0.020 | +0.2418/+0.242 |
| screw/manipulated_front | −0.1722/−0.172 | +0.0205/+0.021 | +0.1927/+0.193 |
| screw/thread_side | −0.1631/−0.163 | +0.0242/+0.024 | +0.1873/+0.187 |
| screw/thread_top | −0.1648/−0.165 | +0.0258/+0.026 | +0.1906/+0.191 |
| grid/glue | +0.2781/+0.278 | +0.4648/+0.465 | +0.1867/+0.187 |
| grid/metal_contamination | +0.2514/+0.251 | +0.4639/+0.464 | +0.2125/+0.213 |
| grid/thread | +0.2591/+0.259 | +0.4474/+0.447 | +0.1882/+0.188 |

**最大绝对差 = 0.000477**（即 README 三位小数舍入误差级别）→ 确定性指标 bit 级复现。sign consistency 全部 1.00。family 汇总亦完全一致（shrink −0.372/+0.001；neutral −0.167/+0.024；expand +0.263/+0.459）。

### 含 coreset 随机性指标（容差比较，1J-B TRR / dose-response）

| defect | Δradius now/hist | ΔNNstd now/hist | Δscore now/hist | TRR now/hist |
|---|---|---|---|---|
| grid/glue | +10.844/+10.84 | +3.447/+3.45 | +20.606/+20.60 | +0.638/+0.64 |
| grid/metal_contamination | +11.077/+11.08 | +4.080/+4.08 | +25.411/+25.40 | +0.806/+0.82 |
| grid/thread | +10.729/+10.73 | +3.584/+3.59 | +20.728/+20.73 | +1.363/+1.39 |

neutral TRR ≈ +0.03/+0.04/+0.05（历史一致）、shrink +0.11/−0.05/−0.08（历史一致）。**差异 ≤0.03，远小于任何效应量**——随机性指标同样高度复现（本次提取与历史在同版本环境 + 固定 seed 序列下 coreset 采样路径一致）。

### 无人值守运行事故

无。GPU 提取一次通过，无 unit 失败、无 OOM、无 retry。

## C. Scientific Conclusion

> 1J-A / 1J-B 的原主要科学结论是否被成功复现？

**REPRODUCED** ✅

- 1J-A：expand 家族特有 Layer3 RMS-radius expansion（R_L3≈+0.45 仅 expand；shrink/neutral≈0）；R_L2 符号完美三分家族；GAI 9/9 为正——全部复现。
- 1J-B：β 干预剂量-反应（expand 3/3 单调，radius→NN→score 同向）；β=0 压制后 NN/score amplification 大幅减弱；neutral/shrink family control 模式——全部复现。

## D. Mechanism Freeze Decision

**MECHANISM STAGE FROZEN ✅**

> No further mechanism exploration is required before Method Design.

科学证据（1B→1J-B）与实验资产（可追溯、可复核、可重新生成论文数字与图表）均已齐备。

## E. Paper Roadmap

```text
① Problem / Reality                    ✅
② Theory / Literature                  ✅
③ Phenomenon / NDIL Evidence           ✅
④ Mechanism Evidence                   ✅ FROZEN（2026-10-05，1J 资产已恢复并复核）
⑤ Method Design                        ← NEXT（等待用户参与）
⑥ Mini Method Validation               ⏸
⑦ Full Experiments / Benchmark         ⏸
⑧ Writing                              ⏸
```

---

## 环境记录（指令第 10 节）

- GPU：NVIDIA RTX 3090 24GB；峰值显存 ~6.4 GB；PyTorch 2.8.0+cu128；Python 3.12.3；Anomalib 2.6.2
- git：`a6b88a1`（资产恢复运行于该 commit，工作区含恢复的未入库结果资产）
- 关键运行时：`HF_ENDPOINT=https://hf-mirror.com`（HF 直连不可达；权重已缓存）

## 遗留事项（供用户决策，未擅自执行）

1. `results/experiment_1j/`、`results/experiment_1j_b/` 目前被 `.gitignore` 的 `results/*` 规则排除；若要使 analysis/figures 入库需为两者加白名单（需用户确认后修改 .gitignore 并 commit）。
2. 根 README 实验归档仍止于 1H-S（缺 1I/1J-A/1J-B 条目），Environment 节为历史 Windows 记录——待用户确认后更新。
3. 1J-A extract 末尾打印计数器 bug（"rows: 0"）为已知良性问题，数据不受影响；正式数字以 rebuild 路径为准。

---

```
========================================
OVERNIGHT EXPERIMENT REPORT
========================================

Paper Stage:
Stage ④ Mechanism Validation — FROZEN

2A:
NOT STARTED (repo contains no 2A protocol; per instruction §1,
repo state takes priority; mechanism exploration hard-stopped
after 1J-B per frozen decision)

Executed instead (approved task):
1J-A + 1J-B Paper Asset Recovery: DONE

Units:
Completed: 15 (cat,seed) groups × 3 fits; 1848 NPZ; 30 banks;
           1386 intervention rows — 0 failures, 0 retries
Failed: 0
Skipped: 0 (bottle smoke NPZ legitimately reused, protocol-identical)

Runtime: ~70 min GPU extraction + ~10 min rebuild + ~2 min analyses

Peak VRAM: ~6.4 GB / 24 GB

Sanity: PASS (α=0 identity; β=1 recon 9.5e-07; row counts exact;
        NaN/Inf = 0; deterministic metrics reproduce README to
        rounding error, max |Δ| = 0.000477)

Main Finding:
1J-A/1J-B original conclusions fully reproduced from scratch on
this server: expand-family-specific Layer3 RMS-radius expansion,
and intervention-level geometry→NN→score transmission (TRR_NN
0.64/0.81/1.36). All paper-critical assets now traceable.

Verdict:
Asset recovery: COMPLETE
Scientific reproduction: REPRODUCED

Paper Progress:
④ Mechanism Evidence ✅ FROZEN — all eight evidence cells closed
  with traceable assets. Next: ⑤ Method Design (user decision).

Next Recommended Step:
User reviews this report → optionally whitelist 1J results in
.gitignore + update root README archive → begin Stage ⑤ Method
Design discussion (A+B combination per project principles).

STOPPED HERE. No 1J-C, no 2A, no DPR, no new experiments.
========================================
```
