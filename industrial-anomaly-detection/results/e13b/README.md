# E13-B · R1 Cross-Seed Stability Validation

**日期**：2026-10-14
**论文阶段**：方法海选 → 第一轮稳定性验证
**Verdict**：**HOLD**（R1 整体稳健为正，Cube 类别未达 PASS 门槛）

## 一句话结论
R1 的 seed0 增益**基本可复现**（8/9 cell 正，overall mean ΔAUROC +0.0132 / ΔAP +0.0130），但 **Cube 类别 3-seed mean ΔAUROC +0.00265 未达 +0.005 的 PASS 门槛**（根源是 Cube seed1 一个负 cell），因此判定 **HOLD**，不扩大类别。

## 9-cell 结果（view120 / frozen DINOv2-B + 1-NN-max）

| category | seed0 | seed1 | seed2 | 3-seed mean ΔAUROC |
|---|---|---|---|---|
| Bird | +0.01061 | +0.03049 | +0.03647 | **+0.02586** |
| Car | +0.01959 | +0.00785 | +0.00626 | **+0.01123** |
| Cube | +0.01600 | **−0.01229** | +0.00423 | **+0.00265** |

- overall mean ΔAUROC = **+0.01325**（std 0.01459），mean ΔAP = **+0.01300**
- positive AUROC cells = **8/9**，positive AP cells = **8/9**（唯一负 = Cube seed1）
- 无 collapse

## decision memo 八问八答
1. **seed0 增益可复现？** 是。seed1 mean +0.0087、seed2 mean +0.0157 均正，overall +0.0132 与 seed0 +0.0154 同量级。
2. **overall mean ΔAUROC / ΔAP？** +0.01325 / +0.01300。
3. **9 cell 几正几负？** 8/9 正 AUROC、8/9 正 AP（唯一负 = Cube seed1）。
4. **方差最大类别？** Cube（std 0.0142），其次 Bird（0.0135）；Car 最稳（0.0073）。
5. **Car 提升仍存在？** 是，但幅度缩小（seed0 +0.0196 → seed1/2 +0.008/+0.006），仍 3/3 正。
6. **某 seed 崩掉？** 是——Cube seed1 ΔAUROC −0.01229（该 seed 的 frozen R0 异常高 0.87203，adapter 退化）。
7. **值得扩类别？** 按 HOLD 规则暂不扩大，先人工裁决（Cube 的 seed 不稳定是主要顾虑）。
8. **PASS/HOLD/STOP？** **HOLD**（PASS 因 Cube per-cat mean +0.00265 < +0.005 未达标；HOLD 全满足）。

## 关键文件
- `e13b_protocol.json`、`e13b_results.csv`、`e13b_summary.json`、`e13b_decision_memo.md`、`README.md`、`figures/`（4 张图）、`logs/`
- 复现：`python scripts/e13b_screen.py --categories Bird,Car,Cube --seeds 1,2` + `scripts/e13b_analysis.py`

## 对论文证据链的贡献
把「R1 是否跨 seed 稳定」填上**部分肯定**答案：正收益整体可复现（8/9、+0.013），但 Cube 类别存在单 seed 退化，未达扩大类别的 PASS 标准。这是 R1 从 KEEP 到可发表方法之间需补齐的最后一块稳定性证据。
