# E13-B · R1 Cross-Seed Stability Validation — Decision Memo

## Verdict: **HOLD**（R1 整体稳健为正，但 Cube 类别未达 PASS 门槛）

## 1. 实验题目
E13-A 最简单 KEEP 候选 R1（Linear Residual Adapter）在 3 个随机种子下是否稳定提升 Bird/Car/Cube 的 AUROC 与 AP？

## 2. 日期
2026-10-14

## 3. 论文阶段
从方法海选进入第一轮稳定性验证。

## 4. 要验证的核心论断
「R1 的 seed0 增益（mean ΔAUROC +0.01540 / ΔAP +0.01171）不是单 seed 偶然性。」

## 5. 实验目的
排除 R1 的 KEEP 是单 seed 运气；验证跨 seed 稳定增益。

## 6. 数据与参数
- 3 categories × 3 seeds = 9 cells；seed0 复用 E13-A（0 GPU）；seed1/2 新增（6 cells）。
- 种子规则：bank = `random.Random(20261009+seed).sample(pool,300)`；训练 `torch.manual_seed(seed)`+`np.random.default_rng(seed)`。
- R1 超参数完全复用 E13-A（L=L_cons+0.1·L_preserve，AdamW lr=1e-3，±15% mild 增强，≤300 step，early-stop 30）。R0=frozen 1-NN-max（每 seed 各自 bank 重算）。

## 7. 实验过程
Git 收尾（E13-A `80929ef` / E12-A `143bd02`）→ 协议冻结 → Bird seed1 smoke（5 项 PASS）→ 6 新 cell（screen `e13b_main`，12.3 min，0 failed）。

## 8. 结果（Fact）

### 8.1 9-cell ΔAUROC / ΔAP 矩阵

| category | seed0 ΔAUROC/ΔAP | seed1 ΔAUROC/ΔAP | seed2 ΔAUROC/ΔAP | 3-seed mean ΔAUROC |
|---|---|---|---|---|
| Bird | +0.01061/+0.01048 | +0.03049/+0.02890 | +0.03647/+0.03176 | **+0.02586** |
| Car | +0.01959/+0.01295 | +0.00785/+0.00922 | +0.00626/+0.01128 | **+0.01123** |
| Cube | +0.01600/+0.01170 | **−0.01229/−0.00321** | +0.00423/+0.00389 | **+0.00265** |

### 8.2 汇总
- overall mean ΔAUROC = **+0.01325**（std 0.01459，min −0.01229，max +0.03647）
- overall mean ΔAP = **+0.01300**（std 0.01104）
- positive AUROC cells = **8/9**，positive AP cells = **8/9**（唯一负 = Cube seed1）
- 无 collapse（pairwise_cos 0.33–0.59，全 < 0.98）

### 8.3 R0 baseline 的 seed 敏感性（重要事实）
frozen R0 本身高度 bank 敏感：Car R0 0.78552→0.79985→0.81399，Cube R0 0.85410→0.87203→0.86528。Cube seed1 的 R0=0.87203 是全矩阵最高的 frozen baseline，而 R1=0.85974 反而退化 → 负 Δ。

## 9. 异常情况
无 NaN / 无 OOM / 无失败 / 无 collapse。唯一异常 = Cube seed1 负 cell（ΔAUROC −0.012），其 R0 异常偏高。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. 8/9 cell 正、overall mean ΔAUROC +0.0132、mean ΔAP +0.0130。
2. Bird 最稳健（3/3，mean +0.0259）；Car 3/3 正但 seed1/2 增益缩至 +0.006~+0.008；Cube 有 1 个负 seed。
3. R0 本身 bank 敏感（同一 frozen 模型，Car R0 跨 seed 差 0.028）。

**Interpretation**
1. **seed0 增益基本可复现**：seed1 mean +0.0087、seed2 mean +0.0157 均为正，overall +0.0132 与 seed0 的 +0.0154 同量级。
2. **Cube 是脆弱类别**：3-seed mean +0.00265 < +0.005，未达 PASS 门槛；根源是 Cube seed1 的 R0 异常高（0.872），adapter 在已极强 baseline 上反而退化。
3. **Car 增益真实但幅度缩小**：从 seed0 +0.0196 降到 seed1/2 +0.008/+0.006，但 3/3 正、方差最小（std 0.0073）。
4. R1 的收益分布「Bird 最强、Car 次之、Cube 不稳」，比 E13-A 单 seed 观察到的「3/3 均 ≥+0.010」更保守。

**Hypothesis（未验证）**
- 猜测 Cube seed1 负值源于 bank 采样：该 seed 的 300 normal 恰好构成极强 frozen baseline，illumination-consistent adaptation 没有额外信息可挖掘。是否稳定需更多 seed 或对 bank 采样的敏感性分析。

## 11. 结论
R1 整体稳健为正（8/9、overall +0.0132/+0.0130），但 Cube 类别未达 PASS 的 per-category +0.005 门槛 → **HOLD**。

## 12. 出口条件判断
- PASS 需 per-category 3-seed mean ΔAUROC ≥ +0.005：Cube +0.00265 **未达标**（其余三项 PASS：overall +0.0132≥0.010、8/9≥7、mean ΔAP +0.0130≥0.005）→ 不满足 PASS。
- HOLD 全满足：overall +0.0132≥0.0075、8/9≥6、per-cat mean 全 ≥0、mean ΔAP +0.0130≥0 → **HOLD**。

## 13. 对论文证据链的贡献
本实验把「R1 是否跨 seed 稳定」这一格填上**部分肯定**答案：R1 的正收益整体可复现（8/9、均值 +0.013），但 Cube 类别存在单 seed 退化，尚未达到可扩大类别的 PASS 标准。这是 R1 从「KEEP」到「可发表方法」之间需要补齐的最后一块稳定性证据。

## 14. 下一步（建议，不自动执行）
- **HOLD**：保留 R1，但不扩大类别；先人工裁决。
- 可选裁决方向（不自动执行）：(a) 对 Cube seed1 做 bank 采样敏感性诊断（是否 bank 混杂）；(b) 增加 seed 数量看 Cube 是否回归均值；(c) 接受「R1 在 Bird/Car 稳定、Cube 不稳」的保守结论。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e13b_screen.py --categories Bird,Car,Cube --seeds 1,2`（screen `e13b_main`）+ `scripts/e13b_analysis.py`
- 输出：`results/e13b/{e13b_protocol.json, e13b_results.csv, e13b_summary.json, e13b_decision_memo.md, README.md, figures/}`
- seed0 复用 E13-A；无 test-label 调参；无超参数修改。
