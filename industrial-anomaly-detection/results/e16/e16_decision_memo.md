# E16 · R1 Signal Qualification — Decision Memo

## Verdict: **GO**（5/5 预注册判据全部满足）

> 本轮是**资格赛**：只回答「R1 值不值得成为论文方法」，不搜索新 detector、不调参救火。

## 1. 实验题目
R1 illumination-consistent adaptation 在 Bird / Car 上的正收益，是否足够稳定、简单、可解释，从而具备进入论文方法阶段的资格？

## 2. 日期
2026-10-14

## 3. 论文阶段
方法资格赛（qualification）——major detector-family search CLOSED 之后，对唯一残余正信号的定性判定。

## 4. 要验证的核心论断
R1 能否 (1) 在 Bird/Car 稳定正收益；(2) 在 Cube control 不造成明显退化；(3) 同时改善或保持 illumination robustness；(4) 保持结构简单、具备应用型论文价值。

## 5. 实验目的
决定 R1 是否进入论文方法阶段（GO→E17），或降级（HOLD→极小 refinement），或终止（STOP→重定义问题）。

## 6. 数据与参数
- 3 categories × 3 seeds × 2 methods（F0 / R1）
- F0/R1 的 AUROC/AP = **REUSED**（E13-A/E13-B，禁止重跑）；d′/robustness/runtime/peak = **NEW**
- 复现身份校验：9/9 cell 的 AUROC/AP 与历史**逐位一致（`repro_ok=True`，容差 1e-6）**
- 环境：RTX 3090 / PyTorch 2.8.0+cu128 / seed 规则 `random.Random(20261009+seed)` + `torch.manual_seed(seed)`

## 7. 实验过程
P0 只读审计 → `git push`（29 commits，`1caecfc..55542e9`）→ protocol.md 冻结 → Bird/seed0 smoke PASS → 9 cell 确定性复现（40.5 min，0 failed，0 REPRO_FAIL）→ paired bootstrap + diagnostic。

## 8. 结果（Fact）

### 8.1 Raw metrics（view120；AUROC/AP = REUSED，其余 = NEW）

| category | seed | AUROC F0→R1 | AP F0→R1 | d′ F0→R1 | rob_feat F0→R1 | rob_score F0→R1 |
|---|---|---|---|---|---|---|
| Bird | 0 | 0.74725→0.75786 | 0.78863→0.79911 | 0.916→0.957 | 0.0760→0.0121 | 1.328→0.328 |
| Bird | 1 | 0.75093→0.78142 | 0.79216→0.82107 | 0.923→1.023 | 0.0766→0.0125 | 1.336→0.308 |
| Bird | 2 | 0.75576→0.79223 | 0.79174→0.82349 | 0.939→1.083 | 0.0761→0.0118 | 1.387→0.321 |
| Car | 0 | 0.78552→0.80511 | 0.84207→0.85502 | 1.042→1.100 | 0.0517→0.0100 | 0.986→0.281 |
| Car | 1 | 0.79985→0.80770 | 0.85393→0.86315 | 1.096→1.126 | 0.0513→0.0098 | 0.961→0.263 |
| Car | 2 | 0.81399→0.82025 | 0.85405→0.86533 | 1.112→1.136 | 0.0520→0.0101 | 0.962→0.285 |
| Cube | 0 | 0.85409→0.87010 | 0.91686→0.92856 | 1.440→1.500 | 0.0361→0.0083 | 0.885→0.220 |
| Cube | 1 | 0.87203→0.85974 | 0.92662→0.92342 | 1.515→1.431 | 0.0346→0.0081 | 0.913→0.227 |
| Cube | 2 | 0.86528→0.86952 | 0.92290→0.92679 | 1.482→1.498 | 0.0357→0.0076 | 0.886→0.217 |

### 8.2 预注册判据逐条

| # | 判据 | 实测 | 结果 |
|---|---|---|---|
| 1 | Bird+Car 6 cell 中 ≥4/6 ΔAUROC>0 | **6/6** | ✅ |
| 2 | Bird+Car mean ΔAUROC ≥ +0.010 | **+0.018545** | ✅ |
| 3 | Bird+Car mean ΔAP ≥ 0 | **+0.017433** | ✅ |
| 4 | robustness 总体方向改善 | mean Δrob_feat **−0.0529**；mean Δrob_score **−0.8625**；最差 cell 相对变化 **−0.704**（全部改善） | ✅ |
| 5 | Cube mean ΔAUROC ≥ −0.005 | **+0.002647** | ✅ |

→ **GO**

### 8.3 Per-category（3-seed mean）

| category | mean ΔAUROC | mean ΔAP | mean Δd′ |
|---|---|---|---|
| Bird | **+0.02586** | +0.02371 | +0.0948 |
| Car | **+0.01123** | +0.01115 | +0.0375 |
| Cube | +0.00265 | +0.00413 | −0.0028 |

### 8.4 Paired bootstrap（test-image resampling，detectable-filtered，3000 draws）

| cell | ΔAUROC | 95% CI | P(Δ>0) |
|---|---|---|---|
| Bird\|0 | +0.01061 | [−0.0076, +0.0298] | 0.866 |
| Bird\|1 | +0.03049 | **[+0.0121, +0.0491]** | 0.999 |
| Bird\|2 | +0.03647 | **[+0.0180, +0.0561]** | 1.000 |
| Car\|0 | +0.01959 | **[+0.0026, +0.0361]** | 0.988 |
| Car\|1 | +0.00785 | [−0.0126, +0.0273] | 0.780 |
| Car\|2 | +0.00626 | [−0.0110, +0.0245] | 0.770 |
| Cube\|0 | +0.01600 | **[+0.0008, +0.0314]** | 0.979 |
| Cube\|1 | −0.01229 | [−0.0291, +0.0044] | 0.072 |
| Cube\|2 | +0.00423 | [−0.0120, +0.0208] | 0.705 |

### 8.5 Diagnostic（A–E，seed 内平均）

| 代号 | 指标 | Bird | Car | Cube |
|---|---|---|---|---|
| A | normal dispersion（pairwise cos ↑ = 更聚集） | 0.889 → **0.985** | 0.902 → **0.980** | 0.910 → **0.965** |
| A | patch std | 1.353 → **2.005** | 1.419 → **2.067** | 1.446 → **2.064** |
| B | illumination-group centroid distance | 0.0652 → **0.0074** | 0.0334 → **0.0056** | 0.0314 → **0.0111** |
| C | good-vs-defect centroid distance | 0.00190 → 0.00017 | 0.00047 → 0.00010 | 0.00226 → 0.00050 |
| D | illumination shift magnitude | 0.2378 → **0.1172** | 0.1826 → **0.0966** | 0.1378 → **0.0777** |
| E | adaptation centroid drift | 0.1817 | 0.1696 | 0.2490 |

## 9. 异常情况
- 无 OOM / NaN / 无失败 cell / 9-9 `repro_ok=True`。
- **工程/实现修正 1 处（记于 protocol.md 附注 1）**：首次身份校验失败（失配 ~4e-6）。决定性诊断证明 DINO 前向与打分在同进程内**完全确定性**（`max|Δ|=0.0`）；真因是历史 CSV 为 `f"{x:.5f}"` 的 5 位小数展示格式（有损 ~4e-6），权威全精度记录在 partial JSON。修正 = history loader 改读 partial JSON。**1e-6 容差、GO/HOLD/STOP 判据、R1 超参、指标定义全部未变**，且修正发生在任何 d′/robustness 数值被读出之前。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. Bird+Car 6/6 ΔAUROC 为正，mean +0.0185，mean ΔAP +0.0174（全为正）。
2. Cube mean +0.0026（3 seed：+0.0160 / −0.0123 / +0.0042），≥ −0.005 门槛，未受损。
3. 配对 bootstrap：9 cell 中 **4 个 CI 排除 0**（Bird\|1、Bird\|2、Car\|0、Cube\|0，全为正）；Bird 的两格最硬。
4. 两类鲁棒性指标**全部 9 cell 同向改善**：特征级 1−cos 由 0.035–0.077 降到 0.008–0.013；分数级归一化漂移由 0.89–1.39 降到 0.22–0.33。
5. d′ 8/9 改善（唯一例外 Cube seed1 −0.084）。
6. Diagnostic：**illumination shift D 与 illumination-group centroid distance B 的类别排序（Bird > Car > Cube）与 ΔAUROC 排序完全一致**；而 A（normal dispersion）**升高**（0.89→0.98 的 pairwise cos，std 1.35→2.0）。

**Interpretation**
1. R1 在光照敏感类别（Bird/Car）产生**一致、跨 seed、AP 同向**的正收益；在 Cube（control）不造成明显退化。满足应用型论文的「稳定 + 简单 + 不伤 control」三条件。
2. **判据 4 需谨慎解读**：特征级 rob_feat 正是 R1 的训练目标（L_cons），其改善**部分由构造保证**，不构成独立证据。更有说服力的是：(a) **分数级 rob_score**（异常分数对光照的稳定性，非直接优化目标）下降 67–77%；(b) **diagnostic B/D**（illumination-group centroid distance 与 shift magnitude）大幅下降（B 降 65–89%，D 降 50–56%）。
3. **"consistent with" illumination-shift 假说**：shift 越大的类别收益越大（Bird 0.238→收益 +0.026；Car 0.183→+0.011；Cube 0.138→+0.0026），排序完全一致。
4. **一个反直觉的事实必须如实记录**：R1 **提高**了 bank normal 特征的聚集度（pairwise cos 0.89→0.98，patch std 1.35→2.0）——即它并非"增加多样性"，而是把 normal 流形**收紧**并**放大尺度**。因此本轮**不支持**"dispersion reduction 导致增益"的简单叙事；支持的是"**illumination-induced shift reduction**"。good/NG 质心距离（C）也下降，但分数级分离度（d′）上升 8/9，说明可分性来自**分数空间**而非原始质心。
5. R1 极简：`z' = z + Wz`（589,824 参数，zero-init 线性残差），≤300 step 于**缓存特征**上训练（44–124 step 早停），单 cell ≈ 4.4 min，峰值显存 5.4 GB。

**Hypothesis（未验证，不得当结论）**
- 猜测 R1 学到的是"把 normal 流形沿光照扰动方向压缩、同时保持缺陷方向"的线性映射；E 显示 adaptation 确实显著移动了表示（drift 0.16–0.25）。
- 本轮**不能**外推为 universal robustness，也不能声称因果机制。

## 11. 结论
R1 通过全部 5 条预注册 GO 判据。**GO**：R1 具备进入论文方法阶段的资格。

## 12. 出口条件判断
- GO 五条全部满足（§8.2）→ **GO**。
- HOLD 分支未触发（Bird+Car mean +0.0185 不在 [+0.005,+0.010) 带；Bird/Car 两类都稳；robustness 判据成立）。
- STOP 分支未触发（mean ≥ +0.005；Bird/Car 无符号翻转；去掉各自最好 seed 后 Bird 仍 +0.0205、Car 仍 +0.0071，均 ≥ +0.005；Cube 未被损害）。

## 13. 对论文证据链的贡献
Major detector-family search 已 CLOSED，唯一残余正信号是 R1。本实验把这一格从「seed0 KEEP 但 Cube 不稳」升级为 **「跨 3 seed 稳定、AP 同向、control 类别不受伤、robustness 双向改善、且与 illumination-shift 幅度排序一致」**，并明确记录了两条限制（rob_feat 部分自证；normal dispersion 反升）。R1 由此获得论文方法资格。

## 14. 下一步（按协议，**不自动执行**）
```
NEXT = E17 Lightweight Refinement
```
建议仅 3–5 个非常小配置，且必须等人工裁决。**本轮不自动跑 E17。**

## 15. 复现信息
- 命令：`python scripts/e16_qualify.py --categories Bird,Car,Cube --seeds 0,1,2`（screen `e16_qualify`）+ `scripts/e16_analysis.py` + `scripts/e16_bootstrap.py`
- 输出：`results/e16/{protocol.md, audit.md, metrics_raw.csv, summary.csv, diagnostics.csv, bootstrap.json, verdict.json, e16_partial.json, decision_memo.md, README.md, figures/, logs/}`
- 总耗时：9 cell 40.5 min（+ smoke ~3 min）；无 test-label 调参；无超参修改。
