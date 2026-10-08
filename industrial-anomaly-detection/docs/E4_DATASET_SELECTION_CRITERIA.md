# E4 DATASET SELECTION CRITERIA — PRE-REGISTERED

**建立时间**：2026-10-08（E4-0C，LL-IAD vs M²AD protocol audit）
**起始 HEAD**：`31d4602`（branch `main`；`origin/research/industrial-anomaly` = `31d4602`）
**性质**：**预注册判据**。本文件在任何 LL-IAD / M²AD 资料被检视**之前**写定。
**冻结声明**：以下 MUST / SHOULD / 决策规则一经写入，在本轮审计中**不得因任何数据集的具体情况而修改**。
若事后发现判据本身有缺陷，只能**另立修订文件**并注明修订发生在何时、基于什么证据。

**科研状态（恢复确认）**：

```text
CURRENT STAGE = Phase II — Paper Validation
METHOD SEARCH = CLOSED
FROZEN METHOD = X6c
```

**本轮待解决的主问题**：
> 为 E4 Frozen Held-out External Validation 找到一个能够尽可能**原义计算 Exp16 冻结指标 ΔR + Δd′**，
> 同时明确包含 illumination variation 的外部工业异常数据集。

**本轮已排除（保留历史，不删除）**：
- MVTec AD 2 — `pending / 暂不作主 E4`（E4-1 预审：public test 无 illumination condition label；private/mixed 无本地 GT；ΔR 无法原义计算）。

---

## A. MUST 条件（任一 FAIL ⇒ 原则上不得成为主 E4）

| ID | 条件 |
|---|---|
| **MUST-1** | 存在正常样本。 |
| **MUST-2** | 存在异常 / defect 样本。 |
| **MUST-3** | 存在**明确 illumination variation**。不能只是论文说"环境复杂"。必须能够识别至少 `regular/well-lit` vs `shifted/low-light/other illumination`。 |
| **MUST-4** | illumination condition 对**本地 evaluation 可识别**（folder / filename / metadata / illumination ID 至少一种）。 |
| **MUST-5** | **GT / labels 本地可获得**。不得像 MVTec AD 2 private test 一样全部依赖隐藏 GT。 |
| **MUST-6** | 能够构造 Exp16 的 normal-score robustness comparison，即原则上可以计算 **`ΔR`**。若不能逐字复用，必须明确说明差异。 |
| **MUST-7** | 能够构造 normal-vs-defect separation，即原则上可以计算 **`d′`**，并比较 Original / B2 / X6c 的 **`Δd′`**。 |
| **MUST-8** | **数据集没有参与 X6c 的开发和选择**（真正 held-out）。 |

## B. SHOULD 条件（0 = 不满足 / 1 = 部分满足 / 2 = 明确满足）

`paired illumination`、`same-object pairing`、`pixel mask`、`defect type`、`illumination ID`、
`multiple illumination levels`、`normal-only training compatible`、`industrial setting`、
`multiple categories`、`recognized publication`、`official/public access`、`manageable size`、
`PatchCore compatibility`、`reproducible local evaluation`、`fixed-view illumination analysis possible`。

## C. 评分与决策规则

1. **MUST 任一 FAIL ⇒ 不得成为主 E4**（记录为 FAIL，不得用 SHOULD 分数抵消）。
2. 两者均通过 MUST ⇒ 按以下优先级**机械**决策（**不得**依据"哪个结果可能更好"）：
   1. ΔR / Δd′ 原义兼容性 → 2. illumination isolation → 3. pairing quality → 4. leakage control →
   5. local GT → 6. industrial relevance → 7. compute / storage → 8. publication recognition。
3. 任一关键条件不清楚 ⇒ **宁可 C**。不得为推进项目强选数据集。

## D. Verdict 取值（三选一）

- **A — LL-IAD WIN**：8 MUST 全 PASS；ΔR + Δd′ 至少 Equivalent；pairing / GT 足够明确；无不可控 leakage。
- **B — M²AD WIN**：同上。
- **C — NEITHER**：任一关键条件不清楚。

---

## E. ANNEX — 冻结指标的精确形式（**在检视任何数据集之前**，仅由本仓库 artifact 读出）

本节只记录**本仓库既有实现**的事实，与 LL-IAD / M²AD 无关；用于判定 MUST-6 / MUST-7。
来源：`scripts/experiment5a_h_analysis.py`（`unit_metrics`、`d_prime`、`SHIFTS`）、
`scripts/experiment5a_h_runner.py`（shift 施加方式）、`scripts/experiment14_metrics.py::verdict`、
`scripts/experiment16_analysis.py`。

### E.1 冻结的 illumination 条件

```python
SHIFTS = ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]
```

由 `runner` 通过 `apply_photometric(img, "brightness"|"gamma", level)` **对同一张 clean 图像现场合成**施加。
（sanity：`brightness_0.7` 的身份校验在 `runner` 中硬编码。）

### E.2 `ΔR` 的精确定义

对每个 `(category, seed, method)`：

```text
μ_g, σ_g  ←  clean_good 的分数（test 集 normal，未加扰动）           # normal-only 校准
z_i(shift) = (score_i(shift_good) − μ_g) / σ_g                       # shift_good = 同一批 clean normal 图 + 合成扰动
mean_abs_delta_z(method) = mean_{shift ∈ SHIFTS} | mean_i z_i(shift) |
ΔR(method) = mean_abs_delta_z(method) − mean_abs_delta_z(Original)   # 同 (category, seed) 配对
```

### E.3 `d′` / `Δd′` 的精确定义

```text
d′(defect_type) = (mean(clean_defect[defect_type]) − μ_g) / sqrt((std(clean_defect[dt])² + σ_g²)/2)
mean_dprime     = mean_{defect_type} d′(defect_type)
Δd′             = mean_dprime(method) − mean_dprime(Original)         # 同 (category, seed) 配对
```

### E.4 由上述定义直接推出的两条**硬性数据要求**（本轮 MUST 判定的技术依据）

| 要求 | 由何指标决定 |
|---|---|
| **(a) test 集必须有 normal 样本**（用于 μ_g/σ_g 与 shift_good 的底图） | ΔR 与 d′ 共同要求 |
| **(b) test 集必须有 **clean defect 样本 + defect type 标签**（`mean_dprime` 按 defect type 取平均） | d′ / Δd′ 要求 |

### E.5 ⚠️ 必须在协议层面裁决的一点（**FACT，非判据修改**）

按 E.1–E.3，冻结实现中的 `shift_good` 是**项目自己对 clean normal 施加的 synthetic photometric 变换**，
而不是数据集自带的真实光照条件。因此：

- **就"能否计算出 ΔR 这个数"而言**：只要数据集有 test normal，ΔR 就**永远可算**——
  数据集是否包含真实 illumination variation，**不是 ΔR 可计算性的必要条件**。
- **就"论文 claim 的外部效度"而言**：若仍使用**同 4 个冻结 synthetic 变换**，则 E4 检验的是
  「同一组成像扰动作用在**未见过的图像/样本**上，trade-off 是否复现」，
  而非「**未见过的真实光照条件**下是否复现」。二者科学含义不同。

> **本判据文件不作裁决**：MUST-3 / MUST-4 仍按用户原文执行（要求数据集**本身**含可识别 illumination variation）。
> 上述 FACT 只用于在审计报告中**平行给出两种口径**的结论，供人工裁决。**不得**据此改写 MUST-3 / MUST-4。

---

## F. 本轮禁止事项（与任务书一致）

0 GPU；不运行 PatchCore / Original / B2 / X6c；不下载完整图像数据集；不按模型结果挑类别；
不修改 X6c / C2 / C6 / Exp16 thresholds；不开 X7/X8/X9；不开始正式 E4；不 push。
