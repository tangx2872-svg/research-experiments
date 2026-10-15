# E20 — Paper-era Extension Audit（总报告）

**系列定位**：E20 是 E18 协议中预留的"论文收敛期"编号（`results/e18/protocol.md` §0：E18–E20 禁止修改方法参数）。
本目录承载论文写作期提出的**证据补充审计**任务；全部为**只读盘点 / 协议设计**，不新增 GPU 训练、不下载数据、不修改任何冻结方法。

**系列状态（2026-10-15）**：

| 任务 | 内容 | 状态 | commit |
|---|---|---|---|
| **E20-A** | M²AD 类别与场景扩展预检（路线 A） | **完成**（0 GPU/0 下载） | 待提交（本目录 + `scripts/e20a_*`） |
| **E20-B** | MVTec AD 2 独立数据集预检（路线 B） | **完成**（0 GPU/0 下载） | `0664943` |

> 两任务分工：**E20-A = 路线 A 资源盘点 + 3 个 mini-experiment 候选**；**E20-B = 路线 B 结构/许可核验 + 功效模拟 + AD2 mini-experiment 设计（含 D1–D6 决策清单）**。
> 两者**独立成立、结论并列**；路线选择由研究负责人裁决（见下"联合裁决面板"）。

---

## E20-A — M²AD 扩展预检（路线 A）

**报告**：[`m2ad_expansion_review.md`](m2ad_expansion_review.md)
**一句话结论**：本地已具备三条低-中成本扩展路径；**当前 E16/E18 的真实数据覆盖 = 3 类 × 1 view(120) × detectable-filtered test 图**（Bird 653 / Car 656 / Cube 678，见 `expansion_candidates.csv`）；其余 7 类像素零、metadata 全；一切候选选择待人工批准。

### 关键事实速览

| 项 | 值 |
|---|---|
| 本地原图 | 仅 Bird（15 GB，12 views × 10 illum 全量） |
| 本地缓存（l0_448） | Bird 6400 / Car 4000 / Cube 4000 图；test 覆盖 4 views（090/120/240/330）× 10 illum |
| 官方 metadata | **10 类全部本地**（train 3600/类、test 8400/类、12 views × 10 illum） |
| 其余 7 类（Dice…Tube） | 像素零；view120 评测集 639–688 图/类 |
| 扩展获取通道 | `e8_fetch.py` streaming selective extraction（已验证；≈22 min/类；磁盘 ≈0.8–2.7 GB/类） |
| 磁盘余量 | 16 GB（全量解压不可行；streaming 落缓存可行） |
| GPU 评估成本 | E16 实测 4.5 min/cell（F0+R1-ILA+diagnostics，view120） |

### 三个候选方案（详见主报告 §3）

| ID | 内容 | 证据增益 | 成本 | 风险 |
|---|---|---|---|---|
| **M1** | 新类别泛化（Dice/Doll/Holder 候选，人工批准） | 最高（类别级） | 网络 ≈22 min/类 + GPU 14–28 min | 中（可达性/gate） |
| **M2** | 4-view 测试扩展（Bird/Car/Cube，零获取） | 中高（跨视角+功效） | 纯 GPU ≈0.5–2 h | 低 |
| **M3** | Bird 12-view 全视角（本地原图） | 中高（单类最大化） | CPU 0.5–1 h + GPU ≈1 h | 低 |

## E20-B — MVTec AD 2 预检（路线 B）

**报告**：[`mvtec_ad2_preflight.md`](mvtec_ad2_preflight.md) · 功效模拟：[`auroc_power_sim.json`](auroc_power_sim.json)
**建议决策**：**HOLD**（两个前置条件：(i) 人工完成官方表单/许可并获取数据；(ii) 批准 §5.3 D1–D5 协议决策）。

### 关键事实速览

| 项 | 值 |
|---|---|
| 可用口径 | 仅 TESTpub 可本地评（image-level AUROC/AP，**非官方指标**；光照/缺陷类型均无标注 → 逐光照分解不可做） |
| 三候选场景 | Vial（最合适，291/41/140）· Sheet Metal（137/19/114，4:1 长宽比）· Fruit Jelly（263/37/80，20 good 功效弱） |
| 下载 | 需**人工浏览器表单/许可**；三场景压缩 ≈3.5 GB，磁盘峰值 ≈7 GB（16 GB 余量可容） |
| 统计功效（4,000 次 bootstrap 实测） | AD2 单场景 ΔAUROC 噪声半宽 **≈0.04–0.07**（远大于 M²AD 效应 +0.010~+0.024）；三场景合并 ≈0.03；**必须 pooled 为主分析** |
| 方法冻结 | F0 / R1-ILA（BETA=0.1）零改动；需批准 D1–D6（输入几何/灰度→RGB/bank=全部 train/统计口径等） |
| GPU 预算 | ≈45–60 min（3 seeds × 3 场景）；显存 <4 GB |

---

## 联合裁决面板（研究负责人）

| 维度 | 路线 A（E20-A） | 路线 B（E20-B） |
|---|---|---|
| 证据增益 | 类别级/视角级泛化（同数据集） | 跨数据集（最强攻击面） |
| 统计分辨力 | **高**：M2 扩到 4-view/2700 图后噪声半宽可压到 ≈0.009（对照 M²AD 现状 ≈0.018） | **受限**：单场景半宽 0.04–0.07，须 pooled（≈0.03） |
| 获取成本 | 0（M2/M3）～22 min/类（M1） | 人工表单下载 + ≈3.5 GB；磁盘峰值 ≈7 GB |
| 协议改动 | 无（M1/M2/M3 全部在冻结协议内） | 需 D1–D6 新决策 + 新预注册 |
| 主要风险 | 网络可达性（M1）；无结构性阻断 | 协议语义降级（无光照分解）、功效弱、人工环节 |

**互补性**：两者不互斥——M2（0 成本、纯 GPU）可先执行以增强主结果统计分辨力；M1 与 E20-B 路径可并行推进（M1 待网络复核 + 类别批准；E20-B 待人工下载 + D1–D5 批准）。

### 待批准事项（不自动执行）

1. 路线 A 排序：先 M2（0 成本）还是先批准 M1（含网络）？**M1 类别选择必须人工指定**（E8 预注册顺序 Dice → Doll → Holder 可作依据）。
2. M1 执行前复核：hf-mirror 可达性/带宽；磁盘峰值；新脚本变体预注册（去 E16 历史断言）。
3. 路线 B：是否启动数据获取（CC BY-NC-SA 非商业）＋批准 D1–D5；详见 `mvtec_ad2_preflight.md` §7（含是否允许"下载→解压→删 zip"、是否叠加 AU-PRO 等 8 项）。
4. 两路线如都批准：执行顺序建议 A-M2 →（B 数据到货后）B →（批准后）A-M1。

---

## 产物索引

| 文件 | 归属 | 内容 |
|---|---|---|
| `m2ad_expansion_review.md` | E20-A | 主报告（Task 1–4 全文） |
| `inventory_assets.json` | E20-A | 全资产盘点（缓存/GT/metadata/磁盘） |
| `inventory_summary.csv` | E20-A | 缓存 × 类别 × Good/NG 汇总（58 行） |
| `inventory_crosscheck.json` | E20-A | 缓存 vs 官方 split 逐视图核对（0 unmatched） |
| `category_view_coverage.csv` | E20-A | 10 类 × {train,test} × 12 views 官方计数（240 行） |
| `expansion_candidates.csv` | E20-A | 类别级扩展候选表（10 行） |
| `mini_experiment_proposals.csv` | E20-A | M1/M2/M3 方案元数据 |
| `mvtec_ad2_preflight.md` | E20-B | 路线 B 预检报告（commit `0664943`） |
| `auroc_power_sim.json` | E20-B | AD2 测试规模的 bootstrap 功效模拟 |

脚本（只读/可重跑）：`scripts/e20a_inventory.py` · `e20a_crosscheck.py` · `e20a_candidates.py` · `e20b_auroc_power_sim.py`

### 约束执行记录

E20-A：0 GPU · 0 下载/解压 · 0 网络请求 · 未修改任何冻结脚本（仅新增 `e20a_*`）· 未覆盖任何历史结果 · 未自动选择候选。
E20-B：未下载/未访问下载资源/未接受许可；仅 CPU bootstrap（8.8 s）。
