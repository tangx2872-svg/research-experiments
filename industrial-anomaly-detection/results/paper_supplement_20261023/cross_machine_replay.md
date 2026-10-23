# 跨机独立复算记录（cross-machine replay）

> 日期：2026-10-23 ｜ 目的：证明 2026-10-23 审计脚本在 Linux（服务器）与 Windows（本地）两套环境下可复现。
> 方法：脚本复制到临时目录、`OUTDIR` 改到 `/tmp/replay_out`，跑完即删，**未触碰仓库内任何已入库产物**。
> 环境差异：服务器 = Linux / Python 3.12.3 / numpy **2.3.2** / sklearn 1.9.1；本地 = Windows / Python 3.11.16 / numpy **2.4.6** / sklearn 1.9.1。

## 1. 复算结果对照

| 产物 | 与仓库版本比对 | 判定 |
|---|---|---|
| `sample_alignment_audit.json` | **逐字节相同** | ✅ byte-identical |
| `audit_a3_main_table_reconciliation.json` | **逐字节相同** | ✅ byte-identical |
| `audit_a3_specimen_cluster_bootstrap.json` | **逐字节相同** | ✅ byte-identical |
| `audit_a2_aggregation_reconciliation.json` | 3 个浮点末位差，max\|Δ\|=**3.5e-18**（相对 3.2e-15） | ✅ value-identical（浮点累积序差异，量级为机器精度） |
| `metric_reconciliation.csv` | 数值**全等**（max\|Δ\|=0.0，39/39 OK），仅**行尾** CRLF vs LF | ✅ value-identical（换行符平台相关） |

## 2. 关键复现值（服务器侧独立得到，与本地方向一致）

- **A1**：39/39 cell，`max_absdiff=0.0`。
- **A2**：PCB `+0.01835185` / MetalParts `−0.01725116` / 11-split `−0.00106797` / 子集等权 `+0.00055035`，全部 match。
- **A3**：macro `0.007596`、median `0.010796`、Doll `std=0.004638`。
- **bootstrap（specimen 级）**：Doll `−0.03549`，CI `[−0.09135, +0.01686]`，P(>0)=`0.101`。

## 3. 换行符平台差异（重要，供后续跨机对账口径）

`metric_reconciliation.csv` 的换行符是平台相关的：

- Windows 首次生成是 CRLF，被 git 规范化成 LF 后入库；
- 服务器新跑又是 CRLF。

因此**跨机不能用字节比对这份 CSV**，须用「值比对」或「直接与冻结 cell 对账」。

## 4. 可复现性结论（可写入 HANDOFF 的口径）

> JSON 产物 3/5 字节一致、2/5 数值一致到 1e-18；CSV 值一致但行尾依赖平台。
> 所有关键标量（AUROC/AP/mean/std/CI）在两套 numpy 版本下均逐位一致，审计脚本跨平台可复现成立。
