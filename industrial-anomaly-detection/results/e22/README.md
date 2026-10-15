# E22-P0/P1 — M2AD 7 个新类别 `l0_448` 缓存获取（R1-ILA 跨类别扩展的数据前置阶段）

**日期**：2026-10-15（18:33:06 → 21:13:28，含 13 s dry-run）
**执行**：`scripts/e22_manifest.py`（P0，元数据盘点）、`scripts/e22_fetch.py`（P1，流式选择性取数）
**上游 commit**：`8957237`（E21: validate R1-ILA across M2AD views）
**输出目录**：`results/e22/`、缓存写入 `data/m2ad/e7_cache/l0_448/<Category>/`（gitignored）

---

## 1. 论文阶段

证据链中的**跨类别外部有效性**扩展阶段。E20-A 盘点给出的三条扩展路径（M1 新类别 / M2 4-view / M3 Bird 12-view）中，本实验执行 **M1 的第一步：数据获取**。尚未进入 GPU 评估阶段。

## 2. 本实验要服务的那句话（先说清楚它不检验什么）

**本实验不检验任何性能断言，也不提供任何性能证据。**

它只把下面这句话从「不可检验」变为「可检验」：

> R1-ILA（DINOv2-B bank 归一化精修）在 M2AD 上的增益不限于 Bird / Car / Cube 三类。

该断言的真假，必须由后续 GPU 单元格（新类别 × 冻结 seeds 的 F0 与 R1-ILA）判定。E22 只负责消除"数据不可得"这一替代解释。

## 3. 实验目的

在不整包解压（单类 zip 15.0–17.5 GB 十进制，磁盘仅剩 16 GB）的前提下，只获取**冻结 R1-ILA 协议实际读取的像素**，并逐类施加硬 gate，使后续 GPU 阶段可执行：

- bank 池：`sorted(train_specs)[::3][:10]` × 12 views × 10 illum = **1200 图/类**（冻结规则，`e4d1_m2ad_loader.bank_and_val_specimens`）
- 评测集：view120 test 官方划分全部原始记录（**700 图/类**，Holder 690）
- **不获取**：`l0_256`、`clahe_mild/medium`、GT mask、原图（R1 增强在 448 图上施加，见 `e13a_screen.prep_image`）

## 4. 数据与参数

| 项 | 值 |
|---|---|
| 数据集 | M2AD（HF 镜像 `hf-mirror.com/datasets/ChengYuQi99/M2AD`），元数据 `data/m2ad/jsons_extract/meta_unsupervised.json` |
| 新类别（7） | Dice, Doll, Holder, Motor, Ring, Teapot, Tube |
| 缓存种类 | 仅 `l0_448` |
| 派生路径 | `TF.resize(rgb, [448,448], antialias=True, BICUBIC)` → WebP lossless q100 method 0（与 `e8_fetch.write_image_caches` 逐字一致，`.png` 扩展名、WebP 容器，与既有 Car/Cube 缓存签名相同） |
| 随机性 | 取数本身无随机性（路径全部由官方元数据决定）；下游 bank 抽样使用冻结 seed `BANK_SUBSET_SEED=20261009` |
| 环境 | conda `base`，Python 3.12.3，PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128，Anomalib 2.6.2，OpenCV 5.0.0，RTX 3090 24 GB（本实验 CPU/网络密集，未使用 GPU） |

## 5. 实验过程

```
# P0 元数据盘点（只读）
python scripts/e22_manifest.py
# P1 dry-run（范围确认，无下载）
python scripts/e22_fetch.py --categories Dice,Doll,Holder,Motor,Ring,Teapot,Tube --dry-run
# P1 正式取数（后台，脱离会话）
setsid nohup /root/miniconda3/bin/python -u scripts/e22_fetch.py \
  --categories Dice,Doll,Holder,Motor,Ring,Teapot,Tube \
  > results/e22/logs/fetch_20261015_183306.log 2>&1 &
```

- dry-run 确认：pool=1200（7/7 类）、test120=700（Holder 690）、缺失=1900/1890，磁盘 16.2 GB 可用。
- 后台 PID 335905；`results/e22/status.json` 在每个类别完成后原子写入；逐成员进度每 250 成员打印（members / kept / streamed / 速率 / 已用时间 / 剩余磁盘）。
- resume：重新运行前先扫描既有 `l0_448` 文件，已存在者从待取集合中剔除；单成员解码/写盘失败即中止（不静默跳过）。
- gate：取数后逐文件校验存在性 + 448×448 + 色彩模式。

## 6. 结果

### 6.1 逐类（事实）

| 类别 | pool | test120 | 归档(GiB) | 流式读取(GiB) | 耗时 | 缓存体积 | gate missing | bad_shape |
|---|---|---|---|---|---|---|---|---|
| Dice | 1200 | 700 | 15.54 | 15.53 | 16m44s | 401 MB | 0 | 0 |
| Doll | 1200 | 700 | 15.68 | 15.68 | 17m04s | 390 MB | 0 | 0 |
| Holder | 1200 | 690 | 16.27 | 16.27 | 18m56s | 391 MB | 0 | 0 |
| Motor | 1200 | 700 | 13.93 | 13.93 | 16m25s | 413 MB | 0 | 0 |
| Ring | 1200 | 700 | 15.43 | 15.43 | 25m57s | 385 MB | 0 | 0 |
| Teapot | 1200 | 700 | 14.68 | 14.68 | 42m59s | 359 MB | 0 | 0 |
| Tube | 1200 | 700 | 15.56 | 15.56 | 20m28s | 381 MB | 0 | 0 |
| **合计** | 8400 | 4890 | 107.08 | 107.08 | **2h38m33s** | **2.626 GiB / 13,290 文件** | 0 | 0 |

- 磁盘：16.2 GB → 13.3 GB 可用（净增 ≈2.9 GB，含 1.0 MB 清单与 40 KB 日志）。
- 平均吞吐 ≈11.5 MiB/s（波动 6–16 MB/s；Teapot 段带宽下降导致 43 min）。
- **早停收益≈0**：想要的成员分散在归档中，最后一个成员几乎都在归档尾部，实际流式读取量≈整包（≥99.9%）。

### 6.2 独立复核（不依赖脚本自身 gate）

- 文件数逐类精确匹配预期：1900/1900/1890/1900/1900/1900/1900；零字节文件 0 个；每类抽样 3 张解码均为 448×448 RGB。
- 容器签名与既有缓存一致：`RIFF….WEBP`（Car 与 Dice 头部字节相同）。
- **pool 规则等价**：磁盘上 10 个 Good specimen 目录与 `estimate_scope()["bank_specimens"]` 完全一致（Dice 例：`000,009,012,019,022,025,029,032,039,048`），每 specimen 恰好 12 views × 10 illum = 120 图。
- **test 视图纯度**：非 pool 文件全部为 `A120_*`（无其他 views 混入）。
- **resume 复核**：直接调用 `fetch_category(..., dry_run=True)`，7/7 类报告 `present=1900/1890, missing=0`（不写状态文件）。
- 完整性清单：`results/e22/cache_l0_448_manifest.csv`（13,290 行 path/size/md5）+ 聚合
  `sha256 = 1569cc57f8c316056b3f445c13779e36aee6e861433c020c3d0a683a16e47812`
  （见 `cache_l0_448_aggregate_sha256.txt`；该 sha256 覆盖每个文件的 path+size+md5，可在缓存损坏或迁移后复验）。

### 6.3 冻结记录

`results/e22/status_fetch_20261015.json`（取数结束时的 `status.json` 快照，含逐类 streamed_gb / seconds / gate 明细 / free_gb_after）；原始 stdout：`results/e22/logs/fetch_20261015_183306.log`（390 行）。

## 7. 异常情况

### 7.1 【实现缺陷，已修复，无数据影响】status 文件写冲突

- **bug 位置**：`e22_fetch.stream_e22` 把流式进度写入 `results/e22/status.json` —— 与 `main` 的 master 记录同一路径。
- **为什么是 bug**：流式进行中，master 记录（已完成类别清单）被逐成员状态临时覆盖；若进程在类别中途被杀，崩溃时刻的 master 记录丢失。
- **影响范围**：仅崩溃期 bookkeeping。缓存像素与 gate 完全不受影响；resume 依据是既有缓存文件扫描（已单独验证 present=1900/missing=0），因此**无需重跑，历史结果无效力损失**。
- **修复**：`write_status(state, path=None)`，流式进度改写入 `results/e22/stream_progress.json`（2026-10-15）；已验证修复后写流式状态时 master `status.json` 的 md5 不变。
- **修订说明**：本次 7 类数据由**修复前**的脚本修订产生（数据路径与缓存生成代码路径在修复中未被触碰）；修复后的文件已纳入版本控制，后续 resume 运行将使用新修订。

### 7.2 网络吞吐波动

Ring 25m57s、Teapot 42m60s（其他类 16–20 min），吞吐 6–16 MB/s，无重试、无数据错误，仅影响耗时。

### 7.3 【需下游注意】NG 目录名与 `image_anomaly` 标签存在 100 条不一致（Dice 例）

- **事实**：Dice view120 test 的 700 图按目录名统计为 Good 200 / NG 500（50 个缺陷 specimen × 10 illum），按元数据 `image_anomaly` 统计为 0→300 / 1→400；`detectable != False` 过滤后官方口径为 Good 300 / NG 388（= 6.1 的 `category_manifest.csv` 记法）。
- **核对**：冻结评测代码的标签取自元数据 —— `e16_qualify.py:305` `labels = [int(m["image_anomaly"]) …]`，目录名不参与判类 → **对协议无影响**。
- **解释（假设，未验证）**：这 100 图大概率是"缺陷实体存在但从 view120 不可见"的图像级正常样本（M2AD 的 image-level 标签随视角变化）；本实验未做验证，仅记录该结构。
- **风险提示**：任何按目录名（`Good/` vs `NG/`）判类、绕开元数据的下游代码，在 M2AD 上会得到 300/500 的错误混淆矩阵；若后续新增统计脚本，必须从元数据取标签。

## 8. 分析

- **成本模型（事实）**：新类别取数成本 = **整包流式 ≈15.5 GiB 归档 / 类**，20–25 min（网络好）/ 43 min（网络差）；磁盘 ≈380–410 MB/类。早停不可依赖（成员分散）。→ 未来新增类别按 ~25 min/类 + 0.4 GB/类 预算，仍受磁盘 16 GB 上限约束（当前余 13.3 GB）。
- **GPU 预算（估算，待确认）**：按 E16 实测 4.5 min/cell（F0+R1-ILA+diagnostics，view120），7 类 × 3 seeds = 21 cells ≈ **1.6 h GPU**；这是估算，不含新脚本变体的调试开销。
- **口径一致性（事实）**：pool 抽样规则、view120 test 范围、448 缓存派生路径均与冻结协议一致（6.2 逐条验证），未改动任何 R1-ILA 参数。
- **未产生任何性能结论**：本实验没有运行任何模型，不应被引用为"跨类别泛化"的证据。

## 9. 结论

7 个新类别的 R1-ILA 输入缓存已在 AutoDL 服务器上完整落盘并通过逐类硬 gate 与独立复核；「R1-ILA 增益仅限 3 类」这一替代解释中的**数据不可得**分支被消除。协议、参数、指标口径均未改动。

## 10. 出口条件判断

- 预设 GO 条件：每类 `gate.n_missing == 0` 且 `bad_shape == 0`。
- 实际：7/7 全绿 → **数据层 GO**。
- 若任一类出现缺失/坏图：停在数据层修复，不得进入 GPU 阶段（未触发）。
- 注意：该 GO **仅限数据层**，不代表任何科学结论成立。

## 11. 对论文证据链的贡献

- **完成的格子**：跨类别外部有效性扩展的**输入可获取性**（3 类 → 10 类的数据前置条件成立）。
- **被排除的替代解释**：外部读者质疑"R1-ILA 结论无法在其他 M2AD 类别上检验，可能因为数据拿不到"—— 已排除。
- **下一格**：新类别上的 F0 与 R1-ILA 单元格（含多 seed 一致性）：只有它才能回答第 2 节的断言。

## 12. 下一步（需用户裁决，不自动执行）

1. **是否执行新类别 GPU 单元格**：需要用户指定类别子集与 seeds（多 seed 不得只报最好一次）。⚠️ 实现约束：`e16_qualify.run_cell` 含历史复现断言 `hist[(cat,seed)]`（`load_history()` 无新类别条目）→ 必须新建变体脚本（例如 `e22_eval.py`），**禁止修改 `e16_qualify.py` 本体**；R1-ILA/ILA 参数沿用 E17 冻结值（`BETA=0.1`）。
2. **是否同时补 E18 风格 baseline**：PatchCore/ResNet 线读 `l0_256`（`e18p1_patchcore_blocker.py` 已记录 `l0_256` 对 MVTec 基线不等价），因此新类别的 baseline 单元格需要另行审计缓存依赖，本实验的 `l0_448`-only 资产不覆盖它。
3. 备选路径 M2（4-view 扩展，零获取成本）与 M3（Bird 12-view）仍待裁决，见 `results/e20/README.md`。

## 13. 复现命令与文件清单

| 文件 | 说明 |
|---|---|
| `results/e22/README.md` | 本文件 |
| `results/e22/category_manifest.csv` | P0：10 类的元数据/本地覆盖盘点（7 类像素零） |
| `results/e22/status_fetch_20261015.json` | P1 冻结记录（逐类 streamed_gb / seconds / gate / free_gb_after） |
| `results/e22/cache_l0_448_manifest.csv` | 13,290 个缓存文件的 path/size/md5 清单 |
| `results/e22/cache_l0_448_aggregate_sha256.txt` | 聚合 sha256（`1569cc57…e47812`） |
| `results/e22/logs/fetch_20261015_183306.log` | 原始 stdout（含逐类早停/ gate 行） |
| `scripts/e22_manifest.py`、`scripts/e22_fetch.py` | P0/P1 脚本 |

```bash
# 完整性复验（迁移/备份后）
python - <<'PY'
import hashlib, csv
rows = list(csv.DictReader(open("results/e22/cache_l0_448_manifest.csv")))
agg = hashlib.sha256("\n".join(f"{r['rel_path']}:{r['bytes']}:{r['md5']}" for r in rows).encode()).hexdigest()
print(agg)
PY
```

**术语**：`l0_448` = 448×448 BICUBIC 无损 WebP 缓存；`pool` = bank 池 1200 图；`test120` = view120 评测集原始记录；`gate` = 取数后存在性+尺寸+模式校验。
