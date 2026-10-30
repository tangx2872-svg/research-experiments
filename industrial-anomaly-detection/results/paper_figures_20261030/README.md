# 正式论文图导出（2026-10-30）

**任务性质**：图件重制（presentation only）。**不新增实验、不训练、不推理、不重新评分、不修改任何已有结果文件。**
**输入**：4 个已冻结的 CSV（E21 / E22 / E23 的既有分析产物，原样复制到 `inputs/`）。
**输出**：3 张图 × (300 dpi PNG + 矢量 PDF)，外加脚本、输入、核对记录与本 README。

---

## 1. 三张图与论文图号

| 新文件（`figures/`） | 论文图号（用户指定） | 内容 | 原运行脚本 | 原图 |
|---|---|---|---|---|
| `delta_auroc_10cat_horizontal.png` / `.pdf` | **图 2** | 10 类 ΔAUROC 横向条形图：3 seed 均值 + 样本标准差误差线 + 逐 seed 散点 | `scripts/e22_figures.py`（Fig 2 分支） | `results/e22/figures/fig2_delta_auroc_10cat.*` |
| `doll_f0_vs_r1_per_image_scatter.png` / `.pdf` | **图 3** | Doll 三 seed 的 F0/R1 逐图得分散点（detectable 过滤后 670 图/seed） | `scripts/e23_figures.py`（Fig 1） | `results/e23/figures/fig1_doll_f0_vs_r1_scatter.*` |
| `multiview_delta_auroc_heatmap.png` / `.pdf` | **图 4** | Bird / Car / Cube × 4 视角 ΔAUROC 热图（seed 0） | `scripts/e21_figures.py`（Fig 2） | `results/e21/figures/fig2_dauroc_heatmap.*` |

图号映射同时以机读形式给出：`FIGURE_MAP.csv`。
**注**：论文图号由用户指定；仓库内 `results/paper_ready/figure_manifest.csv` 只记录 `main_A` / `supp_D` / `supp_E` 的 paper-asset 编号，不含正式图序。

## 2. 输入数据（`inputs/`，原样复制，未改动）

| 文件 | 来源实验 | 用途 |
|---|---|---|
| `main_table_10cat_3seed.csv` | E22 | 图 2 的条形高度（`d_auroc_mean`）与误差线（`d_auroc_std`） |
| `all10_3seed_cells.csv` | E22（view120） | 图 2 的逐 seed 散点；图 3 面板标题中的 AUROC |
| `doll_per_image_delta.csv` | E23 | 图 3 的全部 2010 行逐图得分（`detectable == True` 过滤后每 seed 670 图） |
| `analysis_cells.csv` | E21 | 图 4 的 category × view ΔAUROC（只取 `seed == 0`） |

## 3. 口径（与原文严格一致，未改动）

- **误差线** = 3 个 seed 的**样本标准差**（`ddof=1`，E22 `scripts/e22_analysis.py:116` 定义），**不是** bootstrap 区间。
- **图 2**：条形 = 3-seed 均值；散点 = 每个 seed 的 ΔAUROC（`R1 − F0`，view120）；正值为红 `#C44E52`，负值为灰 `#7F7F7F`，零线黑色。
- **图 3**：`detectable != False` 过滤后每 seed **670 图 = 270 正常 + 400 异常**；正常 `#55A868`（alpha 0.45）、异常 `#C44E52`（alpha 0.35）、对角线灰色虚线；x/y 轴按数据范围 `[min×0.98, max×1.02]` 取，三面板共享坐标；面板标题给出该 seed 的 F0→R1 AUROC 与 Δ。
- **图 3 不含 d′**：原图未绘制 d′，按"没有则不额外添加"处理（d′ 只存在于 E22 主表 `d_prime_mean_f0/r1`）。
- **图 4**：seed 0；色标 `RdBu_r`，对称范围 `±max(0.03, max|ΔAUROC|)` = **±0.03**；格内标注 4 位小数；负值单元（Car 240/330、Cube 090/330，共 4 个）**保留显示**。

## 4. 相对原脚本的改动（仅表现层）

| 改动 | 说明 |
|---|---|
| 删除图内运行代号与编号 | 移除 `E21 ·`、`E23 Fig1 —`、`Fig 2 —` 等标题/总标题；正式中英文图题写在 Word 中 |
| 字号统一 | 正文 10 pt、轴标签 10.5 pt、刻度 9.5 pt、图例 9 pt |
| 导出精度 | PNG 300 dpi（原为 120 dpi）；PDF 字体子集嵌入（CIDFontType2 / DejaVu Sans），便于 Word/LaTeX 使用。图 2、图 3 的 PDF **全矢量**；图 4 的色块由 `imshow` 生成，是 600 dpi 内嵌位图，其文字、坐标轴、色标均为矢量 |
| 坐标轴标签微调 | 图 2 的 x 轴说明改为两行排版（原文过长被画布截断），文字内容不变；图 4 的 x 轴标签写成 `view (seed 0)`，把原图标题里的 seed 信息保留在轴上 |

**未改动**：数据、样本过滤、误差线定义、颜色、条形顺序、散点透明度/大小、色标范围、坐标范围。
坐标范围按原脚本实测值**显式锁定**（见 `scripts/export_paper_figures.py` 的 `ORIGINAL_LIMITS`）：

| 图 | xlim | ylim |
|---|---|---|
| 图 2 | −0.048014 … 0.043554 | −0.7525 … 9.7525 |
| 图 3 | 0.094993 … 0.520450 | 0.094993 … 0.520450 |
| 图 4 | −0.5 … 3.5 | 2.5 … −0.5 |

## 5. 数值一致性核对（`checks/`）

**（a）与原运行脚本的 artist 级比对**（`checks/verify_parity.py`，只读：把原脚本的 `save` 换成捕获钩子，不写任何文件）

```
PASS  delta_auroc_10cat_horizontal       max|diff| = 4.42e-07
PASS  doll_f0_vs_r1_per_image_scatter    max|diff| = 4.23e-07
PASS  multiview_delta_auroc_heatmap      max|diff| = 0
```

比对对象：条形宽度/中心、逐 seed 散点 x 位置、全部散点 offset（670×3×2 个坐标）、热图矩阵、坐标范围。
残余 4e-7 的来源：导出读的是 **6 位小数的 CSV 表**（与论文表格同一套数字），原脚本读的是**全精度 per-cell JSON**；差值比图上任何可见精度低 3 个数量级，也低于已披露的输入量化不确定性（≤0.0017）。

**（b）图 3 的 AUROC 独立重算**（在 `export_paper_figures.py` 内自动执行）：用 670 张逐图得分按秩重算 AUROC，与表中存储值比对，`max|diff| = 4.4e-07`（CSV 舍入）。

**（c）图 2 的数值**（`checks/export_payload.json`）：9/10 类为正，Doll 为唯一负值（−0.0385 ± 0.0046）。

## 6. 运行方法

```bash
# 在仓库根目录下执行（依赖：numpy + matplotlib，base 环境即可，无需 GPU）
cd results/paper_figures_20261030
python scripts/export_paper_figures.py          # 生成 figures/*.png + *.pdf + checks/export_payload.json

# 可选：数值一致性核对（需要仓库内的原始脚本 scripts/e2{1,2,3}_figures.py，只读）
python checks/verify_parity.py
```

`export_paper_figures.py` 支持 `--inputs` / `--out` / `--checks` 指定目录；`--no-limits-check` 可关闭坐标范围锁定（默认开启）。

## 7. 产物清单

```
paper_figures_20261030/
├── figures/   3 × {png,pdf}   （300 dpi / 矢量）
├── scripts/   export_paper_figures.py
├── inputs/    4 个 CSV（总计 264 KB）
├── checks/    verify_parity.py · parity_vs_original.json · export_payload.json
├── FIGURE_MAP.csv
├── README.md
└── paper_figures_20261030.tar.gz   ← 打包件（自身哈希记录在同级 SHA256SUMS.txt，无法自指嵌入包内）
```

不含：原始数据集、符号链接目录、模型权重、cache、逐图打分缓存。

## 8. 使用注意

1. 图题（中英文）、图注、编号一律在 Word 中写；图内不再出现任何运行代号或编号。
2. 图 3 的面板标题保留各 seed 的 AUROC 数值，属于必要的种子与统计说明，不要删除。
3. 图 4 的负值单元（Car 240/330、Cube 090/330）必须保留；E21 结论为 HOLD，图注需说明这是 seed 0 的单种子结果。
4. 请勿对 PNG 二次裁切或用白块遮挡；如需改版，直接改 `scripts/export_paper_figures.py` 后重跑。
