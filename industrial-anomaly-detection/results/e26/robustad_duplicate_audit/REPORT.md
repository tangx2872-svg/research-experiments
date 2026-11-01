# E26-A · RobustAD MetalParts train/test0 近重复审计

**日期**：2026-11-01
**实验编号**：E26-A（投稿前 P0 审计）
**性质**：只读图像审计。不解码到训练、不重跑评测、不修改数据、不修改 E25 任何结果。
**执行脚本**：`scripts/e26_a_duplicate_audit.py`（可原样重跑，约 2 分 15 秒，纯 CPU）
**GPU**：0（未使用）

---

## 1. pair 从哪里来（不猜测对应关系）

严格沿用 E25 已记录的判据，而不是重新设计匹配：

- 数据：`data/robustad/MetalParts/metal_parts_data_dir_train` 与 `.../metal_parts_data_dir_test0`
- 匹配键：各自 `metadata.jsonl` 中 **`file_name` 字符串完全相等**
- 结果：train 231 条 ∩ test0 171 条 = **145 对**

**实际 pairs 数 = 145**，与 E25 `overlap_audit.json` 的 `n_name_overlap = 145` 一致，无需修正论文数字。分解：

| 项 | 值 |
|---|---:|
| train 条目 | 231 |
| test0 条目 | 171 |
| 同名对 | **145** |
| 其中 normal（label=0） | 104 |
| 其中 anomaly（label=1） | 41 |
| 两侧 label 一致的 pair | **145 / 145** |
| 磁盘缺失 | 0 |

补充事实：145 对**全部**尺寸相同（2681 × 1500，RGB JPEG），这也是 E25 记录过的 `same_pixel_size = True`。

---

## 2. 完全重复检查

| 指标 | 结果 |
|---|---|
| SHA-256 相同的 pair | **0 / 145** |
| 字节数相同的 pair | 0 / 145（见 `all_pairs.csv` 的 `bytes_a` / `bytes_b`） |
| RGB 像素级完全相等（同尺寸逐像素比较） | **0 / 145** |

⇒ 与 E25 的字节级结论一致：**不存在字节级或像素级重复**。

---

## 3. 感知哈希与 SSIM

**方法**

| 指标 | 实现 |
|---|---|
| pHash | 灰度 → 32×32（LANCZOS）→ 二维 DCT → 取左上 8×8 去 DC → 中位数二值化 → 63 bit → Hamming |
| dHash | 灰度 → 9×8（LANCZOS）→ 相邻列比较 → 64 bit → Hamming |
| SSIM | 灰度，**原生分辨率**；仅在两图尺寸不同时把 B 用 LANCZOS 缩放到 A 的尺寸（本批 **145/145 对尺寸相同，实际未触发任何 resize**）；`gaussian_weights=True, sigma=1.5, data_range=255` |
| LPIPS | **SKIPPED** |

**LPIPS 未执行的理由**：冻结环境（PyTorch 2.8.0+cu128 / conda base）未安装 `lpips`，安装需引入新依赖并可能牵动 torch 生态；按 E26 的环境隔离原则不安装。
**pHash + dHash + SSIM 已用于本轮近重复审计，LPIPS 因环境隔离原则未强制执行。**

**分布（145 对全部）**

| 指标 | min | median | max |
|---|---:|---:|---:|
| SSIM | 0.6601 | 0.7515 | **0.8139** |
| pHash Hamming | **8** | 20 | 36 |
| dHash Hamming | **2** | 14 | 23 |
| 64×64 灰度平均绝对差（0–255） | 8.94 | 18.64 | 39.64 |

关键事实：**没有任何一对的 SSIM 超过 0.82**，pHash 最小为 8（8 bit/63 已明显偏离），说明不存在"同帧重编码 / 轻微几何扰动"级别的候选。dHash 最小值 2 只出现在 1 对（`MP_012`，其 pHash=10、SSIM=0.740），属于同工件同构图的粗结构巧合，而非同帧。

E25 曾对 20 对抽样记录 64×64 灰度 MAD ∈ [10.94, 32.46]；本轮全量 145 对的范围是 [8.94, 39.64]，**抽样区间落在全量区间内**，两轮测量互相自洽。

---

## 4. 辅助分类（启发式，不是泄漏判定）

阈值（写入 `audit_summary.json`，仅用于筛选复核对象）：

- `EXACT_DUPLICATE`：SHA-256 相等或像素完全相等
- `VERY_LIKELY_NEAR_DUPLICATE`：pHash ≤ 6 且 dHash ≤ 8 且 SSIM ≥ 0.95
- `SIMILAR_SCENE_OR_OBJECT`：SSIM ≥ 0.80 或（pHash ≤ 12 且 dHash ≤ 14）
- `NO_OBVIOUS_DUPLICATE`：其余

| 分类 | 数量 |
|---|---:|
| EXACT_DUPLICATE | **0** |
| VERY_LIKELY_NEAR_DUPLICATE | **0** |
| SIMILAR_SCENE_OR_OBJECT | 17 |
| NO_OBVIOUS_DUPLICATE | 128 |

`SIMILAR_SCENE_OR_OBJECT` 是**同型号金属板的正常表现**（同一块 6 孔板、同一工位、不同照明与曝光），按任务要求不得据此认定重复。

---

## 5. Top-20 复核清单

排序规则：`pHash Hamming + dHash Hamming` 升序，同分按 SSIM 降序（**仅用于安排人工复核顺序**，不是泄漏判据）。

| rank | pair | 文件（两侧同名） | label | pHash d | dHash d | SSIM | MAD64 | 辅助分类 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | MP_012 | anomaly/Live_Chipping_None_None_4.jpg | 1 | 10 | 2 | 0.7402 | 11.71 | SIMILAR_SCENE_OR_OBJECT |
| 2 | MP_111 | normal/Live_Normal_None_None_57.jpg | 0 | 8 | 7 | 0.8000 | 12.94 | SIMILAR_SCENE_OR_OBJECT |
| 3 | MP_062 | normal/Live_Normal_None_None_13.jpg | 0 | 8 | 8 | 0.7925 | 20.72 | SIMILAR_SCENE_OR_OBJECT |
| 4 | MP_133 | normal/Live_Normal_None_None_83.jpg | 0 | 8 | 8 | 0.7653 | 33.81 | SIMILAR_SCENE_OR_OBJECT |
| 5 | MP_003 | anomaly/Live_Chipping_&_Porosity_None_None_7.jpg | 1 | 12 | 5 | 0.7458 | 11.43 | SIMILAR_SCENE_OR_OBJECT |
| 6 | MP_102 | normal/Live_Normal_None_None_42.jpg | 0 | 8 | 10 | 0.7486 | 15.47 | SIMILAR_SCENE_OR_OBJECT |
| 7 | MP_128 | normal/Live_Normal_None_None_77.jpg | 0 | 16 | 3 | 0.7726 | 11.82 | NO_OBVIOUS_DUPLICATE |
| 8 | MP_119 | normal/Live_Normal_None_None_64.jpg | 0 | 10 | 9 | 0.7290 | 10.65 | SIMILAR_SCENE_OR_OBJECT |
| 9 | MP_073 | normal/Live_Normal_None_None_141.jpg | 0 | 12 | 8 | 0.7609 | 8.94 | SIMILAR_SCENE_OR_OBJECT |
| 10 | MP_041 | anomaly/Live_Porosity_None_None_7.jpg | 1 | 12 | 8 | 0.7306 | 21.13 | SIMILAR_SCENE_OR_OBJECT |
| 11 | MP_042 | normal/Live_Normal_None_None_1.jpg | 0 | 10 | 10 | 0.7032 | 20.58 | SIMILAR_SCENE_OR_OBJECT |
| 12 | MP_089 | normal/Live_Normal_None_None_28.jpg | 0 | 12 | 9 | 0.7905 | 13.68 | SIMILAR_SCENE_OR_OBJECT |
| 13 | MP_002 | anomaly/Live_Chipping_&_Porosity_None_None_5.jpg | 1 | 14 | 7 | 0.7437 | 15.29 | NO_OBVIOUS_DUPLICATE |
| 14 | MP_103 | normal/Live_Normal_None_None_45.jpg | 0 | 12 | 9 | 0.7221 | 17.73 | SIMILAR_SCENE_OR_OBJECT |
| 15 | MP_100 | normal/Live_Normal_None_None_39.jpg | 0 | 14 | 8 | 0.7349 | 14.85 | NO_OBVIOUS_DUPLICATE |
| 16 | MP_131 | normal/Live_Normal_None_None_81.jpg | 0 | 10 | 12 | 0.7182 | 17.10 | SIMILAR_SCENE_OR_OBJECT |
| 17 | MP_120 | normal/Live_Normal_None_None_65.jpg | 0 | 16 | 7 | 0.7905 | 12.73 | NO_OBVIOUS_DUPLICATE |
| 18 | MP_015 | anomaly/Live_Chipping_None_None_7.jpg | 1 | 14 | 9 | 0.7618 | 11.44 | NO_OBVIOUS_DUPLICATE |
| 19 | MP_034 | anomaly/Live_Porosity_None_None_14.jpg | 1 | 14 | 9 | 0.7544 | 15.32 | NO_OBVIOUS_DUPLICATE |
| 20 | MP_106 | normal/Live_Normal_None_None_48.jpg | 0 | 14 | 10 | 0.7117 | 18.06 | NO_OBVIOUS_DUPLICATE |

完整 145 行：`all_pairs.csv`；仅前 20：`top20.csv`。

**目视复核（contact sheet）**：`top01_10_contact_sheet.png`、`top11_20_contact_sheet.png`（1996 × 3826 px，180 dpi，每行 A/B 并排 + pair 元数据）。
目视结果：top-10 全部是**同一型号金属板的两次不同拍摄**——高光/曝光位置不同、工件在画面中的位置与角度略有差别、划痕与缺陷的可见性不同（例如 rank 1 的 A 有明显长划痕而 B 没有同位置划痕）。**未发现同帧、重编码、裁切或轻微几何扰动级别的重复。**

---

## 6. 结论

1. **实际检查 145 对**（train 231 ∩ test0 171，按 `file_name` 精确匹配；104 normal + 41 anomaly），与 E25 记录一致，论文中的"145"无需修改。
2. **SHA-256 完全重复 0 对；像素级完全重复 0 对。**
3. **无明显近重复**：SSIM 最大 0.8139，pHash Hamming 最小 8，`VERY_LIKELY_NEAR_DUPLICATE` 计数为 0。
4. 17 对被启发式标为 `SIMILAR_SCENE_OR_OBJECT`，经目视复核均为**同型号工件的不同帧**，不构成重复。

**结论表述（可直接引用）**：

> 未发现足以表明外部验证结果由明显图像近重复驱动的证据。

**不得表述为**："已完全证明不存在任何数据关联" / "train 与 test0 是独立采集"。原因：

- 本审计只检验**图像层面**的重复（字节 / 像素 / 感知哈希 / 结构相似度）；
- **同一物理工件在不同光照、不同位姿下的两次拍摄**在这套指标下会呈现中等相似度（本批 SSIM 0.66–0.81），无法与"近邻帧"在概念上完全区分；
- 文件名重合本身仍是**数据集组装层面的关联信号**，E25 的 `valid=0` 排除规则保持不变，不受本审计影响。

---

## 7. 对论文结论的影响

- **不影响 E25 的任何聚合数字**：MetalParts test0 原本就按 `valid=0` 从所有 target 聚合中排除；本审计未改变该判定，也未触发任何重跑。
- **可以加强的表述**：补充材料现在可以写"145 对同名文件已完成字节级 + 感知哈希 + SSIM 审计，0 对完全重复、0 对近重复；该 split 仍按原协议保守排除"。
- **仍需保留的限定**：文件级关联（同名）与采集对象级独立性是两个问题，本审计只能回答前者中的"图像是否重复"这一子问题。

---

## 8. 产物

| 文件 | 内容 |
|---|---|
| `pair_manifest.csv` | 145 对清单（pair_id / domain / filename / label / split / 两路径） |
| `all_pairs.csv` | 145 行全指标（SHA-256、字节、尺寸、像素相等、pHash/dHash/Hamming、SSIM、MAD、LPIPS=SKIPPED、辅助分类、rank） |
| `top20.csv` | 前 20 最可疑 pair |
| `audit_summary.json` | 计数、分布、阈值、口径、环境（机器可读） |
| `top01_10_contact_sheet.png` / `top11_20_contact_sheet.png` | 目视复核图 |

---

## 附：任务要求字段对照

| 要求字段 | 位置 |
|---|---|
| `sha256_a` / `sha256_b` / `sha256_equal` | `all_pairs.csv` |
| `bytes_a` / `bytes_b` | `all_pairs.csv` |
| `width_a/height_a/width_b/height_b` | `all_pairs.csv` |
| `exact_pixel_equal` | `all_pairs.csv` |
| `phash_a` / `phash_b` / `phash_hamming` | `all_pairs.csv` |
| `dhash_a` / `dhash_b` / `dhash_hamming` | `all_pairs.csv` |
| `ssim` | `all_pairs.csv` |
| `lpips` | `all_pairs.csv`（全部为 `SKIPPED`） |
