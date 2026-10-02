# Change Log — Experiment 1E category 参数化改造（2026-10-02）

## 修改的文件

1. `scripts/experiment1b_defect_sensitivity.py`（+57 / -22）
2. `scripts/scan_mvtec_categories.py`（新增）

## 只是泛化 category（未改任何科学计算）

- `category="bottle"` 硬编码 → `--category` CLI 参数（默认 `DEFAULT_CATEGORY="bottle"`，向后兼容）
- `DEFECT_TYPES` 硬编码列表 → `discover_defect_types(category)` 自动扫描 `<root>/<category>/test/`，排除 `good`，字典序固定排序
- `make_validation_split` / `fit_model` / `run_screening` 全部接受 `category` 参数
- validation split 协议不变：train/good 固定 seed 划 20 张 validation，剩余进 memory bank

## 核心科学计算完全不变

- α-IN 实现（`falpha_patchcore.py`）：**零改动**
- backbone `wide_resnet50_2`、layers `["layer2","layer3"]`：不变
- coreset_sampling_ratio=0.1、num_neighbors=9：不变
- α grid `[0,0.25,0.5,0.75,1.0]`：不变
- seeds 协议、z-score 公式、d' 公式、anomaly score 定义、preprocessing：不变
- 阈值规则 `max_validation_score`：不变

## 静态自测结果（不训练）

- bottle defect 自动发现 = `['broken_large','broken_small','contamination']` ✅（与 1B/1C 一致）
- seed=0 validation split = 20/189，前 3 名 `017/026/043` 与 1C 记录完全一致 ✅
- 不存在的 category 返回空列表（不再抛 FileNotFoundError）✅
- CLI `--category` 参数正确加入 ✅
- 两个脚本 `py_compile` 通过 ✅
- α-IN 文件 git diff 为空（未改动）✅

## 发现并修复的 bug

1. `discover_defect_types` 原实现对不存在的 category 目录会抛 `FileNotFoundError`（`iterdir()` 无守卫），
   已加 `test_root.is_dir()` 守卫，返回空列表。
2. `run_screening` 新增前置守卫：category 数据目录不存在时给出清晰中文报错，
   而非等到 fit 阶段才报晦涩错误。
3. `scan_mvtec_categories.py` 的 `--categories` 参数原只支持逗号分隔，
   已改为同时支持空格和逗号（协议用法示例是空格分隔）。

## 未做（等待数据）

- 未下载任何第三方数据
- 未启动 smoke test
- 未启动正式 1E Pilot
