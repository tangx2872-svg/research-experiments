# E15 — EfficientAD Teacher-Student Feasibility Audit（READ-ONLY）

日期：2026-10-14 · HEAD：87098a8（E14 已提交）

## E15 AUDIT REPORT

| 项 | 值 |
|---|---|
| implementation | `third_party/EfficientAD/`（nelson1425/EfficientAD，作者官方仓库，Apache-2.0） |
| repo / commit | `github.com/nelson1425/EfficientAD` @ `fcab514`（2023-05-28） |
| weights | **teacher_small.pth (10.8MB) + teacher_medium.pth (32MB) 已随仓库克隆，本地就绪**（DINO teacher，无需联网下载） |
| dependencies | torch（**torch 2.8 下已实测可 build + forward**）、torchvision、tifffile、tqdm、sklearn、scipy、Pillow |
| external_assets | **ImageNet 可选**：`--imagenet_train_path none` → `pretrain_penalty=False`（仅跳过 AE 预训练惩罚，不改变 teacher-student 主机制） |
| dataset_compatibility | 需适配（EfficientAD 用 MVTec 风格 `train/good` + `test/{good,anomaly}` 目录；M2AD 需 symlink farm，纯工程） |
| GPU_memory_estimate | 极小（PDN-small：teacher 2.7M / student 4.3M / AE 1.1M 参数，batch=1，256 输入；已实测 forward 通） |
| runtime_estimate_per_category | **70000 train steps × batch=1 ≈ 10–15 min**（small 模型） |
| disk_free | 16 GB |
| historical_overlap | **无**（EfficientAD 从未正式测试） |
| risk | **低**（模型已实测在 torch 2.8 可跑；仅剩数据适配 + 分数提取） |
| verdict | **GO** |

## 关键工程确认（已完成实测）
- `get_pdn_small(384)` / `get_pdn_small(768)` / `get_autoencoder(384)` 在 torch 2.8 下正常构建。
- `teacher_small.pth` 可正常 `load_state_dict`（8 keys）。
- 前向通过：teacher → (1,384,56,56)、student → (1,768,56,56)、AE → (1,384,56,56)。
- image-level score = `max(0.5*map_st + 0.5*map_ae)`；label 由 `defect_class=='good'` 判定。

## 偏离官方默认的说明（记录，非魔改）
1. `pretrain_penalty=False`（无 ImageNet）：跳过 AE 预训练惩罚，主机制（teacher-student 蒸馏 + AE 重构）不变。
2. 数据用 symlink farm 把 M2AD 映射成 MVTec 风格（与 E4-D1/E14 同法，纯工程）。
3. 最终 AUROC/AP 用项目官方 `image_anomaly` + raw `detectable` 口径（与 E11–E14 一致），而非 EfficientAD 目录标签口径。
