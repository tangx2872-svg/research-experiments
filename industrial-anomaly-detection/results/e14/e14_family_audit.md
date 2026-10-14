# E14 — Anomaly-Oriented Representation Family Audit（READ-ONLY）

日期：2026-10-14 · HEAD：0645cc7（E13-B2 已提交）

## 核心审计结论（务必先读）

**F1 / F2 并不改变 representation。** F1（AnomalyDINO）仍是 DINOv2 backbone，F2（SoftPatch）仍是 WideResNet50（PatchCore 家族）。两者的「family 差异」只在 scoring / memory 机制，不在 backbone。唯一真正换 paradigm 的是 F3（EfficientAD，teacher-student），但本轮因成本 SKIP。

因此 E14 的实际含义是「anomaly-oriented scoring / memory 家族对比」，而非「换 representation」。这一结论直接决定 E14 对「突破 DINOv2 ceiling」的科学意义有限（F1 用同一 DINOv2，F2 是已穷尽的 PatchCore 线）。

## 逐 family 审计

### F0 — Current Reference（复用）
- DINOv2-B final layer，1-NN-max，300-bank = E13 R0。0 GPU。数值：Bird 0.74725 / Car 0.78552 / Cube 0.85410。

### F1 — AnomalyDINO
| 项 | 值 |
|---|---|
| implementation | `third_party/AnomalyDINO/`（WACV 2025，Apache-2.0）+ 忠实实现 `scripts/e7_pipelines.py::run_dinov2` |
| 机制 | DINOv2 448 → patch 特征 → L2-normalized kNN（k=1）→ image score = **mean_top1p**（top-1% patch 距离均值）；可选 PCA background masking / reference rotation |
| local_repo | `third_party/AnomalyDINO/`（src/{backbones,detection,post_eval,utils}.py） |
| weights | DINOv2（已缓存，需离线 loader，`torch.hub.load` 会挂 github） |
| dependencies | torch、faiss-cpu（已装）、numpy |
| historical_overlap | **高**：E7 P06 = DINOv2-B→NN 已跑（忠实 mean_top1p，弱于 PatchCore）；E11–E13 已穷尽 frozen DINOv2 kNN 并确立 Cube ceiling |
| estimated_runtime | ~2–3 min/category（frozen 前向 + 精确 kNN） |
| VRAM | < 4 GB（bank 300×1024×768） |
| protocol_compatibility | 高（复用 M2AD bank/test 接口；E7 已适配） |
| risk | **同 DINOv2 backbone → 预期复现 Cube ceiling**；仅 scoring（mean_top1p）与 F0 不同 |
| recommendation | **GO**（便宜，官方 mean_top1p 是相对 R0 max 的合法家族对比；但明确标注：非 representation 变更） |

### F2 — SoftPatch
| 项 | 值 |
|---|---|
| implementation | `third_party/SoftPatch/`（NeurIPS 2022，无 LICENSE）+ 忠实实现 `scripts/e7_pipelines.py::run_softpatch` |
| 机制 | WideResNet50(L2+L3) PatchCore 表示 → 官方 `WeightedGreedyCoresetSampler` soft patch weighting（nearest / LOF）→ score = kNN距离 × coreset_weight，聚合 max |
| local_repo | `third_party/SoftPatch/src/`（softpatch.py / sampler.py / backbones.py） |
| weights | torchvision `wide_resnet50_2`（已缓存，无网络） |
| dependencies | torch、torchvision、faiss |
| historical_overlap | **高**：E7 P10/P11 = SoftPatch nearest/LOF 已跑（**弱于纯 PatchCore −0.04~−0.07**）；E1–E16 已穷尽 PatchCore 线 |
| estimated_runtime | ~3–5 min/category（PatchCore 拟合 + SoftPatch 加权 + 打分） |
| VRAM | < 6 GB |
| protocol_compatibility | 高（复用 M2AD；E7 已适配） |
| risk | **soft weighting 为噪声 bank 设计，干净 bank 只是扰动距离场（E7 实证）→ 预期仍弱于 PatchCore** |
| recommendation | **GO（低优先，带 E7 警告）**；非 representation 变更，预期难 KEEP |

### F3 — EfficientAD
| 项 | 值 |
|---|---|
| implementation | **不在 third_party**（需 clone；github 现可达，raw 也通） |
| 机制 | teacher-student 蒸馏：DINO teacher 特征 ← 轻量 student 重建，anomaly score = student-teacher discrepancy |
| weights | teacher = DINO（需下载 DINO-v1 权重）；student 需训练 |
| dependencies | torch + 官方 repo 依赖 |
| historical_overlap | **无**（唯一真正新 paradigm） |
| estimated_runtime | 下载 + 每类 student 训练（数分钟）+ 推理 |
| VRAM | 未知（轻量 student） |
| protocol_compatibility | 需适配 M2AD + 训练管线 |
| risk | **teacher 仍是 DINO（DINOv1），可能与 DINOv2 有类似 ceiling；工程 >45 min；第三优先** |
| recommendation | **SKIP（本轮）**：成本超预算、第三优先；作为 E14-B 候选（若需要真正换 paradigm） |

## 建议进入正式快筛的 family
**F1（AnomalyDINO）** 与 **F2（SoftPatch）** 两个（均便宜、权重/实现本地就绪）。F3 本轮 SKIP。

## 历史重复检查（无完全等价）
- F1 的 mean_top1p 与 F0 的 max 不同 → 非等价。
- F2 的官方 soft weighting 在 E14 的 Bird/Car/Cube × view120 × 300-bank × seed0 协议下首次正式比较（E7 是不同协议）→ 非严格等价，但 E7 已给出强负先验。
