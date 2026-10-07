# Stage 0.5 — Protocol Audit（209 vs 189）

9B-R §11 发现 `Engine.fit` 内部 `dm.setup("fit")` 会重置 runner 的 train 过滤，使 memory bank 实际含 209 张（含 20 张 val）。本阶段用 **corrected-189**（在 Lightning 再次 setup 后重新施加过滤，并用 `Patchcore.training_step` 计数证明实际嵌入图像数 = 189）对比 historical-209。

**CASE: P-A**（方向一致=True，trade-off 判断不变=True）

详情见 `protocol_audit.json` / `protocol_audit.csv`。
