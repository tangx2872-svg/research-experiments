# 已解决的运行事件记录（原 ERROR_REPORT.md）

本文件由 Experiment 10 的安全护栏自动生成（连续 ≥3 个 unit 失败 → 停止新 GPU 任务）。
**三个事件都已定位并修复，最终 57/57 units 全部成功（0 failed）。** 保留此记录以求透明。

## 事件 1 — orchestrator 日志句柄被 GC（bug-1）
`experiment10_batch.launch()` 打开日志文件后未持有引用，file 对象被 GC 关闭 → worker 启动早期死亡且**无日志**。
修复：模块级 `_HANDLES` 持有引用。影响：s1_frontier 的 6 个 unit 未启动（无任何计算发生）。

## 事件 2 — `%g` 截断配置名（bug-2）
`experiment10_candidates._an()` 用 `"%g"` 只保留 6 位有效数字，导致 `0.40091275 → a040091`，
与实际传入名 `a040091275` 不匹配 → runner 校验 `unknown 9B config` 并 SystemExit；
一个坏名字会杀掉整个 worker 组。修复：改用 `repr(float(a))`。影响：s1_frontier 的 6 个 unit 未运行。

## 事件 3 — 同 stage 不同 backend 日志撞车（bug-3，命名问题）
`s2a/s3/s4` 同时用 7ao 与 9b 两个 backend 但共用 `{stage}_w{wid}.log` → 输出交错（不影响数值）。
修复：日志名含 backend。

## 事件 4 — 单 worker 混 seed（bug-4）
Stage 4 的 8 个单元跨 seeds {1,2}，轮询分组把两个 seed 混进同一 worker → runner 拒绝
（`single seed per worker required`），8 个 unit 全部未运行。
修复：orchestrator 改为**按 seed 分组**后再轮询；并以两个 seed-specific 批次重跑成功。

**结论**：以上均为**编排/命名层缺陷**，未影响任何已产出结果的数值正确性；
所有失败 unit 均以原始冻结配置重跑并成功（`0 failed`，见 `progress.json`）。
