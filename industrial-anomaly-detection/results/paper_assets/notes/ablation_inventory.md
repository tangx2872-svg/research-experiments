# Ablation Inventory（只盘点已有结果，不运行新实验）

| Method | Ablation | Status | Evidence / Source |
|---|---|---|---|
| Original | α=0 baseline | **already available** | Exp10 `P10_ORIG_a000` + Exp14/15/16 A0；50 单元 |
| Adaptive B2 | category-adaptive β_c（去除 = Fixed C2） | **already available** | Exp11（Fixed C2 9/15 vs B2 12/15）；β_c 表在 `experiment11_candidates.beta_table()` |
| Adaptive B2 | L2-only vs Full | **already available** | Exp11 `abl/`（`E11_B2_L2only`，已核验与 Exp14 A6 bit-exact） |
| C2 | w=0.35 固定权重 | **already available** | Exp15/16（完整 w 曲线：w∈{0.20,0.35,0.50,0.65,0.80}） |
| C6 | max 规则 | **already available** | Exp15/16 |
| X6c | 组成分解：X6c = mean(C2, C6) | **already available** | Exp15/16 三者的独立单元结果（可做组成分解） |
| X6c | 端点 ablation：w=0（=B2）与 w=1（=Original） | **already available** | Exp15 §O3（免费端点） |
| 全部 | 层特异性 ablation（L2/L3 强度） | **partially available** | Exp14 A3/A5/A6 + Exp15 X4a/X4b（均被淘汰，但构成 negative ablation 证据） |
| 全部 | held-out 数据集 ablation | **missing** | 需 E4（`held_out_validation_plan.md`） |
| 全部 | 更多 seeds（>10）/ 多数据集统计功效 | **missing** | 若审稿人要求更强统计，扩到 seeds 10–19 或第二数据集 |

**结论**：主表所需 ablation **已基本齐备（0 GPU 可得）**；缺失项仅为 E4 外部验证与更多 seeds 的统计功效，
均属 deferred（未来验证），不影响当前冻结决策。
