# Experiment 8B — sanity report

| ID | status | detail |
| --- | --- | --- |
| S1_feature_shape | PASS | layer2=(512,32,32), layer3=(1024,16,16) 全部 cell 一致（extract 侧 assert） |
| S2_channel_ordering | PASS | 与 1J 逐 channel 对齐（S13 corr≈1.0, 同 index 同 channel） |
| S3_layer_identity | PASS | layer3 未 upsample；与 1J 命名一致（同 pooler 约定） |
| S4_original_score_equivalence | PENDING | PENDING（由 experiment8b_probe_analyze.py 在 Phase B 后写入：FULL mask vs 历史 α=0，max_rel_diff） |
| S5_image_mask_alignment | PASS | top-5% patch-NN 格点与 mask overlap>0.5 的重叠 z>3：n=60/60 rows(15 cells x 2 img x 2 layers)，z_min=4.51 > 3，但绝对 overlap 最小仅 0.047（screw/manipulated_front layer3），已记录 |
| S6_paired_illumination_alignment | PASS | paired identity 条件 Δ≡0（15/15 cells，layer2+layer3，逐位相等） |
| S7_no_test_leakage | PASS | ranking=oracle-diagnostic；score-level 结果标 oracle_probe_only=true |
| S8_D_finite | PASS | n=23040 |
| S9_I_finite | PASS | I_mean/brightness/gamma 全部有限 |
| S10_scale_normalization_valid | PASS | sigma_eff=sqrt(sigma^2+eps^2)，eps=1e-3*median(sigma) 已冻结 |
| S11_coverage_complete | PASS | cells=15/15, channel rows=23040 |
| S12_random_baseline_reproducible | PASS | 同 seed null 逐位一致 |
| S13_source_consistency | PASS | n=240 全部 corr>=0.9999 且 max|Δ|<=0.05 |

primary sanity FAIL count (excl. PENDING) = 0
