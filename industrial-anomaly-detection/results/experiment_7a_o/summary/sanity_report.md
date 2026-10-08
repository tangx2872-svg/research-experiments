# Experiment 7A-O — sanity report

frozen protocol: `results/experiment_7a_o/reference/policy_freeze.json` (sha256 `e046a89a852da9aad55d2eafeea1c3bc05a5aff640be4338f7e515e53150f224`)

| # | check | status | detail |
|---|---|---|---|
| 1 | S1_dataset_completeness | PASS | 5 categories train/test/ground_truth present |
| 2 | S2_mask_completeness | PASS | 401 defect masks matched to test images |
| 3 | S3_seed_completeness | PASS | seeds observed = [0, 1, 2] |
| 4 | S4_no_duplicate_units | PASS | 92 units, 92 unique |
| 5 | S5_no_nan_in_primary_metrics | PASS | 624 primary metric values (d-prime / |dz| / image AUROC); NaN count = 0 |
| 5b | S5b_secondary_metric_nan_accounted_for | PASS | secondary (pixel AUROC / AUPRO) NaN = 16 — all from the 5A per_image_scores.csv-sourced baseline entries, whose source asset never stored pixel metrics; primary metrics unaffected |
| 6 | S6_no_inf_in_metrics | PASS | Inf count = 0 over 1040 values |
| 7 | S7_image_count_consistency | PASS | per-category n_rows identical across all configs/seeds: {'bottle': [183], 'cable': [402], 'grid': [182], 'hazelnut': [290], 'screw': [344]} |
| 11 | S11_frozen_protocol_hash_unchanged | PASS | freeze sha256 = e046a89a852da9aa… (recomputed; recorded in reference/policy_freeze.sha256) |
| 12 | S12_no_target_dependent_parameter_generation | PASS | frozen parameter grid contains no target-derived quantity |
| 13 | S13_no_per_category_tuning | PASS | 16 configs; all have a single spec across categories/seeds |
| 14 | S14_no_hidden_dense_sweep | PASS | configs seen = 16; unfrozen extras = [] |
| 15 | S15_deterministic_config_serialization | PASS | configs/module_specs.csv matches the frozen parameter grid |
| 16 | S16_memory_bank_dimensions_recorded | PASS | 92 units with embedding_dim/bank recorded |
| 17 | S17_runtime_recorded | PASS | runtime_seconds present for all units |
| 18 | S18_peak_vram_recorded | PASS | peak GPU memory present for all units |
| 19 | S19_promotion_rule_mechanical | PASS | 14 candidate tiers recomputed identically |
| 20 | S20_no_failed_config_silently_excluded | PASS | 0 unit(s) with non-OK status, all retained and reported: [] |

**18/18 PASS**
