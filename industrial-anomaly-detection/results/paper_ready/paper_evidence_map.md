# Paper evidence map (claim → evidence → files → boundary)

> Consolidated 13-section map, including E24 and E25: `results/paper_ready/EVIDENCE_MAP.md` (hand-curated; this file covers Claims A–F only).

### Claim A — R1-ILA shows broadly positive cross-category behaviour on M²AD
Evidence: E22 (10 categories × 3 seeds × view120): 9/10 positive mean ΔAUROC, macro mean 0.00760, median 0.01080, 7/10 positive across all seeds; protocol verdict HOLD (Doll triggers the severe-regression exclusion).
Files: `results/e22/README.md`, `main_table_10cat_3seed.csv`, `bootstrap_e22.json`, `verification.csv` (36/36), `results/paper_ready/paper_main_table.csv`.
Boundary: cannot be written as universal robustness or as a statistically significant improvement on all categories; Doll must be reported.

### Claim B — The benefit is related to illumination conditions
Evidence: E16/E18/E19/E22 diagnostics: illumination shift (D) decreases for every category after adaptation; normal-score spread across illumination groups is reduced in 9/10 categories; shift ↔ gain association is weakly positive (n = 10).
Files: `results/e16/e16_decision_memo.md`, `results/e18/statistical_analysis.md`, `results/e19/` (77/77 checks), `results/e22/shift_gain_association.json`, `results/e23/category_comparison.csv`.
Boundary: association only (n = 10, exploratory); no causal claim; the normal-dispersion narrative is NOT supported (dispersion increases in the diagnostic).

### Claim C — The preservation term prevents degenerate behaviour
Evidence: E17 BETA ablation (0 / 0.05 / 0.1 / 0.2 / 0.5 × Bird/Car/Cube): BETA=0 yields the lowest d′ on Bird, Car and Cube (tied-lowest on Cube with BETA=0.2), while its nominal AUROC does not rank it worst on every category — Cube shows the highest nominal AUROC (0.90904) accompanied by the documented collapse (pairwise cos 0.988). BETA=0.1 was frozen as the final method; no challenger passed the +0.003 gate.
Files: `results/e17/README.md`, `results/e17/summary.csv`, `results/paper_ready/paper_ablation_table.csv`.
Boundary: do not state 'BETA=0 is the worst detector on all three categories' (AUROC-based wording); the collapse must be mentioned whenever Cube's nominal AUROC is quoted.

### Claim D — Multi-view generalization is partial rather than universal
Evidence: E21 (4 views × 3 categories × seed0): 8/12 positive cells, macro pooled ΔAUROC +0.00142 (CI [−0.0032, +0.0062]), ΔAP −0.00165, verdict HOLD; Car 240/330 and Cube 090/330 negative; view120 reproduction bit-identical to E16/E18/E19.
Files: `results/e21/README.md`, `results/e21/analysis_cells.csv`, `results/paper_ready/paper_multiview_table.csv` (all 12 cells retained).
Boundary: HOLD, seed0 only; not a viewpoint-invariance claim.

### Claim E — Doll defines the applicability boundary
Evidence: E22 (Doll −0.03850, CI [−0.0661, −0.0140], p = 0.001, 0/3 seeds) + E23 descriptive audit (EXPLAINED_PATTERN: anomaly-side drop exceeds normal-side drop and d′ falls in 3/3 seeds; broad across illumination groups; Doll has the lowest F0 AUROC and the lowest d′ among the 10 categories; its illumination shift is close to the category median).
Files: `results/e23/README.md`, `e23_verdict.json`, `doll_separation_summary.csv`, `doll_illumination_breakdown.csv`, `doll_vs_others.json`, `figures/` (Fig1–Fig5), `results/paper_ready/paper_limitations.md` (L1).
Boundary: no causal explanation, no repair, no universal claim; the audit states a consistent but small-magnitude pattern.

### Claim F — The final method is a single frozen configuration
Evidence: E16 (GO) → E17 (FREEZE_CURRENT_R1) → E18/E19 (main table, paper assets) → E22 (10-category validation); parameters only from `results/e17/final_method.json`; no tuning in any later experiment; frozen core files unchanged (verified in E22 V31 and E23).
Files: `results/e17/final_method.json`, `results/e18/README.md`, `results/e19/README.md`, `results/e22/protocol.md`.
Boundary: parameter claims must cite `final_method.json`; never hand-copy hyperparameters.

### Claim G — External evaluation on RobustAD is conditionally positive (E25)
Evidence: E25 re-trains the frozen adapter on each RobustAD sub-dataset (PCB, MetalParts) and scores the same 180-normal bank with F0 (frozen DINOv2-B 1-NN-max) and R1; 39 cells = 13 splits × 3 adapter seeds, all status OK, per-cell AUROC/AP recomputation max|d| = 0.0; 33 of them (11 valid target splits × 3 seeds) enter the target aggregates. PCB +0.01835 (5/5 splits positive, all 3 seeds), MetalParts −0.01725 (2/6 positive; background/scale negative in all 3 seeds); overall macro ΔAUROC −0.001068 split-weighted vs +0.000550 sub-dataset-weighted (full precision), descriptive t-interval [−0.00333, +0.00119], macro ΔAP −0.00544.
Files: `results/e25/README.md` (§15), `multiseed_summary.csv`, `multiseed_analysis.json`, `cells/*.json`, `figures/e25_multiseed_domain_delta.*`, `results/paper_ready/EVIDENCE_MAP.md` §7–§9.
Boundary: re-trained per sub-dataset, so NOT direct transfer; no overall gain; not caused by a specific mechanism (scale/background/geometry hypotheses untested); 2 of 3 sub-datasets.

### Claim H — The frozen augmentation setting is an audited, non-arbitrary choice (E24)
Evidence: E24 scans 6 strengths × 3 development categories (seed0) and 7 augmentation-type subsets, plus 3-seed re-runs of the top two strengths, on the SAME frozen pipeline (reproduction vs E18 `Ours`/`F0`: 9/9 checks, max|d| = 0.0). S15 ranks first under the pre-registered rule (3/3 positive, worst +0.010614, mean +0.015401); S05–S15 is a plateau; S10 is +0.00274 higher over 3 seeds but the gap is 0.19× the seed std; gamma alone (A2 +0.019067) exceeds the full set (A6 +0.015401) at seed0 only.
Files: `results/e24/README.md`, `tables/strength_ranked.csv`, `tables/type_ranked.csv`, `tables/stability_multiseed.csv`, `final_strength_analysis.json`, `figures/`.
Boundary: audit only — KEEP_FROZEN; S10 and A2 are candidates, never the frozen method; single seed for the type ablation; no mechanistic claim about gamma.
