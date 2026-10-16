# Paper evidence map (claim → evidence → files → boundary)

### Claim A — R1-ILA shows broadly positive cross-category behaviour on M²AD
Evidence: E22 (10 categories × 3 seeds × view120): 9/10 positive mean ΔAUROC, macro mean 0.00760, median 0.01079, 7/10 positive across all seeds; protocol verdict HOLD (Doll triggers the severe-regression exclusion).
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
