# E5-1 — Broad Module Composition Screening (results)

Frozen reference: Original AUROC=0.76766, d'=1.20676, R_all=0.67543

| Candidate | AUROC | dAUROC | d' | dd' | R_all | R_ratio | Decision |
|---|---|---|---|---|---|---|---|
| Original | 0.76766 | +0.00000 | 1.20676 | +0.00000 | 0.67543 | 1.0000 | REFERENCE |
| C01 PIAD-Retinex (input) | 0.78043 | +0.01277 | 1.30552 | +0.09876 | 0.69617 | 1.0307 | **HOLD** |
| C06 SimpleNet (post-concat) | 0.48820 | -0.27946 | -0.05534 | -1.26210 | 0.50851 | 0.7529 | **STOP** |
| C07 ReConPatch (post-concat) | 0.55555 | -0.21211 | 0.14363 | -1.06313 | 0.67286 | 0.9962 | **STOP** |
| C08 CRAD (memory) | 0.46197 | -0.30569 | -0.11035 | -1.31711 | 0.31423 | 0.4652 | **STOP** |

## Ranks (not AUROC-only)

| Candidate | Detection | Robustness | Preservation | Overall |
|---|---|---|---|---|
| C01 | 1 | 4 | 1 | **HOLD** |
| C06 | 3 | 2 | 3 | **STOP** |
| C07 | 2 | 3 | 2 | **STOP** |
| C08 | 4 | 1 | 4 | **STOP** |
