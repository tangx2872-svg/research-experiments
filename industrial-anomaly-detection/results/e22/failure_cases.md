# E22 — Failure-case analysis (all negative results retained)

Generated from 30/30 cells. Every negative cell listed, none removed.

## 1. Worst category (3-seed mean ΔAUROC)

- **Doll**: ΔAUROC -0.03850 (std 0.00464), ΔAP -0.01302, positive seeds 0/3.
- F0 AUROC 0.71250 → R1 AUROC 0.67401

## 2. Best category (for contrast)

- **Bird**: ΔAUROC +0.02586, ΔAP +0.02371, positive seeds 3/3.

## 3. Worst single cell (ΔAUROC / ΔAP)

- **Doll|seed0** ΔAUROC: ΔAUROC -0.04385, ΔAP -0.00947 (F0 0.71941/0.80755 → R1 0.67556/0.79808), bank=46a03f7ed371, steps=44
- **Doll|seed2** ΔAP: ΔAUROC -0.03576, ΔAP -0.02203 (F0 0.70504/0.80981 → R1 0.66928/0.78778), bank=d4efeaf9f33b, steps=49

## 4. Largest cross-seed variability (std of ΔAUROC)

- **Cube**: std 0.01421 (min -0.01229, max +0.01600, positive seeds 2/3)

## 5. All negative cells (ΔAUROC < 0)

| category | seed | ΔAUROC | ΔAP | F0 AUROC | R1 AUROC | bank |
|---|---|---|---|---|---|---|
| Cube | 1 | -0.01229 | -0.00321 | 0.87203 | 0.85974 | f8b4249250 |
| Doll | 0 | -0.04385 | -0.00947 | 0.71941 | 0.67556 | 46a03f7ed3 |
| Doll | 1 | -0.03588 | -0.00756 | 0.71306 | 0.67719 | b60e2ca632 |
| Doll | 2 | -0.03576 | -0.02203 | 0.70504 | 0.66928 | d4efeaf9f3 |
| Motor | 1 | -0.00025 | -0.00258 | 0.82981 | 0.82956 | b4c1ffbf6b |

## 6. Categories with a negative 3-seed mean

| category | ΔAUROC mean | ΔAP mean | positive seeds | CI (seed-mean, ΔAUROC) |
|---|---|---|---|---|
| Doll | -0.03850 | -0.01302 | 0/3 | [-0.0661, -0.0140] |

## 7. Seed-stability description (descriptive labels only; no category removed)

| category | positive seeds (AUROC) | std ΔAUROC | label |
|---|---|---|---|
| Bird | 3/3 | 0.013535 | seed_sensitive |
| Car | 3/3 | 0.007279 | stable_positive |
| Cube | 2/3 | 0.014214 | seed_sensitive |
| Dice | 3/3 | 0.000775 | stable_positive |
| Doll | 0/3 | 0.004638 | stable_negative |
| Holder | 3/3 | 0.004453 | stable_positive |
| Motor | 2/3 | 0.007646 | mixed |
| Ring | 3/3 | 0.003634 | stable_positive |
| Teapot | 3/3 | 0.007286 | stable_positive |
| Tube | 3/3 | 0.004989 | stable_positive |

**Rule for the descriptive labels** (defined in README): `stable_positive` = 3/3 positive seeds and std ≤ 0.01; `stable_negative` = 0/3 positive; `seed_sensitive` = std > 0.01; `mixed` otherwise. Labels are descriptive; they never remove a category.
