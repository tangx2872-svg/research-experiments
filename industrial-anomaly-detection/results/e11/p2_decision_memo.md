# E11-P2 · Decision Memo

## Result (official image_anomaly + detectable, view120, 300-bank, unified 1-NN-max)

| backbone | Bird | Car | Cube | mean |
|---|---|---|---|---|
| ResNet (PatchCore emb) | 0.6845 | 0.6225 | 0.6086 | 0.6385 |
| DINOv2-S | 0.7388 | 0.7579 | 0.8332 | 0.7766 |
| DINOv2-B | 0.7473 | 0.7855 | 0.8541 | 0.7956 |

DINOv2-B vs ResNet: +0.063 / +0.163 / +0.246 (mean +0.157). DINOv2-S: mean +0.138.

## Answers (P2.5)
- **A. Does DINOv2 improve over ResNet?** YES, strongly and consistently (3/3 categories).
- **B. Category-specific?** NO - consistent across Bird/Car/Cube (no category reverses).
- **C. Normal score varies with illumination?** YES (illumination 07 is the hard condition for all backbones).
- **D. Anomaly/normal overlap?** ResNet heavily overlaps; DINOv2 separation is 3-6x larger.
- **E. Was E7's DINOv2 failure representation or scoring?** Evidence now points to **scoring/aggregation +
  label protocol**, NOT representation: with the SAME clean scoring, DINOv2 >> ResNet. E7 used
  (i) top-1% mean instead of max, (ii) included the CLS token, and (iii) legacy object_anomaly labels.
  This is a *reasonable inference*, not a formal causal proof.

## Verdict: **GO** (direction, not method)

There is strong evidence that the **foundation representation (DINOv2) is the dominant lever** for
image-level AD under illumination variation, far larger than any E5-E10 cheap PatchCore modification
(whose gains were all category-specific and <=+0.02). The E11 direction should be
**DINOv2-family methods** (Dinomaly, INP-Former, a DINOv2 + illumination-invariant head), NOT further
ResNet/PatchCore modifications.

## Caveats (GO does NOT mean "proven")
- View120 only (not full 12-view Invariant). Single seed. Frozen features (no trained head).
- The unified scoring (1-NN-max) is a deliberately simple diagnostic; the final method may use a
  different scoring.
- No claim of novelty or publication readiness is made here.

## Recommended next experiment (single, not executed)
E11-P3: replace the PatchCore/ResNet representation with frozen DINOv2-B and add a **normal-only
illumination-conditioning / invariant scoring head**, benchmarked on the full official 12-view
Invariant protocol, comparing against the frozen DINOv2-B 1-NN-max baseline established here.
(Awaiting human approval; P3 NOT auto-started.)
