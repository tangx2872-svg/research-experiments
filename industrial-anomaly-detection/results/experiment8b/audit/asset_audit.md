# Experiment 8B — P0 asset audit

- normal bank assets: 30/30
- per-channel sigma (30 banks, 23040 ch): min=0 p01=0.00795 median=0.82 max=6.97; n(<1e-3)=62 n(<1e-6)=43
- 1J defect-side npz: 1848 (a0=924, a1=924)
- illumination-side: NO feature-level illumination asset: brightness/gamma perturbation only stored as per-image anomaly scores in 7A-O Q3/Q4; no paired feature representation exists
- feature files in results outside 1J: []

## frozen primary 9 defects: images with zero defect patch (overlap>0.5)
- bottle/contamination: imgs=21 mask_ok=21 empty_mask=0 l2_zero=0 l3_zero=2
- cable/bent_wire: imgs=13 mask_ok=13 empty_mask=0 l2_zero=0 l3_zero=0
- grid/glue: imgs=11 mask_ok=11 empty_mask=0 l2_zero=1 l3_zero=2
- grid/metal_contamination: imgs=11 mask_ok=11 empty_mask=0 l2_zero=0 l3_zero=8
- grid/thread: imgs=11 mask_ok=11 empty_mask=0 l2_zero=0 l3_zero=4
- hazelnut/print: imgs=17 mask_ok=17 empty_mask=0 l2_zero=0 l3_zero=0
- screw/manipulated_front: imgs=24 mask_ok=24 empty_mask=0 l2_zero=0 l3_zero=17
- screw/thread_side: imgs=23 mask_ok=23 empty_mask=0 l2_zero=5 l3_zero=19
- screw/thread_top: imgs=23 mask_ok=23 empty_mask=0 l2_zero=1 l3_zero=11

## test coverage
- bottle: good=20 defects=[('broken_large', 20), ('broken_small', 22), ('contamination', 21)]
- cable: good=58 defects=[('bent_wire', 13), ('cable_swap', 12), ('combined', 11), ('cut_inner_insulation', 14), ('cut_outer_insulation', 10), ('missing_cable', 12), ('missing_wire', 10), ('poke_insulation', 10)]
- grid: good=21 defects=[('bent', 12), ('broken', 12), ('glue', 11), ('metal_contamination', 11), ('thread', 11)]
- hazelnut: good=40 defects=[('crack', 18), ('cut', 17), ('hole', 18), ('print', 17)]
- screw: good=41 defects=[('manipulated_front', 24), ('scratch_head', 24), ('scratch_neck', 25), ('thread_side', 23), ('thread_top', 23)]
