# Experiment 1E Dataset Inventory

| category | train good | test good | defect types | defect samples | total test | masks missing | resolution |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 209 | 20 | 3 | 63 | 83 | 0 | 900x900 |
| cable | 224 | 58 | 8 | 92 | 150 | 0 | 1024x1024 |
| hazelnut | 391 | 40 | 4 | 70 | 110 | 0 | 1024x1024 |
| screw | 320 | 41 | 5 | 119 | 160 | 0 | 1024x1024 |
| grid | 264 | 21 | 5 | 57 | 78 | 0 | 1024x1024 |

## Defect types 明细

### bottle
- broken_large: 20
- broken_small: 22
- contamination: 21

### cable
- bent_wire: 13
- cable_swap: 12
- combined: 11
- cut_inner_insulation: 14
- cut_outer_insulation: 10
- missing_cable: 12
- missing_wire: 10
- poke_insulation: 10

### hazelnut
- crack: 18
- cut: 17
- hole: 18
- print: 17

### screw
- manipulated_front: 24
- scratch_head: 24
- scratch_neck: 25
- thread_side: 23
- thread_top: 23

### grid
- bent: 12
- broken: 12
- glue: 11
- metal_contamination: 11
- thread: 11
