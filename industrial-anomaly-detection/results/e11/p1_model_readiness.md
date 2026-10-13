# E11-P1 · Dinomaly / MSFlow Minimal Runnability

Official code: `third_party/M2AD` @ `a606aa0` (no LICENSE).

## Dependencies (checked)
| dep | status | needed by |
|---|---|---|
| timm | OK | framework |
| mmcv / mmdet | MISSING | framework (trainer/data/evaluator) |
| adeval | MISSING | metrics |
| imgaug / mmselfsup | MISSING | framework |
| FrEIA | MISSING | **MSFlow** (flow_models) |

## Method readiness
| method | backbone | weight | train | blocker |
|---|---|---|---|---|
| Dinomaly | dinov2reg_vit_base_14 | MISSING (Baidu link) | 100e | reg weight + mmdet framework |
| MSFlow | wide_resnet50_2 | CACHED | 100e | FrEIA + mmdet framework |
| INP-Former | dinov2reg_vit_base_14 | MISSING | 100e | reg weight |
| RD++ | wide_resnet50_2 + decoder | CACHED | 100e | DTD (DRAEM synthetic) |
| CDO | mshrnet32 | MISSING | 100e | mshrnet32 + DTD |

## Minimal runnability status
- **Framework**: NOT runnable without installing mmcv/mmdet/adeval (~1-2 GB, risks current env). NOT installed this round.
- **Core models**: Dinomaly core is torch+local (importable in principle) but needs reg checkpoint;
  MSFlow core needs FrEIA.
- **No import/init/forward smoke performed** — would require installing missing deps; deferred
  (recorded as BLOCKED, not a false result).

## Weight download watchlist
- `dinov2reg_vit_base_14` (~1.1 GB) — official M2AD README gives a **Baidu** link (likely unreachable); HF mirror may host `facebook/dinov2-with-registers`. Verify before download.
- `mshrnet32` (CDO) — timm HRNet backbone.
- `DTD` (~600 MB) — for DRAEM synthetic anomaly (RD++/CDO).
