# E11-P2 · Cache Validation

## Verdict: PASS — both caches are lossless RGB images (NOT precomputed features)

| cache | content | format | mode | shape | dtype | range | count (Bird/Car/Cube) |
|---|---|---|---|---|---|---|---|
| `l0_256` | RGB image | PNG | RGB | 256x256x3 | uint8 | [0,~255] | 4000 / 4000 / 4000 |
| `l0_448` | RGB image | WEBP-lossless (`.png` extension, PIL detects by magic bytes) | RGB | 448x448x3 | uint8 | [0,~255] | 4000 / 4000 / 4000 |

## Per-item check
- **format/shape**: both are reconstructible RGB uint8 images. l0_448 was written as WebP-lossless with a
  `.png` suffix (E8 `write_image_caches` bug); `PIL.Image.open` detects the true format by content, so it
  decodes correctly as WEBP. Verified `PIL_format=WEBP`, shape (448,448,3).
- **RGB/BGR**: RGB (PIL). No BGR.
- **normalization range**: uint8 [0,255]; the backbones' own transforms (ToTensor+ImageNet norm for DINOv2,
  load_image_as_tensor for ResNet) are applied at inference, identical to E8/E10.
- **resize/crop**: BICUBIC square resize (256/448). DINOv2 `_dinov2_prepare` Resize(448)+crop is a no-op on
  already-448x448 images; patch grid = 32x32 (448/14) for both S and B.
- **ID alignment**: cache rel path == metadata `img_path` exactly (verified: `Bird/Good/000/A000_I01.png`
  exists in both caches). 1:1 with metadata.
- **normal train coverage**: 300-bank is a subset of the 1200 pool; all present in caches.
- **view coverage**: view120 test (700 img) fully cached for Bird/Car/Cube.
- **duplicates / train-test leakage**: bank = train-split Good specimens; test = test-split; disjoint by the
  frozen E7-1 scope (verified in E8 sanity gate S4: 0 specimen overlap). No cache-level mixing.

## Caveat (declared, not hidden)
The DINOv2 input is the 448x448 BICUBIC resize of the 1024 original (l0_448), which is byte-identical to the
E7 `_dinov2_prepare` smaller-edge-448 resize for square 1024 images. ResNet input is l0_256 (256x256 BICUBIC),
identical to E8/E10. Both are the official-image preprocessed inputs; no original 1024 bytes are used, and
no further difference is introduced beyond the frozen resize (documented E8 D2 deviation).
