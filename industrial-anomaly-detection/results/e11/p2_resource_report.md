# E11-P2 · Resource Report

- **GPU**: RTX 3090 (24 GB). Peak VRAM 6.9 GB (resnet/Cube, 1536-d bank 307200 patches).
- **Runtime per unit** (300 bank + 700 test, view120): resnet ~123 s (includes PatchCore fit ~90 s),
  dinov2s ~30 s, dinov2b ~55 s. Total formal run ~11 min.
- **GPU-hours consumed**: ~0.7 GPU-h total (smoke + 2 bug-fix partial runs + 1 clean full run).
- **Disk**: 16 GB free, unchanged (no new large caches; 1 corrupt l0_448 file rebuilt in place).
- **Downloads**: 0 bytes this round (DINOv2 weights already cached: vits14 88 MB, vitb14 346 MB).
- **Incident log**:
  1. l0_448 corrupt file `Bird/Good/029/A030_I08.png` (0-byte, E8 Stage-A interrupt) -> rebuilt
     from the local Bird original (bit-exact WebP-lossless 448 resize). Scanned all l0_256/l0_448;
     only this 1 file was corrupt.
  2. torch.hub transient `RemoteDisconnected` on a repeated model load -> fixed with per-backbone
     in-memory model cache + retry (no re-download; weights were already local).
- **Fidelity**: frozen PatchCore ResNet embedding reused from E10 (bit-identical extractor); DINOv2
  uses the E7 `_dinov2_prepare` semantics (448 BICUBIC + ImageNet norm) on l0_448.
