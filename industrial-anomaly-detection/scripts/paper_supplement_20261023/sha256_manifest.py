#!/usr/bin/env python
"""Generate / verify a per-file SHA-256 manifest for cross-machine data transfer.

Purpose (paper supplement 2026-10-23):
    The DINOv2-line input data (M2AD l0_448 cache, RobustAD images, DINOv2 weights)
    lives on the remote Linux server only. To move it to the Windows local machine
    we need a per-file checksum manifest so a partially-corrupted transfer can be
    localized to the exact file (unlike the single aggregate hash used by E22).

Two subcommands, both platform-neutral (no Windows/Linux hardcoded paths):

    generate  ROOT  OUT_MANIFEST
        Walk ROOT recursively, write "sha256  <hex>  <relpath>" lines (sorted).
        relpath uses forward slashes so the manifest is byte-identical across OSes.

    verify    ROOT  MANIFEST
        Re-walk ROOT and compare against MANIFEST. Reports missing / size-mismatch /
        hash-mismatch / unexpected files. Exit 0 iff everything matches.

The manifest line format is:
    <hex64>  <two spaces>  <relpath/with/forward/slashes>

This keeps the two spaces as a fixed separator (paths never contain "  "), so the
same manifest can be diffed verbatim across Linux and Windows.

Exit codes:
    0 = OK (generate always 0; verify 0 iff all files present & matching)
    1 = any mismatch / unexpected file / missing file / usage error
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

SEP = "  "  # two spaces; relpath is normalized to forward slashes, never contains "  "


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(root: Path) -> dict[str, Path]:
    """Return {relpath -> absolute path} for every file under root, forward-slash relpaths."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = p
    return out


def generate(root: Path, manifest: Path) -> int:
    files = _walk(root)
    if not files:
        print(f"[generate] WARNING: no files under {root}", file=sys.stderr)
    lines = []
    for rel in sorted(files):
        digest = _sha256(files[rel])
        lines.append(f"{digest}{SEP}{rel}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[generate] wrote {len(lines)} entries -> {manifest}")
    return 0


def verify(root: Path, manifest: Path) -> int:
    if not manifest.exists():
        print(f"[verify] FAIL: manifest not found: {manifest}", file=sys.stderr)
        return 1
    expect: dict[str, str] = {}
    for ln, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if SEP not in line:
            print(f"[verify] FAIL: bad manifest line {ln}: {line!r}", file=sys.stderr)
            return 1
        digest, rel = line.split(SEP, 1)
        expect[rel] = digest

    actual = _walk(root)
    # If the manifest itself lives inside root, never flag it as "extra".
    actual.pop(manifest.resolve().relative_to(root.resolve()).as_posix(), None)

    missing = sorted(set(expect) - set(actual))
    extra = sorted(set(actual) - set(expect))
    bad = []
    for rel in sorted(set(expect) & set(actual)):
        d = _sha256(actual[rel])
        if d != expect[rel]:
            bad.append(rel)

    print(f"[verify] root={root}")
    print(f"[verify] expect={len(expect)}  actual={len(actual)}  "
          f"missing={len(missing)}  extra={len(extra)}  mismatch={len(bad)}")
    for r in missing:
        print(f"  MISSING  {r}")
    for r in extra:
        print(f"  EXTRA    {r}")
    for r in bad:
        print(f"  MISMATCH {r}")

    if missing or extra or bad:
        print("[verify] RESULT: FAIL")
        return 1
    print("[verify] RESULT: OK (all files present and byte-identical)")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "verify"):
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "generate" and len(sys.argv) == 4:
        return generate(Path(sys.argv[2]), Path(sys.argv[3]))
    if cmd == "verify" and len(sys.argv) == 4:
        return verify(Path(sys.argv[2]), Path(sys.argv[3]))
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
