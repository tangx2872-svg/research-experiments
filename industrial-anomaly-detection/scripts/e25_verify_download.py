#!/usr/bin/env python
"""E25 - verify the RobustAD local copy against the HF tree listing.

Checks, in order:
  1. every kept remote file (image / jsonl, masks excluded) exists locally with the exact size
  2. no unexpected local file
  3. every entry in each split's metadata.jsonl resolves to a local file
  4. metadata label agrees with the directory token (normal/ -> 0, anomaly/ -> 1)

Writes results/e25/download_verification.json and one results/e25/split_inventory.csv row
per split. Exit code 1 on any FAIL.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e25_download_robustad import DEST, KEEP_SUFFIX, fetch_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e25"
SUBSETS = ["PCB", "MetalParts"]


def local_files() -> set:
    return {str(p.relative_to(DEST)) for p in DEST.rglob("*") if p.is_file()}


def main() -> int:
    remote = []
    for sub in SUBSETS:
        tree = fetch_tree(sub)
        files = [x for x in tree if x.get("type") == "file"]
        keep = [x for x in files
                if x["path"].endswith(KEEP_SUFFIX) and "/masks/" not in x["path"]]
        remote += [(x["path"], int(x.get("size", 0))) for x in keep]
        print(f"[tree] {sub}: remote_files={len(files)} kept={len(keep)}", flush=True)

    missing, bad_size = [], []
    for path, size in remote:
        dst = DEST / path
        if not dst.exists():
            missing.append(path)
        elif dst.stat().st_size != size:
            bad_size.append({"path": path, "remote": size, "local": dst.stat().st_size})
    print(f"[files] remote={len(remote)} missing={len(missing)} bad_size={len(bad_size)}",
          flush=True)

    expect = {p for p, _ in remote}
    actual = local_files()
    extra = sorted(actual - expect)
    part = [p for p in actual if p.endswith(".part")]
    print(f"[files] extra_local={len(extra)} part_files={len(part)}", flush=True)

    splits = sorted({str(Path(p).parent) for p, _ in remote if p.endswith(".jsonl")})
    inv_rows, meta_bad, meta_missing = [], [], []
    for split in splits:
        md = DEST / split / "metadata.jsonl"
        if not md.exists():
            inv_rows.append({"split": split, "n_metadata": 0, "n_normal": 0,
                             "n_anomaly": 0, "n_local_images": 0})
            print(f"[split] {split}: METADATA MISSING", flush=True)
            continue
        recs = [json.loads(line) for line in md.read_text().splitlines() if line.strip()]
        n_ok = n_ng = 0
        for r in recs:
            rel = f"{split}/{r['file_name']}"
            if rel not in actual:
                meta_missing.append(rel)
            expect_lab = 0 if r["file_name"].startswith("normal/") else 1
            if int(r["label"]) != expect_lab:
                meta_bad.append({"path": rel, "label": r["label"]})
            n_ok += int(r["label"]) == 0
            n_ng += int(r["label"]) == 1
        inv_rows.append({"split": split, "n_metadata": len(recs), "n_normal": n_ok,
                         "n_anomaly": n_ng,
                         "n_local_images": sum(1 for p in actual
                                               if p.startswith(split + "/")
                                               and p.rsplit(".", 1)[-1] in ("jpg", "jpeg", "png"))
                         })
        print(f"[split] {split}: metadata={len(recs)} (ok={n_ok} ng={n_ng})", flush=True)

    ok = not (missing or bad_size or meta_missing or meta_bad or part)
    payload = {"subset": SUBSETS, "remote_files": len(remote), "missing": missing[:50],
               "n_missing": len(missing), "bad_size": bad_size[:50],
               "n_bad_size": len(bad_size), "extra_local": extra[:50],
               "n_extra_local": len(extra), "n_part_files": len(part),
               "n_metadata_missing": len(meta_missing),
               "n_label_inconsistent": len(meta_bad),
               "label_inconsistent": meta_bad[:20], "splits": inv_rows, "ok": ok}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "download_verification.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "split_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()) if inv_rows else ["split"])
        w.writeheader()
        w.writerows(inv_rows)
    print(f"[verify] ok={ok} -> results/e25/download_verification.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
