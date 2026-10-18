#!/usr/bin/env python
"""E25 - RobustAD downloader (hf-mirror, HTTP/1.1, resumable, no masks).

Why this exists: on this server huggingface.co is unreachable and the hub client's
pagination links point back to huggingface.co, so `snapshot_download` stalls. This
script talks to hf-mirror.com directly with a small thread pool.

Only image-level data is needed, therefore `/masks/` files are skipped.

Usage:
  python scripts/e25_download_robustad.py --subsets PCB,MetalParts --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ("https://hf-mirror.com/api/datasets/AmazonScience/RobustAD/tree/main/"
       "{}?expand=false&recursive=true&limit=1000")
RES = "https://hf-mirror.com/datasets/AmazonScience/RobustAD/resolve/main/{}"
DEST = ROOT / "data" / "robustad"
KEEP_SUFFIX = (".jpg", ".jpeg", ".png", ".jsonl")
HDRS = {"User-Agent": "curl/8.5.0"}

_lock = threading.Lock()
_state = {"done": 0, "bytes": 0, "fail": [], "t0": time.time()}
DELAY = 0.0  # seconds slept before each HTTP request (rate-limit guard)


def fetch_tree(sub: str) -> list:
    url = API.format(sub)
    out: list = []
    for _ in range(20):
        req = urllib.request.Request(url, headers=HDRS)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out += json.loads(r.read())
                link = r.headers.get("Link", "")
        except Exception as exc:  # noqa: BLE001
            print(f"[tree] {sub} error {type(exc).__name__}: {exc}", flush=True)
            break
        nxt = None
        for part in (link or "").split(","):
            if 'rel="next"' in part:
                nxt = part.split("<", 1)[1].split(">", 1)[0]
        if not nxt:
            break
        url = nxt.replace("https://huggingface.co", "https://hf-mirror.com")
    return out


def download_one(path: str, size: int, attempt: int = 0) -> bool:
    dst = DEST / path
    if dst.exists() and dst.stat().st_size == size:
        with _lock:
            _state["done"] += 1
            _state["bytes"] += size
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = RES.format(urllib.parse.quote(path))
    try:
        if DELAY:
            time.sleep(DELAY)
        req = urllib.request.Request(url, headers=HDRS)
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if tmp.stat().st_size != size:
            raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
        os.replace(tmp, dst)
        with _lock:
            _state["done"] += 1
            _state["bytes"] += size
        return True
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        # 429 = mirror rate limit: retry more times with a much longer backoff
        limit = 6 if code == 429 else 3
        if attempt < limit:
            time.sleep(15 * (attempt + 1) if code == 429 else 2 + attempt * 3)
            return download_one(path, size, attempt + 1)
        with _lock:
            _state["fail"].append(f"{path} :: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="PCB,MetalParts")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N files")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to sleep before each request (use after HTTP 429)")
    args = ap.parse_args()
    global DELAY
    DELAY = max(0.0, args.delay)

    items = []
    for sub in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        tree = fetch_tree(sub)
        files = [x for x in tree if x.get("type") == "file"]
        keep = [x for x in files
                if x["path"].endswith(KEEP_SUFFIX) and "/masks/" not in x["path"]]
        items += [(x["path"], int(x.get("size", 0))) for x in keep]
        print(f"[tree] {sub}: files={len(files)} kept={len(keep)} "
              f"size={sum(x.get('size', 0) for x in keep)/1e9:.2f} GB", flush=True)
    if args.limit:
        items = items[:args.limit]
    total = len(items)
    total_bytes = sum(s for _, s in items)
    already = sum(1 for p, s in items if (DEST / p).exists()
                  and (DEST / p).stat().st_size == s)
    print(f"[plan] files={total} total={total_bytes/1e9:.2f} GB "
          f"already_complete={already} workers={args.workers}", flush=True)

    q = [(p, s) for p, s in items if not ((DEST / p).exists()
                                          and (DEST / p).stat().st_size == s)]
    q.sort(key=lambda x: x[1])  # small files first -> faster feedback

    def worker():
        while True:
            with _lock:
                if not q:
                    return
                p, s = q.pop(0)
            download_one(p, s)
            with _lock:
                d, b = _state["done"], _state["bytes"]
            if d % 25 == 0:
                el = time.time() - _state["t0"]
                rate = b / max(el, 1e-9) / 1e6
                remain = (total_bytes - b) / max(b / max(el, 1e-9), 1e-9) / 60
                print(f"[dl] {d}/{total} files ({100*d/total:.1f}%) "
                      f"{b/1e9:.2f}/{total_bytes/1e9:.2f} GB {rate:.2f} MB/s "
                      f"elapsed={el/60:.1f}m eta={remain:.1f}m", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    el = time.time() - t0
    print(f"[dl] finished in {el/60:.1f} min | done={_state['done']}/{total} | "
          f"bytes={_state['bytes']/1e9:.2f} GB | failures={len(_state['fail'])}", flush=True)
    for f in _state["fail"][:20]:
        print("  FAIL", f, flush=True)
    return 1 if _state["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
