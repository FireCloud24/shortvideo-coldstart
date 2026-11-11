"""Download MicroLens-100K into data/raw/ and write a SHA-256 manifest.

MicroLens (Ni et al., 2023, arXiv:2309.15379) forbids redistributing modified
copies, so the data is always fetched from the authors' server and never
committed. Re-running skips files that are already present and recorded in the
manifest with the same size.

    python scripts/download.py                 # everything (covers are ~690 MB)
    python scripts/download.py --skip-covers   # interactions and metadata only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/"

# name -> short description. All files live directly under BASE_URL.
FILES = {
    "MicroLens-100k_pairs.csv": "interactions: user,item,timestamp (ms), with header",
    "MicroLens-100k_title_en.csv": "titles (English): item, title; no header",
    "tags_to_summary.csv": "one category per item: item,category; no header",
    "MicroLens-100k_likes_and_views.txt": "item<TAB>likes<TAB>views; no header",
    "readme.txt": "the authors' format notes",
    "MicroLens-100k_covers.zip": "cover images, {item}.jpg",
}
COVERS = "MicroLens-100k_covers.zip"
MANIFEST = "manifest.json"
# Identify honestly as a script. If the server refuses it, download the files
# by hand from https://recsys.westlake.edu.cn/ into data/raw/ and re-run with
# --adopt to record them in the manifest.
USER_AGENT = "shortvideo-coldstart-downloader/0.1 (research use)"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(raw_dir: Path) -> dict:
    path = raw_dir / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": BASE_URL, "files": {}}


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    (raw_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def is_done(raw_dir: Path, name: str, manifest: dict) -> bool:
    """A file is done when it exists and its size matches the manifest entry."""
    path = raw_dir / name
    entry = manifest["files"].get(name)
    return path.exists() and entry is not None and path.stat().st_size == entry["bytes"]


def fetch(url: str, dest: Path, opener=urllib.request.urlopen) -> None:
    """Stream url to dest, resuming a partial .part file when the server allows it."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with opener(req) as resp:
        resumed = start and getattr(resp, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        done = start if resumed else 0
        total = resp.headers.get("Content-Length")
        total = int(total) + done if total else None
        last = time.time()
        with open(part, mode) as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"    {done / 1e6:.0f} MB{pct}", flush=True)
    part.replace(dest)


def download_all(raw_dir: Path, names: list[str], opener=urllib.request.urlopen, adopt: bool = False) -> dict:
    """Fetch each missing file. With adopt=True, a file that is already on disk
    (downloaded by hand) is hashed and recorded instead of being fetched."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)
    for name in names:
        if is_done(raw_dir, name, manifest):
            print(f"skip  {name} (already downloaded)")
            continue
        dest = raw_dir / name
        if adopt and dest.exists():
            print(f"adopt {name} (found on disk)")
        else:
            print(f"fetch {name}")
            fetch(BASE_URL + name, dest, opener)
        manifest["files"][name] = {
            "url": BASE_URL + name,
            "bytes": dest.stat().st_size,
            "sha256": sha256_of(dest),
        }
        save_manifest(raw_dir, manifest)  # after every file, so a crash loses nothing
    save_manifest(raw_dir, manifest)
    return manifest


def extract_covers(raw_dir: Path) -> int:
    """Unzip covers into data/raw/covers/ (flat), skipping files already there."""
    out = raw_dir / "covers"
    out.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(raw_dir / COVERS) as z:
        for info in z.infolist():
            name = Path(info.filename).name  # drop any folder prefix; also blocks path traversal
            if info.is_dir() or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            target = out / name
            if not target.exists() or target.stat().st_size != info.file_size:
                target.write_bytes(z.read(info))
            n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--skip-covers", action="store_true", help="do not download the cover images")
    p.add_argument("--no-extract", action="store_true", help="download the covers zip but do not unzip it")
    p.add_argument("--adopt", action="store_true", help="record files already placed in the folder by hand")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    names = [n for n in FILES if not (args.skip_covers and n == COVERS)]
    manifest = download_all(raw_dir, names, adopt=args.adopt)
    print(f"manifest: {raw_dir / MANIFEST} ({len(manifest['files'])} files)")
    if COVERS in names and not args.no_extract:
        print(f"extracted {extract_covers(raw_dir)} covers to {raw_dir / 'covers'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
