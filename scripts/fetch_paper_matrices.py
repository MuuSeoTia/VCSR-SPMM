#!/usr/bin/env python3
import os
from pathlib import Path
import traceback
import ssgetpy
from scipy.io import mmwrite

ROOT = Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))
ROOT.mkdir(parents=True, exist_ok=True)

PAPER_MATRICES = [
   ("GHS_indef","boyd2"),      
    ("Mittelmann","neos3"),
]


def have_mtx(p: Path) -> bool:
    return any(x.suffix == ".mtx" for x in p.glob("**/*.mtx"))

def fetch_one(group, name):
    outdir = ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)
    if have_mtx(outdir):
        print(f"[SKIP] {group}/{name} already exists")
        return
    try:
        print(f"[INFO] {group}/{name}: querying SuiteSparse...")
        result = ssgetpy.search(group=group, name=name)
        if not result:
            print(f"[WARN] {group}/{name}: not found in database")
            return
        entry = result[0]
        print(f"[INFO] Downloading tarball for {group}/{name}")
        entry.download(destpath=str(outdir), extract=True)
        mtx_files = list(outdir.glob("**/*.mtx"))
        if mtx_files:
            print(f"[OK] {group}/{name} -> {mtx_files[0]}")
            return
        # fallback: API fetch if no tarball extracted
        print(f"[WARN] No .mtx found; API fallback for {group}/{name}")
        matdata = ssgetpy.fetch(group=group, name=name)
        mmwrite(outdir / f"{name}.mtx", matdata["A"])
        print(f"[OK] {group}/{name} -> {outdir / (name + '.mtx')}  (API fallback)")
    except Exception as e:
        print(f"[ERROR] {group}/{name}: {e}")
        traceback.print_exc()

def main():
    print(f"[INFO] Destination: {ROOT}")
    for g, n in PAPER_MATRICES:
        fetch_one(g, n)
    print("\n[INFO] Completed downloads:")
    for p in ROOT.glob("**/*.mtx"):
        print(" ", p)

if __name__ == "__main__":
    main()
