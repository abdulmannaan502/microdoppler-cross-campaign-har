"""Stage 1 cache: parse every raw .dat file once, range-compress + MTI, and store the
complex range-time matrix (48 bins x n_chirps, complex64) plus a manifest CSV.

usage: python scripts/build_cache.py <extracted_dataset_root> [--jobs 10]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.gla848 import index_dataset, read_dat  # noqa: E402
from src.radar_processing.frontend import range_compress, mti  # noqa: E402

CACHE = ROOT / "data" / "cache" / "rt"
WINDOW = "rect"


def campaign_id(rec, root):
    # campaign = first-level folder under the dataset root (sub-rooms kept as a separate column)
    rel = rec.path.relative_to(root).parts
    return rel[0], "/".join(rel[1:-1])


def one(rec, root, cache=CACHE, window=WINDOW):
    camp, sub = campaign_id(rec, root)
    out = cache / camp.replace(" ", "_") / (rec.path.stem + ".npy")
    row = dict(file=str(rec.path.relative_to(root)), campaign=camp, room=sub, subject=rec.subject,
               activity=rec.activity, repetition=rec.repetition, cache=str(out.relative_to(ROOT)))
    try:
        hdr, chirps = read_dat(rec.path)
        row.update(fc=hdr["fc"], tsweep=hdr["tsweep"], nts=hdr["nts"], bw=hdr["bw"],
                   n_chirps=chirps.shape[0], duration_s=chirps.shape[0] * hdr["tsweep"])
        if not out.exists():
            rt = mti(range_compress(chirps, window=window))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, rt)
        row["ok"] = True
    except Exception as e:  # keep going; failures are reported in the manifest
        row.update(ok=False, error=repr(e))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--window", default="rect", choices=["rect", "hann"])
    a = ap.parse_args()
    if a.window != "rect":
        WINDOW = a.window
        CACHE = ROOT / "data" / "cache" / f"rt_{a.window}"
    root = Path(a.root)
    recs = index_dataset(root)
    print(f"indexed {len(recs)} files", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(r, root, CACHE, WINDOW) for r in recs)
    df = pd.DataFrame(rows)
    mf = ROOT / "experiments" / "manifests" / ("gla848_manifest.csv" if a.window == "rect" else f"gla848_manifest_{a.window}.csv")
    df.to_csv(mf, index=False)
    print(df.groupby("campaign").agg(n=("file", "size"), ok=("ok", "sum"),
                                     subjects=("subject", "nunique")).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {mf}")
