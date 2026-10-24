"""Stage 2: from cached MTI range-time matrices, compute model-input spectrogram images for
every front-end configuration in configs/frontends.yaml. One STFT per file is shared by all
configurations. Output: data/cache/images/<name>.npy  [N, 128, 128] float16, aligned with
experiments/manifests/gla848_index.csv (manifest rows with ok == True, original order).

usage: python scripts/build_images.py [--jobs 10] [--only name1,name2]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.radar_processing.frontend import (stft_bins, make_image, range_gate,  # noqa: E402
                                           range_bin_m, select_window)

OUT = ROOT / "data" / "cache" / "images"
WIN_S = 5.0


def one(cache_path, bw, tsweep, cfgs):
    rt = select_window(np.load(ROOT / cache_path), int(round(WIN_S / tsweep)))
    P = stft_bins(rt)
    bm = range_bin_m(bw)
    imgs = {name: make_image(P, bin_m=bm, **c).astype(np.float16) for name, c in cfgs.items()}
    _, centre = range_gate(P, "track", bin_m=bm)
    e = P.sum((1, 2))
    r_m = (np.arange(len(e)) + 1) * bm
    infix = (r_m >= 3.75) & (r_m <= 11.25)
    diag = dict(bin_m=bm, track_median_m=float((np.median(centre) + 1) * bm),
                track_iqr_m=float(np.subtract(*np.percentile(centre, [75, 25])) * bm),
                energy_centroid_m=float((e * r_m).sum() / e.sum()),
                frac_energy_in_fixed=float(e[infix].sum() / e.sum()))
    return imgs, diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--variant", default="", help="e.g. hann: use gla848_manifest_hann.csv, write <name>_hann.npy")
    a = ap.parse_args()
    cfgs = yaml.safe_load(open(ROOT / "configs" / "frontends.yaml"))
    if a.only:
        cfgs = {k: v for k, v in cfgs.items() if k in a.only.split(",")}
    man = pd.read_csv(ROOT / "experiments" / "manifests" / (f"gla848_manifest_{a.variant}.csv" if a.variant else "gla848_manifest.csv"))
    ref = pd.read_csv(ROOT / "experiments" / "manifests" / "gla848_manifest.csv")
    assert (man.file.values == ref.file.values).all(), "variant manifest must keep the sample order"
    man = man[man.ok].reset_index(drop=True)
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=2)(delayed(one)(p, bw, ts, cfgs)
                                             for p, bw, ts in zip(man.cache, man.bw, man.tsweep))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cfgs:
        np.save(OUT / (f"{name}_{a.variant}.npy" if a.variant else f"{name}.npy"), np.stack([r[0][name] for r in res]))
    if not a.only and not a.variant:
        diag = pd.DataFrame([r[1] for r in res])
        pd.concat([man, diag], axis=1).to_csv(ROOT / "experiments" / "manifests" / "gla848_index.csv", index=False)
    print(f"built {list(cfgs)} for {len(man)} files in {time.time() - t0:.0f}s")
