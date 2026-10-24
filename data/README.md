# Data

The raw data are **not** included in the project archive (2 GB compressed / 16 GB extracted).

- Dataset: *Radar signatures of human activities*, University of Glasgow, https://researchdata.gla.ac.uk/848/, DOI 10.5525/gla.researchdata.848
- Licence: CC BY 4.0. Cite the dataset and Shah et al. / Fioranelli et al. when you use it.
- Get it: `bash scripts/get_data.sh` (downloads, checks SHA-256 `652f9adc…ae61` and extracts to `data/raw/gla848/extracted`).

Derived caches (re-created by the scripts and not shipped):
- `data/cache/rt/` — complex MTI range-time matrices (48 bins × slow time), `scripts/build_cache.py`
- `data/cache/images/<frontend>.npy` — 128×128 model inputs per front-end, `scripts/build_images.py`

Shipped manifests: `experiments/manifests/gla848_manifest.csv` (per-file header and labels) and
`gla848_index.csv` (plus range diagnostics). Row order defines sample indices in all result files.
