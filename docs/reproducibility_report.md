# Reproducibility Report

## Platforms
| Results | Platform |
|---|---|
| **All neural-network results in the paper** | Vast.ai instance 57526252: NVIDIA RTX 4090 24 GB (driver 595.99, max CUDA 13.2), AMD Threadripper PRO (12.8 vCPU allotted), Ubuntu container `vastai/pytorch`, Python `/venv/main` with torch 2.14.1+cu132, torchvision 0.29.1, numpy 2.5.4, scikit-learn 1.9.1, pandas 3.1.0 (`remote/env.txt`). Workers ran 4–6 conditions concurrently on one GPU |
| SVM / RF baselines, preprocessing, analysis, figures, LaTeX | Local laptop (below) |
| Superseded runs (`results/runs_superseded/`) | Local laptop, RTX 3050 4 GB, torch 2.6.0+cu124. The AdaBN numbers there are biased by the batch-order bug |

**Cross-platform replication of source-model scores (same code, seeds 0–4).** Local vs. remote LOCO macro-F1 (src):
LightCNN fixed_max 0.918 vs 0.908; dual-reference 0.914 vs 0.917; ResNet-18 0.898 vs 0.905. Differences of up to
≈1 point arise from GPU/library non-determinism alone. Effects smaller than this are therefore not interpreted.
Seeds 5–9 were added for the key comparison (fixed_max vs dual-reference) to increase power.

## Local environment
- Windows 10 Home 22H2 (10.0.19045), Git-Bash
- Python 3.12 venv. torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0 (`requirements.txt`; full freeze in `requirements.lock.txt`)
- GPU: NVIDIA GeForce RTX 3050 Laptop (4 GB), driver 610.88. CPU: 12 logical cores
- LaTeX: MiKTeX 25.12, IEEEtran.cls / IEEEtran.bst 1.14

## Data provenance
- Archive SHA-256 `652f9adc2c9921f9fa08ba2de2d40e3f357d89a91f1b13ac790bc68f4f54ae61` (checked by `scripts/get_data.sh`)
- `experiments/manifests/gla848_manifest.csv`: one row per parsed file (header values, labels, cache path)
- `experiments/manifests/gla848_index.csv`: adds range diagnostics. Its row order defines every sample index used in `results/runs/*_preds.npz`

## What is controlled
| Item | How |
|---|---|
| Seeds | `--seeds 0,1,2,3,4`. Seed fixes numpy/torch RNGs, the subject-group permutation (subject protocol), the validation split, the batch order and the enrollment person order |
| Splits | Deterministic functions of the manifest and seed (`scripts/run_experiment.py: folds, val_split`) |
| Hyper-parameters | Fixed for all conditions (`cfg` in `run_experiment.py`). Nothing tuned on test folds. The enrollment recipe uses nested selection |
| Front-ends | `configs/frontends.yaml` |
| Conditions run | `experiments/configs/queue_*.txt` (each line is one condition) |
| Outputs | Per condition: `results/runs/<tag>.csv` (metrics per fold × seed × BN mode), `<tag>_preds.npz` (probabilities and test indices), `<tag>.json` (args, versions, GPU, date) and `experiments/logs/<tag>.log` |
| Paper numbers | `scripts/analyze.py` → `results/tables/*.csv` → `scripts/make_tables.py` → `arxiv/tables/*.tex` (including `numbers.tex` macros used in the text). No number in the paper is typed by hand |

## Known sources of non-determinism
- cuDNN convolution kernels and fp16 autocast are not bit-deterministic, so re-runs give slightly different per-fold scores. See the re-run check below.
- `joblib` parallel preprocessing is deterministic (no randomness in the front-end).

## Re-run check
See `docs/repro_check.md`: clean-checkout tests pass and paper tables regenerate byte-identically; the full pipeline ran on a fresh remote container.

## Exploratory runs not used in reported results
- `results/runs/pilot_*`: 1-seed fp32 pilots that motivated the dual-reference hypothesis (disclosed in the paper).
- `results/runs/enroll_smoke.csv`: first enrollment recipe (full fine-tuning, lr 3e-4). It degraded results and led to nested selection.

## Unit tests
`python -m pytest -q tests`: 6 tests (file-name parsing, ASCII reader, range/Doppler localisation on a synthetic point target, robustness of the tracking gate to range offset, gain invariance of the intensity references, AdaBN batch-order invariance regression test).
