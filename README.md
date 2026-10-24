# Moving micro-Doppler activity recognition to new sites

Code, experiment records and paper source for **"Moving Micro-Doppler Activity Recognition to New Sites: A
Leave-One-Campaign-Out Study of Low-Cost Remedies"** (Abdul Mannaan Sayyad, VIT-AP University).
The study asks how a radar micro-Doppler activity classifier transfers to an **unseen campaign** (new site, cohort
and set-up) on the public University of Glasgow 5.8 GHz FMCW dataset, and whether low-cost remedies help: range
gating, intensity referencing, a dual-reference input, augmentation, ImageNet pre-training, label-free test-time
adaptation (AdaBN, TENT) and enrollment of a few labelled residents.

**Headline results (LightCNN, macro-F1):** random split 0.948, subject-disjoint 0.929, leave-one-campaign-out 0.912
(laboratories 0.932, care sites 0.860). No remedy narrows the ≈7-point care-site gap. Negative results are
reported as such. Preprint: *arXiv link to be added after posting.*

## Layout
```
src/data/gla848.py               dataset reader / file-name parser
src/radar_processing/frontend.py range FFT, MTI, STFT, range gates, intensity references
src/models/nets.py               LightCNN, ResNet-18 (scratch / ImageNet), MobileNetV2, AdaBN, TENT
src/models/features.py           hand-crafted micro-Doppler features (SVM / RF baselines)
src/training/train.py            training loop, augmentations
src/evaluation/metrics.py        macro-F1, ECE, NLL
scripts/                         download, caches, runners, queues, analysis, tables, figures
configs/frontends.yaml           all front-end variants
experiments/configs/queue_*.txt  every condition that was run (remote_w*.txt: GPU-host worker queues)
experiments/manifests/           per-file manifest and range diagnostics
experiments/logs/                raw training logs
results/runs/                    per-condition metrics (fold × seed × adaptation mode) and predictions
results/tables/, results/statistical_analysis.md, results/figures/
results/runs_superseded/         runs repeated after an evaluation bug fix (kept for transparency; do not use)
arxiv/                           paper source (main.tex, generated tables and figures, references.bib)
docs/, literature/               dataset report, proposal, reproducibility notes, literature review
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.12 (Windows: .venv/Scripts/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m pytest -q tests
bash scripts/get_data.sh                                  # 2 GB download, SHA-256 checked, 16 GB extracted
python scripts/build_cache.py data/raw/gla848/extracted --jobs 10
python scripts/build_images.py --jobs 10
python scripts/run_queue.py experiments/configs/queue_main.txt      # also queue_classical.txt, queue_arch.txt
python scripts/run_enroll.py                              # site-enrollment experiment
python scripts/analyze.py && python scripts/make_tables.py && python scripts/make_figures.py
cd arxiv && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
A single condition: `python scripts/run_experiment.py --protocol loco --frontend fixed_max --model lightcnn --seeds 0,1,2,3,4`.
Neural results were produced on one RTX 4090; re-running on other hardware changes fold-averaged scores by up to ≈1
point (`docs/reproducibility_report.md`). Every number in the paper is generated from `results/` by the scripts above.

## Data
Raw data are not included. Dataset: *Radar signatures of human activities*, University of Glasgow, CC BY 4.0,
DOI 10.5525/gla.researchdata.848 (see `data/README.md`).

## Notes on integrity
An evaluation bug (class-sorted batches in AdaBN/TENT) was found during self-review, fixed, covered by a regression
test and all affected runs were repeated (`DECISION_LOG.md`, `results/runs_superseded/`). AI assistance (Anthropic
Claude) was used for literature search, code development, running experiments and drafting; the author reviewed and
takes responsibility for the content.

## Licence and citation
Code: MIT (`LICENSE`). Paper: CC BY 4.0. Please cite via `CITATION.cff`.
