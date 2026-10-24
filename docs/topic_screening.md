# Topic Screening (Phase 2) — 2026-10-24

Rapid, deadline-aware screen (7 days to the 31 Oct 2026 deadline). Scores use the rubric in the
master prompt: Relevance /15, Novelty /20, Data /15, Feasibility /15, Measurable /15, Baselines /10,
Time /10. Scores are judgement calls informed by the searches logged in
`literature/literature_search_log.md`; they are a triage tool, not a measurement.

Hard constraints applied: data must be downloadable today without credentials we may not enter
(Kaggle, IEEE DataPort and SDMS/MSTAR all need logins), and the full study must fit an RTX 3050 (4 GB)
in roughly 3 days of compute.

| # | Topic | Data (verified?) | Rel | Nov | Data | Feas | Meas | Base | Time | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Cross-environment generalisation of micro-Doppler HAR: radar-aware front-end normalisation vs. learned adaptation (leave-one-campaign-out)** | Glasgow "Radar signatures of human activities", CC BY 4.0, direct download (verified) | 12 | 13 | 14 | 13 | 14 | 9 | 8 | **83** | 7 campaigns / 5 sites give a natural real domain shift; UDA works exist but mostly on mmWave with few environments; the front-end side has not been isolated under a strict venue-out protocol |
| 2 | Evaluation-protocol leakage study (window/subject/campaign splits) on Glasgow HAR | same | 11 | 10 | 14 | 15 | 13 | 9 | 10 | 82 | Strong but on its own reads as a "negative result" paper; folded into #1 as a secondary contribution |
| 3 | Calibration and selective prediction for radar HAR under shift | same | 10 | 12 | 14 | 14 | 12 | 8 | 9 | 79 | Folded into #1 as an analysis (ECE under shift) |
| 4 | Learned CFAR detector for K-distributed / non-Gaussian clutter | simulation only | 14 | 8 | 8 | 14 | 13 | 9 | 9 | 75 | Explicitly in the CfP, but heavily studied; simulation-only weakens real-world claims |
| 5 | Drone vs. bird micro-Doppler classification | DIAT-uSAT, Real Doppler RAD-DAR (login-gated: Kaggle/DataPort) | 14 | 10 | 6 | 10 | 12 | 8 | 7 | 67 | Data access needs credentials we may not enter |
| 6 | LPI radar waveform recognition (Choi-Williams + CNN) | simulation | 12 | 5 | 9 | 14 | 12 | 9 | 9 | 70 | Saturated literature |
| 7 | Automotive radar point-cloud classification (RadarScenes) | Zenodo, CC BY-NC-SA, ~ tens of GB | 13 | 9 | 12 | 7 | 12 | 8 | 5 | 66 | Too large for the time budget |
| 8 | Range-Doppler object detection (CARRADA/RADDet) | public, large | 13 | 8 | 12 | 7 | 12 | 8 | 5 | 65 | Compute and time heavy |
| 9 | SAR ATR robustness (MSTAR) | SDMS registration required | 12 | 6 | 7 | 12 | 12 | 9 | 7 | 65 | Access and saturation |
| 10 | Interference mitigation for automotive FMCW (synthetic) | simulation | 14 | 9 | 8 | 12 | 12 | 7 | 8 | 70 | Plausible but simulation-only |
| 11 | Knowledge distillation for edge radar HAR | Glasgow | 10 | 7 | 14 | 14 | 12 | 8 | 9 | 74 | Incremental |
| 12 | Few-shot radar HAR | Glasgow | 10 | 7 | 14 | 13 | 12 | 7 | 8 | 71 | Crowded |
| 13 | Self-supervised micro-Doppler pretraining | Glasgow | 10 | 9 | 14 | 9 | 11 | 7 | 6 | 66 | Too slow to tune properly in a week |
| 14 | Explainable radar HAR (saliency) | Glasgow | 9 | 7 | 14 | 14 | 8 | 6 | 9 | 67 | Hard to evaluate quantitatively |
| 15 | Radar anomaly detection for falls (one-class) | Glasgow | 11 | 9 | 12 | 12 | 11 | 7 | 8 | 70 | Fall class is missing from some campaigns |

## Top five
1. #1 Cross-environment generalisation with radar-aware front-end normalisation (83)
2. #2 Split-protocol leakage study (82)
3. #3 Calibration under shift (79)
4. #4 Learned CFAR in non-Gaussian clutter (75)
5. #11 Distillation for edge HAR (74)

## Decision
Topic #1 is selected, with #2 and #3 folded in as supporting analyses. They share the same data and
pipeline, and together they answer one question: *what actually breaks when a micro-Doppler activity
classifier is moved to a new room or site, and how much of that can a radar front-end fix without
target labels?*
