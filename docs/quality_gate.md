# Research Quality Gate (Phase 8) — 2026-10-24 22:15 IST

An internal diagnostic, **not** an acceptance prediction.

| Question | Answer |
|---|---|
| Is the problem genuinely relevant to radar? | Yes. Real FMCW micro-Doppler data, a deployment-relevant site shift and radar front-end design choices. radar sensing for assisted living |
| Is the contribution distinguishable from prior work? | Partly. The closest work (Shah et al. 2022) used pooled random splits; UDA works target mmWave and add networks. The contribution is evaluative (protocol + controlled comparison + negative results), not a new method |
| Is the dataset suitable? | Yes for the question. Limitations: single radar; site and cohort confounded; 2 care-site campaigns |
| Is the evaluation free of obvious leakage? | Yes. One sample per recording, duration-neutral window, person identity across campaigns, source-only model selection, nested enrollment selection. A pilot that saw test scores is disclosed |
| Are the baselines fair? | Yes. Reference processing reimplemented with metre-defined gates (no strawman); SVM/RF of the closest prior work; three CNN families plus ImageNet pre-training; AdaBN and TENT |
| Are the results reproducible? | Yes to within ≈1 point across platforms (measured). All numbers generated from result files; 6 unit tests; seeds recorded |
| Is there a meaningful improvement or finding? | A finding: a persistent ≈7-point care-site gap that low-cost remedies do not narrow; source validation overstates by ≈9 points; noise referencing is harmful. The improvements are small (≤ 1.7 points) |
| Unresolved bugs? | One evaluation bug (batch order in AdaBN/TENT) was found and fixed, and all affected runs were repeated. None known now |
| Are the conclusions supported? | Yes, with the hedges stated in the paper (cohort-level shift is an inference) |
| Strong enough for serious review? | Methodologically yes. The risk is reviewers who expect a winning method |

## Score (diagnostic)
| Criterion | Max | Score | Justification |
|---|---|---|---|
| Novelty | 20 | 11 | Evaluation-centred; first strict LOCO study of these remedies on this multi-site radar data, but no new algorithm |
| Technical correctness | 20 | 17 | Physically grounded front-end, bug found and fixed, regression tests; single-channel radar limits generality |
| Experimental rigor | 20 | 17 | 5–10 seeds, paired tests + Holm + cluster bootstrap, nested selection, fresh-seed confirmation, cross-platform replication; DANN not run |
| Venue relevance | 15 | 12 | Radar HAR and assisted living fit the target venue; more of a sensing/ML paper than a radar-systems paper |
| Reproducibility | 10 | 9 | Code, configs, manifests, per-fold results, environment; raw data public (download script) |
| Research writing readiness | 10 | 8 | Complete 5-page draft; author details and the human author's review still pending |
| Practical significance | 5 | 4 | Clear guidance for deployment and evaluation practice |
| **Total** | **100** | **78** | |

## Critical weaknesses and handling
1. *No positive method*: framed honestly as an evaluation study. Cannot be "fixed" without new research.
2. *Site/cohort confound*: inherent to the data; stated.
3. *Small effects vs. platform noise*: quantified and used as an interpretation bound.

**Gate decision:** proceed to final packaging. Status: **READY FOR HUMAN REVIEW** (see `docs/review_report.md`). Not "ready for submission" until the authors have filled their details, reviewed and approved the manuscript, and confirmed the policies.
