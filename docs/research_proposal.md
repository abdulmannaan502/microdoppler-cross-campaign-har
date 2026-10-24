# Research Proposal (Phase 4)

## Working title
*What breaks when a micro-Doppler activity classifier is moved to a new site? Leave-one-campaign-out evaluation of radar front-end normalisation and test-time adaptation*

## Problem statement
Radar human-activity recognition (HAR) papers routinely report 90–99 % accuracy, usually with random
or pooled splits in which the same person, room and session appear in both training and test data.
In the intended use (assisted living, care homes), a classifier trained on earlier campaigns is
deployed in a **new room, with new people and a different set-up**. Domain-adaptation papers attack
this with extra networks (adversarial alignment, distillation, source-free fine-tuning), mostly for
mmWave sensors. The **radar front-end** (which range cells are integrated, and what the spectrogram
intensity is referenced to) also decides what the classifier sees. Under a realistic site shift its
role has not been isolated.

## Research questions
- **RQ1 (protocol):** How much do random-file and subject-disjoint splits overestimate cross-campaign performance on real 5.8 GHz FMCW data?
- **RQ2 (front-end):** Does a target-tracking range gate combined with a per-Doppler noise-floor reference (an ordered-statistic, CFAR-like normalisation) improve leave-one-campaign-out macro-F1 over the standard fixed-gate, max-referenced spectrogram, for several classifier families?
- **RQ3 (complementarity):** Are front-end normalisation and label-free test-time BatchNorm adaptation (AdaBN) complementary? Is the front-end gain explained by generic intensity augmentation?
- **RQ4 (reliability):** How does calibration (ECE) degrade under campaign shift, and does either intervention help?

## Hypotheses (falsifiable)
- **H1:** LOCO macro-F1 is lower than random-split macro-F1 by more than the seed-to-seed spread (paired over seeds).
- **H2:** Proposed front-end > fixed_max baseline in LOCO macro-F1 (paired Wilcoxon over campaign×seed, p < 0.05), for at least two of three CNN architectures.
- **H3:** Front-end + AdaBN ≥ either alone. Generic augmentation on the baseline front-end does not reach the proposed front-end.
- **H4:** ECE under LOCO is higher than under random splits, and is reduced by the proposed front-end and/or AdaBN.

Each hypothesis can fail. A failure will be reported.

## Method
Signal chain per recording (Ancortek 5.8 GHz, B = 400 MHz, T_c = 1 ms, 128 samples/chirp):
1. Range FFT per chirp, positive-beat half, DC bin dropped. Keep 48 range cells (0.375 m each, about 18 m).
2. MTI: 4th-order Butterworth high-pass along slow time (3.75 Hz), as in the dataset reference code.
3. STFT per range cell: Hann window of 200 ms, 20 ms hop, 256-point FFT. Keep 128 Doppler bins (±250 Hz, ±6.46 m/s).
4. **Range gate**: (a) *fixed* cells 10–30 (reference code); (b) *all* cells; (c) **track**: for each frame, r*(t) = argmax_r Σ_{|f|>f_0} P_r(f,t), median-filtered over 9 frames, and S(f,t) = Σ_{|r−r*(t)|≤W} P_r(f,t) with W = 4 cells (±1.5 m).
5. Time axis resampled to 128 frames.
6. **Power reference**: (a) *max*: 10log10 S − max, clipped to [−40, 0] dB; (b) *noise*: relative to a scalar 20th-percentile floor; (c) **noise_f**: relative to N(f) = 20th percentile over time of each Doppler bin, clipped to [0, 40] dB. This is an ordered-statistic CFAR-style normalisation along slow time. It removes per-recording gain, range-dependent path loss and the Doppler-dependent residual of the MTI/clutter spectrum.

Proposed front-end = track + noise_f.

## Baselines
- Front-ends: fixed_max (reference practice), fixed_z (per-image z-score), all_max, plus ablations track_max, track_noise and fixed_nf.
- Classifiers: SVM and RF on centroid/bandwidth moments (the feature family of Shah et al. 2022); LightCNN (≈0.1 M parameters); ResNet-18; MobileNetV2.
- Adaptation: AdaBN (label-free; uses only unlabeled target-campaign data). Generic augmentation (shift, gain, noise, time-mask).

## Evaluation
- Protocols: random 5-fold (file-level), subject-disjoint 5-fold, LOCO over 7 campaigns.
- Metrics: macro-F1 (primary), accuracy, ECE (15 bins), NLL, confusion matrices, per-campaign F1, latency and parameter count.
- Five seeds. Paired Wilcoxon signed-rank over (campaign, seed) pairs, Holm correction across comparisons, bootstrap 95 % CIs.

## Closest prior work and difference
See `literature/literature_review.md` (novelty matrix).

## Risks
- Campaign-level differences may be dominated by *population* (age) rather than environment. The two cannot be fully separated in this dataset, so claims will say "campaign shift (site + cohort)".
- The 20th-percentile floor may remove a persistent signal (for example the torso line during long walks). This is tested with the noise / noise_f / percentile sensitivity ablation.
- Only one radar type and one frequency band, so the conclusions are scoped accordingly.

## Outcome of the hypotheses (2026-10-24, after all experiments)
| Hypothesis | Outcome |
|---|---|
| H1 protocol inflation | **Supported**: random 0.948 → subject 0.929 → LOCO 0.912 (LightCNN); same ordering for SVM/RF |
| H2 tracking + noise-referenced front-end beats reference processing | **Refuted**: tracking −0.007, noise referencing −0.017 (worse in 6/7 campaigns), Hann window −0.010 |
| H2' (post-pilot) dual-reference input | **Refuted**: +0.002 over 10 seeds; −0.0045 on fresh seeds; worse than a 2-seed ensemble |
| H3 front-end and AdaBN complementary; generic aug. does not match | **Refuted / reversed**: front-end changes add nothing; generic augmentation + AdaBN is the best LightCNN setting (+0.017) |
| H4 ECE higher under LOCO; reduced by remedies | **Partly supported**: ECE doubles at care sites (0.045 → 0.089); AdaBN/TENT barely change it |
| Added: few-person enrollment closes the care-site gap | **Refuted**: no gain beyond unlabeled AdaBN |
