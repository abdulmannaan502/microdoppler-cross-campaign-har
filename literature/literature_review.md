# Literature Review — Cross-site generalisation of radar micro-Doppler HAR

Search log: `literature_search_log.md` (Semantic Scholar + OpenAlex, 2026-10-24). DOI verification:
`doi_verification.csv` (all 30 DOIs resolved through doi.org on 2026-10-24). Full text was read for
Shah et al. 2022 (open access, `fulltext/`). Other entries were assessed from abstracts and publisher
metadata, and are marked as such. Preprints are flagged.

## 1. Foundations
- **Micro-Doppler modelling**: Chen et al., IEEE TAES 2006 [chen2006microdoppler] (peer-reviewed). Physical basis: body-part velocities modulate the Doppler shift f_D = 2v/λ.
- **Classical features + SVM**: Kim & Ling, IEEE TGRS 2009 [kim2009svm]. Hand-crafted Doppler features (torso frequency, bandwidth, period).
- **Deep CNNs on spectrograms**: Kim & Moon, IEEE GRSL 2016 [kim2016dcnn]. Seyfioğlu et al., IEEE TAES 2018 [seyfioglu2018cae] (convolutional autoencoder for similar aided/unaided activities).
- **Surveys / overviews**: Gurbuz & Amin, IEEE SPM 2019 [gurbuz2019spm]. Li et al., Remote Sensing 2019 [li2019survey]. Fioranelli et al., Electron. Lett. 2019 [fioranelli2019health]. Shah & Fioranelli, IEEE AESM 2019 [shah2019review].
- **CFAR / ordered statistics**: Rohling, IEEE TAES 1983 [rohling1983cfar]. Motivates the percentile noise-floor reference.

## 2. The Glasgow dataset and its benchmarks
- **Dataset**: Fioranelli et al., Enlighten 848 [gla848], CC BY 4.0.
- **Human Activity Radar Challenge**: Yang et al., IEEE JBHI 2023 [yang2023challenge]. 12 teams and 188 entries benchmarked on this data. The abstract describes the approaches but not a cross-site protocol (full text not accessed).
- **Data portability** (closest prior work, full text read): Shah et al., Neural Comput. Appl. 2022 [shah2022portability]. Same radar, nine locations including five care homes, 99 subjects, 1,453 signatures. SVM/KNN on centroid and bandwidth moments, AlexNet transfer, CNN and autoencoder, best accuracy ≈ 88 %. **Protocol:** pooled data, 70/30 hold-out repeated 30 times, separate analyses for young and mature cohorts. No leave-one-location-out test, so the reported numbers mix seen persons and rooms into the test set.
- **Continuous HAR**: Shrestha et al., IEEE Sensors J. 2020 [shrestha2020bilstm] (Bi-LSTM on the continuous recordings).

## 3. Domain shift and adaptation in radar HAR
- **Li et al., IEEE RadarConf 2020** [li2020uda]: adversarial UDA from simulated to measured spectrograms (81.6 %). This is a sim-to-real shift, not site-to-site.
- **Li et al., IEEE BIBM 2025** [li2025crossenv]: mmWave cross-environment UDA with feature-fusion and distillation, 12 transfer tasks, 94.2 %. Needs unlabeled target data at training time and a dedicated network. Abstract only.
- **Liu et al., ICASSP 2024** [liu2024sfda] and **Si et al., Pattern Recognit. 2025/26** [si2025sfda]: source-free DA with balanced clustering or self-supervision on mmWave HAR. Abstract only.
- **Tanaka et al., arXiv 2025** [tanaka2025crossenv] (*preprint*): people counting with FMCW MIMO across two rooms. Amplitude preprocessing (sigmoid weighting) reduced RMSE by 50 %; transfer learning was needed for large shifts. This is the closest in spirit (preprocessing vs learning) but for a different task.
- **AdaBN**: Li et al., Pattern Recognit. 2018 [li2018adabn]. **TENT**: Wang et al., ICLR 2021 [wang2021tent] (*arXiv version cited*). **DANN**: Ganin et al., JMLR 2016 [ganin2016dann] (*arXiv version cited*).

## 4. Pre-processing, robustness and normalisation
- **Ayaz et al., Sensors 2025** [ayaz2025sp] and **Pervasive Mob. Comput. 2026** [ayaz2026bench]: compare TR, RD, STFT and SPWVD maps with CNN and transfer models. Random/within-dataset splits; no site shift.
- **Zhou et al., IEEE TRS 2024** [zhou2024corruption]: corruption-robustness taxonomy (temporal, Doppler, intensity) using *synthetic* corruptions. Augmentation and adversarial training help. Real site shift is not studied.
- **Nguyen et al., PLoS ONE 2024** [nguyen2024noise]: denoising against *added* white Gaussian noise.
- **Sadeghi-Adl & Ahmad, Sensors 2023** [adl2023whitening]: replaces BN by whitening layers, within-dataset evaluation.
- **BatchNorm**: Ioffe & Szegedy 2015 [ioffe2015bn] (*arXiv*). **Calibration**: Guo et al., ICML 2017 [guo2017calibration] (*arXiv*). **ResNet** [he2016resnet], **MobileNetV2** [sandler2018mobilenetv2].

## 5. Gap
1. On real multi-site radar HAR data, reported results come from pooled random splits [shah2022portability] or within-dataset splits [ayaz2025sp, ayaz2026bench]. No study we found quantifies **random → subject → leave-one-campaign-out** inflation with seeds and paired statistics.
2. Robustness studies use **synthetic** corruptions [zhou2024corruption, nguyen2024noise]. Adaptation studies add **target-trained networks** [li2025crossenv, liu2024sfda, si2025sfda]. The cheaper question is untested on real cross-site data: can **front-end intensity referencing / range gating** or **label-free BN re-estimation** close the gap?
3. Two practical confounds of this dataset have not been reported as far as we found: the **duration shortcut** (walk 10 s vs 5 s) and the **100 MHz subset**.

## 6. Novelty matrix
| Existing work | Main method | Known limitation | Our difference | Evidence required |
|---|---|---|---|---|
| Shah et al. 2022 | SVM/KNN/CNN/AE, pooled 70/30 | Seen persons/rooms in test | Strict LOCO, 7 campaigns, 5 seeds | Random vs subject vs LOCO table |
| Yang et al. 2023 | Challenge benchmark | Single split | Cross-campaign protocol on the same data | LOCO per-campaign F1 |
| Zhou et al. 2024 | Synthetic corruption robustness | Synthetic shift only | Real site/cohort shift | LOCO with care-home campaigns |
| Li 2025, Liu 2024, Si 2025 | UDA / SFDA networks | Extra training on target data | Label-free, training-free (front-end, AdaBN) | Paired comparison vs baseline |
| Tanaka 2025 (preprint) | Amplitude preprocessing for counting | Different task, 2 rooms | Micro-Doppler HAR, max- vs noise-referenced, dual-reference input | Ablation table |
| Ayaz 2025/2026 | Representation comparison | No site shift | Representation × protocol interaction | LOCO ablations |

The novelty is deliberately narrow: an evaluation-driven study plus a cheap representation (the dual reference). It is not a new network.
