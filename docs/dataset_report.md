# Dataset Report (Phase 5)

## Source
- **Name:** Radar signatures of human activities (University of Glasgow, INSHEP project EP/R041679/1)
- **Record:** https://researchdata.gla.ac.uk/848/ — DOI 10.5525/gla.researchdata.848
- **Licence:** CC BY 4.0 (stated on the record page for all files). Redistribution with attribution is allowed. The project ZIP still ships a download script rather than the 2 GB archive.
- **Archive:** `Dataset_848.7z`, 2,060,197,481 bytes, SHA-256 `652f9adc2c9921f9fa08ba2de2d40e3f357d89a91f1b13ac790bc68f4f54ae61` (downloaded 2026-10-24); 16.0 GB uncompressed.
- **Benchmark use:** Human Activity Radar Challenge (Yang et al., IEEE JBHI 2023). Cross-location study by Shah et al. (Neural Comput. Appl. 2022), which used pooled random 70/30 splits.

## Sensor
Ancortek FMCW radar, f_c = 5.8 GHz, chirp T_c = 1 ms (PRF 1 kHz), 128 complex samples per chirp, Yagi antennas (≈17 dB),
P_out ≈ +18 dBm. Bandwidth is 400 MHz (range bin 0.375 m) for 1,691 files and **100 MHz (1.5 m) for 60 West Cumbria files**. This was read from each file header and is not mentioned in the readme. Wavelength λ = 5.17 cm, so 1 m/s ↔ 38.7 Hz Doppler.

## Content (after parsing; `experiments/manifests/gla848_manifest.csv`)
| Campaign | Site | Files | Subjects | Classes |
|---|---|---|---|---|
| 1 Dec 2017 | UoG lab | 360 | 20 | 6 |
| 2 Mar 2017 | UoG lab | 47 | 4 | 6 |
| 3 Jun 2017 | UoG lab | 161 | 9 | 6 |
| 4 Jul 2018 | UoG common room | 288 | 16 | 6 |
| 5 Feb 2019 UoG | UoG lab | 305 | 17 | 6 |
| 6 Feb 2019 NG Homes | care housing, 3 rooms, older adults (age 24–88) | 301 | 20 | 5 (+3 falls) |
| 7 Mar 2019 West Cumbria | Age UK centre, older adults | 289 | 20 | 5 |
| **Total** | | **1751** | **106 person-campaigns** | |

Activities: 1 walk, 2 sit down, 3 stand up, 4 pick up object, 5 drink, 6 fall.

## Integrity checks
- 1,754 `.dat` files in the archive. **Three were excluded** because the leading activity digit contradicts the `Ayy` code: `3P03A02R1`, `2P34A03R1` and `6P01A05R03`. The label is ambiguous, so they are not used.
- All 1,751 remaining files parsed. Headers are consistent apart from the bandwidth difference above.
- **Durations:** walking recordings are 10 s (20 s for 8 NG Homes files); all other activities are 5 s. **Duration is therefore a label shortcut.** We cut every recording to the 5 s window with maximum post-MTI moving energy. This is label-agnostic and identical for all front-ends.
- **Subject IDs are re-used across campaigns** (for example P03 appears in Mar 2017 and Feb 2019), so person identity = (campaign, subject).

## Range geometry (from `gla848_index.csv`, tracked target range, median per campaign)
| Campaign | Median target range | Moving energy inside the fixed 3.75–11.25 m gate |
|---|---|---|
| Lab campaigns (1–5) | 5.3–6.8 m | 75–93 % |
| NG Homes | 3.4 m | **14 %** |
| West Cumbria (400 / 100 MHz) | 4.9 / 3.0 m | 65 % / 31 % |

The care-site campaigns place the person much closer to the radar, which is outside the range interval of the reference processing script.

## Leakage analysis
| Risk | Status |
|---|---|
| Adjacent windows of one recording in train and test | Eliminated: one sample per recording |
| Same person in train and test | Present in the *random* protocol (by design, to measure inflation); eliminated in *subject* and *LOCO* protocols |
| Same session/room in train and test | Present in *random* and *subject*; eliminated in *LOCO* |
| Duration shortcut | Eliminated by the 5 s window |
| Hyper-parameter selection on test | Model selection uses a subject-disjoint validation split of the training campaigns only |
| Augmented copies crossing splits | Augmentation is on-the-fly inside training batches only |
| Person and site confound | **Cannot be separated**: each campaign has its own cohort. LOCO measures campaign shift = site + cohort + set-up |

## Primary task
5-class (walk, sit, stand, pick, drink), N = 1,656 recordings. Classes are near-balanced (≤ 3 % imbalance per campaign). Falls are excluded from the LOCO study because two campaigns have none or only 3.
