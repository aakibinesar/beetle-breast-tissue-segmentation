# Results log

Append new rows at the bottom; do not edit old rows. Scores are the public-leaderboard `overall_dice` on the hidden test set. Weights of every blend were fixed before the score was seen. "Org" means the organisers' released nnU-Net 5-fold ensemble (not trained by me). Leaderboard ranks are deliberately not recorded.

## All submissions (12 of 12 allowed per 28 days, in order)

| # | date | entry | mine / Org (%) | overall_dice |
|---|---|---|---|---|
| 1 | 2026-10-11 | Org alone (reproduction, no training) | 0 / 100 | 0.8666 |
| 2 | 2026-10-11 | Org with flip averaging | 0 / 100 | 0.8667 |
| 3 | 2026-10-19 | v2: Phikon ViT-B, 154 slides | 100 / 0 | 0.8388 |
| 4 | 2026-10-19 | v2 + Org | 50 / 50 | 0.8684 |
| 5 | 2026-10-19 | v2 + Org (Org carries more than half; not a headline) | 30 / 70 | 0.8789 |
| 6 | 2026-10-24 | v3: v2 continued with stain jitter, 218 slides | 100 / 0 | 0.8240 |
| 7 | 2026-11-01 | pv2: Phikon-v2, 218 slides, stain jitter | 100 / 0 | 0.8269 |
| 8 | 2026-11-01 | average of v2 and pv2 | 100 / 0 | 0.8459 |
| 9 | 2026-11-01 | average of v2, v3, pv2, pv2abl (1/4 each) | 100 / 0 | 0.8429 |
| 10 | 2026-11-01 | v2 (25%) + pv2 (25%) + Org (50%) | 50 / 50 | 0.8753 |
| 11 | 2026-11-03 | pv2big: Phikon-v2, 154 + 113 slides, no jitter | 100 / 0 | 0.8464 |
| 12 | 2026-11-03 | pv2big + Org | 50 / 50 | **0.8839** |

Rule used for #12, fixed after #11 and before #12: if pv2big alone scored above 0.8459 (the best own result), blend it 50/50 with Org; between 0.830 and 0.8459, use one sixth each of pv2big, v2 and pv2 plus 50% Org; below 0.830, no further blend. A file for the second case was built but not submitted.

Note on #12: pv2big is the Phikon-v2 model trained on the original 154 slides plus 113 further development slides, without stain jitter. Alone (#11) it scored 0.8464, higher than each earlier single model (v2 0.8388, v3 0.8240, Phikon-v2 with jitter 0.8269) and 0.0005 above the best earlier own-only result (#8, 0.8459), which is within noise. It was blended because of the rule above, which was fixed before #12 was submitted. #12 is 0.0173 above Org alone (#1) and 0.0086 above #10.


## Per-class and per-centre scores (where recorded)

| entry | overall | other | non-inv | invasive | necrosis | Biopticka | SCDC | UWMed | ILC inv. | NST inv. |
|---|---|---|---|---|---|---|---|---|---|---|
| Org alone (#1) | 0.8666 | 0.9370 | 0.6523 | 0.7755 | 0.5110 | 0.8791 | 0.8487 | 0.8708 | 0.7791 | 0.7742 |
| Org with flips (#2) | 0.8667 | 0.9373 | 0.6511 | 0.7744 | 0.5231 | 0.8798 | 0.8485 | 0.8708 | 0.7783 | 0.7731 |
| v2 (#3) | 0.8388 | 0.9193 | 0.6432 | 0.7440 | 0.5505 | 0.7989 | 0.8261 | 0.9001 | 0.7964 | 0.7240 |
| v2 + Org 50/50 (#4) | 0.8684 | 0.9369 | 0.6925 | 0.7824 | 0.5836 | 0.8507 | 0.8496 | 0.9097 | 0.8304 | 0.7647 |
| v2 + Org 30/70 (#5) | 0.8789 | 0.9402 | 0.7066 | 0.7991 | 0.5784 | 0.8916 | 0.8540 | 0.8906 | 0.8001 | 0.7988 |
| v3 (#6) | 0.8240 | 0.9202 | 0.5945 | 0.7052 | 0.5592 | 0.7756 | 0.8216 | 0.8845 | 0.7404 | 0.6922 |

Per-class and per-centre numbers of #7 to #12 will be added later.

## Validation (patient-disjoint, 1,715 patches; in-distribution, did not predict the test)

| model | mean-class Dice | overall | necrosis |
|---|---|---|---|
| v2 | 0.851 | 0.921 | 0.630 |
| v3 (selected epoch) | 0.862 | 0.926 | 0.652 |
| pv2 (selected epoch) | 0.890 | 0.943 | 0.706 |
| pv2abl | 0.883 | 0.939 | n/a |
| pv2big (epoch 2 of 3) | 0.912 | 0.956 | 0.774 |

## Compute

Kaggle T4 wall-clock per training run: v3 about 4.5 h, pv2 about 5.9 h, pv2big 5.5 h (v2 about 3.7 h for its second stage). Inference for 170 regions with four-fold averaging: about 25 min for ViT-B, 44 min for Phikon-v2.
