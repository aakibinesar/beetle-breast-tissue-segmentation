# Results log

Append new rows at the bottom; do not edit old rows. Scores are the public-leaderboard `overall_dice` on the hidden test set. Weights of every blend were fixed before the score was seen. "Org" means the organisers' released nnU-Net 5-fold ensemble (not trained by me). Leaderboard ranks are deliberately not recorded.

**12 submissions have been made in total (limit 12 per 28 days). This version lists submissions 1 to 6; the later ones will be added in a later update.**

## Submissions (in order)

| # | date | entry | mine / Org (%) | overall_dice |
|---|---|---|---|---|
| 1 | 2026-10-11 | Org alone (reproduction, no training) | 0 / 100 | 0.8666 |
| 2 | 2026-10-11 | Org with flip averaging | 0 / 100 | 0.8667 |
| 3 | 2026-10-19 | v2: Phikon ViT-B, 154 slides | 100 / 0 | 0.8388 |
| 4 | 2026-10-19 | v2 + Org | 50 / 50 | 0.8684 |
| 5 | 2026-10-19 | v2 + Org (Org carries more than half; not a headline) | 30 / 70 | 0.8789 |
| 6 | 2026-10-24 | v3: v2 continued with stain jitter, 218 slides | 100 / 0 | 0.8240 |

## Per-class and per-centre scores

| entry | overall | other | non-inv | invasive | necrosis | Biopticka | SCDC | UWMed | ILC inv. | NST inv. |
|---|---|---|---|---|---|---|---|---|---|---|
| Org alone (#1) | 0.8666 | 0.9370 | 0.6523 | 0.7755 | 0.5110 | 0.8791 | 0.8487 | 0.8708 | 0.7791 | 0.7742 |
| Org with flips (#2) | 0.8667 | 0.9373 | 0.6511 | 0.7744 | 0.5231 | 0.8798 | 0.8485 | 0.8708 | 0.7783 | 0.7731 |
| v2 (#3) | 0.8388 | 0.9193 | 0.6432 | 0.7440 | 0.5505 | 0.7989 | 0.8261 | 0.9001 | 0.7964 | 0.7240 |
| v2 + Org 50/50 (#4) | 0.8684 | 0.9369 | 0.6925 | 0.7824 | 0.5836 | 0.8507 | 0.8496 | 0.9097 | 0.8304 | 0.7647 |
| v2 + Org 30/70 (#5) | 0.8789 | 0.9402 | 0.7066 | 0.7991 | 0.5784 | 0.8916 | 0.8540 | 0.8906 | 0.8001 | 0.7988 |
| v3 (#6) | 0.8240 | 0.9202 | 0.5945 | 0.7052 | 0.5592 | 0.7756 | 0.8216 | 0.8845 | 0.7404 | 0.6922 |

## Validation (patient-disjoint, 1,715 patches; in-distribution, did not predict the test)

| model | mean-class Dice | overall | necrosis |
|---|---|---|---|
| v2 | 0.851 | 0.921 | 0.630 |
| v3 (selected epoch) | 0.862 | 0.926 | 0.652 |

## Compute

Kaggle T4 wall-clock: v2 second stage about 3.7 h, v3 about 4.5 h. Inference for 170 regions with four-fold averaging: about 25 min.
