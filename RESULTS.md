# Results log

Scores are the public-leaderboard `overall_dice` on the hidden test set; dates are the "Created" dates shown on the leaderboard. Weights of every blend were fixed before the score was seen. "Organisers' nnU-Net" is the organisers' released 5-fold ensemble, which I ran but did not train. The models are described in section 2 of the [README](README.md).

## Submissions

| # | date | entry | own / organisers (%) | overall_dice |
|---|---|---|---|---|
| 1 | 2026-10-11 | Organisers' nnU-Net alone (reproduction, no training) | 0 / 100 | 0.8666 |
| 2 | 2026-10-11 | Organisers' nnU-Net with flip averaging | 0 / 100 | 0.8667 |
| 3 | 2026-10-19 | Phikon ViT-B, 154 slides | 100 / 0 | 0.8388 |
| 4 | 2026-10-19 | Phikon ViT-B + organisers' nnU-Net | 50 / 50 | 0.8684 |
| 5 | 2026-10-19 | Phikon ViT-B + organisers' nnU-Net | 30 / 70 | 0.8789 |
| 6 | 2026-10-24 | Phikon ViT-B + jitter (Phikon ViT-B continued with stain jitter, 218 slides) | 100 / 0 | 0.8240 |
| 7 | 2026-10-31 | Phikon-v2 + jitter, 218 slides | 100 / 0 | 0.8269 |
| 8 | 2026-10-31 | average of Phikon ViT-B and Phikon-v2 + jitter | 100 / 0 | 0.8459 |
| 9 | 2026-11-01 | average of four own models: Phikon ViT-B, Phikon ViT-B + jitter, Phikon-v2 + jitter, Phikon-v2 154 slides (1/4 each) | 100 / 0 | 0.8429 |
| 10 | 2026-11-01 | Phikon ViT-B (25%) + Phikon-v2 + jitter (25%) + organisers' nnU-Net (50%) | 50 / 50 | 0.8753 |
| 11 | 2026-11-02 | Phikon-v2, extra slides (154 + 113 slides, no jitter) | 100 / 0 | 0.8464 |
| 12 | 2026-11-02 | Phikon-v2, extra slides + organisers' nnU-Net | 50 / 50 | **0.8839** |

Before #12 was submitted, I decided that the model of #11 would be blended 50/50 with the organisers' nnU-Net if it scored above 0.8459 alone, the best own-only result at that point (#8). It scored 0.8464.

## Per-class and per-centre scores

| # | entry | overall | other | non-invasive | invasive | necrosis | Biopticka | SCDC | UWMed | ILC invasive | NST invasive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Organisers' nnU-Net alone | 0.8666 | 0.9370 | 0.6523 | 0.7755 | 0.5110 | 0.8791 | 0.8487 | 0.8708 | 0.7791 | 0.7742 |
| 2 | Organisers' nnU-Net with flips | 0.8667 | 0.9373 | 0.6511 | 0.7744 | 0.5231 | 0.8798 | 0.8485 | 0.8708 | 0.7783 | 0.7731 |
| 3 | Phikon ViT-B | 0.8388 | 0.9193 | 0.6432 | 0.7440 | 0.5505 | 0.7989 | 0.8261 | 0.9001 | 0.7964 | 0.7240 |
| 4 | Phikon ViT-B + organisers 50/50 | 0.8684 | 0.9369 | 0.6925 | 0.7824 | 0.5836 | 0.8507 | 0.8496 | 0.9097 | 0.8304 | 0.7647 |
| 5 | Phikon ViT-B + organisers 30/70 | 0.8789 | 0.9402 | 0.7066 | 0.7991 | 0.5784 | 0.8916 | 0.8540 | 0.8906 | 0.8001 | 0.7988 |
| 6 | Phikon ViT-B + jitter | 0.8240 | 0.9202 | 0.5945 | 0.7052 | 0.5592 | 0.7756 | 0.8216 | 0.8845 | 0.7404 | 0.6922 |
| 7 | Phikon-v2 + jitter | 0.8269 | 0.9103 | 0.6130 | 0.7538 | 0.4937 | 0.8248 | 0.8161 | 0.8412 | 0.7839 | 0.7421 |
| 8 | average of Phikon ViT-B and Phikon-v2 + jitter | 0.8459 | 0.9211 | 0.6540 | 0.7696 | 0.5342 | 0.8209 | 0.8297 | 0.8931 | 0.8234 | 0.7488 |
| 9 | average of four own models | 0.8429 | 0.9204 | 0.6469 | 0.7632 | 0.5397 | 0.8179 | 0.8214 | 0.8959 | 0.8190 | 0.7418 |
| 10 | Phikon ViT-B + Phikon-v2 + jitter + organisers (25/25/50) | 0.8753 | 0.9398 | 0.7036 | 0.8001 | 0.5675 | 0.8834 | 0.8527 | 0.8899 | 0.8150 | 0.7948 |
| 11 | Phikon-v2, extra slides | 0.8464 | 0.9160 | 0.6420 | 0.7805 | 0.6123 | 0.8335 | 0.8344 | 0.8747 | 0.8017 | 0.7720 |
| 12 | Phikon-v2, extra slides + organisers 50/50 | **0.8839** | 0.9393 | 0.7157 | 0.8251 | 0.6334 | 0.8921 | 0.8577 | 0.9023 | 0.8438 | 0.8182 |

Biopticka, SCDC and UWMed are the three centres of the test set. ILC is invasive lobular carcinoma and NST is invasive carcinoma of no special type.

## Validation (patient-disjoint, 1,715 patches; in-distribution, did not predict the test)

| model | mean-class Dice | overall | necrosis |
|---|---|---|---|
| Phikon ViT-B | 0.851 | 0.921 | 0.630 |
| Phikon ViT-B + jitter (selected epoch) | 0.862 | 0.926 | 0.652 |
| Phikon-v2 + jitter (selected epoch) | 0.890 | 0.943 | 0.706 |
| Phikon-v2, 154 slides | 0.883 | 0.939 | n/a |
| Phikon-v2, extra slides (epoch 2 of 3) | 0.912 | 0.956 | 0.774 |

## Compute

Kaggle T4 wall-clock per training run: Phikon ViT-B + jitter about 4.5 h, Phikon-v2 + jitter about 5.9 h, Phikon-v2 with extra slides 5.5 h (Phikon ViT-B about 3.7 h for its second stage). Inference for 170 regions with four-fold averaging: about 25 min for Phikon ViT-B, 44 min for Phikon-v2.
