# BEETLE breast-tissue segmentation: pathology foundation models and a blend with the organisers' nnU-Net

Code and write-up for my entries to [BEETLE](https://beetle.grand-challenge.org/) (Grand Challenge): 4-class tissue segmentation (other, non-invasive epithelium, invasive epithelium, necrosis) on 170 hidden H&E regions from three centres. The leaderboard metric is `overall_dice`.

**Status: ongoing.** This version lists the results up to 2026-10-24 (submissions 1 to 6). I have made 12 submissions in total; the later ones are added to [RESULTS.md](RESULTS.md) in a later update, and the README is revised then.

**Results listed so far (public leaderboard, hidden test set)**

| entry | overall_dice |
|---|---|
| organisers' released nnU-Net 5-fold ensemble, run by me, no training | 0.8666 |
| my model v2 alone (Phikon ViT-B) | 0.8388 |
| my model v3 alone (v2 continued with stain jitter) | 0.8240 |
| 50% v2 + 50% the organisers' nnU-Net | 0.8684 |

A blend with the organisers' model at 70% scored 0.8789; I do not use it as a headline because my model carries less than half. **None of my own models alone beats the organisers' reproduced result.** The blend weights were fixed before any score was seen and were not tuned on the leaderboard.

## 1. Task and constraints
- Train on the public BEETLE development set (Zenodo, CC BY-NC-SA 4.0); the 170 evaluation regions (about 2048 to 2560 px) have hidden labels, so every test number comes from uploading a zip of label PNGs.
- Limit: 12 submissions per 28 days.
- Compute: Kaggle free tier only (about 30 GPU hours per week; T4 GPUs; 12 h per session). Slides are read in Kaggle notebooks by HTTP range request, so the 120 GB of slides never touch my disk.
- No external data. Slides named after TCGA cases were left out of training. Splits are by patient.

## 2. Models (`src/`, `kernels/`)
| name | backbone | what was trained | training data | test score (alone) |
|---|---|---|---|---|
| v2 | Phikon ViT-B/16 (Owkin) | last 4 blocks + light CNN-skip decoder; stage 1 on 72 slides, stage 2 on 154 | 154 slides, 26,343 patches | 0.8388 |
| v3 | same, continued | stain (H&E) jitter in optical-density space | 218 slides | 0.8240 |

The repository also contains the Phikon-v2 (DINOv2 ViT-L/16) module and trainer; their results are added in a later update.

- **Patches:** 512 px at 0.5 µm/px, class-balanced centres per slide for training; a patient-disjoint validation set of 23 slides (1,715 patches) for model selection.
- **Training:** AdamW, cosine schedule, fp16; a class-rarity sampler; backbone learning rate 3e-5. The checkpoint was chosen by mean per-class Dice on the validation set (for v3 together with a held-out stain-shift set).
- **Inference:** overlapping 512 px windows (stride 256, cosine taper), four-fold flip/rotation averaging, arg-max over the four classes.
- **Organisers' model:** the released 5-fold nnU-Net ensemble from the BEETLE repository, run by me on Kaggle (`kernels/22_infer`); I did not train it. With flip averaging its score was 0.8667, so flip averaging changes nothing measurable.
- **Blends:** per-pixel class probabilities of my model and of the organisers' model are averaged with weights fixed in advance, then an arg-max is taken (`kernels/50_blend`, `ensemble/`).

## 3. What I found
- **Validation did not predict the test.** v3 had a higher in-distribution validation score than v2 (mean-class Dice 0.862 against 0.851) and a lower test score (0.8240 against 0.8388).
- **Stain jitter did not help.** v3 scored lower than v2 at all three centres.
- **Where we lose:** v2 was better than the organisers' model on necrosis (+0.040), on the UWMed centre (+0.029) and on ILC invasive tumour (+0.017), and worse on the Biopticka centre (-0.080) and on NST invasive tumour (-0.050). So most of the gap is one centre, which fits a domain shift.
- **Stain statistics (weak support, not proof):** the test regions are farther in colour statistics from the nearest training slide (median 0.66) than training slides are from each other (0.43). The per-region correlation with disagreement with the organisers is weak (Spearman 0.18).
- **The two models complement each other.** The 50/50 blend is above the organisers' model alone by only +0.002; the 30/70 blend by +0.012.

## 4. Limits
- 170 regions from three centres: differences of about 0.01 or less should not be read as real. I did not estimate confidence intervals.
- "Best of N" is optimistic: 12 submissions in total, of which 6 are listed in this version.
- The blends contain the organisers' released model; my own models alone score below it.
- I used part of the 396 development slides and no external data; free-tier compute limited training to a few epochs per model. Top entries on the leaderboard score higher.

## 5. Repository layout
| path | content |
|---|---|
| `src/` | the two segmentation modules (Phikon ViT-B, Phikon-v2) |
| `kernels/20_fetch`, `22_infer`, `23_tta` | fetch of the evaluation regions and organisers' model; their nnU-Net run (no training) |
| `kernels/30_wsiprobe`, `31_prep`, `33_prep2`, `38_prep3`, `39_prep4`, `42_prep5`, `45_prep6` | slide probing and patch extraction (slide selections in `selection*.json`) |
| `kernels/41_pack`, `43_pack2`, `46_pack3` | packing of the patches into a few large shard files |
| `kernels/32_train*`, `62_train_pv2` | trainers (ViT-B; Phikon-v2) with `build.sh` generators and local tests |
| `kernels/36_infer_seg`, `64_infer_pv2`, `66_probs_vitb` | inference and saving of probabilities |
| `kernels/50_blend`, `65_blend_pv2`, `ensemble/` | blends with the organisers' model; CPU ensembling from saved probabilities |
| `kernels/34_check`, `35_scale`, `37_maskstats`, `51_stain`, `52_stain2`, `53_priors`, `63_pv2check` | data checks and the stain / class-prior analyses |
| `select_shift.py`, `shift.json` | the held-out stain-shift patients |
| `RESULTS.md` | submissions with date and weights |

## 6. Reproducing
Set `KAGGLE_USER`, then generate a notebook, for example the ViT-B trainer: `cd kernels/32_train && ./build.sh <name> ...` (the header of each `build.sh` lists its arguments), and push it with `kaggle kernels push -p kernels/32_train --accelerator NvidiaTeslaT4`. The notebooks read the slides and the organisers' model from Zenodo and from earlier notebook outputs; adapt the slugs to your account. Weights and probabilities are not included.

## 7. Data and licence note
BEETLE data and the organisers' model belong to the organisers (CC BY-NC-SA 4.0) and are not included. Phikon and Phikon-v2 are used under Owkin's non-commercial licence. No data, weights or probabilities are redistributed here.

## 8. Credits
- BEETLE challenge, data, evaluation and the released nnU-Net baseline: the BEETLE organisers (Grand Challenge, Zenodo record 16812932). nnU-Net: Isensee et al.
- Backbones: Phikon and Phikon-v2 (Owkin). Libraries: PyTorch, transformers, nnU-Net.
- All code here was written for this project; no code from other participants' submissions was used.
