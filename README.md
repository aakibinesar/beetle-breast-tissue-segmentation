# BEETLE breast-tissue segmentation: pathology foundation models and ensembling

Code and write-up for my entries to [BEETLE](https://beetle.grand-challenge.org/) (Grand Challenge): four-class tissue segmentation (other, non-invasive epithelium, invasive epithelium, necrosis) on 170 hidden H&E regions from three centres. The metric is overall Dice (`overall_dice`). **Status: ongoing**, last updated 2026-11-13.

## Summary
I fine-tuned pathology foundation models (Phikon ViT-B and Phikon-v2) for this task and combined predictions at three levels: flip and rotation averaging within each model, averaging between my models, and a 50/50 blend with the organisers' released nnU-Net. The blend scores 0.8843 overall Dice on the hidden test set, against 0.8666 for the organisers' model alone and 0.8480 for my best single model. The models make different mistakes (mine are stronger on necrosis and on one centre, theirs on another), which is why averaging them helps. Everything ran on Kaggle's free GPUs.

| entry | overall Dice |
|---|---|
| organisers' released nnU-Net 5-fold ensemble (run by me, not trained by me) | 0.8666 |
| my best single model (Phikon-v2, fine-tuned by me) | 0.8480 |
| **50% my Phikon-v2 model + 50% the organisers' nnU-Net** | **0.8843** |

I made 14 submissions in total; all are listed in [RESULTS.md](RESULTS.md).

## 1. Task and constraints
- **Data:** the public BEETLE development set (Zenodo, CC BY-NC-SA 4.0). The full development set has 587 slides, of which 396 are in the public download; my models used between 154 and 267 of them, split by patient, with no external data and no slides named after TCGA cases.
- **Evaluation:** the 170 evaluation regions (about 2048 to 2560 px) have hidden labels, so every test number comes from uploading a zip of label images (at most 12 uploads per 28 days).
- **Compute:** Kaggle's free tier (T4 GPUs, about 30 GPU hours per week, 12 hours per session). Slides are read inside Kaggle notebooks by HTTP range request, so the 120 GB of slides never touch my disk.

## 2. Method and why
**Models.** Pathology foundation models pretrained on histology transfer well with limited data, and fine-tuning only their last blocks plus a light convolutional decoder fits on one T4. Fine-tuning the last four blocks beat a frozen backbone on validation (overall Dice 0.912 against 0.881), so every model unfreezes its last blocks.

| model | what was trained | training data | test score (alone) | name in the code |
|---|---|---|---|---|
| Phikon ViT-B | Phikon ViT-B/16 (Owkin), last 4 blocks + CNN-skip decoder | 154 slides, 26,343 patches | 0.8388 | v2 |
| Phikon ViT-B + jitter | same, continued with stain (H&E) jitter | 218 slides | 0.8240 | v3 |
| Phikon-v2 + jitter | Phikon-v2 (DINOv2 ViT-L/16), last 6 of 24 blocks + same decoder, stain jitter | 218 slides | 0.8269 | pv2 |
| Phikon-v2, 154 slides | as above, original slides, no jitter (an ablation) | 154 slides | not submitted alone | pv2abl |
| Phikon-v2, extra slides | Phikon-v2, last 6 blocks, no jitter | 154 + 113 further slides, 43,772 patches | 0.8464 | pv2big |
| Phikon-v2, extra slides + colour matching | same model, colour-matched at test time (below) | same | 0.8480 | pv2big, notebook `67_probs_norm` |

- **Training:** 512 px patches at 0.5 µm/px, class-balanced centres per slide, a class-rarity sampler, AdamW with a cosine schedule in fp16. The checkpoint was chosen by mean per-class Dice on a patient-disjoint validation set of 23 slides.
- **Inference:** overlapping 512 px windows with a cosine taper, four-fold flip and rotation averaging, arg-max over the four classes. Class probabilities are saved so that ensembles can be built on CPU afterwards (`ensemble/`).

**Ensembling at three levels, and why.** Averaging predictions helps when the members make different mistakes.
- *Within one model:* four-fold flip and rotation averaging. On the organisers' model it changed nothing measurable (0.8667 against 0.8666).
- *Between my own models:* averaging Phikon ViT-B and Phikon-v2 + jitter gave 0.8459, above either alone (0.8388 and 0.8269). Averaging four models gave 0.8429: the two added models were weaker and made similar mistakes, so they diluted the better ones.
- *With the organisers' nnU-Net:* Phikon ViT-B was better than the organisers' model on necrosis (+0.040), on the UWMed centre (+0.029) and on ILC invasive tumour (+0.017), and worse on the Biopticka centre (-0.080), so I expected complementary errors. The 50/50 blend scores above both sides. Against the organisers alone it adds +0.113 on necrosis, +0.065 on non-invasive and +0.061 on invasive epithelium, and +0.017, +0.004 and +0.033 at the three centres (Biopticka, SCDC, UWMed). The organisers trained on the BEETLE development set, which has 587 slides against the 396 in the public download, so part of the gain may be information my models never saw; I could not test that.

My model carries at least half of the weight in every blend, so that the result reflects its contribution, and the weights were fixed in advance, not tuned on the leaderboard.

**Test-time colour matching.** The test regions are farther in colour from the nearest training slide (median distance 0.66) than the training slides are from each other (0.43), and the Biopticka centre was the weakest. I therefore shifted and scaled each region's tissue colour in CIELAB to the average of the training patches before inference, with no labels and no learning on the test images. It moved the same model from 0.8464 to 0.8480 alone and the blend from 0.8839 to 0.8843, and Biopticka from 0.8335 to 0.8438, which is still below the organisers' 0.8791. So colour alone does not explain that centre's gap.

## 3. What I learned
- **Validation did not predict the test.** Validation Dice rose from 0.851 to 0.912 across models, while single-model test scores stayed between 0.824 and 0.848. The limiting factor is the shift between the development slides and the test centres, mostly one centre.
- **Neither more data nor stain jitter moved the overall score.** 113 further slides left it level with the earlier two-model average (0.8464 against 0.8459) but changed its profile, with necrosis much better (0.612 against 0.534). Stain jitter lowered the score at all three centres.
- **Ensemble members must differ and be strong.** The gain from blending grew with my model's quality but not in proportion: +0.002 with Phikon ViT-B, +0.009 with two models, +0.017 with Phikon-v2 on extra slides.

## 4. Constraints, and what I would do with more resources
Constraints: free-tier compute limited every model to a few epochs; only 396 of the 587 development slides are public; I used no gated foundation models; and the test set has 170 regions from three centres, so differences of about 0.01 or less are not meaningful (I did not estimate confidence intervals).

With more resources I would:
- fine-tune larger pathology foundation models (entries at the top of the leaderboard describe such models), with LoRA;
- train on all development slides with patient-level cross-validation, which would also give out-of-fold predictions for learning the blend weights instead of fixing them;
- address the stain and scanner shift directly, with stronger augmentation or per-centre normalisation, and train for longer;
- estimate confidence intervals with a bootstrap over regions.

## 5. Repository layout
| path | content |
|---|---|
| `src/` | the two segmentation modules (Phikon ViT-B, Phikon-v2) |
| `kernels/20_fetch`, `22_infer`, `23_tta` | fetch of the evaluation regions and the organisers' model; their nnU-Net run (no training) |
| `kernels/30_wsiprobe`, `31_prep`, `33_prep2`, `38_prep3`, `39_prep4`, `42_prep5`, `45_prep6` | slide probing and patch extraction (slide selections in `selection*.json`) |
| `kernels/41_pack`, `43_pack2`, `46_pack3` | packing of the patches into a few large shard files |
| `kernels/32_train*`, `62_train_pv2` | trainers (Phikon ViT-B; Phikon-v2) with `build.sh` generators and local tests |
| `kernels/36_infer_seg`, `64_infer_pv2`, `66_probs_vitb`, `67_probs_norm` | inference and saving of probabilities (`67_probs_norm`: with test-time colour matching) |
| `kernels/50_blend`, `65_blend_pv2`, `ensemble/` | blends with the organisers' model; CPU ensembling from saved probabilities |
| `kernels/34_check`, `35_scale`, `37_maskstats`, `51_stain`, `52_stain2`, `53_priors`, `63_pv2check` | data checks and the stain and class-prior analyses |
| `select_shift.py`, `shift.json` | the held-out stain-shift patients |
| `RESULTS.md` | every submission, with date, weights and per-class scores |

## 6. Reproducing
Set `KAGGLE_USER`, then generate a notebook, for example the Phikon-v2 trainer: `cd kernels/62_train_pv2 && ./build.sh <name> ...` (the header of each `build.sh` lists its arguments), and push it with `kaggle kernels push -p kernels/62_train_pv2 --accelerator NvidiaTeslaT4`. The notebooks read the slides and the organisers' model from Zenodo and from earlier notebook outputs; adapt the slugs to your account. On today's Kaggle image the Phikon-v2 checkpoints load only after their attention layers are renamed to the current library's names (`remap_dino_keys` in `kernels/67_probs_norm/stain_norm.py`); the older inference notebooks do not include this. Weights and probabilities are not included.

## 7. Data and licence note
BEETLE data and the organisers' model belong to the organisers (CC BY-NC-SA 4.0) and are not included. Phikon and Phikon-v2 are used under Owkin's non-commercial licence. No data, weights or probabilities are redistributed here.

## 8. Credits
- BEETLE challenge, data, evaluation and the released nnU-Net baseline: the BEETLE organisers (Grand Challenge, Zenodo record 16812932). nnU-Net: Isensee et al.
- Backbones: Phikon and Phikon-v2 (Owkin). Libraries: PyTorch, transformers, nnU-Net.
- All code here was written for this project; no code from other participants' submissions was used.
