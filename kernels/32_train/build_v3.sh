#!/bin/sh
# The planned BEETLE retrain ("v3"): warm start from v2, the verified 41,572-patch pack (39,857 train / 1,715 val), stain jitter 0.1,
# stain-shift validation (patients chosen by ../../select_shift.py -> ../../shift.json), checkpoint by the average of normal and stain-shift mean-class Dice.
# 5 epochs, batch 8, LR 5e-4 (decoder) / 3e-5 (backbone), last 4 blocks trainable, rarity-weighted sampler  (same recipe as the v2 stage, plus the three changes).
# Push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4      (expected about 5-5.5 h; the script stops itself at 7.5 h and keeps the best checkpoint)
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
[ -f ../../shift.json ] || { echo "shift.json missing: run beetle-52-stain2, download slides_stain.csv to out_stain2/, then python select_shift.py"; exit 1; }
PATCH_SRCS=beetle-43-pack2 STAIN=0.1 SHIFT_JSON=../../shift.json WATCHDOG=0 ./build.sh v3 5 8 5e-4 3e-5 4 0 1 v2 both
