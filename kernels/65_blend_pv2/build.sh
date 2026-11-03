#!/bin/sh
# Phikon-v2 blend notebook = kernels/50_blend/driver.py (unchanged) with src/beetle_seg_pv2.py in place of beetle_seg.py. One GPU run writes: own (Phikon-v2 model alone), org (organisers' nnU-Net alone = baseline reproduction),
# blend_wXX (XX = percent weight of OUR model) zips, and saves all probabilities (probs/*.npz) so other weightings can be made later on a CPU without another GPU run.
# usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-62-train-pv2b> <N_TTA 1-4> <comma-separated weights of OUR model, e.g. 0.5,0.7>
#   Attach ONLY the final training session as the train source (it holds seg_best_<tag>.pt). Our rule: a blend must give OUR model at least half (0.5 or more).
#   Push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4      (internet is on: the organisers' nnU-Net code is pip-installed from GitHub; expected 2-3 GPU h)
NAME=$1; SRC=$2; NT=${3:-4}; WS=${4:-0.5,0.7}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'N_TTA, TAG, WEIGHTS = %s, "%s", (%s,)\nimport os\n' "$NT" "$NAME" "$WS"; cat ../../src/beetle_seg_pv2.py; printf '\n'; cat ../50_blend/driver.py; } > blend_pv2.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-65-blend-pv2-$NAME", "title": "beetle-65-blend-pv2-$NAME", "code_file": "blend_pv2.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$KUSER/beetle-20-fetch", "$KUSER/beetle-60-phikonv2-fetch", "$KUSER/$SRC"], "model_sources": []}
JSON
