#!/bin/sh
# ViT-B (beetle_seg.py) inference + probability saving.  usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-32-train-v3> <N_TTA 1-4>
#   Writes beetle_own_<name>.zip (hard labels, same as 36_infer_seg) AND probs/*.npz (uint8 softmax x 255) for CPU-side ensembles. Attach only the one training notebook that holds the wanted seg_best_*.pt.
#   Push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4      (expected ~0.4-0.5 GPU h for ViT-B with 4-way TTA)
NAME=$1; SRC=$2; NT=${3:-4}; MOD=${MODULE:-beetle_seg}      # MODULE=beetle_seg_pv2 for the Phikon-v2 models (then beetle-60-phikonv2-fetch is attached as well)
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
EXTRA=''; [ "$MOD" = "beetle_seg_pv2" ] && EXTRA='"'"$KUSER"'/beetle-60-phikonv2-fetch", '
{ printf 'N_TTA, TAG, SMOKE = %s, "%s", 0\nimport os\n' "$NT" "$NAME"; cat ../../src/$MOD.py; printf '\n'; cat driver.py; } > probs_vitb.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-66-probs-$NAME", "title": "beetle-66-probs-$NAME", "code_file": "probs_vitb.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$KUSER/beetle-20-fetch", $EXTRA"$KUSER/$SRC"], "model_sources": []}
JSON
