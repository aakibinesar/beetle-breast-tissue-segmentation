#!/bin/sh
# Phikon-v2 inference notebook.  usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-62-train-pv2b> <N_TTA 1-4>
#   env SMOKE=2 [SMOKE_CPU=1]: smoke mode - pretrained Phikon-v2 + RANDOM decoder on the first 2 ROIs (no checkpoint, no zip, the train slug is ignored); SMOKE_CPU=1 runs it on a CPU notebook (no GPU quota)
#   Attach ONLY the final training session as a source (it holds seg_best_<tag>.pt; session B copies session A's best checkpoint when it resumes).
#   Push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4      (SMOKE_CPU=1: no accelerator flag)
NAME=$1; SRC=$2; NT=${3:-4}; SM=${SMOKE:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'N_TTA, TAG, SMOKE = %s, "%s", %s\nimport os\n' "$NT" "$NAME" "$SM"; cat ../../src/beetle_seg_pv2.py; printf '\n'; cat driver.py; } > infer_seg_pv2.py
KS='"'"$KUSER"'/beetle-20-fetch", "'"$KUSER"'/beetle-60-phikonv2-fetch"'; [ "$SM" = "0" ] && KS="$KS, \"$KUSER/$SRC\""
GPU=true; [ "${SMOKE_CPU:-0}" = "1" ] && GPU=false
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-64-infer-pv2-$NAME", "title": "beetle-64-infer-pv2-$NAME", "code_file": "infer_seg_pv2.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": $GPU, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
JSON
