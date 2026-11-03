#!/bin/sh
# usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-32-train-s1uf4> <N_TTA 1-4>
NAME=$1; SRC=$2; NT=${3:-4}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'N_TTA, TAG = %s, "%s"\nimport os\n' "$NT" "$NAME"; cat ../../src/beetle_seg.py; printf '\n'; cat driver.py; } > infer_seg.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-36-infer-seg-$NAME", "title": "beetle-36-infer-seg-$NAME", "code_file": "infer_seg.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$KUSER/beetle-20-fetch", "$KUSER/$SRC"], "model_sources": []}
JSON
