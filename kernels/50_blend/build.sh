#!/bin/sh
# usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-32-train-v2> <N_TTA 1-4> <comma-separated weights of OUR model, e.g. 0.3,0.5,0.7>
NAME=$1; SRC=$2; NT=${3:-4}; WS=${4:-0.3,0.5,0.7}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'N_TTA, TAG, WEIGHTS = %s, "%s", (%s,)\nimport os\n' "$NT" "$NAME" "$WS"; cat ../../src/beetle_seg.py; printf '\n'; cat driver.py; } > blend.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-50-blend-$NAME", "title": "beetle-50-blend-$NAME", "code_file": "blend.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$KUSER/beetle-20-fetch", "$KUSER/$SRC"], "model_sources": []}
JSON
