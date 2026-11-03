#!/bin/sh
# BEETLE extra training slides (prep6): the development slides never used so far (non-TCGA, patients not in the validation or stain-shift sets), cut into 512 px / 0.5 um patches exactly like prep5 (CPU notebook, internet on, data CC BY-NC-SA 4.0).
# usage: build.sh <name> <PART> <NPARTS> <DEADLINE_MIN> <SMOKE 0|1>      e.g. ./build.sh p0 0 5 170 0   |   ./build.sh smoke 0 1 20 1
NAME=$1; PART=$2; NPARTS=$3; DL=$4; SMOKE=${5:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'SMOKE, PART, NPARTS, DEADLINE_MIN = %s, %s, %s, %s\n' "$SMOKE" "$PART" "$NPARTS" "$DL"
  printf "SEL_JSON = r'''"; python -c "import json;print(json.dumps(json.load(open('selection5.json'))),end='')"; printf "'''\n"
  cat prep6_body.py; } > prep6.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-45-prep6-$NAME", "title": "beetle-45-prep6-$NAME", "code_file": "prep6.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON
