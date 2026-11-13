#!/bin/sh
# Phikon-v2 inference with TEST-TIME COLOUR MATCHING + probability saving.  usage: build.sh <name> <TRAIN_KERNEL_SLUG e.g. beetle-62-train-pv2big> <N_TTA 1-4>
#   Writes beetle_own_<name>.zip (hard labels), probs/*.npz (uint8 softmax x 255, for CPU-side blends), target_stats.json, norm_stats.csv and montage/*.png (left: original ROI, right: colour-matched ROI).
#   Sources: beetle-20-fetch (ROIs), beetle-43-pack2 + beetle-46-pack3 (the packed training patches, for the target colour statistics), beetle-60-phikonv2-fetch, the training notebook (checkpoint).
#   env NORM_ONLY=1   CPU notebook, no model, no checkpoint (the train slug is ignored): target statistics + before/after statistics + montages. Look at them BEFORE spending GPU quota. Push without an accelerator flag.
#   env SMOKE=2 [SMOKE_CPU=1]  pretrained Phikon-v2 + RANDOM decoder on the first 2 ROIs (pipeline / speed / memory check, no checkpoint, no zip); SMOKE_CPU=1 runs it on a CPU notebook.
#   env EXCLUDE_JSON=<json list of {sid: patientNNN_wsiM}>  slides left out of the training patches when the target statistics are measured; default = the 85 slides that pv2big did not use.
#   Full run: push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4      (expected ~0.75 GPU h with 4-way TTA, like the pv2big inference)
NAME=$1; SRC=$2; NT=${3:-4}; SM=${SMOKE:-0}; NO=${NORM_ONLY:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
EXJ=${EXCLUDE_JSON:-../38_prep3/selection4.json}
{ printf 'N_TTA, TAG, SMOKE, NORM_ONLY = %s, "%s", %s, %s\nimport os\n' "$NT" "$NAME" "$SM" "$NO"
  printf 'EXCLUDE_SLIDES = %s\n' "$(python -c 'import json,sys;print(sorted(e["sid"] for e in json.load(open(sys.argv[1]))))' "$EXJ")"
  cat ../../src/beetle_seg_pv2.py; printf '\n'; cat stain_norm.py; printf '\n'; cat driver.py; } > probs_norm.py
KS='"'"$KUSER"'/beetle-20-fetch", "'"$KUSER"'/beetle-43-pack2", "'"$KUSER"'/beetle-46-pack3"'
[ "$NO" = "0" ] && KS="$KS, \"$KUSER/beetle-60-phikonv2-fetch\""
[ "$NO" = "0" ] && [ "$SM" = "0" ] && KS="$KS, \"$KUSER/$SRC\""
GPU=true; { [ "${SMOKE_CPU:-0}" = "1" ] || [ "$NO" = "1" ]; } && GPU=false
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-67-probs-norm-$NAME", "title": "beetle-67-probs-norm-$NAME", "code_file": "probs_norm.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": $GPU, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
JSON
