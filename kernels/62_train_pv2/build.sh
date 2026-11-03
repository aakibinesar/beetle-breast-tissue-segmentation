#!/bin/sh
# BEETLE Phikon-v2 (DINOv2 ViT-L/16) training notebook builder.   usage: build.sh <name> <EPOCHS> <BS> <LR_DEC> <LR_BB> <UNFREEZE_LAST> <LIMIT 0=all>
#   env STAIN=0.1 SHIFT_JSON=../../shift.json   stain jitter + held-out stain-shift patients (as in the v3 recipe); PATCH_SRCS=beetle-43-pack2 (default)
#   env STOP_AFTER=3      session A: stop (cleanly, exit 0) after that many epochs and keep resume_<name>.pt; session B: RESUME_TAG=<name of session A> (attaches beetle-62-train-<A> as a source; A must have ended COMPLETE)
#   env TIME_BUDGET_H=11  stop cleanly at an epoch boundary if the next epoch would not fit in this many hours (Kaggle's session cap is 12 h)
#   env BB_DELAY=0.1      fraction of all steps with backbone LR 0 (the fresh decoder trains alone first); 0 = off
#   env SHIFT_STOP=2      stop when the held-out stain-shift score fell this many epochs in a row (0 = off)
#   env TIMING=30         GPU timing test: 30 training steps + a little evaluation, print the projected hours, exit (use LIMIT=240 EPOCHS=1)
#   env EXCLUDE_JSON=<json list of {sid: patientNNN_wsiM}>  whole slides left out of the TRAINING patches (ablation: ../38_prep3/selection4.json = the 85 slides added after v2)
#   env SEED=0  BALANCE=1  SELECT=both  WATCHDOG=0 (seconds; 0 = off)
# Push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4
NAME=$1; EP=$2; BS=$3; LRD=$4; LRB=$5; UF=$6; LIM=${7:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
PATCH_SRCS=${PATCH_SRCS:-beetle-43-pack2}; RES=${RESUME_TAG:-}
ALL_SRCS="$PATCH_SRCS,beetle-60-phikonv2-fetch"; [ -n "$RES" ] && ALL_SRCS="$ALL_SRCS,beetle-62-train-$RES"
KS=$(echo "$ALL_SRCS" | sed 's/[^,]*/"'"$KUSER"'\/&"/g')
WD=${WATCHDOG:-0}
{ [ "$WD" != "0" ] && printf 'import faulthandler; faulthandler.dump_traceback_later(%s, exit=True)\n' "$WD"
  printf 'EPOCHS, BS, LR_DEC, LR_BB, UNFREEZE_LAST, LIMIT, TAG, BALANCE, SELECT, STAIN = %s, %s, %s, %s, %s, %s, "%s", %s, "%s", %s\n' "$EP" "$BS" "$LRD" "$LRB" "$UF" "$LIM" "$NAME" "${BALANCE:-1}" "${SELECT:-both}" "${STAIN:-0}"
  printf 'SEED, STOP_AFTER, RESUME_TAG, TIME_BUDGET_H, BB_DELAY, TIMING, SHIFT_STOP = %s, %s, "%s", %s, %s, %s, %s\n' "${SEED:-0}" "${STOP_AFTER:-0}" "$RES" "${TIME_BUDGET_H:-11}" "${BB_DELAY:-0.1}" "${TIMING:-0}" "${SHIFT_STOP:-2}"
  [ -n "${EXCLUDE_JSON:-}" ] && printf 'EXCLUDE_SLIDES = %s
' "$(python -c 'import json,sys;print(sorted(e["sid"] for e in json.load(open(sys.argv[1]))))' "$EXCLUDE_JSON")"
  [ -n "${SHIFT_JSON:-}" ] && printf 'SHIFT_PATIENTS = %s\n' "$(python -c 'import json,sys;print(sorted(json.load(open(sys.argv[1]))["patients"]))' "$SHIFT_JSON")"
  cat ../../src/beetle_seg_pv2.py; printf '\n'; cat driver.py; } > train_seg_pv2.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-62-train-$NAME", "title": "beetle-62-train-$NAME", "code_file": "train_seg_pv2.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
JSON
