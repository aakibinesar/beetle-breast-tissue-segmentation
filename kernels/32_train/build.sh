#!/bin/sh
# usage: build.sh <name> <EPOCHS> <BS> <LR_DEC> <LR_BB> <UNFREEZE_LAST> <LIMIT> [BALANCE 1|0] [INIT_TAG] [SELECT mean_class|overall|both]
#   LIMIT 0 = all patches; BALANCE = class-rarity patch sampling;
#   INIT_TAG  = warm start from seg_best_<INIT_TAG>.pt of the notebook beetle-32-train-<INIT_TAG> (empty = start from the Phikon weights);
#   SELECT    = metric that chooses the saved best epoch (default mean_class = average of the four per-class Dice; "overall" reproduces the old behaviour).
#   Patch sources: default = beetle-41-pack (all patches packed into shards; GPU sessions stalled when 7 loose-folder sources were mounted);
#   override with e.g.  PATCH_SRCS=beetle-31-prep,beetle-33-prep2  ./build.sh ...   to reproduce the first runs.
NAME=$1; EP=$2; BS=$3; LRD=$4; LRB=$5; UF=$6; LIM=${7:-0}; BAL=${8:-1}; INIT=${9:-}; SEL=${10:-mean_class}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
PATCH_SRCS=${PATCH_SRCS:-beetle-41-pack}      # ONE notebook output holding all patches as a few big shard files (see kernels/41_pack); the loose-folder sources still work: PATCH_SRCS=beetle-31-prep,beetle-33-prep2,beetle-39-prep4-p0,...
ALL_SRCS="$PATCH_SRCS"; [ -n "$INIT" ] && ALL_SRCS="$ALL_SRCS,beetle-32-train-$INIT"
KS=$(echo "$ALL_SRCS" | sed 's/[^,]*/"'"$KUSER"'\/&"/g')
WD=${WATCHDOG:-0}      # env WATCHDOG=<seconds>: diagnostic - after that long the run prints where it is stuck and exits (0 = off, use 0 for real runs)
# env SHIFT_JSON=<file with {"patients": [...]}>: stain-shift validation - those patients' slides are held out of training and evaluated separately (made by select_shift.py); SELECT=both then picks the epoch by the average of the normal-validation and the stain-shift mean-class Dice.
# env STAIN=<sigma>: stain (H&E) jitter strength in optical-density space, e.g. STAIN=0.1 (0 = off, the default; the v2 model was trained without it)
{ [ "$WD" != "0" ] && printf 'import faulthandler; faulthandler.dump_traceback_later(%s, exit=True)
' "$WD"; printf 'EPOCHS, BS, LR_DEC, LR_BB, UNFREEZE_LAST, LIMIT, TAG, BALANCE, INIT, SELECT, WATCHDOG, STAIN = %s, %s, %s, %s, %s, %s, "%s", %s, "%s", "%s", %s, %s\n' "$EP" "$BS" "$LRD" "$LRB" "$UF" "$LIM" "$NAME" "$BAL" "$INIT" "$SEL" "$WD" "${STAIN:-0}"; [ -n "${SHIFT_JSON:-}" ] && printf 'SHIFT_PATIENTS = %s
' "$(python -c 'import json,sys;print(sorted(json.load(open(sys.argv[1]))["patients"]))' "$SHIFT_JSON")"; cat ../../src/beetle_seg.py; printf '\n'; cat driver.py; } > train_seg.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/beetle-32-train-$NAME", "title": "beetle-32-train-$NAME", "code_file": "train_seg.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [$KS], "model_sources": ["afiltowk/phikon/pytorch/default/1"]}
JSON
