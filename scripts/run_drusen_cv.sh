#!/bin/bash
# Orchestrates k-fold cross-validation for the Drusen finetuning (Phase 2).
#
# For each fold i = 0..K_FOLDS-1:
#   1. Finetune from scratch (RESUME=0) on drusen-fold{i}-{train,valid}.csv,
#      starting from the shared Phase-1 Mogon pretrain checkpoint.
#   2. Archive the fold's checkpoints/best_checkpoint out of the shared
#      experiment folder into a fold-specific location, so the next fold's
#      finetune run doesn't overwrite them.
#   3. Run inference on drusen-fold{i}-test.csv against the archived
#      best_checkpoint, writing inference_result.json next to it.
#
# Prerequisite: dataset/splits/create_drusen_cv_splits.py has already been run
# to produce drusen-fold{i}-{train,valid,test}.csv for i=0..K_FOLDS-1.
#
# Usage:
#   K_FOLDS=5 PRETRAIN_CHECKPOINT=/checkpoints/final-models/drusen-unet/pretrain-mogon \
#       bash scripts/run_drusen_cv.sh

set -e

# Same Docker mountpoint paths as scripts/run.sh — this script calls run.sh as
# a subprocess per fold (which sets these for itself), but needs them in its
# own shell too, for the archiving step and the final aggregation call below.
export PROJECT_ROOT="/workspace"
export DATA_ROOT="/data"
export INFERENCE_CHECKPOINT_FOLDER="/checkpoints/final-models"

K_FOLDS="${K_FOLDS:-5}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:?Set PRETRAIN_CHECKPOINT to the Phase-1 Mogon pretrain checkpoint dir}"

export MODEL="unet"
export DATA="fundus"
export RESUME=0
export PRETRAINED_CHECKPOINT="$PRETRAIN_CHECKPOINT"

EXPERIMENT_PATH="$PROJECT_ROOT/experiments/fundus-unet"
CV_ARCHIVE_ROOT="$INFERENCE_CHECKPOINT_FOLDER/drusen-unet/cv"

for ((i = 0; i < K_FOLDS; i++)); do
    echo "=== Fold $i/$((K_FOLDS - 1)): Finetuning ==="
    export FOLD="$i"
    export FUNCTION="finetune"
    bash scripts/run.sh
    FOLD_EXIT=$?
    if [[ $FOLD_EXIT -ne 0 ]]; then
        echo "Fold $i finetuning failed (exit $FOLD_EXIT) — aborting CV run."
        exit $FOLD_EXIT
    fi

    echo "=== Fold $i: Archiving checkpoint ==="
    FOLD_ARCHIVE_DIR="$CV_ARCHIVE_ROOT/fold${i}"
    mkdir -p "$FOLD_ARCHIVE_DIR"
    mv "$EXPERIMENT_PATH/checkpoints" "$FOLD_ARCHIVE_DIR/checkpoints"
    mv "$EXPERIMENT_PATH/best_checkpoint" "$FOLD_ARCHIVE_DIR/best_checkpoint"

    echo "=== Fold $i: Running inference on the held-out test split ==="
    export FUNCTION="inference"
    bash scripts/run.sh
    FOLD_EXIT=$?
    if [[ $FOLD_EXIT -ne 0 ]]; then
        echo "Fold $i inference failed (exit $FOLD_EXIT) — aborting CV run."
        exit $FOLD_EXIT
    fi
done

echo "=== All $K_FOLDS folds complete. Aggregating results. ==="
python3 "$PROJECT_ROOT/experiments/fundus-unet/aggregate_cv_results.py" \
    --cv-root "$CV_ARCHIVE_ROOT" --k-folds "$K_FOLDS"
