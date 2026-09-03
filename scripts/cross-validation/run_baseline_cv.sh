#!/bin/bash
# Orchestrates k-fold cross-validation for the ResNet50 baseline classifier on
# the Drusen data, reusing the same folds as the diffusion CV pipeline
# (scripts/run_drusen_cv.sh).
#
# Unlike the diffusion classifier, the baseline has no separate pretrain/
# finetune stages: each fold trains directly from ImageNet-pretrained weights
# (PRETRAINED=true, RESUME=0), so there is no shared pretrain checkpoint to
# pass in.
#
# For each fold i = 0..K_FOLDS-1:
#   1. Train from ImageNet-pretrained weights (RESUME=0) on
#      drusen-fold{i}-{train,valid}.csv.
#   2. Archive the fold's checkpoint_{variant}/best_checkpoint_{variant} out of
#      the shared experiment folder into a fold-specific location, so the next
#      fold's training run doesn't overwrite them.
#   3. Run inference on drusen-fold{i}-test.csv against the archived
#      best checkpoint, writing inference_result.json next to it.
#
# Prerequisite: dataset/splits/create_drusen_cv_splits.py has already been run
# to produce drusen-fold{i}-{train,valid,test}.csv for i=0..K_FOLDS-1 (same
# splits used for the diffusion classifier's CV run).
#
# Usage:
#   K_FOLDS=5 VARIANT=resnet50 bash scripts/run_baseline_cv.sh
#
# Set START_FOLD (default 0) to resume from a later fold, same as
# run_drusen_cv.sh.

set -e

export PROJECT_ROOT="/workspace"
export DATA_ROOT="/data"
export INFERENCE_CHECKPOINT_FOLDER="/checkpoints/final-models"

K_FOLDS="${K_FOLDS:-5}"
START_FOLD="${START_FOLD:-0}"

export MODEL="baseline"
export DATA="fundus"
export BACKBONE="${BACKBONE:-resnet}"
export VARIANT="${VARIANT:-resnet50}"
export RESUME=0

EXPERIMENT_PATH="$PROJECT_ROOT/experiments/fundus-classifier"
CHECKPOINT_NAME="checkpoint_${VARIANT}"
BEST_CHECKPOINT_NAME="best_checkpoint_${VARIANT}"
CV_ARCHIVE_ROOT="$INFERENCE_CHECKPOINT_FOLDER/fundus-classifier/cv"

for ((i = START_FOLD; i < K_FOLDS; i++)); do
    echo "=== Fold $i/$((K_FOLDS - 1)): Training ==="
    export FOLD="$i"
    export FUNCTION="train"
    bash scripts/run.sh
    FOLD_EXIT=$?
    if [[ $FOLD_EXIT -ne 0 ]]; then
        echo "Fold $i training failed (exit $FOLD_EXIT) — aborting CV run."
        exit $FOLD_EXIT
    fi

    echo "=== Fold $i: Archiving checkpoint ==="
    FOLD_ARCHIVE_DIR="$CV_ARCHIVE_ROOT/fold${i}"
    mkdir -p "$FOLD_ARCHIVE_DIR"
    # Same idempotent-move pattern as run_drusen_cv.sh: clear any stale archive
    # from an earlier interrupted attempt first, since /workspace and
    # /checkpoints are separate Docker volumes (mv can't atomically rename
    # across them, and a non-empty target would otherwise nest instead of
    # replacing).
    rm -rf "$FOLD_ARCHIVE_DIR/$CHECKPOINT_NAME" "$FOLD_ARCHIVE_DIR/$BEST_CHECKPOINT_NAME"
    mv -T "$EXPERIMENT_PATH/$CHECKPOINT_NAME" "$FOLD_ARCHIVE_DIR/$CHECKPOINT_NAME"
    mv -T "$EXPERIMENT_PATH/$BEST_CHECKPOINT_NAME" "$FOLD_ARCHIVE_DIR/$BEST_CHECKPOINT_NAME"

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
python3 "$PROJECT_ROOT/results/general-scripts/aggregate_cv_results.py" \
    --cv-root "$CV_ARCHIVE_ROOT" --k-folds "$K_FOLDS" \
    --checkpoint-subdir "$BEST_CHECKPOINT_NAME"
