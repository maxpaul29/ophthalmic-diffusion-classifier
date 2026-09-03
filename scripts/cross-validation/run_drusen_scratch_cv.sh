#!/bin/bash
# Orchestrates k-fold cross-validation for the UNet diffusion classifier
# trained from scratch on the Drusen data (no Mogon Phase-1 pretrained
# checkpoint), reusing the same folds as run_drusen_cv.sh, so the two can be
# compared mean +/- std over the same folds instead of on a single split.
#
# This uses train.py (FUNCTION=train), not finetune.py: finetune.py refuses to
# run without either RESUME=1 or a pretrained_checkpoint (see its guard at the
# top of main()), by design, since it is meant to continue an existing run.
# train.py is the actual from-scratch entrypoint, and uses its own default
# learning rate/epoch count (scripts/unet/fundus-unet.sh: 1e-4 / 500 epochs
# for FUNCTION=train, vs. 1e-5 / 400 epochs for FUNCTION=finetune) — so this
# compares each approach's own standard training recipe, not just the
# initialization in isolation. Keep this in mind when interpreting results.
#
# For each fold i = 0..K_FOLDS-1:
#   1. Train from scratch (RESUME=0, no PRETRAINED_CHECKPOINT) on
#      drusen-fold{i}-{train,valid}.csv.
#   2. Archive the fold's checkpoints/best_checkpoint out of the shared
#      experiment folder into a fold-specific location under a separate
#      "drusen-unet-scratch" archive root, so it never collides with the
#      pretrained-finetuning CV run's checkpoints.
#   3. Run inference on drusen-fold{i}-test.csv against the archived
#      best_checkpoint, writing inference_result.json next to it.
#
# Prerequisite: dataset/splits/create_splits_scripts/create_drusen_cv_splits.py
# (or create_drusen_cv_from_holdout.py) has already been run (same splits used
# for run_drusen_cv.sh).
#
# Usage (this script is normally invoked for you via `CROSS_VALIDATION=1` in
# scripts/run.sh, not called directly — see DOCKER.md Section 3/5):
#   K_FOLDS=5 bash scripts/cross-validation/run_drusen_scratch_cv.sh
#
# Set START_FOLD (default 0) to resume from a later fold, same as
# run_drusen_cv.sh.

set -e

export PROJECT_ROOT="/workspace"
export DATA_ROOT="/data"
export INFERENCE_CHECKPOINT_FOLDER="/checkpoints/final-models"

K_FOLDS="${K_FOLDS:-5}"
START_FOLD="${START_FOLD:-0}"

export MODEL="unet"
export DATA="fundus"
export RESUME=0
export PRETRAINED_CHECKPOINT=""
export DRUSEN_MODEL_DIR="drusen-unet-scratch"

EXPERIMENT_PATH="$PROJECT_ROOT/experiments/fundus-unet"
# Must match the path scripts/unet/fundus-unet.sh builds for CHECKPOINT_FOLDER
# when FOLD is set: $INFERENCE_CHECKPOINT_FOLDER/$DRUSEN_MODEL_DIR/cv/10-folds-holdout/fold{FOLD}/best_checkpoint.
CV_ARCHIVE_ROOT="$INFERENCE_CHECKPOINT_FOLDER/$DRUSEN_MODEL_DIR/cv/10-folds-holdout"

for ((i = START_FOLD; i < K_FOLDS; i++)); do
    echo "=== Fold $i/$((K_FOLDS - 1)): Training from scratch ==="
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
    rm -rf "$FOLD_ARCHIVE_DIR/checkpoints" "$FOLD_ARCHIVE_DIR/best_checkpoint"
    mv -T "$EXPERIMENT_PATH/checkpoints" "$FOLD_ARCHIVE_DIR/checkpoints"
    mv -T "$EXPERIMENT_PATH/best_checkpoint" "$FOLD_ARCHIVE_DIR/best_checkpoint"

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
    --cv-root "$CV_ARCHIVE_ROOT" --k-folds "$K_FOLDS"
