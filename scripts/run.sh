export PROJECT_ROOT="/workspace"                  # Path to diffusion-classifier repository
export DATA_ROOT="/data"                    # Path to the data directory containing chexpert and mel_isic_balanced
export INFERENCE_CHECKPOINT_FOLDER="/checkpoints/final-models"  # Checkpoint folder for inference

export COMET_PROJECT_NAME="diffusion-classifier"
export COMET_WORKSPACE=""
export COMET_API_KEY=""
export COMET_EXPERIMENT_NAME=""
export USE_COMET=0

# ${VAR:-default} so a value already exported by a calling script (e.g.
# a cross-validation script below setting MODEL/FUNCTION/FOLD before
# re-invoking this same run.sh once per fold) is respected instead of being
# unconditionally clobbered back to these defaults. Edit the default value
# directly (the part after ":-") to change what a container run does.
export MODEL="${MODEL:-unet}"                           # "baseline", "unet", "dit", "sd"
export FUNCTION="${FUNCTION:-train}"                        # "train", "inference", "explain", "finetune" (only supported for funuds-unet)
export DATA="${DATA:-fundus}"                            # "chexpert", "isic", "fundus"

# For the baseline
export BACKBONE="${BACKBONE:-resnet}"                   # (str) Backbone for the classifier ('resnet' or 'efficientnet', 'vit', 'swin')
export VARIANT="${VARIANT:-resnet50}"               # (str) Variant of the backbone ('resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'swin_base_patch4_window7_224', 'vit_base_patch16_224', 'vit_small_patch16_224')


export SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cross-validation: set CROSS_VALIDATION=1 above to run the k-fold CV
# orchestration script matching the selected MODEL/FUNCTION instead of the
# normal single-run workflow below. CROSS_VALIDATION=0 leaves everything
# unchanged.
export CROSS_VALIDATION="${CROSS_VALIDATION:-0}"
if [[ "$CROSS_VALIDATION" == "1" ]]; then
    # The orchestration scripts below call "bash scripts/run.sh" once per fold
    # for the actual train/inference run. Reset CROSS_VALIDATION=0 first, so
    # that inherited-environment re-entry into run.sh takes the normal
    # single-run path instead of re-triggering this dispatch (which would
    # otherwise restart the whole CV script from fold 0 on every fold, forever).
    export CROSS_VALIDATION=0
    export K_FOLDS="${K_FOLDS:-10}"
    export START_FOLD="${START_FOLD:-0}"
    if [[ "$MODEL" == "baseline" && "$DATA" == "fundus" ]]; then
        bash "$SCRIPTS_DIR/cross-validation/run_baseline_cv.sh"
        exit $?
    elif [[ "$MODEL" == "unet" && "$DATA" == "fundus" && "$FUNCTION" == "finetune" ]]; then
        export PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-/checkpoints/final-models/drusen-unet/pretrain-mogon}"
        bash "$SCRIPTS_DIR/cross-validation/run_drusen_cv.sh"
        exit $?
    elif [[ "$MODEL" == "unet" && "$DATA" == "fundus" && "$FUNCTION" == "train" ]]; then
        bash "$SCRIPTS_DIR/cross-validation/run_drusen_scratch_cv.sh"
        exit $?
    else
        echo "Error: CROSS_VALIDATION=1 is only supported for MODEL=baseline (DATA=fundus)," \
             "or MODEL=unet DATA=fundus with FUNCTION=finetune (pretrained) or FUNCTION=train (from scratch)." \
             "Got MODEL=$MODEL DATA=$DATA FUNCTION=$FUNCTION."
        exit 1
    fi
fi

# Baseline classifier
if [[ "$MODEL" == "baseline" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/baseline-classifier/${DATA}-classifier.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else 
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=baseline"
    fi
fi

# UNet model (supports training, finetuning, inference, explain)
if [[ "$MODEL" == "unet" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "finetune" || "$FUNCTION" == "inference" || "$FUNCTION" == "explain" ]]; then
        if [[ "$FUNCTION" == "finetune" && "$DATA" != "fundus" ]]; then
            echo "Error: Finetuning is only supported for the fundus dataset with the UNet model"
            exit 1
        fi
        SCRIPT_PATH="$SCRIPTS_DIR/unet/${DATA}-unet.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=unet"
    fi
fi

# DiT model (supports training and inference)
if [[ "$MODEL" == "dit" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/dit/${DATA}-dit.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=dit"
    fi
fi

# Stable Diffusion model
if [[ "$MODEL" == "sd" ]]; then
    if [[ "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/stable-diffusion/sd-${DATA}-inference.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    elif [[ "$FUNCTION" == "train" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/stable-diffusion/train.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            echo "Note that SD model trains on both datasets at the same time"
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=sd"
    fi
fi
