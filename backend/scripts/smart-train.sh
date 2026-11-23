#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# smart-train.sh — Sequencer training a batteria (o senza)
# Esegue tutti i fold di tutti i modelli in sequenza, poi aggrega e carica su HF.
#
# Usage:
#   ./smart-train.sh                          # Exp 6
#   ./smart-train.sh --exp 7                  # Exp 2 (SupCon)
#   ./smart-train.sh --exp 8                  # Exp 3 (SpHOR)
#   ./smart-train.sh --exp 9                  # Exp 9 (SpHOR, no O. majellensis)
#   ./smart-train.sh --exp 10                 # Exp 10 (MGH-NCD: multi-grano + hard negative + deep f.t.)
#   ./smart-train.sh --model resnet18         # Solo un modello
#   ./smart-train.sh --no-upload             # Senza upload HF finale
#   nohup ./smart-train.sh --exp 8 >> smart-train.log 2>&1 &   # Append log!
#   nohup ./smart-train.sh --exp 9 >> smart-train.log 2>&1 &
#
# Se interrotto (Ctrl+C o crash), riprende dal fold successivo.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config ---
EPOCHS=150
CE_EPOCHS=100
PATIENCE=15
BATCH_SIZE=8
ACCUMULATION=4
UPLOAD=true

EXP=6
EXP6_MODELS=(resnet18 resnet50 convnext_tiny convnext_small dinov2_small dinov2_base)
EXP7_MODELS=(resnet50 resnet18 dinov2_small dinov2_base convnext_tiny convnext_small)
EXP8_MODELS=(resnet18 convnext_tiny convnext_small dinov2_base)
EXP9_MODELS=(convnext_tiny dinov2_small resnet50 resnet18 dinov2_base convnext_small)
EXP10_MODELS=(dinov2_base dinov2_small convnext_tiny convnext_small resnet18 resnet50)

# --- Battery ---
BAT="/sys/class/power_supply/BAT0"
STOP_THRESHOLD=15
START_THRESHOLD=40
battery_level() { cat "$BAT/capacity" 2>/dev/null || echo "100"; }
BATTERY_POLL=30  # secondi tra un controllo e l'altro

# --- Parse args ---
SINGLE_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp) EXP="$2"; shift 2 ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --no-upload) UPLOAD=false ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ "$EXP" -eq 6 ]; then
    EXP_PREFIX="exp1_ce"
    HF_PREFIX="exp1_ce"
    MODELS=("${EXP6_MODELS[@]}")
elif [ "$EXP" -eq 7 ]; then
    EXP_PREFIX="exp2_supcon"
    HF_PREFIX="exp2_supcon"
    MODELS=("${EXP7_MODELS[@]}")
elif [ "$EXP" -eq 8 ]; then
    EXP_PREFIX="exp3_sphor"
    HF_PREFIX="exp3_sphor"
    MODELS=("${EXP8_MODELS[@]}")
elif [ "$EXP" -eq 9 ]; then
    EXP_PREFIX="exp4_sphor_ncd"
    HF_PREFIX="exp4_sphor_ncd"
    MODELS=("${EXP9_MODELS[@]}")
else
    EXP=10
    EXP_PREFIX="exp10_mgh_ncd"
    HF_PREFIX="exp10_mgh_ncd"
    MODELS=("${EXP10_MODELS[@]}")
fi

if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
fi

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

model_dir() { echo "$BACKEND_DIR/experiments/${EXP_PREFIX}_${1}"; }
model_done()  { test -f "$(model_dir "$1")/results.json"; }

next_fold_for() {
    for i in 0 1 2 3 4; do
        if ! test -f "$(model_dir "$1")/fold_${i}.json"; then
            echo "$i"
            return
        fi
    done
    echo "done"
}

training_is_running() { pgrep -f "cli.py.*${EXP_PREFIX}" >/dev/null 2>&1; }

stop_training() {
    echo "[$(date +%H:%M:%S)] Battery $(battery_level)% — stopping..."
    pkill -f "cli.py.*${EXP_PREFIX}" 2>/dev/null || true
    for i in $(seq 1 10); do
        training_is_running || break
        sleep 2
    done
}

start_fold() {
    local model="$1" fold="$2"
    echo "[$(date +%H:%M:%S)] >>> $model fold $fold (battery: $(battery_level)%)"
    local BS="$BATCH_SIZE"
    local ACC="$ACCUMULATION"
    local EPOCHS_PER_MODEL="$EPOCHS"
    local CE_EPOCHS_PER_MODEL="$CE_EPOCHS"
    local SPHOR=""
    local EXTRA=""
    case "$model" in
        dinov2_small|dinov2_base)
            EPOCHS_PER_MODEL=100
            CE_EPOCHS_PER_MODEL=50
            BS=4
            ACC=8
            ;;
    esac
    if [ "$EXP" -eq 8 ] || [ "$EXP" -eq 9 ]; then
        SPHOR="--sphor"
    fi
    if [ "$EXP" -eq 10 ]; then
        SPHOR=""
        EPOCHS_PER_MODEL=150
        CE_EPOCHS_PER_MODEL=150
        EXTRA="--full-finetune --hard-weight 5.0"
        case "$model" in
            dinov2_small|dinov2_base)
                BS=4
                ACC=8
                ;;
            convnext_tiny|convnext_small|resnet18)
                BS=8
                ACC=4
                ;;
            resnet50)
                BS=8
                ACC=4
                EXTRA="$EXTRA --proj-dim 384"
                ;;
        esac
    fi
    cd "$BACKEND_DIR"
    python cli.py --cross-validate \
        --exp "$EXP" --model "$model" --epochs "$EPOCHS_PER_MODEL" \
        --patience "$PATIENCE" --batch-size "$BS" --accumulation "$ACC" \
        --ce-epochs "$CE_EPOCHS_PER_MODEL" --fold "$fold" $SPHOR $EXTRA &
    TRAIN_PID=$!
    CURRENT_MODEL="$model"
    CURRENT_FOLD="$fold"
}

aggregate_model() {
    local model="$1"
    echo "[$(date +%H:%M:%S)] >>> Aggregating $model..."
    cd "$BACKEND_DIR"
    python cli.py --cross-validate \
        --exp "$EXP" --model "$model" --epochs "$EPOCHS" \
        --patience "$PATIENCE" --accumulation "$ACCUMULATION" \
        --ce-epochs "$CE_EPOCHS" --aggregate-only 2>&1 | tail -5
    echo "[$(date +%H:%M:%S)] $model complete!"

    if $UPLOAD; then
        echo "[$(date +%H:%M:%S)] Uploading $model to HuggingFace..."
        HF_TOKEN="${HF_TOKEN:-}" python "$BACKEND_DIR/scripts/hf_tools/upload_models.py" --exp "$HF_PREFIX" 2>&1 | tail -3 || true
    fi
}

find_next() {
    for model in "${MODELS[@]}"; do
        if model_done "$model"; then
            continue
        fi
        local fold=$(next_fold_for "$model")
        if [ "$fold" = "done" ]; then
            echo "aggregate $model"
            return
        fi
        echo "train $model $fold"
        return
    done
    echo "done"
}

# --- Main ---
echo "============================================"
echo "Smart Train"
echo "Exp: $EXP"
echo "Models: ${MODELS[*]}"
echo "Battery: stop < ${STOP_THRESHOLD}%, resume > ${START_THRESHOLD}%"
$UPLOAD && echo "Upload to HF: enabled"
echo "============================================"

trap "echo; echo 'Interrotto. I fold completati non veranno ripetuti.'; exit 0" INT TERM

CURRENT_MODEL=""
CURRENT_FOLD=""
TRAIN_PID=""

while true; do
    LEVEL=$(battery_level)
    if [ "$LEVEL" -lt "$STOP_THRESHOLD" ]; then
        if training_is_running; then
            stop_training
        fi
        echo "[$(date +%H:%M:%S)] Battery ${LEVEL}% — in attesa di ${START_THRESHOLD}%..."
        sleep "$BATTERY_POLL"
        continue
    fi

    if [ "$LEVEL" -lt "$START_THRESHOLD" ] && ! training_is_running && [ -z "${TRAIN_PID:-}" ]; then
        echo "[$(date +%H:%M:%S)] Battery ${LEVEL}%. In attesa di ${START_THRESHOLD}%..."
        sleep "$BATTERY_POLL"
        continue
    fi

    # Check if training just finished
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        wait "$TRAIN_PID" 2>/dev/null || true
        if [ -n "$CURRENT_FOLD" ] && [ "$CURRENT_FOLD" != "aggregate" ]; then
            test -f "$(model_dir "$CURRENT_MODEL")/fold_${CURRENT_FOLD}.json" && \
                echo "[$(date +%H:%M:%S)] $CURRENT_MODEL fold $CURRENT_FOLD done"
        fi
        TRAIN_PID=""
        CURRENT_MODEL=""
        CURRENT_FOLD=""
    fi

    read -r action arg1 arg2 < <(find_next)

    if [ "$action" = "done" ]; then
        echo "[$(date +%H:%M:%S)] All models complete!"
        if $UPLOAD; then
            echo "[$(date +%H:%M:%S)] Final HF upload & README update..."
            cd "$BACKEND_DIR"
            HF_TOKEN="${HF_TOKEN:-}" python scripts/hf_tools/upload_models.py --exp "$HF_PREFIX" 2>&1 | tail -5 || true
        fi
        break
    fi

    if ! training_is_running && [ -z "$TRAIN_PID" ]; then
        if [ "$action" = "aggregate" ]; then
            aggregate_model "$arg1"
        elif [ "$action" = "train" ]; then
            start_fold "$arg1" "$arg2"
        fi
    fi

    sleep "$BATTERY_POLL"
done
