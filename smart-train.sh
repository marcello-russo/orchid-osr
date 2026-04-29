#!/bin/bash
# smart-train.sh — Battery-aware training manager for all remaining models
# Stops training when battery < STOP, resumes when battery > START.
# No charging status check — only battery level matters.
# Automatically moves to the next model when all folds complete.
#
# Usage:
#   ./smart-train.sh                      # Train all remaining models
#   ./smart-train.sh --model convnext_tiny # Only one model

# --- Config ---
STOP_THRESHOLD=40    # Stop training below this %
START_THRESHOLD=80   # Resume training above this %
CHECK_INTERVAL=120
EXP=6
EPOCHS=100
PATIENCE=15
OVERFIT_THRESHOLD=0
ACCUMULATION=4

ALL_MODELS=(resnet18 resnet50 convnext_tiny dinov2_small dinov2_base convnext_small)

# --- Battery ---
BAT="/sys/class/power_supply/BAT0"
battery_level() { cat "$BAT/capacity"; }
is_charging() { grep -q "Charging\|Full" "$BAT/status"; }

# --- Training helpers ---
model_dir() { echo "experiments/exp6_clean_split_${1}"; }

model_done() {
    podman exec pytorch_app test -f "$(model_dir "$1")/results.json" 2>/dev/null
}

next_fold_for() {
    # Retry to avoid race condition: process exits but file not visible yet
    for attempt in 1 2 3; do
        local missing=""
        for i in 0 1 2 3 4; do
            if ! podman exec pytorch_app test -f "$(model_dir "$1")/fold_${i}.json" 2>/dev/null; then
                missing="$i"
                break
            fi
        done
        if [ -z "$missing" ]; then
            echo "done"
            return
        fi
        [ "$attempt" -lt 3 ] && sleep 2
    done
    echo "$missing"
}

training_is_running() {
    podman exec pytorch_app pgrep -f "cli.py.*--model" >/dev/null 2>&1
}

stop_training() {
    echo "[$(date +%H:%M:%S)] Battery $(battery_level)% — waiting for current epoch to finish..."
    # Wait for the next epoch boundary (checkpoint saved)
    local log="experiments/exp6_clean_split_${CURRENT_MODEL}/exp6_clean_split_${CURRENT_MODEL}.log"
    timeout 360 podman exec pytorch_app bash -c "
        tail -f '$log' 2>/dev/null | grep -m 1 'Epoch [0-9]*/100'
    " 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Epoch boundary reached. Killing training..."
    podman exec pytorch_app pkill -f "cli.py" 2>/dev/null || true
    for i in $(seq 1 10); do
        training_is_running || break
        sleep 2
    done
    TRAIN_PID=""
    echo "[$(date +%H:%M:%S)] Training stopped (battery: $(battery_level)%)."
}

start_fold() {
    local model="$1" fold="$2"
    echo "[$(date +%H:%M:%S)] $model fold $fold — starting (battery: $(battery_level)%)..."
    podman exec pytorch_app python cli.py --cross-validate \
        --exp "$EXP" --model "$model" --epochs "$EPOCHS" \
        --patience "$PATIENCE" --overfit-threshold "$OVERFIT_THRESHOLD" \
        --accumulation "$ACCUMULATION" --fold "$fold" &
    TRAIN_PID=$!
    CURRENT_MODEL="$model"
    CURRENT_FOLD="$fold"
}

# Aggregate results after all folds done
aggregate_model() {
    local model="$1"
    echo "[$(date +%H:%M:%S)] Aggregating $model results..."
    podman exec pytorch_app python cli.py --cross-validate \
        --exp "$EXP" --model "$model" --epochs "$EPOCHS" \
        --patience "$PATIENCE" --overfit-threshold "$OVERFIT_THRESHOLD" \
        --accumulation "$ACCUMULATION" --aggregate-only 2>&1 | tail -5
    echo "[$(date +%H:%M:%S)] $model complete!"
}

# --- Parse args ---
SINGLE_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "============================================"
echo "Smart Train — battery-aware manager"
echo "Stop < ${STOP_THRESHOLD}%, resume > ${START_THRESHOLD}% (battery only)"
echo "Models: ${MODELS[*]}"
echo "============================================"

CURRENT_MODEL=""
CURRENT_FOLD=""
TRAIN_PID=""

# Find next work item: (model, fold) or ("aggregate", model) or ("done", "")
find_next() {
    for model in "${MODELS[@]}"; do
        if model_done "$model"; then
            continue
        fi
        local fold=$(next_fold_for "$model")
        if [ "$fold" = "done" ]; then
            # All folds done but no results.json — need aggregate
            echo "aggregate $model"
            return
        fi
        echo "train $model $fold"
        return
    done
    echo "done"
}

# --- Main loop ---
while true; do
    LEVEL=$(battery_level)
    CHARGING=$(is_charging && echo "yes" || echo "no")

    # Check if training process just finished
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        wait "$TRAIN_PID" 2>/dev/null
        TRAIN_PID=""

        if [ -n "$CURRENT_MODEL" ] && [ "$CURRENT_FOLD" != "aggregate" ]; then
            if podman exec pytorch_app test -f "$(model_dir "$CURRENT_MODEL")/fold_${CURRENT_FOLD}.json" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] $CURRENT_MODEL fold $CURRENT_FOLD completed! (battery: ${LEVEL}%)"
            fi
        fi
        CURRENT_MODEL=""
        CURRENT_FOLD=""
    fi

    # Find next task
    read -r action arg1 arg2 <<< "$(find_next)"

    if [ "$action" = "done" ]; then
        echo "[$(date +%H:%M:%S)] All models complete!"
        break
    fi

    # Battery check
    if [ "$LEVEL" -lt "$STOP_THRESHOLD" ]; then
        # Battery critical — always stop
        if training_is_running; then
            stop_training
        fi
        echo "[$(date +%H:%M:%S)] Battery ${LEVEL}% — paused. Waiting for ${START_THRESHOLD}%..."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    if [ "$LEVEL" -lt "$START_THRESHOLD" ]; then
        # Mid zone — keep running if already training, don't start new
        if ! training_is_running && [ -z "$TRAIN_PID" ]; then
            echo "[$(date +%H:%M:%S)] Battery ${LEVEL}%. Waiting for ${START_THRESHOLD}%..."
        fi
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # Battery OK — start next task
    if ! training_is_running && [ -z "$TRAIN_PID" ]; then
        if [ "$action" = "aggregate" ]; then
            aggregate_model "$arg1"
        elif [ "$action" = "train" ]; then
            start_fold "$arg1" "$arg2"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
