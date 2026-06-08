#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="logs/cloud-train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cloud-train-$(date +%Y%m%d-%H%M).log"

EXPERIMENTS=(7 8 9 10)
MODELS=("resnet18" "resnet50" "convnext_tiny" "convnext_small" "dinov2_small" "dinov2_base")
declare -A EXP_FLAGS=([7]="" [8]="--sphor" [9]="--sphor" [10]="--sphor")
declare -A EXP_NAMES=([7]="exp7_supcon" [8]="exp8_sphor" [9]="exp9_ncd" [10]="exp10_mgh_ncd")
RESUME=false
FILTER_EXP=""
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=true; shift ;;
        --exp) FILTER_EXP="$2"; shift 2 ;;
        --model) FILTER_MODEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Usage: $0 [--resume] [--exp N] [--model NAME] [--dry-run]"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# Phase 1: Dataset
log "=== Phase 1: Dataset ==="
if [ ! -d "dataset/cropped_256x512/train" ]; then
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log "[DRY-RUN] Would download dataset"
    else
        log "Downloading dataset..."
        python scripts/hf_tools/download_cache.py 2>&1 | tee -a "$LOG_FILE"
        log "Dataset download complete"
    fi
else
    log "Dataset already present"
fi

# Phase 2: Training
log "=== Phase 2: Training ==="
START_TIME=$(date +%s)
TOTAL_RUNS=0
COMPLETED=0
FAILED=0

for exp_id in "${EXPERIMENTS[@]}"; do
    [[ -n "$FILTER_EXP" && "$exp_id" != "$FILTER_EXP" ]] && continue
    exp_name="${EXP_NAMES[$exp_id]}"
    flags="${EXP_FLAGS[$exp_id]}"
    log "=== EXP $exp_id: $exp_name ==="

    for model in "${MODELS[@]}"; do
        [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]] && continue
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        results_file="experiments/${exp_name}_${model}/results.json"
        exp_log="$LOG_DIR/${exp_name}_${model}.log"

        if $RESUME && [ -f "$results_file" ]; then
            f1=$(python3 -c "import json; r=json.load(open('$results_file')); print(f'{r.get(\"mean_f1\", \"N/A\"):.4f}')" 2>/dev/null || echo "?")
            log "[SKIP] $exp_name / $model (F1: $f1)"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        log "[START] $exp_name / $model"
        if [ "${DRY_RUN:-false}" = "true" ]; then
            log "[DRY-RUN] python cli.py --cross-validate --exp $exp_id --model $model $flags"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        set +e
        python cli.py --cross-validate --exp "$exp_id" --model "$model" $flags 2>&1 | tee "$exp_log"
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ] && [ -f "$results_file" ]; then
            f1=$(python3 -c "import json; r=json.load(open('$results_file')); print(f'{r.get(\"mean_f1\", \"N/A\"):.4f}')" 2>/dev/null || echo "?")
            acc=$(python3 -c "import json; r=json.load(open('$results_file')); print(f'{r.get(\"mean_accuracy\", \"N/A\"):.4f}')" 2>/dev/null || echo "?")
            log "[OK]   $exp_name / $model — F1: $f1, Acc: $acc"
            COMPLETED=$((COMPLETED + 1))
        else
            log "[FAIL] $exp_name / $model — exit code $EXIT_CODE"
            FAILED=$((FAILED + 1))
        fi
    done
done

# Phase 3: Report
END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 3600 ))
DURATION_MIN=$(( ((END_TIME - START_TIME) % 3600) / 60 ))

log ""
log "=== REPORT ==="
log "Total: $TOTAL_RUNS | OK: $COMPLETED | FAIL: $FAILED"
log "Time: ${DURATION}h ${DURATION_MIN}m"
log "Log: $LOG_FILE"

# Summary JSON (env-var passing avoids set -u conflicts with Python variables)
export _PY_EXPS="${EXPERIMENTS[*]}"
export _PY_MODELS="${MODELS[*]}"
export _PY_TOTAL="$TOTAL_RUNS"
export _PY_COMPLETED="$COMPLETED"
export _PY_FAILED="$FAILED"
export _PY_LOG_DIR="$LOG_DIR"

python3 << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json, os

exp_ids = [int(x) for x in os.environ['_PY_EXPS'].split()]
models = os.environ['_PY_MODELS'].split()
exp_names = {7: 'exp7_supcon', 8: 'exp8_sphor', 9: 'exp9_ncd', 10: 'exp10_mgh_ncd'}
results = {
    'total': int(os.environ['_PY_TOTAL']),
    'completed': int(os.environ['_PY_COMPLETED']),
    'failed': int(os.environ['_PY_FAILED']),
    'experiments': {}
}
for eid in exp_ids:
    for m in models:
        ename = exp_names.get(eid, f'exp{eid}')
        f = f'experiments/{ename}_{m}/results.json'
        try:
            r = json.load(open(f))
            results['experiments'][f'exp{eid}_{m}'] = {
                'f1': r.get('mean_f1'), 'acc': r.get('mean_accuracy')
            }
        except Exception:
            pass
json.dump(results, open(os.path.join(os.environ['_PY_LOG_DIR'], 'summary.json'), 'w'), indent=2)
print('Summary saved')
PYEOF
