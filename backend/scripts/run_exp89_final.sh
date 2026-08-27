#!/usr/bin/env bash
# Run exp8 (closed-set, all 6 classes) and exp9 (O. garganica hidden novel) for all 6 models.
#
# Scientific setup:
#   - SupCon + SpHOR (--sphor), LR per-architecture from model registry
#     (resnet 0.001, convnext 0.0001, dinov2 0.0002).
#   - Hard-negative mining DISABLED (--hard-weight 0): it is the feature of
#     Exp 10 (MGH-NCD), not of exp8/9. Profs advised against it; class
#     balancing is already active via augmentDataPathBalanced (484 img/class).
#   - 150 SupCon epochs, patience 15 (early stopping stops ~epoch 20);
#     Phase 2: 100 CE epochs on frozen backbone (patience 15).
#
# Robust: logs to experiments/<exp_dir>/train.log, skips models with a complete
# results.json unless FORCE=1. Resume-safe (per-fold JSONs reused by kfold).
#
# Usage:
#   ./scripts/run_exp89_final.sh               # exp8 + exp9, all models
#   ./scripts/run_exp89_final.sh --exp 9       # only exp9
#   ./scripts/run_exp89_final.sh --model convnext_small   # one model
#   FORCE=1 ./scripts/run_exp89_final.sh --model convnext_tiny  # rerun anyway
set -euo pipefail
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
cd "$(cd "$(dirname "$0")/.." && pwd)"

FORCE="${FORCE:-0}"
EXPS="8 9"
MODELS="resnet18 resnet50 convnext_tiny convnext_small dinov2_small dinov2_base"
declare -A EXP_NAME=([8]="exp3_sphor" [9]="exp4_sphor_ncd")
# Per-architecture gradient accumulation (effective batch = batch × accum)
declare -A ACCUM=([resnet18]=4 [resnet50]=4 [convnext_tiny]=4 [convnext_small]=4 [dinov2_small]=1 [dinov2_base]=4)
# Batch size per architecture
declare -A BATCH=([resnet18]=8 [resnet50]=8 [convnext_tiny]=8 [convnext_small]=8 [dinov2_small]=8 [dinov2_base]=8)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp) EXPS="$2"; shift 2 ;;
        --model) MODELS="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) echo "Usage: $0 [--exp 8|9] [--model NAME] [--force]"; exit 1 ;;
    esac
done

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "=== PIPELINE EXP8+EXP9 INIZIATA ===  $(date '+%Y-%m-%d %H:%M:%S')"
log "Modelli: $MODELS | Esperimenti: $EXPS"

for exp in $EXPS; do
    exp_dir_prefix="${EXP_NAME[$exp]}"
    for m in $MODELS; do
        out_dir="experiments/${exp_dir_prefix}_${m}"
        results_file="${out_dir}/results.json"
        train_log="${out_dir}/train.log"

        # NCD pipeline (OSR/clustering) runs after exp9 training.
        # exp8 is closed-set: garganica is known, so no OSR evaluation.
        if [ "$exp" = "9" ]; then
            ncd_cmd="python cli.py --ncd --exp 9 --model $m"
        else
            ncd_cmd="true"
        fi

        if [ -f "$results_file" ] && [ "$FORCE" != "1" ]; then
            # Per exp9, richiedi anche ncd_results.json: se il training è finito
            # ma l'NCD non è mai stato eseguito, bisogna rifarlo.
            if [ "$exp" = "9" ] && [ ! -f "${out_dir}/ncd_results.json" ]; then
                log "RESUME exp${exp} ${m}: results.json ok ma ncd_results.json mancante, riprendo dall'NCD"
            else
                f1=$(python3 -c "import json;r=json.load(open('$results_file'));print(f'{r.get(\"mean_f1\",0):.4f}')" 2>/dev/null || echo "?")
                log "SKIP  exp${exp} ${m}: results.json presente (F1 $f1). FORCE=1 per rifare"
                continue
            fi
        fi

        mkdir -p "$out_dir"
        log "START exp${exp} ${m} → ${out_dir}"

        # Train + test (cross-validation with aggregate)
        set +e
        python cli.py --cross-validate --exp "$exp" --model "$m" \
            --sphor --hard-weight 0 \
            --epochs 150 --patience 15 --ce-epochs 100 \
            --batch-size "${BATCH[$m]}" --accumulation "${ACCUM[$m]}" \
            > "$train_log" 2>&1
        rc=$?
        set -e

        if [ $rc -ne 0 ]; then
            log "FAIL exp${exp} ${m}: exit $rc (log: $train_log)"
            continue
        fi

        # NCD pipeline for exp9 (novel class O. garganica)
        if [ "$exp" = "9" ]; then
            log "NCD  exp9 ${m}"
            set +e
            $ncd_cmd > "${out_dir}/ncd.log" 2>&1
            ncd_rc=$?
            set -e
            if [ $ncd_rc -ne 0 ]; then
                log "FAIL NCD exp9 ${m}: exit $ncd_rc (log: ${out_dir}/ncd.log)"
                continue
            fi
        fi

        log "DONE exp${exp} ${m}"
    done
done
log "All experiments finished."
