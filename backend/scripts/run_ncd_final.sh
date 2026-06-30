#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Run NCD for exp8 and exp10 with fixed LedoitWolf centering.
# Saves logs to experiments/<exp_dir>/ncd_batch.log
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

EXPS="8 10"
MODELS="resnet18 resnet50 convnext_tiny convnext_small dinov2_small dinov2_base"

for exp in $EXPS; do
    for m in $MODELS; do
        if [ "$exp" = "8" ]; then exp_dir="exp8_sphor_${m}"; fi
        if [ "$exp" = "10" ]; then exp_dir="exp10_mgh_ncd_${m}"; fi
        log="experiments/${exp_dir}/ncd_batch.log"
        echo "🚀 START exp${exp} $m"
        timeout 600 python cli.py --ncd --exp "$exp" --model "$m" > "$log" 2>&1
        if grep -q "COMPLETED" "$log" 2>/dev/null; then
            echo "✅ DONE exp${exp} $m"
        else
            echo "❌ FAIL exp${exp} $m — check $log"
        fi
    done
done

python scripts/ncd_comparison.py > experiments/ncd_batch_summary.log 2>&1
echo "🎉 All done!"
