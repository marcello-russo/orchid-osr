#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Run NCD pipeline for all remaining exp8 and exp10 models.
# Skips models that already have new-format results.
# Output: each model writes to experiments/<exp>_<model>/ncd_results.json
# Then combines results in experiments/ncd_comparison.json

set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

MODELS="resnet18 resnet50 convnext_tiny convnext_small dinov2_small dinov2_base"

run_ncd() {
  local exp=$1 model=$2
  local exp_name
  if [ "$exp" -eq 8 ]; then exp_name="exp8_sphor"; else exp_name="exp10_mgh_ncd"; fi
  local exp_dir="experiments/${exp_name}_${model}"
  local ncd_file="$exp_dir/ncd_results.json"

  # Skip if already has new-format results (detectors key present)
  if python -c "import json; r=json.load(open('$ncd_file')); assert 'detectors' in r" 2>/dev/null; then
    echo "✅ SKIP  exp${exp} $model — già in nuovo formato"
    return 0
  fi

  local logfile="/tmp/ncd_exp${exp}_${model}.log"
  echo "🚀 START exp${exp} $model → log: $logfile"
  
  nohup python cli.py --ncd --exp "$exp" --model "$model" > "$logfile" 2>&1 &
  local pid=$!
  echo "   PID: $pid"

  # Poll every 30s until done
  while kill -0 $pid 2>/dev/null; do
    sleep 30
  done

  # Verify
  if python -c "import json; r=json.load(open('$ncd_file')); assert 'detectors' in r" 2>/dev/null; then
    echo "✅ DONE  exp${exp} $model"
  else
    echo "❌ FAIL  exp${exp} $model — check $logfile"
    return 1
  fi
}

# Exp 8
for m in $MODELS; do run_ncd 8 "$m"; done

# Exp 10
for m in $MODELS; do run_ncd 10 "$m"; done

# Comparison
echo ""
echo "📊 Comparison table..."
python scripts/ncd_comparison.py
echo ""
echo "🎉 All done!"
