#!/bin/bash
set -e
cd /home/user/Documenti/orchid/orchid-osr/backend

# Carica HF_TOKEN per upload automatico risultati
if [ -f ../.env ]; then
  export HF_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2)
fi

run() {
  local exp_name="$1"
  local exp_num="$2"
  local model="$3"
  shift 3
  if [ -f "experiments/${exp_name}/results.json" ]; then
    echo "=== ${exp_name} già completato, salto ==="
    return
  fi
  echo "=== ${exp_name} ==="
  python cli.py --cross-validate --exp "$exp_num" --model "$model" "$@"
  
  # NCD pipeline per exp 8 e 10
  if [ "$exp_num" = "8" ] || [ "$exp_num" = "10" ]; then
    ncd_file="experiments/${exp_name}/ncd_results.json"
    if [ ! -f "$ncd_file" ]; then
      echo "=== NCD per ${exp_name} ==="
      python cli.py --ncd --exp "$exp_num" --model "$model" --osr-ensemble
    else
      echo "=== NCD già presente per ${exp_name}, salto ==="
    fi
  fi
}

# ConvNeXt exp 6 già fatto (F1 0.892 Tiny, 0.883 Small)
# ResNet exp 7-8-10 in cloud

run "exp7_supcon_convnext_tiny"  7 convnext_tiny   --epochs 100 --batch-size 8 --accumulation 4
run "exp7_supcon_convnext_small" 7 convnext_small  --epochs 100 --batch-size 8 --accumulation 4
run "exp8_sphor_convnext_tiny"   8 convnext_tiny   --epochs 100 --batch-size 8 --accumulation 4 --sphor
run "exp8_sphor_convnext_small"  8 convnext_small  --epochs 100 --batch-size 8 --accumulation 4 --sphor
run "exp10_mgh_ncd_convnext_tiny"  10 convnext_tiny   --epochs 100 --batch-size 8 --accumulation 4 --full-finetune --hard-weight 5.0
run "exp10_mgh_ncd_convnext_small" 10 convnext_small  --epochs 100 --batch-size 8 --accumulation 4 --full-finetune --hard-weight 5.0

# NCD comparison table
echo "=== NCD Comparison ==="
python cli.py --ncd-comparison
echo "=== TUTTO COMPLETATO ==="
