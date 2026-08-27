#!/bin/bash
# run_dino_fullft.sh — Lancia training DINOv2 in full fine-tuning
# Usage: ./run_dino_fullft.sh [exp] [model] [tag_suffix]

EXP=${1:-6}
MODEL=${2:-dinov2_small}
TAG=${3:-fullft}

cd /home/user/Documenti/orchid/orchid-osr/backend

echo "=== [$(date)] Starting Exp $EXP — $MODEL — tag=$TAG (full FT) ==="
echo "Log: /tmp/exp${EXP}_${MODEL}_${TAG}.log"

nohup python3 -u cli.py \
    --cross-validate \
    --exp "$EXP" \
    --model "$MODEL" \
    --tag "$TAG" \
    --full-finetune \
    --epochs 150 \
    --patience 15 \
    --accumulation 1 \
    > "/tmp/exp${EXP}_${MODEL}_${TAG}.log" 2>&1 &

PID=$!
echo "PID: $PID"
echo "$PID" > "/tmp/exp${EXP}_${MODEL}_${TAG}.pid"

echo "[$(date)] Experiment avviato in background (PID=$PID)"
echo "Per monitorare: tail -f /tmp/exp${EXP}_${MODEL}_${TAG}.log"
echo "Per uccidere: kill $PID"