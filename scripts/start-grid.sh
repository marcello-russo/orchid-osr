#!/usr/bin/env bash
# Avvia (o riprende) il grid SpHOR con freno GPU e watchdog di alimentazione.
#
# Le fold già complete vengono saltate e una fold interrotta riparte dal suo
# checkpoint: rilanciare questo comando dopo un reboot non rifà il lavoro.
#
#   scripts/start-grid.sh                       # II-A + II-B, DINOv2-Small
#   EXPS=3 MODELS=resnet18,resnet50 scripts/start-grid.sh
#   CLOCK=1100 PAUSE_AT=35 scripts/start-grid.sh
#   CLOCK=max scripts/start-grid.sh      # nessun blocco: boost fino a 40 W
set -euo pipefail
cd "$(dirname "$0")/.."

EXPS="${EXPS:-4,5}"
MODELS="${MODELS:-dinov2_small}"
CLOCK="${CLOCK:-900}"
PAUSE_AT="${PAUSE_AT:-50}"
STATE="${STATE:-/tmp/opencode}"

mkdir -p "$STATE"

# CLOCK=max toglie il blocco: la GPU boosta fino al limite di potenza (40 W).
if [ "$CLOCK" = "max" ]; then
  ./scripts/gpu-throttle.sh --reset >/dev/null 2>&1 || echo "attenzione: reset del clock non riuscito" >&2
elif ! ./scripts/gpu-throttle.sh --clock "$CLOCK" >/dev/null 2>&1; then
  echo "attenzione: freno GPU non applicato (serve sudo -n)" >&2
fi

nohup python scripts/run_grid.py --experiments "$EXPS" --models "$MODELS" > "$STATE/grid.log" 2>&1 &
echo $! > "$STATE/grid.pid"
sleep 10

nohup python scripts/power-watchdog.py --pidfile "$STATE/grid.pid" \
  --pause-at "$PAUSE_AT" --resume-at 60 --unplugged-confirmations 1 \
  > "$STATE/watchdog.log" 2>&1 &
echo $! > "$STATE/watchdog.pid"

echo "grid avviato: pid $(cat "$STATE/grid.pid"), log $STATE/grid.log"
echo "watchdog: pid $(cat "$STATE/watchdog.pid"), pausa <=${PAUSE_AT}% ripresa >=60%"
