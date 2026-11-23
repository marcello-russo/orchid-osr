#!/usr/bin/env bash
# Limita la potenza della GPU per far durare la batteria durante i training lunghi.
#
# Su questa macchina il caricatore non tiene il passo: con la GPU libera la
# batteria scende anche con l'AC collegato. `nvidia-smi -pl` qui non viene
# applicato (il driver riporta `power.limit: N/A`), quindi il freno vero e' il
# clock: `-lgc` blocca la frequenza e la potenza scende con lei.
#
# Misurato su Quadro T2000 (batch 8, DINOv2-Small):
#   libera   ~1200 MHz   ~29 W   66 °C
#   900 MHz  900 MHz     ~21.5 W 61 °C
#
# Uso:
#   scripts/gpu-throttle.sh --status
#   scripts/gpu-throttle.sh --clock 900     # blocca a 900 MHz
#   scripts/gpu-throttle.sh --reset         # toglie il blocco
#
# Serve sudo (senza password su questa macchina). Il blocco non sopravvive a un
# reboot: non c'e' niente da ripristinare a mano.
set -euo pipefail

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/orchid"
STATE_FILE="$STATE_DIR/gpu-throttle.state"

status() {
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader
}

require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    echo "serve sudo senza password (nvidia-smi -lgc non funziona da utente)" >&2
    exit 1
  fi
}

case "${1:-}" in
  --status)
    status
    ;;
  --clock)
    clock="${2:?uso: --clock <MHz>}"
    if ! [[ "$clock" =~ ^[0-9]+$ ]]; then
      echo "--clock vuole un numero di MHz" >&2
      exit 1
    fi
    require_sudo
    mkdir -p "$STATE_DIR"
    # ricorda cosa c'era prima, per poterlo dire in --reset
    nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits > "$STATE_FILE"
    sudo -n nvidia-smi -lgc "${clock},${clock}" >/dev/null
    echo "clock bloccato a ${clock} MHz (prima: $(cat "$STATE_FILE") MHz)"
    status
    ;;
  --reset)
    require_sudo
    sudo -n nvidia-smi -rgc >/dev/null
    rm -f "$STATE_FILE"
    echo "blocco rimosso"
    status
    ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
