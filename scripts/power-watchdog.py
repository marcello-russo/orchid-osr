#!/usr/bin/env python3
"""Watchdog di alimentazione: mette in pausa il training quando la batteria scende.

Su questo portatile il caricatore non tiene il passo della GPU: con l'AC
collegato la batteria **scende** comunque. Il watchdog guarda la capacita' e,
sotto una soglia, ferma il processo (SIGSTOP sul processo e su tutti i suoi
discendenti, worker del DataLoader compresi) e lo riprende (SIGCONT) quando la
batteria e' tornata sopra una soglia piu' alta.

SIGSTOP/SIGCONT e non un checkpoint: il processo resta congelato con il suo
contesto CUDA e il suo stato, quindi non si perde nulla e non si corrompe
niente. Il costo e' che la GPU resta occupata (memoria) mentre e' in pausa.

Uso:
    scripts/power-watchdog.py --pidfile /tmp/opencode/rerun.pid
    scripts/power-watchdog.py --pid 1234 --pause-at 35 --resume-at 60
    scripts/power-watchdog.py --once --capacity-source /tmp/fake-battery.json

Il file di stato (default `~/.local/state/orchid/power-watchdog.json`) serve a
un watchdog riavviato: se il processo era in pausa, lo sa e non lo lascia
fermo per sempre.

Nessuna dipendenza oltre la standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "orchid"
STATE_FILE = STATE_DIR / "power-watchdog.json"
LOG_FILE = STATE_DIR / "power-watchdog.log"

DEFAULT_PAUSE_AT = 35
DEFAULT_RESUME_AT = 60
DEFAULT_UNPLUGGED_PAUSE_AT = 90
DEFAULT_INTERVAL = 15.0
CRITICAL_AT = 10


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Lettura dello stato di alimentazione
# --------------------------------------------------------------------------- #
def read_power_sysfs() -> dict:
    """Capacita', stato e AC da /sys/class/power_supply (Linux)."""
    base = Path("/sys/class/power_supply")
    capacity = None
    status = None
    for bat in sorted(base.glob("BAT*")):
        try:
            capacity = int((bat / "capacity").read_text().strip())
            status = (bat / "status").read_text().strip()
            break
        except (OSError, ValueError):
            continue

    ac_online = None
    for ac in sorted(base.glob("A*")):
        try:
            ac_online = int((ac / "online").read_text().strip())
            break
        except (OSError, ValueError):
            continue

    return {"capacity": capacity, "status": status, "ac_online": ac_online}


def read_power_file(path: str) -> dict:
    """Sorgente alternativa (test): JSON {"capacity": 42, "status": "...", "ac_online": 1}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "capacity": data.get("capacity"),
        "status": data.get("status"),
        "ac_online": data.get("ac_online"),
    }


# --------------------------------------------------------------------------- #
# Decisione (pura, testabile)
# --------------------------------------------------------------------------- #
def decide(
    capacity: int | None,
    status: str | None,
    ac_online: int | None,
    paused: bool,
    pause_at: int = DEFAULT_PAUSE_AT,
    resume_at: int = DEFAULT_RESUME_AT,
    unplugged_pause_at: int = DEFAULT_UNPLUGGED_PAUSE_AT,
) -> str:
    """Restituisce "pause", "resume" o "keep".

    Isteresi: si mette in pausa sotto `pause_at`, si riprende sopra `resume_at`.
    In piu' si mette in pausa subito se la batteria sta *scaricando* e non e'
    quasi piena (cavo staccato): senza carico la macchina si spegnerebbe in
    fretta.
    """
    if capacity is None:
        return "keep"
    discharging = bool(status) and status.lower().startswith("discharg")
    unplugged = ac_online == 0
    if not paused:
        if capacity <= pause_at:
            return "pause"
        # Cavo staccato: si ferma subito, la macchina non ha una fonte.
        # "Discharging" con l'AC online e' il caso di questa macchina (il
        # caricatore non tiene il passo): li' vale la sola isteresi, altrimenti
        # il training non partirebbe mai sotto il 90%.
        if unplugged and capacity < unplugged_pause_at:
            return "pause"
        return "keep"
    if capacity >= resume_at and (not unplugged):
        return "resume"
    return "keep"


# --------------------------------------------------------------------------- #
# Albero dei processi
# --------------------------------------------------------------------------- #
def children_of(pid: int) -> list[int]:
    out: list[int] = []
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            kids = (task / "children").read_text().split()
            out.extend(int(k) for k in kids)
    except OSError:
        pass
    return out


def collect_tree(pid: int) -> list[int]:
    """Discendenti (profondita' prima) + il pid, saltando i pid spariti."""
    tree: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        if not Path(f"/proc/{current}").exists():
            continue
        tree.append(current)
        stack.extend(children_of(current))
    return tree


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def signal_tree(pids: list[int], sig: int) -> int:
    sent = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            log(f"permesso negato su pid {pid}")
    return sent


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(pid: int, paused: bool) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": pid, "paused": paused, "ts": time.time()}), encoding="utf-8")
        tmp.replace(STATE_FILE)
    except OSError as exc:
        log(f"stato non salvato: {exc}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pausa il training quando la batteria scende.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="PID del processo da sorvegliare")
    target.add_argument("--pidfile", help="File che contiene il PID")
    parser.add_argument("--pause-at", type=int, default=DEFAULT_PAUSE_AT, help="Soglia di pausa (%%)")
    parser.add_argument("--resume-at", type=int, default=DEFAULT_RESUME_AT, help="Soglia di ripresa (%%)")
    parser.add_argument(
        "--unplugged-pause-at",
        type=int,
        default=DEFAULT_UNPLUGGED_PAUSE_AT,
        help="In pausa se sta scaricando e la batteria e' sotto questa soglia (%%)",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Secondi tra i controlli")
    parser.add_argument(
        "--confirmations",
        type=int,
        default=None,
        help="Letture consecutive concordi prima di agire (default: 2, 1 con --once)",
    )
    parser.add_argument(
        "--unplugged-confirmations",
        type=int,
        default=4,
        help="Conferme per la regola 'cavo staccato': su questa macchina l'AC sfarfalla (default: 4)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Non invia segnali, stampa solo le decisioni")
    parser.add_argument("--once", action="store_true", help="Un solo controllo (per test e cron)")
    parser.add_argument(
        "--capacity-source",
        default="auto",
        help="auto (sysfs) oppure il percorso di un JSON con capacity/status/ac_online",
    )
    parser.add_argument("--state-file", default=str(STATE_FILE), help="Dove salvare lo stato")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    global STATE_FILE
    STATE_FILE = Path(args.state_file)

    # `--once` e' un controllo singolo (test, cron): non ha senso chiedergli
    # due conferme, altrimenti non agisce mai.
    confirmations = args.confirmations if args.confirmations is not None else (1 if args.once else 2)

    pid = args.pid
    if pid is None:
        try:
            pid = int(Path(args.pidfile).read_text().strip())
        except (OSError, ValueError) as exc:
            log(f"pidfile illeggibile ({exc}): esco")
            return 2

    if pid <= 0:
        # pid 0 in `kill` significa "tutto il mio process group": non e' mai
        # quello che si vuole da un watchdog.
        log(f"pid non valido ({pid}): esco")
        return 2

    state = load_state()
    paused = bool(state.get("paused")) and state.get("pid") == pid and alive(pid)
    log(
        f"watchdog su pid {pid} (pausa ≤{args.pause_at}%, ripresa ≥{args.resume_at}%, "
        f"staccato <{args.unplugged_pause_at}%, ogni {args.interval:.0f}s"
        + (", dry-run" if args.dry_run else "")
        + ")"
        + (" [era in pausa]" if paused else "")
    )

    stop = {"flag": False}

    def _stop(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    pending: str | None = None
    pending_count = 0

    while not stop["flag"]:
        if not alive(pid):
            log(f"pid {pid} terminato: esco")
            save_state(pid, paused=False)
            return 0

        power = read_power_file(args.capacity_source) if args.capacity_source != "auto" else read_power_sysfs()
        action = decide(
            power["capacity"],
            power["status"],
            power["ac_online"],
            paused,
            pause_at=args.pause_at,
            resume_at=args.resume_at,
            unplugged_pause_at=args.unplugged_pause_at,
        )
        detail = f"batteria {power['capacity']}% {power['status']} ac={power['ac_online']}"

        # Debounce: una lettura sola non basta. Un cavo che fa contatto per un
        # istante non deve fermare un training.
        if action == "keep":
            pending, pending_count = None, 0
        elif action == pending:
            pending_count += 1
        else:
            pending, pending_count = action, 1
        needed = max(1, confirmations)
        # La regola "cavo staccato" scatta sopra la soglia di capacita': qui il
        # sensore AC sfarfalla (ac=0 per decine di secondi con status=Charging),
        # quindi pretende piu' conferme della soglia normale.
        if action == "pause" and power["capacity"] is not None and power["capacity"] > args.pause_at:
            needed = max(needed, args.unplugged_confirmations)
        if action != "keep" and pending_count < needed:
            if args.dry_run:
                log(f"[dry-run] {action}? ({detail}) — in attesa di conferma")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        if action == "pause":
            if args.dry_run:
                log(f"[dry-run] PAUSA ({detail})")
            else:
                pids = collect_tree(pid)
                sent = signal_tree(pids, signal.SIGSTOP)
                paused = True
                save_state(pid, paused=True)
                log(f"PAUSA ({detail}) — fermati {sent} processi: la batteria deve risalire a {args.resume_at}%")
        elif action == "resume":
            if args.dry_run:
                log(f"[dry-run] RIPRESA ({detail})")
            else:
                pids = collect_tree(pid)
                sent = signal_tree(pids, signal.SIGCONT)
                paused = False
                save_state(pid, paused=False)
                log(f"RIPRESA ({detail}) — ripartiti {sent} processi")
        else:
            if args.dry_run:
                log(f"[dry-run] keep ({detail})")
            if power["capacity"] is not None and power["capacity"] <= CRITICAL_AT:
                log(f"CRITICO: batteria {power['capacity']}% — attacca il caricatore")

        if args.once:
            return 0
        time.sleep(args.interval)

    log("watchdog terminato su richiesta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
