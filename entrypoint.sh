#!/bin/bash
set -e

# Use persistent storage on HF Spaces (/data) — dataset, cache HF, users
DATA_DIR="${DATA_DIR:-/data}"

# --- Privilegi ----------------------------------------------------------------
# Il volume /data arriva dal host come root:root, quindi il primo avvio deve
# partire da root per sistemare i permessi; subito dopo questo script si
# ri-esegue come utente non privilegiato (`app`, uid 10001) e da lì in poi il
# server gira senza privilegi. Un `chown -R` su GB di cache a ogni redeploy
# sarebbe un ritardo di minuti: si ricorre solo sulle voci di primo livello che
# ancora non appartengono all'utente applicativo.
APP_UID="${APP_UID:-10001}"
if [ "$(id -u)" = "0" ]; then
    if [ -d "$DATA_DIR" ]; then
        if [ "$(stat -c %u "$DATA_DIR")" != "$APP_UID" ]; then
            chown "$APP_UID:$APP_UID" "$DATA_DIR"
        fi
        for entry in "$DATA_DIR"/* "$DATA_DIR"/.[!.]*; do
            [ -e "$entry" ] || continue
            if [ "$(stat -c %u "$entry")" != "$APP_UID" ]; then
                chown -R "$APP_UID:$APP_UID" "$entry"
            fi
        done
    fi
    exec setpriv --reuid="$APP_UID" --regid="$APP_UID" --clear-groups "$0" "$@"
fi

# --- Volume -----------------------------------------------------------------
# Dataset, esperimenti, utenti e DB vivono su /data. Le directory del progetto
# che li contengono diventano symlink al volume.
link_to_data() {
    # Un `ln -sfn` su una directory che esiste già crea il link *dentro*
    # (dataset/dataset): il volume resterebbe vuoto e gli upload finirebbero nel
    # layer dell'immagine. Si toglie la directory solo se è vuota, così un
    # dataset vero in locale non viene mai distrutto.
    local name="$1"
    mkdir -p "$DATA_DIR/$name"
    [ -L "$name" ] && return 0
    if [ ! -e "$name" ]; then
        ln -s "$DATA_DIR/$name" "$name"
    elif [ -d "$name" ] && [ -z "$(ls -A "$name")" ]; then
        rmdir "$name" && ln -s "$DATA_DIR/$name" "$name"
    fi
}

if [ -d "$DATA_DIR" ]; then
    export HF_HOME="$DATA_DIR/.hf_cache"
    # Miniature della galleria dataset: sul volume, così sopravvivono ai redeploy
    export THUMBNAILS_DIR="$DATA_DIR/.thumbnails"
    link_to_data dataset
    link_to_data experiments

    # Utenti sul volume persistente (il backup su hub e' disattivato: vedi auth.py).
    # Al primo avvio il file viene creato vuoto al primo salvataggio;
    # il superadmin e' bootstrappato dalle env SUPERADMIN_*.
    export USERS_PATH="$DATA_DIR/users.json"
    # JWT secret path: persist in /data per sopravvivere ai restart
    export JWT_SECRET_PATH="$DATA_DIR/.jwt_secret"
    # Metadati (esperimenti, utenti, tentativi di login) sul volume: senza
    # questo il DB resta nel layer scrivibile e il lockout anti brute-force si
    # azzera a ogni redeploy.
    export DB_PATH="$DATA_DIR/orchid.db"
fi

# Check frontend dist (prima di operazioni lente)
FRONTEND_DIST="${FRONTEND_DIST:-../frontend/dist}"
if [ ! -d "$FRONTEND_DIST" ]; then
    echo "⚠️  $FRONTEND_DIST non trovata. La SPA non sarà servita."
    echo "   Esegui 'cd frontend && npm run build' prima di avviare."
fi

# Avvia server SUBITO
EXTRA_ARGS=""
if [ "$API_RELOAD" = "true" ]; then
    EXTRA_ARGS=" --reload"
fi

python3 server.py$EXTRA_ARGS &
SERVER_PID=$!

# Sync experiments/models in background (non blocca il server)
if [ -d "$DATA_DIR" ]; then
    (
        echo "Download experiment results..."
        PYTHONPATH=/workspace/backend python3 scripts/sync_experiments.py || echo "Experiment sync failed (non-fatal)"
    ) &
fi

# Dataset non scaricato localmente: le immagini vengono servite via proxy HF CDN

wait $SERVER_PID
