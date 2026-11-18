#!/usr/bin/env bash
# Deploy sulla VPS di produzione (/srv/orchid).
#
# Perché non parte dalla CI: l'SSH della VPS è aperto solo su `tailscale0`
# (default deny, vedi docs/plans/2026-11-10-hardening.md in `personale/actions`)
# e il runner self-hosted `central` è registrato solo su `marcello-russo/actions`.
# Dall'account member di `squidslab/OrchID` non si possono creare né secret né
# runner, quindi il deploy parte da questa macchina (nella tailnet).
#
# Contratto identico a `personale/actions` (docs/vps-deploy.md): il tag è
# l'unica fonte di verità. La CI ha già costruito e pubblicato
# `ghcr.io/squidslab/orchid-app:<tag>`; qui si prendono `compose.yml` ed
# `edge.yml` **dal tag** (non dal working tree: un rollback deve riportare
# anche la config di allora) e si fa solo `pull` + `up -d`.
#
# Uso:
#   scripts/deploy-vps.sh            # ultimo tag del repo
#   scripts/deploy-vps.sh v0.2.3     # tag esplicito (anche rollback)
#   ORCHID_DEPLOY_HOST=altro ssh alias/host (default: vps-orchid)
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
    echo "errore: $*" >&2
    exit 1
}

TAG="${1:-}"
if [ -z "$TAG" ]; then
    TAG=$(git describe --tags --abbrev=0 2>/dev/null || true)
    [ -n "$TAG" ] || die "nessun tag nel repo: creane uno (git tag -a vX.Y.Z) o passalo esplicito"
fi

git rev-parse -q --verify "refs/tags/${TAG}^{commit}" >/dev/null \
    || die "tag $TAG inesistente in locale: git fetch --tags"
COMMIT=$(git rev-parse --short "${TAG}^{commit}")
git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1 \
    || die "tag $TAG non è su origin: la CI non può aver pubblicato l'immagine"

HOST="${ORCHID_DEPLOY_HOST:-vps-orchid}"
REMOTE_DIR=/srv/orchid
HEALTH_URL="${ORCHID_HEALTH_URL:-https://osr.marrusso.dev/health}"

# Asset dal tag: stessa configurazione dell'immagine che si sta deployando.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git archive "$TAG" compose.yml edge.yml | tar -x -C "$STAGE" \
    || die "il tag $TAG non contiene compose.yml/edge.yml"
[ -s "$STAGE/compose.yml" ] || die "compose.yml vuoto nel tag $TAG"

echo "==> deploy $TAG ($COMMIT) su $HOST:$REMOTE_DIR"
scp -q "$STAGE/compose.yml" "$STAGE/edge.yml" "$HOST:$REMOTE_DIR/"

# Lo smoke post-deploy viaggia col tag (non col working tree). I tag vecchi non
# lo hanno: il rollback a una versione precedente deve continuare a funzionare,
# quindi la sua assenza non è un errore.
git archive "$TAG" scripts/smoke-live.py 2>/dev/null | tar -x -C "$STAGE" 2>/dev/null || true
if [ -f "$STAGE/scripts/smoke-live.py" ]; then
    # shellcheck disable=SC2029  # voluto: REMOTE_DIR è una costante nostra
    ssh "$HOST" "mkdir -p '$REMOTE_DIR/bin'"
    scp -q "$STAGE/scripts/smoke-live.py" "$HOST:$REMOTE_DIR/bin/smoke-live.py"
fi

# L'healthcheck autorevole è in loopback sull'host: la VPS ha IP datacenter e
# Cloudflare (WAF della zona) risponde con una challenge agli IP non residenziali,
# quindi `curl https://osr.marrusso.dev` da lì non è affidabile.
# shellcheck disable=SC2029  # voluto: TAG/COMMIT espansi qui, quotati sopra
ssh "$HOST" "TAG='$TAG' COMMIT='$COMMIT' bash -s" <<'REMOTE'
set -eu
cd /srv/orchid

PORT=$(sed -n 's/^ORCHID_PORT=//p' .env | tail -n1)
[ -n "${PORT:-}" ] || { echo "errore: ORCHID_PORT assente in /srv/orchid/.env" >&2; exit 1; }
PREV=$(sed -n 's/^ORCHID_VERSION=//p' .env | tail -n1)
echo "versione precedente: ${PREV:-nessuna}"

set_version() {
    # `sed` sostituisce solo una riga che esiste già: su un host nuovo (o se il
    # .env è stato scritto a mano senza quella riga) l'aggiornamento sarebbe un
    # no-op e `compose.yml` si fermerebbe su ${ORCHID_VERSION:?...}, facendo
    # sembrare "impossibile da deployare" un tag che va benissimo.
    grep -q '^ORCHID_VERSION=' .env || printf 'ORCHID_VERSION=\n' >> .env
    sed -i "s|^ORCHID_VERSION=.*|ORCHID_VERSION=$1|" .env
}

# Un pull fallito (tag senza immagine, GHCR giù) non deve toccare il servizio
# vivo: il container vecchio resta su, e la versione in .env torna coerente.
set_version "$TAG"
if ! docker compose pull; then
    echo "errore: pull di $TAG fallita, il servizio resta sulla versione precedente" >&2
    [ -n "${PREV:-}" ] && set_version "$PREV"
    exit 1
fi
if ! docker compose up -d --remove-orphans; then
    echo "errore: up fallito, ripristino ${PREV:-lo stato precedente}" >&2
    [ -n "${PREV:-}" ] && set_version "$PREV" && docker compose up -d --remove-orphans
    exit 1
fi

for i in $(seq 1 36); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/health"; then
        echo "health locale OK dopo ~$((i * 5))s"
        # Registro di ciò che gira davvero (stesso ruolo di `bin/.deployed`
        # in personale/actions): scritto solo a deploy sano.
        printf '{"tag":"%s","commit":"%s","ts":"%s"}\n' "$TAG" "$COMMIT" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > .deployed.tmp && mv .deployed.tmp .deployed
        docker compose ps

        # "Sano" non è "giusto": /health risponde 200 anche con la dist sbagliata
        # o il CDN delle immagini giù, e un errore mascherato è spesso un 200 con
        # content-type application/json. Lo smoke guarda i byte delle immagini,
        # i totali del dataset e la versione dichiarata in /openapi.json (un
        # container vecchio con il tag nuovo in .env si vede solo lì).
        if [ -f bin/smoke-live.py ]; then
            if sudo -n true 2>/dev/null; then
                sudo -n python3 bin/smoke-live.py --url "http://127.0.0.1:${PORT}" \
                    --env-file secrets/app.env --expect-version "$TAG" \
                    || {
                        echo "errore: smoke live fallito su $TAG (servizio su, risposte sbagliate)" >&2
                        echo "rollback: scripts/deploy-vps.sh <tag precedente>" >&2
                        exit 1
                    }
            else
                echo "!! smoke live saltato: sudo senza password non disponibile su $HOST" >&2
            fi
        fi
        exit 0
    fi
    sleep 5
done

echo "healthcheck fallito su http://127.0.0.1:${PORT}/health" >&2
docker compose logs --tail 80 app || true
if [ -n "${PREV:-}" ] && [ "$PREV" != "$TAG" ]; then
    echo "rollback a $PREV" >&2
    set_version "$PREV"
    docker compose pull && docker compose up -d --remove-orphans
fi
exit 1
REMOTE

# Verifica end-to-end dal punto di vista di un utente (da qui: IP residenziale).
# È un avviso, non un errore: la challenge Cloudflare sugli IP datacenter non
# dice nulla sullo stato del deploy.
for i in 1 2 3; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$HEALTH_URL" || echo 000)
    if [ "$code" = "200" ]; then
        echo "==> esterno OK: $HEALTH_URL"
        echo "==> rollback: scripts/deploy-vps.sh <tag precedente>"
        exit 0
    fi
    [ "$i" = "3" ] && echo "!! $HEALTH_URL ha risposto $code (challenge Cloudflare o propagazione DNS?)"
    sleep 5
done
exit 0
