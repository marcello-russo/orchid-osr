#!/usr/bin/env bash
# Deploy sulla VPS di produzione (/srv/orchid).
#
# Perché non parte dalla CI: l'SSH della VPS è aperto solo su `tailscale0`
# (default deny, vedi docs/plans/2026-11-10-hardening.md in `personale/actions`)
# e il runner self-hosted `central` è registrato solo su `marcello-russo/actions`.
# Dall'account member di `squidslab/OrchID` non si possono creare né secret né
# runner, quindi il deploy parte da questa macchina (nella tailnet).
#
# Il contratto è lo stesso di `personale/actions`: build dell'immagine in CI
# (tag `v*` → `ghcr.io/squidslab/orchid-app:<tag>`), e qui solo `pull` + `up -d`,
# con compose.yml ed edge.yml presi dal repo (l'host è cache, non sorgente).
#
# Uso:
#   scripts/deploy-vps.sh            # ultimo tag del repo
#   scripts/deploy-vps.sh v0.2.1     # tag esplicito (anche rollback a un tag vecchio)
#   ORCHID_DEPLOY_HOST=altro ssh alias/host (default: vps-orchid)
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-}"
if [ -z "$TAG" ]; then
    TAG=$(git describe --tags --abbrev=0 2>/dev/null || true)
fi
if [ -z "$TAG" ]; then
    echo "errore: nessun tag nel repo, passane uno esplicito (es. v0.2.0)" >&2
    exit 1
fi

HOST="${ORCHID_DEPLOY_HOST:-vps-orchid}"
REMOTE_DIR=/srv/orchid
HEALTH_URL="${ORCHID_HEALTH_URL:-https://osr.marrusso.dev/health}"

echo "==> deploy $TAG su $HOST:$REMOTE_DIR"
scp -q compose.yml edge.yml "$HOST:$REMOTE_DIR/"

# L'healthcheck autorevole è in loopback sull'host: la VPS ha IP datacenter e
# Cloudflare (WAF della zona) risponde con una challenge agli IP non residenziali,
# quindi `curl https://osr.marrusso.dev` da lì non è affidabile.
ssh "$HOST" "TAG='$TAG' bash -s" <<'REMOTE'
set -eu
cd /srv/orchid

PORT=$(sed -n 's/^ORCHID_PORT=//p' .env | tail -n1)
PREV=$(sed -n 's/^ORCHID_VERSION=//p' .env | tail -n1)
echo "versione precedente: ${PREV:-nessuna}"
sed -i "s|^ORCHID_VERSION=.*|ORCHID_VERSION=$TAG|" .env

docker compose pull
docker compose up -d

for i in $(seq 1 36); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/health"; then
        echo "health locale OK dopo ~$((i * 5))s"
        docker compose ps
        exit 0
    fi
    sleep 5
done

echo "healthcheck fallito su http://127.0.0.1:${PORT}/health" >&2
docker compose logs --tail 80 app || true
if [ -n "${PREV:-}" ]; then
    echo "rollback a $PREV" >&2
    sed -i "s|^ORCHID_VERSION=.*|ORCHID_VERSION=$PREV|" .env
    docker compose pull && docker compose up -d
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
        exit 0
    fi
    [ "$i" = "3" ] && echo "!! $HEALTH_URL ha risposto $code (challenge Cloudflare o propagazione DNS?)"
    sleep 5
done
exit 0

