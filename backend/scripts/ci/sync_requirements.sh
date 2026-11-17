#!/usr/bin/env bash
# Rigenera requirements.txt e requirements-dev.txt da pyproject.toml.
#
# Serve per avere un solo set di versioni condiviso da immagine e CI:
# `pip install -r requirements.txt` installa esattamente ciò che è stato testato,
# invece di risolvere i range di pyproject al momento della build.
#
#   bash scripts/ci/sync_requirements.sh
#
# Richiede `uv` e rete. torch/torchvision restano fuori dal file: li installa il
# Dockerfile dall'indice CPU con versione esplicita (vedi commento in testa).
set -euo pipefail

cd "$(dirname "$0")/../.."

TMP_FULL="$(mktemp)"
TMP_RUNTIME="$(mktemp)"
trap 'rm -f "$TMP_FULL" "$TMP_RUNTIME"' EXIT

uv pip compile pyproject.toml \
    --python-version 3.11 --python-platform linux --extra dev --torch-backend cpu \
    --no-annotate --no-header -q -o "$TMP_FULL"
uv pip compile pyproject.toml \
    --python-version 3.11 --python-platform linux --torch-backend cpu \
    --no-annotate --no-header -q -o "$TMP_RUNTIME"

TORCH_VERSION="$(grep -m1 '^torch==' "$TMP_RUNTIME" | cut -d= -f3)"
TORCHVISION_VERSION="$(grep -m1 '^torchvision==' "$TMP_RUNTIME" | cut -d= -f3)"

{
    cat <<EOF
# Versioni di torch/torchvision per l'indice CPU, uniche per Dockerfile e CI.
# Fuori da requirements.txt di proposito: vanno installate con
#   --index-url https://download.pytorch.org/whl/cpu
# altrimenti un wheel CUDA (~2,5 GB) può finire nell'immagine di produzione.
torch==${TORCH_VERSION/\+cpu/}
torchvision==${TORCHVISION_VERSION/\+cpu/}
EOF
} > torch-pins.txt

{
    cat <<EOF
# Dipendenze di runtime, pinnate esattamente (Python 3.11 / linux, wheel CPU).
#
# Perché un file qui e non solo i range in pyproject.toml: l'immagine di
# produzione e la CI devono installare la STESSA roba, altrimenti un fix
# (o una regressione) può dipendere dal giorno della build invece che dal
# commit. Dependabot aggiorna questo file.
#
# torch/torchvision NON stanno qui: li installa il Dockerfile dall'indice CPU
# con versione esplicita, così un wheel CUDA (~2,5 GB) non può finire
# nell'immagine di produzione. Versioni usate: ${TORCH_VERSION} / ${TORCHVISION_VERSION}.
#
# Generato da pyproject.toml, non si modifica a mano:
#   bash backend/scripts/ci/sync_requirements.sh
EOF
    echo
    grep -vE '^(torch|torchvision)==' "$TMP_RUNTIME"
} > requirements.txt

{
    cat <<'EOF'
# Solo sviluppo/test: pytest, ruff e i loro plugin.
# `-r requirements.txt` mantiene un solo set di versioni per runtime e test.
#
# Generato da pyproject.toml: bash backend/scripts/ci/sync_requirements.sh
EOF
    echo
    echo "-r requirements.txt"
    comm -23 <(sort "$TMP_FULL") <(sort "$TMP_RUNTIME")
} > requirements-dev.txt

echo "torch-pins.txt: $(grep -c '==' torch-pins.txt) pin"
echo "requirements.txt: $(grep -c '==' requirements.txt) pin"
echo "requirements-dev.txt: $(grep -c '==' requirements-dev.txt) pin dev"
