#!/usr/bin/env bash
# Audit delle dipendenze Python con allowlist motivata.
#
#   bash scripts/ci/pip_audit.sh
#
# Le voci tollerate stanno in audit-ignores.txt (formato: ID + commento).
# pip-audit esce con 1 se trova qualcosa che non è in allowlist: in CI questo
# blocca il merge, che è il punto.
set -euo pipefail

cd "$(dirname "$0")/../.."

ARGS=()
while read -r id rest; do
    case "$id" in
        "" | \#*) continue ;;
    esac
    ARGS+=("--ignore-vuln" "$id")
done < audit-ignores.txt

pip-audit -r requirements.txt "${ARGS[@]}" --progress-spinner off -f columns
