#!/usr/bin/env bash
# Local CI — esegue gli stessi identici step di .github/workflows/ci.yml
# Trigger manuale dopo `git push`.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "  ✅ PASS"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ FAIL"; }

echo "═══════════════════════════════════════════════"
echo "  Local CI — $(date)"
echo "  Mirror di .github/workflows/ci.yml"
echo "═══════════════════════════════════════════════"

# ──────────────────────────────────────
#  BACKEND JOB
# ──────────────────────────────────────
echo ""
echo "═══ JOB: backend ═══"

echo "→ pip install .[dev]"
(cd "$ROOT/backend" && pip install -e ".[dev]" -q) && pass || fail

echo "→ ruff check"
(cd "$ROOT/backend" && ruff check app/ tests/ scripts/) && pass || fail

echo "→ ruff format --check"
(cd "$ROOT/backend" && ruff format --check app/ tests/ scripts/ --diff) && pass || fail

echo "→ pytest -x -q --cov=app --cov=scripts --cov-fail-under=30"
(cd "$ROOT/backend" && python3 -m pytest -x -q --cov=app --cov=scripts --cov-report=term --cov-fail-under=30) && pass || fail

# ──────────────────────────────────────
#  DOCKER JOB
# ──────────────────────────────────────
echo ""
echo "═══ JOB: docker ═══"

echo "→ bash -n entrypoint.sh"
bash -n "$ROOT/entrypoint.sh" && pass || fail

echo "→ python3 backend/scripts/ci/check_syntax.py"
python3 "$ROOT/backend/scripts/ci/check_syntax.py" && pass || fail

echo "→ docker build --target cpu -t orchid-osr"
docker build --target cpu -t orchid-osr "$ROOT" && pass || fail

echo "→ smoke test — health endpoint"
container_id=$(docker run -d orchid-osr)
SMOKE_PASS=0
for i in $(seq 1 12); do
  sleep 5
  if docker exec "$container_id" python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:7860/health')
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null; then
    echo "  OK: health endpoint risponde dopo $((i*5))s"
    SMOKE_PASS=1
    break
  fi
  status=$(docker inspect "$container_id" --format='{{.State.Status}}')
  if [ "$status" = "exited" ]; then
    echo "  FAIL: container exited"
    docker logs "$container_id" || true
    break
  fi
done
docker rm -f "$container_id" 2>/dev/null || true
[ "$SMOKE_PASS" -eq 1 ] && pass || fail

# ──────────────────────────────────────
#  FRONTEND JOB
# ──────────────────────────────────────
echo ""
echo "═══ JOB: frontend ═══"

echo "→ npm ci"
(cd "$ROOT/frontend" && npm ci --silent) && pass || fail

echo "→ npm run lint"
(cd "$ROOT/frontend" && npm run lint -- --no-cache 2>/dev/null)
LINT_EXIT=$?
# eslint con errori → codice 1; solo warnings → codice 0
if [ "$LINT_EXIT" -eq 0 ]; then
  pass
else
  fail
fi

echo "→ npm run build"
(cd "$ROOT/frontend" && npm run build) && pass || fail

# ──────────────────────────────────────
#  RIEPILOGO
# ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  Risultati: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
