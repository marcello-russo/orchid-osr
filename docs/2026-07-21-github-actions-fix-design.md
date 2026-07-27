# GitHub Actions Fix — Design Doc

## Objective

Fix CI pipeline issues: eliminate obsolete HF Spaces sync, improve Docker smoke test with real health check, add Docker layer caching, pin ruff version, raise coverage threshold, add automatic VPS deployment.

## Changes

### 1. Eliminare `sync-hf.yml`
Il file `.github/workflows/sync-hf.yml` è obsoleto — il deployment è sulla VPS (`orchid-osr.klad.ml`), non più su HF Spaces. Rimuovere il file.

### 2. Docker smoke test via `/health`
Sostituire il polling loop attuale (controlla a 5/15/30/45s solo se il container non è uscito):

```yaml
- run: docker build --target=cpu -t orchid-osr .
- name: Smoke test — avvio container
  run: |
    container_id=$(docker run -d orchid-osr)
    for i in 5 15 30 45; do
      sleep $i
      status=$(docker inspect "$container_id" --format='{{.State.Status}}')
      ...
```

**Nuovo approccio** — avviare il container e fare `curl http://localhost:7860/health` con retry:

```yaml
- run: docker build --target=cpu -t orchid-osr .
- name: Smoke test — health endpoint
  run: |
    container_id=$(docker run -d orchid-osr)
    for i in $(seq 1 12); do
      sleep 5
      if docker exec "$container_id" python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" 2>/dev/null; then
        echo "OK: health endpoint risponde dopo $((i*5))s"
        docker rm -f "$container_id" || true
        exit 0
      fi
      status=$(docker inspect "$container_id" --format='{{.State.Status}}')
      if [ "$status" = "exited" ]; then
        echo "FAIL: container exited (code $(docker inspect "$container_id" --format='{{.State.ExitCode}}'))"
        docker logs "$container_id"
        docker rm -f "$container_id" || true
        exit 1
      fi
    done
    echo "FAIL: health endpoint non risponde dopo 60s"
    docker logs "$container_id" || true
    docker rm -f "$container_id" || true
    exit 1
```

### 3. Docker layer caching
Sostituire `docker build` diretto con `docker/build-push-action@v6`:

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
- name: Build and cache
  uses: docker/build-push-action@v6
  with:
    target: cpu
    tags: orchid-osr
    load: true
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

Nota: `load: true` carica l'immagine nel Docker locale dopo il build (necessario per lo smoke test).

### 4. Ruff pin
`backend/pyproject.toml`: `"ruff>=0.5.0"` → `"ruff~=0.5.0"` (solo patch updates, niente major che potrebbe introdurre breaking changes).

### 5. Coverage threshold
`ci.yml`: `--cov-fail-under=40` → `--cov-fail-under=50`

### 6. Nuovo job deploy (solo su push a main)

Nuovo job `deploy` in `ci.yml` che gira solo dopo merge su `main` (non su PR):

```yaml
deploy:
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  needs: [backend, docker, frontend]
  runs-on: ubuntu-latest
  steps:
    - name: SSH and deploy
      uses: appleboy/ssh-action@v1
      with:
        host: ${{ secrets.VPS_HOST }}
        username: ubuntu
        key: ${{ secrets.VPS_SSH_KEY }}
        script: |
          cd ~/OrchID
          git pull origin main
          cd frontend && npm run build
          cd ..
          sudo docker compose build backend
          sudo docker compose up -d backend nginx
```

Alternativa: usare `ssh` diretto con chiave scritta su file (evita dipendenza appleboy). Ma appleboy/ssh-action è la soluzione standard e più pulita per GitHub Actions.

### 7. SSH key pair
Generata con `ssh-keygen -t ed25519 -f keys/orchid-osr -N "" -C "github-actions-deploy"`.
- `keys/orchid-osr` (privata) → non committata (`.gitignore`), già aggiunta come secret `VPS_SSH_KEY`
- `keys/orchid-osr.pub` (pubblica) → installata in `~/.ssh/authorized_keys` sulla VPS

### 8. Ruff fix su dinov2.py
`backend/app/core/models/dinov2.py:136-140` — riga troppo lunga nel logger, già fixata.

## Files modificati

| File | Modifica |
|------|----------|
| `.github/workflows/sync-hf.yml` | Eliminato |
| `.github/workflows/ci.yml` | Docker caching + smoke test + coverage 50% + deploy job |
| `backend/pyproject.toml` | Ruff pin `>=0.5.0` → `~=0.5.0` |
| `backend/app/core/models/dinov2.py` | Fix formato ruff (già applicato) |
| `.gitignore` | Aggiunto `keys/` (già applicato) |

## Secret GitHub (già settati)

| Secret | Valore |
|--------|--------|
| `VPS_HOST` | `51.38.51.220` |
| `VPS_SSH_KEY` | Chiave privata ed25519 in `keys/orchid-osr` |

## Trigger

- `ci.yml` su **push a main** → backend + docker + frontend + deploy
- `ci.yml` su **PR verso main** → backend + docker + frontend (NO deploy)
- `sync-hf.yml` → eliminato
