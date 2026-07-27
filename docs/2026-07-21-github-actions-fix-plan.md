# GitHub Actions Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the CI/CD pipeline: remove obsolete HF sync, improve Docker smoke test, add layer caching, pin ruff, raise coverage, add VPS auto-deploy.

**Architecture:** Two workflows: `ci.yml` (CI + deploy) and delete `sync-hf.yml`. Backend change: one line in pyproject.toml. Secrets already set in GitHub.

**Tech Stack:** GitHub Actions, Docker, SSH, Python, Node.js

**IMPORTANT — Sessions and branches:** All tasks in this plan are on the `fix/dashboard-kpis` branch which already exists and is pushed. Each task's commit goes on this branch. At the end, a PR will be created.

---

### Task 1: Eliminare sync-hf.yml

**File:**
- Delete: `.github/workflows/sync-hf.yml`

- [ ] **Step 1: Eliminare il file**

Rimuovere il file obsoleto che sincronizzava con HF Spaces (non più usato):

```bash
rm .github/workflows/sync-hf.yml
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/sync-hf.yml
git commit -m "chore: remove obsolete HF Spaces sync workflow"
```

---

### Task 2: Pin ruff version in pyproject.toml

**File:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Leggere la riga attuale**

Aprire `backend/pyproject.toml`. Cercare la riga:
```
"ruff>=0.5.0",
```

- [ ] **Step 2: Cambiare il pin**

Sostituire `">=0.5.0"` con `"~=0.5.0"`.

`>=0.5.0` significa "qualsiasi versione 0.5.0 o superiore" — se esce ruff 1.0 con regole breaking, CI si rompe.
`~=0.5.0` significa "qualsiasi versione >= 0.5.0 e < 0.6.0" — solo patch e minor, niente major breaking.

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml
git commit -m "fix: pin ruff ~=0.5.0 (solo patch, no major breaking)"
```

---

### Task 3: Aggiornare ci.yml — Docker layer caching + smoke test + coverage

**File:**
- Modify: `.github/workflows/ci.yml` — solo il job `docker` (righe 26-59)

**ATTENZIONE:** Vanno cambiate SOLO le righe del job `docker`. I job `backend` e `frontend` rimangono invariati.

- [ ] **Step 1: Leggere il file attuale**

The current `ci.yml` has three jobs: `backend`, `docker`, `frontend`. We only change the `docker` job.

**Current docker job (lines 26-59):**

```yaml
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: bash -n entrypoint.sh
      - name: Python syntax check
        run: python3 backend/scripts/ci/check_syntax.py
      - run: docker build --target=cpu -t orchid-osr .
      - name: Smoke test — avvio container
        run: |
          container_id=$(docker run -d orchid-osr)
          for i in 5 15 30 45; do
            sleep $i
            status=$(docker inspect "$container_id" --format='{{.State.Status}}' 2>/dev/null)
            if [ "$status" = "running" ]; then
              echo "OK: container running after ${i}s"
              docker rm -f "$container_id" || true
              exit 0
            fi
            if [ "$status" = "exited" ]; then
              echo "FAIL: container exited (code $(docker inspect "$container_id" --format='{{.State.ExitCode}}')) after ${i}s"
              echo "--- LOGS ---"
              docker logs "$container_id"
              echo "------------"
              docker rm -f "$container_id" || true
              exit 1
            fi
          done
          echo "FAIL: container still not running after 90s"
          echo "--- LOGS ---"
          docker logs "$container_id" || true
          echo "------------"
          docker rm -f "$container_id" || true
          exit 1
```

- [ ] **Step 2: Modificare il job docker**

Sostituire l'intero job docker (dalla riga `docker:` fino alla riga prima di `frontend:`) con:

```yaml
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: bash -n entrypoint.sh
      - name: Python syntax check
        run: python3 backend/scripts/ci/check_syntax.py
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
              docker logs "$container_id" || true
              docker rm -f "$container_id" || true
              exit 1
            fi
          done
          echo "FAIL: health endpoint non risponde dopo 60s"
          docker logs "$container_id" || true
          docker rm -f "$container_id" || true
          exit 1
```

**Cosa cambia:**
- `docker build` diretto → `docker/build-push-action@v6` con cache GHA (righe 8-14)
- `docker/setup-buildx-action@v3` abilita Buildx (necessario per cache)
- `load: true` serve per avere l'immagine disponibile nel Docker locale dopo build (per lo smoke test)
- Smoke test: invece di aspettare 90s e controllare solo che non crashi, fa `curl` sull'endpoint `/health` ogni 5s fino a 60s
- Se il container esce, mostra i log e fallisce subito (non aspetta)

- [ ] **Step 3: Alzare coverage threshold**

Nel job `backend`, alla riga 24, cambiare:
```yaml
      - run: pytest -x -q --cov=app --cov=scripts --cov-report=term --cov-fail-under=40
```
in:
```yaml
      - run: pytest -x -q --cov=app --cov=scripts --cov-report=term --cov-fail-under=50
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "fix: Docker layer caching, health smoke test, coverage 50%"
```

---

### Task 4: Aggiungere job deploy a ci.yml

**File:**
- Modify: `.github/workflows/ci.yml` (aggiungere job dopo `frontend`)

- [ ] **Step 1: Aggiungere il job deploy**

Alla fine del file `ci.yml`, DOPO la chiusura del job `frontend` (dopo la riga `      - run: npm run build`), aggiungere:

```yaml
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: [backend, docker, frontend]
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to VPS
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ubuntu
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            set -e
            cd ~/OrchidOSR
            git pull origin main
            cd frontend
            npm run build 2>&1
            cd ..
            sudo docker compose build backend 2>&1
            sudo docker compose up -d backend nginx 2>&1
```

**Spiegazione delle condizioni:**
- `if: github.ref == 'refs/heads/main' && github.event_name == 'push'` — gira SOLO quando si fa push diretto su `main` (cioè dopo un merge). NON gira sulle Pull Request.
- `needs: [backend, docker, frontend]` — aspetta che tutti e 3 i job precedenti siano passati.
- `appleboy/ssh-action@v1` — azione standard per SSH. Usa la chiave privata dal secret.
- `2>&1` sull'output — ridirige stderr su stdout per avere log completi in caso di errori.

**Secret necessari:** `VPS_HOST` e `VPS_SSH_KEY` sono già stati settati. Non serve aggiungerli.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: auto-deploy su VPS dopo CI passato su main"
```

---

### Task 5: Verifica finale e PR

- [ ] **Step 1: Verificare che non ci siano errori di sintassi YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"
```

Se `yaml` non è installato:
```bash
pip install pyyaml && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"
```

- [ ] **Step 2: Verificare git status**

```bash
git status
```
Dovrebbe mostrare:
```
On branch fix/dashboard-kpis
nothing to commit, working tree clean
```

- [ ] **Step 3: Push del branch**

```bash
git push origin fix/dashboard-kpis
```

- [ ] **Step 4: Creare la Pull Request**

```bash
gh pr create \
  --base main \
  --head fix/dashboard-kpis \
  --title "fix: CI/CD — rimuovi HF sync, migliora Docker test, aggiungi caching, auto-deploy VPS" \
  --body "## Cosa cambia

### Rimosso
- \`sync-hf.yml\`: HF Spaces non più usato come deployment

### Migliorato \`ci.yml\`
- **Docker smoke test**: ora chiama /health invece di aspettare 90s
- **Layer caching**: \`docker/build-push-action@v6\` con cache GHA (build 2-3x più veloci)
- **Coverage**: threshold 40% → 50%

### Nuovo
- **Job deploy**: dopo merge su main, SSH nella VPS → pull → build → restart

### Altro
- Ruff pin \`~=0.5.0\` (niente major breaking)
- Chiave SSH ed25519 generata e installata
- Secret GitHub VPS_HOST e VPS_SSH_KEY settati
- \`.gitignore\` aggiornato (keys/)
"
```

## Riepilogo commit

| # | Commit | Branch |
|---|--------|--------|
| — | (commit esistenti del PR dashboard KPI) | fix/dashboard-kpis |
| 1 | `chore: remove obsolete HF Spaces sync workflow` | fix/dashboard-kpis |
| 2 | `fix: pin ruff ~=0.5.0 (solo patch, no major breaking)` | fix/dashboard-kpis |
| 3 | `fix: Docker layer caching, health smoke test, coverage 50%` | fix/dashboard-kpis |
| 4 | `feat: auto-deploy su VPS dopo CI passato su main` | fix/dashboard-kpis |
