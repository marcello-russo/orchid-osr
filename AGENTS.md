# AGENTS.md — OrchID OSR (orchid)

Master's thesis project (discussa il 29/10/2026): ultra-fine-grained classification of *Ophrys* orchids via deep learning. Python backend (FastAPI + PyTorch) + React frontend (Vite + TypeScript).

## Working directory

All code lives under `orchid-osr/`. Commands must be relative to that subtree:

- Backend: `orchid-osr/backend/`
- Frontend: `orchid-osr/frontend/`
- Docker root: `orchid-osr/`

## Prerequisites

- Python 3.11+ (`cd backend && pip install -e ".[dev]"`)
- NVIDIA GPU + CUDA for training (CPU-only inference is supported)
- Docker + NVIDIA Container Toolkit for containerized runs
- Dataset e pesi **non sono nel repo** (rimossi l'8/11/2026, elenco completo in `BACKUP-HF-MANIFEST.txt.gz`, ~20 MB di testo compressi a 1,3 MB: `zgrep -i exaltata BACKUP-HF-MANIFEST.txt.gz`): servono `HF_TOKEN` e `orchid hf download`

## Commands

### Backend

```bash
cd backend

# Training (Studio I-A/B/C, II-A/B; --exp prende il numero dell'esperimento)
python cli.py --cross-validate --exp 1 --model resnet18   # Studio I-A: CE baseline
python cli.py --cross-validate --exp 2 --model resnet18   # Studio I-B: SupCon
python cli.py --cross-validate --exp 3 --model resnet18 --sphor  # Studio I-C: SpHOR
python cli.py --cross-validate --exp 4 --model dinov2_small --sphor  # Studio II-A: GCD (hidden O. garganica)
python cli.py --cross-validate --exp 5 --model dinov2_small --sphor  # Studio II-B: 2 novel

# Single fold / aggregation
python cli.py --cross-validate --exp 1 --model resnet50 --fold 0
python cli.py --cross-validate --exp 1 --model resnet50 --aggregate-only

# NCD pipeline (Novel Class Discovery)
orchid ncd --exp 4 --model dinov2_small         # OSR + clustering (GPU)
orchid ncd --stats                              # calibration stats
orchid ncd --delong                             # DeLong report
orchid ncd-comparison                           # cross-model comparison table

# Embedding extraction for latent space viewer
python cli.py --embeddings --exp 3 --model dinov2_small

# Preprocessing pipeline
python cli.py --crop           # Faster R-CNN flower cropping
python cli.py --split          # Clean train/test split with MD5 dedup
python cli.py --resize         # Letterbox resize
python cli.py --pipeline       # All three in sequence

# API server (default :5000)
python server.py               # Use `&` for background, avoid --reload for stability

# Quality gate (tutti e tre enforced da CI)
pytest -m "not slow"                                         # test
ruff check app/ tests/ scripts/ && ruff format --check app/ tests/ scripts/
mypy app/cli --ignore-missing-imports

# Orchid CLI (Typer) — preferred; python cli.py is deprecated alias
orchid --help
orchid train --help
orchid train --exp 4 --model dinov2_small --sphor
orchid figures --all  # rigenera le figure in documents/_figures/ (§3.5 riproducibile)
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # Vite dev server on :5175, proxies /api /classify /health /auth → :5000
npm run build      # tsc -b && vite build → dist/
npm run lint       # eslint .
npm run test       # vitest
```

### Containerized

```bash
cd orchid-osr
docker compose -f compose.dev.yml up --build        # dev: API + SPA su :5000
docker compose -f compose.training.yml run --rm training \
  python cli.py --cross-validate --exp 1 --model resnet18   # training GPU
```

`compose.yml` è il compose **di produzione** (immagine GHCR pinnata, nessun build):
non si usa in locale.

## Deploy

Build in CI, deploy da questa macchina (la VPS è raggiungibile solo via tailnet).
Il tag è l'unica fonte di verità, come in `personale/actions`:

1. `git tag -a vX.Y.Z` + push del tag → workflow `Publish Docker images`:
   build con il `GITHUB_TOKEN` del repo (nessun secret aggiuntivo), push di
   `ghcr.io/squidslab/orchid-app:<tag>` (+ `:sha-<commit>`), **GitHub Release
   con le note generate dai commit** e prune GHCR che tiene le ultime
   `KEEP_VERSIONS` (3) versioni taggate
2. `scripts/deploy-vps.sh [tag]` (default: ultimo tag) → `compose.yml` +
   `edge.yml` **presi dal tag** via `git archive`, scp in `/srv/orchid`,
   `pull` + `up -d` via SSH, healthcheck in loopback sull'host, registro
   `/srv/orchid/.deployed` (tag + commit + istante), rollback automatico alla
   versione precedente se il servizio non torna sano

Regole di versioning (stesse del repo `actions`: nessun `latest`, nessun
deploy da un working tree sporco):

- si tagga **quando la versione deve andare online**, non a ogni commit di
  lavoro: i tag sono un registro di ciò che è stato pubblicato, non una cronologia
  di sviluppo (quella la fanno i commit)
- semver: patch = fix, minor = funzionalità compatibili, major = cambi che
  rompono il contratto (API, formato esperimenti, layout)
- `:sha-<commit>` esiste per tracciare un'immagine a un commit, ma il deploy
  usa sempre un tag versione: così esiste una Release, un'immagine e una
  voce nel registro
- rollback = tag precedente **con immagine ancora in GHCR**: con
  `KEEP_VERSIONS=3` i due tag precedenti sono sempre disponibili

Perché non un workflow di deploy: l'SSH della VPS è aperto solo su `tailscale0`
e il runner self-hosted `central` è registrato solo su `marcello-russo/actions`.
Dall'account member di `squidslab/OrchID` non si possono creare secret né runner,
quindi il passo di deploy è locale. Se l'owner dell'org aggiunge
`DISPATCH_TOKEN` (o un runner), si torna al dispatcher centrale di
`personale/actions` senza cambiare altro.

Sull'host non esiste il sorgente: `compose.yml` ed `edge.yml` arrivano dal repo
nella stessa transazione del deploy. Unica immagine: il backend FastAPI serve
anche la SPA da `frontend/dist`. Layout e regole: `actions/docs/host-layout.md`.

- Live demo: `https://osr.marrusso.dev`
- Tunnel: `orchid-ncd` (Cloudflare), ingress da `edge.yml`
- Backup: `backup@orchid.timer` salva `users.json`; dataset e esperimenti stanno su HuggingFace

## Architecture

### Key paths

```
orchid-osr/
├── backend/
│   ├── cli.py                        # CLI entry point (shim deprecato)
│   ├── server.py                     # FastAPI entry point
│   ├── app/
│   │   ├── config.py                 # pydantic-settings (reads .env)
│   │   ├── core/models/registry.py   # 6 registered models with per-architecture recipes
│   │   ├── core/models/base.py       # ClassifierModel ABC
│   │   ├── core/kfold/               # Stratified fold splitting, SupCon two-phase, aggregation
│   │   ├── core/losses.py            # SupConLoss
│   │   ├── core/ncd/                 # OSR detectors, GCD clustering, evaluation, DeLong
│   │   ├── core/perfold.py           # per-fold test metrics (source of truth for F1)
│   │   ├── preprocessing/            # Cropper (Faster R-CNN), split, resize
│   │   ├── cli/                      # Arg parsing, experiment runner, definitions
│   │   ├── api/                      # FastAPI routes + deps + schemas
│   │   ├── explainability/           # Captum occlusion heatmaps
│   │   ├── figures/                  # Figure della tesi (300 dpi, phash check)
│   │   └── hf/                       # Sync dataset/modelli su HuggingFace
│   ├── scripts/                      # gcd_*, compute_*, one-shot di tesi, hf_tools/
│   └── experiments/                  # Training outputs (gitignored, su HF)
├── frontend/
│   └── src/
│       ├── features/explorer/        # Experiment charts & KPIs
│       ├── features/classify/        # Image upload + classify hooks
│       ├── features/dataset/         # Dataset gallery + distribution
│       ├── pages/                    # Route pages (lazy-loaded except Dashboard/Experiments)
│       ├── locales/                  # EN + IT translations
│       └── components/               # Layout + Shadcn/ui wrappers
```

### Config sources

- **Backend config**: `app/config.py` (`pydantic-settings`, reads `.env`). Model-specific params (input_size, lr, optimizer, weight_decay) are in the model registry at `app/core/models/registry.py`, not in config.
- **Frontend env**: `frontend/.env` → `VITE_API_URL` (empty = relative paths, proxied in dev e same-origin in prod)
- **Root env**: in locale `.env` (segreti + config); in produzione `.env` non segreto in `/srv/orchid` e segreti in `/srv/orchid/secrets/app.env`
- **Cache su volume**: l'entrypoint esporta `HF_HOME=$DATA_DIR/.hf_cache` e `THUMBNAILS_DIR=$DATA_DIR/.thumbnails`; le miniature della galleria restano così tra un redeploy e l'altro (e un errore transitorio del CDN è un 502, non un 404)
- **CORS**: solo dev server (`CORS_ORIGINS`, CSV, default `localhost:5173/5175`); in produzione la SPA è same-origin e l'auth è `Authorization: Bearer`, quindi niente cookie e niente wildcard

### Container
- L'immagine gira come utente **non privilegiato** `app` (uid 10001): l'entrypoint parte da root solo per sistemare i permessi del volume `/data` e poi si ri-esegue con `setpriv`. Lo stage `training` usa invece `APP_UID=42420` (owner di `/workspace` in quell'immagine).
- Nella dist dentro l'immagine i **PNG non entrano** (`find … -name '*.png' -delete`): sono il sorgente 300 dpi della tesi, la SPA serve i `.webp`. Risparmio ~14 MB per immagine.
- Dall'immagine sono esclusi `backend/data/merged_weights/` (1,7 GB di checkpoint usati solo per gli upload su HF), `users.json`, `jwt_secret.txt` e `orchid.db`: in produzione la fonte è il volume `/data` (`USERS_PATH`, `JWT_SECRET_PATH`). Il Dockerfile copia solo `species_counts.json` e `kfold_indices_seed57.json`.
- `ORCHID_VERSION` arriva nel container da `compose.yml`: `/openapi.json` riporta la versione davvero in esecuzione.


### Image size convention

- ResNet18/50: **256×512** (letterbox, preserves aspect ratio)
- ConvNeXt, DINOv2: **224×224**
- Sizes come from the model registry, not hardcoded

## Critical gotchas

### Data leakage
The preliminary experiments (pre-renumbering) had undetected train/test overlap (images from same source with different crops). **All current experiments use the clean MD5-deduplicated split (2.192 train / 300 test, split_manifest.json, seed 57).** `dataset/train_clean` is an intermediate dir — training reads `dataset/cropped_224x224/` and `dataset/cropped_256x512/`.

### GPU vs CPU
- Training requires CUDA. Lo stage Docker `runtime` è **CPU-only** (wheel PyTorch CPU) ed è quello di produzione; per addestrare si usa lo stage `training` via `compose.training.yml`.
- Su questa macchina (Quadro T2000, 4 GB) il batch è fissato a 8: batch 4 fa collassare la loss contrastiva su DINOv2.

### Quality gate
Tutti enforced dalla CI (`.github/workflows/ci.yml`), `pre-commit` anticipa ruff e mypy in locale:

- backend: `pytest -m "not slow"` con copertura; la soglia (`fail_under`) e le esclusioni stanno **solo** in `pyproject.toml` (la CI non la sovrascrive: averla in due posti l'aveva fatta divergere)
- `ruff check` + `ruff format --check` (max 120 righe) **senza ignore per-file**
- `mypy app/cli` bloccante (era `continue-on-error`: rosso per settimane senza fermare nulla)
- frontend: `eslint`, `check-i18n`, `vitest` (110 test) e `tsc -b && vite build`
- dipendenze: `pip-audit` con allowlist motivata in `backend/audit-ignores.txt`
- immagine: smoke test su `/health` + Trivy (CRITICAL, solo CVE con fix disponibile)
- analisi statica di sicurezza: regole `S` di ruff (la famiglia di bandit) dentro il gate che già esiste, con ignore motivati in `pyproject.toml` e `# noqa: S…` sul singolo falso positivo

### Dipendenze e versioni
`backend/requirements.txt` è un lock **esatto** (Python 3.11, wheel CPU) e `requirements-dev.txt` aggiunge pytest/ruff. Immagine e CI installano esattamente quelle versioni:

```bash
bash backend/scripts/ci/sync_requirements.sh   # dopo aver cambiato pyproject.toml
```

torch/torchvision non sono nel lock perché vanno presi dall'indice CPU: le loro versioni stanno in `backend/torch-pins.txt`, lo stesso file usato da Dockerfile e CI, così un wheel CUDA non può finire nell'immagine di produzione e i due ambienti non divergono. Dependabot apre PR settimanali raggruppate (pip diviso in dev-tools/runtime-patch, npm in react/tooling, actions, docker); le major di torch/transformers sono escluse e si valutano a mano.

### Database / state
C'è **un** database SQLite, `backend/data/orchid.db` (`app/core/db.py`): metadati degli esperimenti (`experiments`, `folds`), utenti, inviti e tentativi di login; contiene un fallback su JSON (`users.json`) per retrocompatibilità. `sync_all()` popola le tabelle scandendo `experiments/` (thread periodico ogni 60s) e l'API legge da lì. Il resto è su filesystem: esperimenti JSON, pesi `.pt`, immagini via proxy HF CDN. L'API carica i modelli su richiesta con cache LRU (`MAX_CACHED_MODELS = 2`).

### Debito tecnico dichiarato
- **Tipizzazione**: `mypy` in `strict` è attivo solo dove il codice è tipizzato (`app/cli`); `app/core`, `app/api`, `preprocessing`, `visualization`, `explainability` hanno `ignore_errors` in `pyproject.toml`. Allargare lo scope è un cantiere, non una patch: finché è così lo si dichiara qui e nei commenti, invece di lasciare intendere che tutto sia tipizzato.
- **transformers 4.x**: 6 avvisi di sicurezza hanno fix solo in 5.x, un major che cambia il caricamento dei modelli DINOv2 e va validato con i pesi su GPU. Allowlist motivata in `backend/audit-ignores.txt`; rischio ridotto perché i pesi arrivano da un repo nostro pinnato, non da input utente.
- **`core/training.py` e `core/supcon.py`**: primitive di training usate da `core/kfold/`, non codice morto — ma convivono con le versioni k-fold e vanno lette insieme.
- **Code scanning (CodeQL)**: non attivo. Su un repository privato di un'organizzazione richiede GitHub Advanced Security, che un account *member* non può abilitare: il workflow falliva solo in fase di upload (`Advanced Security must be enabled`). Un job perennemente rosso è peggio di un buco dichiarato, quindi il file è stato tolto e la copertura statica è affidata a ruff `S`. Se GHAS viene attivato, il workflow si riprende dalla storia (`git show 5c05b23:.github/workflows/codeql.yml`, `security-extended` + `build-mode: none`).


### Server vs CLI
- `python server.py` starts the FastAPI API on `0.0.0.0:5000`
- `python cli.py` runs training/preprocessing offline
- Dev: Vite su `:5175` fa proxy di `/api`, `/classify`, `/health`, `/auth` verso `:5000`
- Prod: una sola immagine, FastAPI serve la SPA da `frontend/dist` (fallback su `index.html`)

### HuggingFace integration
- Dataset: `marcellorusso/orchid-osr-dataset`
- Models: `marcellorusso/orchid-osr-models`
- Sync via CLI (logic in `app/hf/`, tested): `orchid hf download|sync|upload-models|upload-ncd` (requires `HF_TOKEN`)
- All'avvio l'entrypoint sincronizza in background esperimenti e pesi nel volume `/data` (con lock, skip per dimensione)

### Studio I-C (SpHOR) and Studio II-A (SpHOR-NCD)
- Studio I-C adds Spherical Orthogonal Prototypes (SpHOR) on top of SupCon. Prototypes are L2-normalized hypersphere embeddings with repulsion loss.
- Studio II-A is identical but excludes *O. garganica* from training (1.629 known images) — hidden novel class for OSR + GCD evaluation.
- `--sphor` per entrambi.

### Studio II-A results (OSR + GCD)
- **OSR**: best AUROC 0.875 (DINOv2-Base + SpHOR + Ensemble), ConvNeXt-Tiny 0.850.
- **GCD** (Vaze et al. 2022; `orchid ncd`): known-class recovery works (SS-KM ACC old up to 0.93), novel discovery fails with the standard semi-supervised method (ACC new ~0). Only DINOv2-Base discovers the hidden species uninformed (K-means ACC new 0.83±0.04).
- GCD protocol: labeled = train embeddings of 5 known classes (1.629); pool = test (300 = 250 old + 50 garganica), L2-normalized. Metrics: ACC old/new/all (global Hungarian), NMI, ARI.

### Studio II-B (due novel nascoste)
*O. garganica* + *O. majellensis* escluse (4 classi note). L'OSR regge (ConvNeXt-Tiny 0.843) mentre la discovery collassa (SS-KM ACC new 0.12–0.38): rilevamento e scoperta sono problemi distinti.

### SupCon mode (Studio I-B)
Two-phase training: (1) contrastive pretraining with KNN validation using `SupConLoss`, (2) MLP head fine-tuning with cross-entropy on frozen backbone.
