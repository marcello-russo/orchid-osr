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
   `/srv/orchid/.deployed` (`tag`, `commit`, `ts` **e `smoke`**: `pending` appena
   il servizio è su, poi `ok`, `failed` o `skipped` — così il file dice sempre
   cosa gira, e separatamente se qualcuno l'ha verificato), rollback automatico
   alla versione precedente se il servizio non torna sano
3. `scripts/smoke-live.py` **dentro il deploy** (stesso tag, eseguito sull'host
   via `sudo -n`): oltre agli status guarda i **byte** delle immagini
   (magic bytes JPEG/WebP, non solo il 200: un errore mascherato è spesso un 200
   con `content-type: application/json`), i totali del dataset e la versione
   dichiarata in `/openapi.json` — così un container vecchio con il tag nuovo in
   `.env` non passa per un deploy. Si può lanciare a mano:
   `scripts/smoke-live.py --url http://127.0.0.1:8000 --env-file /srv/orchid/secrets/app.env --expect-version vX.Y.Z`
   (in locale contro la produzione autentica solo se hai le credenziali; senza,
   salta i controlli autenticati e lo dice). Uno smoke fallito **non** fa
   rollback da solo (un CDN lento non è una regressione): il deploy esce 1, il
   registro scrive `failed`, e la decisione è umana. `tests/test_smoke_live.py`
   legge i due script veri e cade se lo smoke smette di guardare i byte, se il
   deploy smette di chiamarlo o se un tag vecchio senza smoke viene saltato in
   silenzio

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
- se un tag va rifatto (build rotta, commit sbagliato), GitHub converte la
  Release del tag cancellato in **bozza**: il workflow la ripubblica da solo
  (`gh release edit --draft=false`) con le note del commit nuovo, e
  `tests/test_ci_release.py` tiene la guardia. L'immagine, invece, va rifatta a
  mano: `docker-publish.yml` si può rilanciare con `workflow_dispatch` sul tag
  (`gh workflow run docker-publish.yml -f tag=vX.Y.Z`)

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
- Segreti: `/srv/orchid/secrets/app.env` (mode 600), passati via `env_file`.
  Si ruotano con `scripts/rotate-app-secrets.py --env-file /srv/orchid/secrets/app.env`
  (`--rotate jwt|superadmin`, `--dry-run`), che riscrive il file in modo atomico
  preservando mode e owner e **non stampa mai i valori**. `JWT_SECRET` nuovo =
  tutte le sessioni e i guest pass in corso invalidati (è il modo di buttare
  fuori un token compromesso); `SUPERADMIN_PASSWORD` nuova entra in vigore al
  riavvio, quando `ensure_superadmin_exists()` allinea l'hash nel DB. `HF_TOKEN`
  non si genera in locale: si crea sull'account HuggingFace e lo script lo
  rifiuta con la spiegazione. `tests/test_rotate_secrets.py` copre duplicati,
  mode, dry-run e il divieto di stampare i segreti
- Backup: `backup@orchid.timer` salva **`orchid.db`** con l'API di backup SQLite (`type: sqlite`: snapshot coerente anche con WAL, `--verify` con `integrity_check`), non più `users.json`; dataset e esperimenti stanno su HuggingFace. Il job e il suo `backup.json` **vivono sull'host**, fuori dal repo: la voce va cambiata nello stesso deploy che porta il DB in produzione (v0.4.0), altrimenti il timer continua a salvare un file congelato e i dati auth nuovi non finiscono in nessun backup

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
│   │   ├── api/serializers/          # JSON su disco → payload (fold, esperimenti)
│   │   ├── explainability/           # Captum occlusion heatmaps
│   │   ├── figures/                  # Figure della tesi (300 dpi, phash check)
│   │   └── hf/                       # Sync dataset/modelli su HuggingFace
│   ├── scripts/                      # gcd_*, compute_*, one-shot di tesi, hf_tools/
│   └── experiments/                  # Training outputs (gitignored, su HF)
├── frontend/
│   └── src/
│       ├── features/explorer/        # Experiment charts & KPIs
│       ├── features/explorer/detail/ # Tab del dettaglio (grafici, baseline, NCD, log)
│       ├── features/classify/        # Image upload + classify hooks
│       ├── features/dataset/         # Dataset gallery + distribution
│       ├── features/research/        # Tesi: una sezione per esperimento + primitive di prosa
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
- L'immagine gira come utente **non privilegiato** `app` (uid 10001): l'entrypoint parte da root solo per sistemare i permessi del volume `/data` e poi si ri-esegue con `setpriv`. Lo stage `training` usa invece `APP_UID=42420` (owner di `/workspace` in quell'immagine). L'immagine **non dichiara `USER`**: è il processo del server a girare come `app` (PID 1 = 10001), mentre `docker exec` e un `--entrypoint` sovrascritto girano da root — il chown del volume non si può fare altrimenti se non con un init separato.
- L'entrypoint inoltra SIGTERM/SIGINT a uvicorn: senza, `bash` come PID 1 non passa il segnale, `docker stop` aspetta i 30s di grace e poi uccide il processo (exit 137) troncando le richieste in volo.
- Nella dist dentro l'immagine i **PNG non entrano** (`find … -name '*.png' -delete`): sono il sorgente 300 dpi della tesi, la SPA serve i `.webp`. Risparmio ~14 MB per immagine.
- Dall'immagine sono esclusi `backend/data/merged_weights/` (1,7 GB di checkpoint usati solo per gli upload su HF), `users.json` (artefatto legacy, serve solo all'import una-tantum), `jwt_secret.txt` e `orchid.db`: in produzione la fonte è il volume `/data` (`JWT_SECRET_PATH`, `DB_PATH`, più `USERS_PATH` per il solo import). Il Dockerfile copia solo `species_counts.json` e `kfold_indices_seed57.json`.
- `dataset` **non** esiste nel layer: l'entrypoint lo collega a `/data/dataset` (e lo fa solo se non è già un link o se la directory è vuota — un dataset vero in locale non viene mai toccato). Con una directory già presente, `ln -sfn` avrebbe creato `dataset/dataset` lasciando `dataset_dir` vuota.
- `ORCHID_VERSION` arriva nel container da `compose.yml`: `/openapi.json` riporta la versione davvero in esecuzione.
- `index.html` esce con `Cache-Control: no-cache` e un 304 senza corpo quando la copia è ancora buona: è la mappa degli asset, i cui nomi cambiano a ogni build — un indice tenuto in cache dopo un deploy chiederebbe `/assets/index-<hash-vecchio>.js`, che non esiste, e la pagina resterebbe bianca senza nessun errore lato server. La rivalidazione accetta ETag **e** `Last-Modified`, perché Cloudflare sull'HTML toglie l'ETag (verificato: `/favicon.svg` lo conserva, `/` no) e la data è l'unico validatore che sopravvive alla riva. Gli altri file non hanno un `Cache-Control` nostro: `/assets` lo cacha Cloudflare (4h) e i nomi sono hashati, ma `assets/images/` no, quindi niente `immutable`.


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
- `mypy app/cli` bloccante e **reale**: le uniche deroghe sono sulle annotazioni mancanti (`disallow_untyped_defs/calls/incomplete_defs`), nessun modulo dello scope è coperto da `ignore_errors`. Il test `tests/test_ci_mypy.py` confronta lo scope della CI con gli override in `pyproject.toml`: se qualcuno li rimette, il test cade. (Prima era un gate vuoto: tutti i moduli avevano `ignore_errors` e passavano solo i due `__init__.py`.) Il job installa il **lock intero** alla versione esatta di `requirements.txt` — con torch/torchvision presi prima dai pin CPU (`torch-pins.txt`, `--no-deps`): senza fastapi/starlette/typer ogni decoratore vale `Any` e `strict` segnala `untyped-decorator` solo sul runner (è successo con la v0.4.0: verde in locale, 62 errori in CI); senza il pin di torch, che accelerate/peft/captum tirano dentro, arrivavano 3,2 GB di wheel `nvidia-*` e una versione scelta da PyPI il giorno della run. Un gate il cui esito dipende da cosa è installato non è un gate, e `tests/test_ci_mypy.py` sorveglia entrambe le cose.
- frontend: `eslint`, `check-i18n`, `vitest` (110 test) e `tsc -b && vite build`
- dipendenze: `pip-audit` su `requirements.txt` **e** `torch-pins.txt` (torch non passa da Dependabot e Trivy vede solo i CRITICAL), con allowlist motivata in `backend/audit-ignores.txt`; gli strumenti dei gate (mypy, pip-audit) sono pinnati nella CI — un gate il cui esito dipende dalla release di ieri non è un gate
- immagine: smoke test su `/health` **e `GET /`** con il volume `/data` montato (esercita il ramo root→app dell'entrypoint e accorgersi di una dist rotta) + Trivy (CRITICAL, solo CVE con fix disponibile)
- analisi statica di sicurezza: regole `S` di ruff (la famiglia di bandit) dentro il gate che già esiste, con ignore motivati in `pyproject.toml` e `# noqa: S…` sul singolo falso positivo

### Dipendenze e versioni
`backend/requirements.txt` è un lock **esatto** (Python 3.11, wheel CPU) e `requirements-dev.txt` aggiunge pytest/ruff. Immagine e CI installano esattamente quelle versioni:

```bash
bash backend/scripts/ci/sync_requirements.sh   # dopo aver cambiato pyproject.toml
```

torch/torchvision non sono nel lock perché vanno presi dall'indice CPU: le loro versioni stanno in `backend/torch-pins.txt`, lo stesso file usato da Dockerfile e CI, così un wheel CUDA non può finire nell'immagine di produzione e i due ambienti non divergono. Dependabot apre PR settimanali raggruppate (pip diviso in dev-tools/runtime-patch, npm in react/tooling, actions, docker); le major sono tutte escluse, e così ruff (una minor cambia il formatter), tokenizers (lo vincola transformers 4.x) e `pytorch/pytorch` (runtime CUDA): si aggiornano a mano.

Lo stage `training` è un altro ambiente per necessità (Python 3.10, CUDA 12.1, `constraints-training.txt` con numpy<2): i suoi vincoli entrano nella **stessa** risoluzione di `pyproject.toml`, non in un secondo `pip install` che declassa un pacchetto dopo che pip ha già deciso. Il workflow `Training image` lo costruisce una volta a settimana (e sulle PR che toccano Dockerfile/constraint) e importa torch/transformers/peft: senza, un constraints rotto si scopriva solo al primo addestramento, con la GPU accesa.

### Database / state
C'è **un** database SQLite, `backend/data/orchid.db` (`app/core/db.py`), unica fonte di verità per: metadati degli esperimenti (`experiments`, `folds`), utenti, inviti, passi ospite (`users`, `invites`, `guest_passes`) e tentativi di login (`login_attempts`). Le scritture auth passano da `app/core/auth_store.py` con `BEGIN IMMEDIATE` e connessioni brevi: un invito monouso non può essere consumato due volte da due richieste concorrenti. Lo schema è creato una volta per processo (`_SCHEMA_READY`), non a ogni connessione.

Un invito creato dall'API scade **sempre**: 72 ore di default, 30 giorni al massimo (`MAX_INVITE_HOURS`); un passo di difesa (link multi-uso per la discussione) parte da 72 ore (`DEFAULT_DEFENSE_HOURS`), un passo ospite da 24 con cap a 7 giorni (`MAX_GUEST_PASS_HOURS`). "Senza scadenza" esiste solo come `--invite-hours 0` sul CLI, cioè un'azione locale e consapevole, e la UI marca come *scaduto* un codice che non è più spendibile invece di lasciarlo "disponibile". Anche gli inviti importati dal `users.json` legacy ricevono una scadenza (30 giorni) se non l'avevano e non sono già stati usati.

In container `DB_PATH` punta a `/data/orchid.db` (volume nominato): senza, il lockout anti brute-force si azzererebbe a ogni redeploy. `users.json` è **solo** l'artefatto dell'import una-tantum da v0.3.x: l'import è marcato in `auth_meta.legacy_json_imported` e il file resta sul volume come rete di rollback, ma non è più letto né scritto — quindi dopo la migrazione è **stale**: un utente creato dopo il passaggio a SQLite esiste solo in `orchid.db`, e un rollback a v0.3.x è sicuro solo nella finestra di manutenzione (prima cioè che qualcuno si registri). Se il file manca all'avvio l'import **non** viene marcato: il rimedio "rimetto il file e riavvio" continua a funzionare (il marker `absent` scritto dalla v0.4.0 vale come non-import, così anche i DB che l'hanno già visto si possono recuperare). `sync_all()` popola le tabelle degli esperimenti scandendo `experiments/` (thread periodico ogni 60s). Il resto è su filesystem: esperimenti JSON, pesi `.pt`, immagini via proxy HF CDN. L'API carica i modelli su richiesta con cache LRU (`MAX_CACHED_MODELS = 2`).

### Debito tecnico dichiarato
- **Tipizzazione**: `mypy` in `strict` copre `app/cli` (e quindi `app/hf`, `app/figures` importati), `app/api`, `app/preprocessing`, `app/config.py` e i tre moduli auth/DB (`app/core/auth.py`, `app/core/auth_store.py`, `app/core/db.py`). Restano sotto `ignore_errors` in `pyproject.toml` il resto di `app/core` (modelli, kfold, ncd), `visualization`, `explainability` e `scripts`. Allargare lo scope è un cantiere, non una patch: finché è così lo si dichiara qui, invece di lasciare intendere che tutto sia tipizzato. Lo scope dichiarato è reale: `tests/test_ci_mypy.py` replica la precedenza degli override di mypy e fallisce se uno di questi percorsi torna coperto da `ignore_errors = true`.
- **transformers 4.x**: 6 avvisi di sicurezza hanno fix solo in 5.x, un major che cambia il caricamento dei modelli DINOv2 e va validato con i pesi su GPU. Allowlist motivata in `backend/audit-ignores.txt`; rischio ridotto perché i pesi arrivano da un repo nostro pinnato, non da input utente.
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
