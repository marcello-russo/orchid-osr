# OrchID OSR — Cleanup & Hardening Spec

## Problem Statement

Il progetto OrchID OSR ha accumulato debito tecnico e bug minori durante lo sviluppo della tesi. Dopo aver completato gli esperimenti principali (Exp 6–10) e il deployment su HF Spaces, sono emersi diversi problemi infrastrutturali, di qualità del codice e di sicurezza che devono essere risolti prima della discussione della tesi per garantire che la piattaforma sia presentabile, robusta e riproducibile.

I problemi coinvolgono:
- Infrastruttura di deploy (Docker, CI/CD)
- Backend (thread safety, auth, cache)
- Frontend (hardcoded text residuo, chiavi i18n non usate)
- Testing (copertura mancante per training e GPU)

---

## Goals

1. **Deploy riproducibile e robusto**: `docker compose up` funziona senza errori su CPU (HF Spaces) e GPU (locale)
2. **Backend thread-safe**: Nessuna race condition nella cache dei modelli e nel DB
3. **Auth sicura**: Admin credenziali non hardcoded, password policy minima
4. **Zero debito i18n**: Nessuna chiave morta, nessun hardcoded text
5. **Test significativi**: Copertura per DB, early stopping, fold resume, API experiments

---

## Non-Goals

1. **Refactoring architetturale** — non riscriviamo moduli interi (es. training loop, NCD pipeline). Solo fix mirati.
2. **PyTorch GPU upgrade** — rimandato (rischio breaking change). Solo CPU aggiornato.
3. **Autenticazione OAuth2/SSO** — l'auth serve solo per proteggere la demo, non è un requisito di produzione.
4. **Database PostgreSQL/MongoDB** — SQLite + JSON va bene per questa scala (30 esperimenti). Non introduciamo un DB esterno.
5. **Frontend redesign** — nessun cambiamento di layout o UX, solo fix testuali e i18n.

---

## Categorie di intervento

### A. Deploy & Infrastruttura

| # | Cosa | Priorità | Stato |
|---|------|----------|-------|
| A1 | Docker CPU: `--index-url` già fixato | P0 | ✅ Fatto |
| A2 | `docker-compose`: frontend volume mount già fixato | P0 | ✅ Fatto |
| A3 | `.env.example` documentato già fixato | P0 | ✅ Fatto |
| A4 | `entrypoint.sh`: warning se frontend/dist manca | P1 | ❌ Da fare |
| A5 | `docker-compose`: aggiungere `restart: unless-stopped` | P1 | ❌ Da fare |
| A6 | `sync-hf.yml`: migliorare logging in caso di fallimento | P2 | ❌ Da fare |
| A7 | `.dockerignore`: aggiungere `docs/` (già presente) | — | ✅ Fatto |

### B. Backend

| # | Cosa | Priorità | Stato |
|---|------|----------|-------|
| B1 | `deps.py`: `_model_cache` thread-safe con `threading.Lock` | P0 | ❌ Da fare |
| B2 | `config.py`: admin credenziali con validator (non hardcoded) | P1 | ❌ Da fare |
| B3 | `auth.py`: TOCTOU race su users.json | P1 | ❌ Da fare |
| B4 | `db.py`: sync periodico (già fatto) | P0 | ✅ Fatto |
| B5 | `supcon.py`: early stopping bug (già fixato) | P0 | ✅ Fatto |
| B6 | `kfold/__init__.py`: fold atomic write (già fixato) | P0 | ✅ Fatto |
| B7 | `experiments.py`: logger import (già fixato) | P0 | ✅ Fatto |

### C. Frontend

| # | Cosa | Priorità | Stato |
|---|------|----------|-------|
| C1 | i18n: 360 chiavi sincronizzate, zero emoji | P0 | ✅ Fatto |
| C2 | Hardcoded text: Header, Sidebar, ClassifyPage, AuthPage, ecc. | P0 | ✅ Fatto |
| C3 | Dashboard: link `/about` → `/research` | P0 | ✅ Fatto |
| C4 | Exp6/Exp9 rimossi dal selettore embeddings | P0 | ✅ Fatto |
| C5 | `research.*` keys rimosse (dead) | P0 | ✅ Fatto |
| C6 | `classify.explainability` dead key rimossa | P0 | ✅ Fatto |
| C7 | `ErrorBoundary.tsx` hardcoded text | P2 | ❌ Da fare (class component, complesso) |

### D. Test

| # | Cosa | Priorità | Stato |
|---|------|----------|-------|
| D1 | `test_db_ncd.py`: NCD metrics sync (scritto) | P0 | ✅ Fatto |
| D2 | `test_early_stopping.py`: early stopping regression (scritto) | P0 | ✅ Fatto |
| D3 | `test_fold_resume.py`: fold atomic write e resume (scritto) | P0 | ✅ Fatto |
| D4 | `test_api/test_experiments_list.py`: API esperimenti (scritto) | P0 | ✅ Fatto |
| D5 | Test per `deps.py` thread-safe cache | P1 | ❌ Da fare (mock requerito) |
| D6 | Test per `auth.py` users.json race | P2 | ❌ Da fare (mock pesante) |

---

## Dettaglio Requisiti (P0 rimanenti)

### B1 — `_model_cache` thread-safe

**Problema**: `app/api/deps.py` usa `OrderedDict` per cache dei modelli senza lock. Con richieste concorrenti, `popitem(last=False)` può causare `RuntimeError: OrderedDict mutated during iteration`.

**Soluzione**:
```python
import threading
_model_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

def get_model(...):
    with _cache_lock:
        # lookup, move_to_end, popitem, insert
```

**Criteri di accettazione**:
- [ ] Due richieste concorrenti allo stesso modello non causano crash
- [ ] Un modello non in cache viene caricato una sola volta
- [ ] La cache rispetta il limite di `MAX_CACHED` modelli
- [ ] I test mockano il caricamento del modello e verificano la concorrenza

### A4 — entrypoint.sh warning frontend

**Problema**: Se `frontend/dist` non esiste, Docker parte ma serve pagina vuota.

**Soluzione**: Aggiungere check in `entrypoint.sh` dopo l'avvio:
```bash
if [ ! -d "frontend/dist" ]; then
    echo "⚠️  frontend/dist non trovata. Esegui 'cd frontend && npm run build' prima di docker compose up."
fi
```

### A5 — restart policy

**Problema**: Se il container crasha, Docker non lo riavvia.

**Soluzione**: Aggiungere `restart: unless-stopped` a tutti i servizi in `docker-compose.yml`.

### B2 — Admin credenziali con validator

**Problema**: `config.py` ha `admin_username: str = "admin"` e `admin_password: str = "admin"` come default Python. Se `.env` manca, le credenziali sono debolissime.

**Soluzione**: Aggiungere un `@model_validator` che rifiuta i default:
```python
@model_validator(mode="after")
def _validate_admin(self):
    if self.admin_username == "admin" and self.admin_password == "admin":
        raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD must be changed from defaults!")
    return self
```

### B3 — users.json TOCTOU

**Problema**: `_load_data()` → modifica → `_save_data()` non è atomico.

**Soluzione**: Aggiungere `fcntl.flock` o `portalocker` per lock file-level.

---

## Success Metrics

| Metrica | Target | Come si misura |
|---------|--------|----------------|
| `docker compose up` senza errori | 100% | Esecuzione locale |
| Test passanti | 41/41 | `pytest -v` |
| Thread safety | Nessun crash con 10 richieste concorrenti | Test di carico |
| Chiavi i18n sincronizzate | 360/360 | Script di diff |
| Admin credenziali | Non "admin/admin" | Validator all'avvio |

---

## Open Questions

| # | Domanda | Chi risponde |
|---|---------|-------------|
| 1 | La cache thread-safe va testata con `concurrent.futures` o basta un `threading` test? | Engineering |
| 2 | Il lock su `users.json` con `fcntl.flock` funziona su Windows? (HF Spaces è Linux) | Engineering |
| 3 | Dobbiamo preoccuparci della rate limiting sull'auth? (già implementata) | — |

---

## Timeline

1. **B1 (cache lock)**: ~30 min
2. **A4 + A5 (entrypoint + restart)**: ~10 min
3. **B2 (admin validator)**: ~15 min
4. **B3 (users.json lock)**: ~20 min
5. **D5 (test cache)**: ~20 min
6. **Verifica finale**: ~15 min

**Totale stimato**: ~2 ore
