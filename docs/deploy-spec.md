# Deploy Fix Spec

## Problemi identificati

### 1. Frontend non visibile in Docker
**Causa**: `docker-compose.yml` backend service monta solo codice backend via volume, non il frontend buildato. La SPA è buildata nell'immagine Docker al momento del `docker build`. Ogni modifica al frontend richiede `docker compose build`, che è lento (~5 min).

**Soluzione**: Montare `./frontend/dist` nel container:
```yaml
volumes:
  - ./frontend/dist:/workspace/backend/frontend/dist
```
Workflow: `npm run build && docker compose up -d`

### 2. URL CPU PyTorch wheels obsoleto
**Causa**: PyTorch ha cambiato l'URL per i wheel CPU. `https://download.pytorch.org/whl/cpu` non è più supportato per versioni recenti.

**Soluzione**: Aggiornare a:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. GPU PyTorch 2.1.0 obsoleto
**Causa**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` è di ottobre 2023. La codebase usa CUDA 13.0 (da `nvidia-smi`).

**Rischio**: Aggiornare PyTorch potrebbe rompere compatibilità con dipendenze (torchvision, torchaudio).
**Azione rimandata**: Non tocchiamo finché non serve.

### 4. .env non documentato
**Soluzione**: Aggiungere commenti a `.env.example` con ogni variabile spiegata.

## Ordine di esecuzione
1. `.env.example` — zero rischi
2. `Dockerfile` CPU wheel URL — testabile immediatamente
3. `docker-compose.yml` frontend volume mount — testabile immediatamente
