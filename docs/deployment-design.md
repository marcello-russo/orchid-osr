# Deployment Architecture Design — OrchID NCD

## Panoramica

```
┌──────────────────────────────────────────────────────────────────┐
│                        Cloudflare                                │
│                                                                  │
│  ┌──────────────────────────────┐  ┌──────────────────────────┐ │
│  │    Cloudflare Pages          │  │  Cloudflare Tunnel Edge  │ │
│  │  orchid-ncd.pages.dev        │  │  <tunnel-uuid>.cfargot...│ │
│  │                              │  │                          │ │
│  │  Frontend React SPA          │  │  Termine SSL/TLS         │ │
│  │  Build: npm run build        │  │  Connette al container   │ │
│  │  Env: VITE_API_URL =         │  │  cloudflared sul VPS     │ │
│  │    https://<tunnel-id>.      │  │                          │ │
│  │    cfargotunnel.com          │  │                          │ │
│  └──────────────────────────────┘  └──────────┬───────────────┘ │
└─────────────────────────────────────────────────────┬────────────┘
                                                      │ tunnel protetto
                                                      │ (outbound-only)
                                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                 VPS OVH (Dokploy)                                │
│                                                                  │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐  │
│  │  Backend Container          │  │  Tunnel Container          │  │
│  │  orchid-ncd:latest          │  │  cloudflare/cloudflared    │  │
│  │                             │  │                            │  │
│  │  FastAPI + PyTorch CPU      │  │  cloudflared tunnel run    │  │
│  │  Porta 7860 (internal)      │  │  --token $TUNNEL_TOKEN     │  │
│  │                             │  │                            │  │
│  │  /api/* → classificazione   │  │  Connessione persistente   │  │
│  │  /classify → inferenza      │  │  a Cloudflare edge         │  │
│  │  /health → monitoring       │  │  (websocket-based)         │  │
│  │                             │  │                            │  │
│  │  Volumi:                    │  │  Rete: orchid-net           │  │
│  │  /data → pesi, dataset, DB  │  │  Dipende da: backend        │  │
│  └─────────────────────────────┘  └────────────────────────────┘  │
│                                                                  │
│  Network: orchid-net (docke内部r network, backend ↔ tunnel via hostname)              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Modifiche al codice necessarie

Prima del deploy, applicare queste modifiche:

### Dockerfile: rimuovere frontend-builder stage

Poiché il frontend è servito da Cloudflare Pages, non serve più buildarlo nell'immagine Docker. Rimuovere dal `Dockerfile` lo stage `frontend-builder` e la riga `COPY --from=frontend-builder`. Il target `cpu` diventa:

```dockerfile
FROM docker.io/python:3.11-slim AS cpu
WORKDIR /workspace/backend
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY backend/pyproject.toml .
RUN sed -i '/"torch",/d; /"torchvision",/d' pyproject.toml && \
    pip install --no-cache-dir --default-timeout=1000 .
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/scripts/ scripts/
COPY backend/data/ data/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENV API_PORT=7860
EXPOSE 7860
CMD ["./entrypoint.sh"]
```

### Backend main.py: aggiungere CORS middleware

Aggiungere in `app/api/main.py`, prima della creazione delle route:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://orchid-ncd.pages.dev",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

La SPA serving (route `/{full_path:path}`) resta dov'è — il codice già controlla se `frontend/dist/` esiste prima di montarla. Sull'immagine Docker senza frontend, non verrà montata. Nessun errore.

### Frontend api.ts: già pronto

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? "";
```

`VITE_API_URL` vuoto = sviluppo (proxy Vite). Valorizzato = produzione (chiamate assolute al tunnel).

## Perché questa è la soluzione perfetta

| Requisito | Come viene soddisfatto |
|---|---|
| **Zero domini a pagamento** | Tunnel named su `cfargotunnel.com` (gratuito, stabile, incluso in ogni account Cloudflare) |
| **HTTPS end-to-end** | Cloudflare Pages (frontend) + Tunnel edge (backend) |
| **Nessuna porta esposta sul VPS** | cloudflared fa solo connessione uscente verso Cloudflare |
| **URL stabile (non random)** | Named tunnel → `<tunnel-uuid>.cfargotunnel.com` permanente |
| **Frontend veloce** | CDN globale di Cloudflare |
| **Backend sicuro** | Container in rete Docker isolata, invisibile dall'esterno |
| **Costo** | 0€ (salvo VPS ~5€/mese) |

## Setup Cloudflare Tunnel (una tantum)

### Prerequisito: Account Cloudflare gratuito

Se non hai un account Cloudflare, crealo su `https://dash.cloudflare.com/sign-up`. Serve solo per gestire il tunnel — non serve nessun dominio acquistato.

### Comandi di setup (da eseguire UNA VOLTA SOLA)

```bash
# 1. Autentica cloudflared con il tuo account Cloudflare
#    Questo apre il browser per il login OAuth.
#    Genera ~/.cloudflared/cert.pem (le credenziali del tuo account)
docker run -it --rm -v ~/.cloudflared:/home/nonroot/.cloudflared \
  cloudflare/cloudflared:latest tunnel login

# 2. Crea un tunnel named
#    Restituisce un UUID tipo "6c3089b0-4a5f-4b8d-8c9a-0e1f2a3b4c5d"
#    Il tunnel è subito accessibile su https://<UUID>.cfargotunnel.com
#    Genera ~/.cloudflared/<UUID>.json (credenziali del tunnel)
docker run -it --rm -v ~/.cloudflared:/home/nonroot/.cloudflared \
  cloudflare/cloudflared:latest tunnel create orchid-ncd

# 3. Estrai il tunnel token (lo userai in Dokploy come TUNNEL_TOKEN)
#    Il token è la stringa che permette a cloudflared di connettersi
#    senza bisogno del file credentials.json
TOKEN=$(docker run -it --rm -v ~/.cloudflared:/home/nonroot/.cloudflared \
  cloudflare/cloudflared:latest tunnel token orchid-ncd)
echo "TUNNEL_TOKEN=$TOKEN"

# 4. (Opzionale) Verifica l'URL del tunnel
#    Puoi trovarlo nella dashboard Cloudflare → Zero Trust → Tunnels
#    Oppure: cat ~/.cloudflared/<UUID>.json | grep tunnel_id
#    URL: https://<UUID>.cfargotunnel.com
```

**Output**: `TUNNEL_TOKEN` da copiare in Dokploy. URL stabile: `https://<tunnel-uuid>.cfargotunnel.com`

## docker-compose.yml (per Dokploy)

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: cpu
    restart: unless-stopped
    expose:
      - "7860"
    environment:
      - API_PORT=7860
      - HF_TOKEN=${HF_TOKEN}
      - ADMIN_USERNAME=${ADMIN_USERNAME}
      - ADMIN_PASSWORD=${ADMIN_PASSWORD}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - orchid-data:/data
    networks:
      - orchid-net
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  tunnel:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --no-autoupdate run --token ${TUNNEL_TOKEN}
    environment:
      - TUNNEL_TOKEN=${TUNNEL_TOKEN}
      - TUNNEL_METRICS=localhost:9090
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - orchid-net

volumes:
  orchid-data:

networks:
  orchid-net:
    driver: bridge
```

## Frontend — Cloudflare Pages

### Build configuration

| Campo | Valore |
|---|---|
| Framework preset | None |
| Build command | `cd frontend && npm ci && npm run build` |
| Build output directory | `frontend/dist` |
| Root directory | (lasciare vuoto) |

### Environment variable

```
VITE_API_URL = https://<tunnel-uuid>.cfargotunnel.com
```

### GitHub connection

`https://github.com/squidslab/OrchID` → branch `main`

### Cosa succede nel codice

In `frontend/src/lib/api.ts`:

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? "";
// → "https://<tunnel-uuid>.cfargotunnel.com"
// Tutte le chiamate diventano:
// fetch("https://<tunnel-uuid>.cfargotunnel.com/api/experiments")
```

## CI/CD Flow Completo

```
── Push su main ──
        │
        ▼
┌─────────────────────────┐
│  GitHub Actions: CI      │
│  • ruff check + format   │
│  • pytest                │
│  • tsc + eslint          │
│  • docker build (syntax) │
└─────────┬───────────────┘
          │ pass
          ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│  Cloudflare Pages        │     │  Dokploy (VPS)              │
│  • npm ci                │     │  • git pull da GitHub       │
│  • npm run build         │     │  • docker compose build     │
│  • deploy a .pages.dev   │     │  • docker compose up -d     │
│  • VITE_API_URL iniettato│     │  • backend + tunnel partono │
└─────────────────────────┘     └─────────────────────────────┘
          │                              │
          ▼                              ▼
   orchid-ncd.pages.dev          <tunnel-uuid>.cfargotunnel.com
   (CDN globale)                 → backend:7860 (rete interna)
```

## Verifica Funzionamento

```bash
# 1. Backend raggiungibile via tunnel
curl https://<tunnel-uuid>.cfargotunnel.com/health
# → {"status":"ok"}

# 2. Frontend risponde
curl https://orchid-ncd.pages.dev
# → HTML della SPA

# 3. Classifica via browser
# Apri https://orchid-ncd.pages.dev → carica foto → API chiama tunnel → backend
```

## Costi Finali

| Componente | Costo/mese | Note |
|---|---|---|
| VPS OVH 4GB | ~5€ | CPU + RAM per PyTorch CPU |
| Cloudflare Pages | 0€ | Build gratuite, CDN globale |
| Cloudflare Tunnel | 0€ | Banda illimitata, incluso in ogni account Cloudflare |
| Domini | 0€ | `cfargotunnel.com` e `pages.dev` sono gratis di Cloudflare |
| **Totale** | **~5€** | Solo il costo del VPS |
