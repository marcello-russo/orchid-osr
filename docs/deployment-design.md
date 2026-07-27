# Deployment Architecture Design — OrchidOSR NCD

## Architettura

```
Cloudflare Pages ──HTTPS──> orchid-osr.pages.dev
(Frontend React, CDN globale)
      │
      │ VITE_API_URL = https://<random>.trycloudflare.com
      │
      ▼ (browser chiama tunnel URL)
      
Cloudflare Edge ─── HTTPS ─── cloudflared (container)
                                      │ http://backend:7860
                                      ▼
                              Backend container (FastAPI)
                              orchid:latest, rete interna Docker
```

## Componenti

| Componente | Ruolo | Costo |
|---|---|---|
| **Cloudflare Pages** | Serve il frontend React | 0€ |
| **Cloudflare Tunnel (Quick)** | Espone backend via HTTPS con URL temporaneo | 0€ |
| **VPS OVH** | Esegue backend + tunnel via Docker | ~5€/mese |
| **Dokploy** | Gestisce deploy e container | 0€ (self-hosted) |

## Modifiche al codice necessarie

### Dockerfile: rimuovere frontend-builder stage

Il frontend ora va su Cloudflare Pages. Il target `cpu` si semplifica:

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

### Backend main.py: aggiungere CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

La SPA serving (`/{full_path:path}`) resta. Il codice già controlla se `frontend/dist/` esiste — sull'immagine senza frontend non viene montata. Nessun errore.

### Frontend api.ts: già pronto

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? "";
```
`VITE_API_URL` vuoto = sviluppo (proxy Vite). Valorizzato = tunnel URL.

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
    command: tunnel --url http://backend:7860
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

## Quick Tunnel: come funziona

Al primo avvio, `cloudflared tunnel --url http://backend:7860` stampa:

```
2026-07-08T10:00:00Z INF +--------------------------------------------------------------------------------------------+
2026-07-08T10:00:00Z INF |  Your free tunnel has started! Visit it:                                                    |
2026-07-08T10:00:00Z INF |    https://random-string.trycloudflare.com                                                   |
2026-07-08T10:00:00Z INF +--------------------------------------------------------------------------------------------+
```

L'URL `https://random-string.trycloudflare.com` va copiato in Cloudflare Pages come `VITE_API_URL`. Per trovarlo dopo l'avvio, guarda i log del container `tunnel` su Dokploy (o `docker logs <container-id>`).

L'URL cambia SOLO se il container tunnel viene ricreato (deploy). Tra un deploy e l'altro, è stabile.

## Cloudflare Pages setup

| Campo | Valore |
|---|---|
| Framework | None |
| Build command | `cd frontend && npm ci && npm run build` |
| Output | `frontend/dist` |
| Branch | `main` |
| Env variable | `VITE_API_URL` = URL del tunnel |
| Env variable | `NODE_VERSION = 22` (il default di Cloudflare è obsoleto) |

## Costi finali

| Componente | Costo |
|---|---|
| VPS OVH 4GB | ~5€/mese |
| Cloudflare Pages | 0€ |
| Cloudflare Tunnel | 0€ |
| **Totale** | **~5€/mese** |

## Nota: se un giorno volessi un URL stabile

Compra un dominio (`.it` ~1€/anno su Aruba o simili), aggiungilo a Cloudflare, e passa da Quick Tunnel a Named Tunnel con routing DNS. Il passaggio richiede 5 minuti e non cambia nient'altro nello stack.
