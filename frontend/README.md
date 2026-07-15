# 🌸 OrchID — Frontend

React SPA per la classificazione e l'esplorazione dei risultati delle orchidee *Ophrys*.

## Tech Stack

React 19, TypeScript, Vite, Tailwind CSS v4, Shadcn/ui, React Query, Recharts

## Quick Start

```bash
npm install
npm run dev     # → http://localhost:5175
```

Richiede il backend OrchID su porta 5000 (proxy API in `vite.config.ts`).

## Build per produzione

```bash
npm run build   # → frontend/dist/
```

## Pagine

| Route | Pagina |
|-------|--------|
| `/` | Dashboard |
| `/classify` | Classificazione immagini |
| `/explorer` | Explorer esperimenti |
| `/research` | Riassunto della tesi |
| `/dataset` | Galleria dataset |
| `/embeddings` | Spazio latente 2D/3D |
