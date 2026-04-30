[🇬🇧 English](README.md)

# 🌸 OrchID-NCD — Orchid Representation Learning & Novel Class Discovery

> Classificazione visiva ultra-fine-grained e representation learning delle specie di orchidee *Ophrys* tramite self-supervised learning e tecniche contrastive.

---

## 🎯 Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| **F1 — Classificazione** | Baseline ResNet18/50, ConvNeXt-Tiny/Small, DINOv2+LoRA, k-fold CV su clean split, recipe per-architettura, model registry | ✅ Completata |
| **F1b — Explorer** | Dashboard React SPA, classificazione upload, explainability, statistiche specie, selettore modello, dataset explorer con galleria live, tooltip esplicativi, pagina ricerca | ✅ Completata |
| **F2 — Spazio Vettoriale** | Supervised Contrastive Learning (SupCon) per costruire uno spazio vettoriale strutturato | 🔜 Prossima |
| **F3 — Retrieval & Similarità** | Ricerca nearest-neighbor, clustering specie, detection classi sconosciute | 📋 Pianificata |

---

## 🌿 Specie

| | Specie |
|---|--------|
| 🌸 | *Ophrys exaltata* |
| 🌺 | *Ophrys garganica* |
| 🌻 | *Ophrys incubacea* |
| 🌼 | *Ophrys majellensis* |
| 🌷 | *Ophrys sphegodes* |
| 🌱 | *Ophrys sphegodes* (Palena) |

---

## 📊 Risultati

I risultati della Serie 1 (exp6 clean split) saranno pubblicati dopo aver ri-eseguito gli esperimenti con un cropper adeguato per il rilevamento del fiore. Gli esperimenti precedenti (exp 1–5) sono stati invalidati da data leakage, e l'exp 6 deve essere rifatto con un cropper Faster R-CNN per una rilevazione ottimale del fiore.

<details>
<summary>Recipe di training per-architettura</summary>

| Modello | Ottimizzatore | LR | Weight Decay | Motivazione |
|---------|--------------|----|-------------|-------------|
| ResNet18/50 | Adam | 0.001 | 0 | Fine-tuning standard, BatchNorm fornisce regolarizzazione implicita |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 | Recipe del paper ConvNeXt — LayerNorm sensibile a gradienti grandi |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 | Leggera regolarizzazione su adapter LoRA (~0.5% parametri trainable) |

Comuni: batch=8 (effettivo 32 via gradient accumulation ×4), patience=15, ReduceLROnPlateau, max 100 epoche.

</details>

---

## 🚀 Avvio Rapido

### Prerequisiti

- Linux (testato su Ubuntu/Pop!_OS)
- Podman (o Docker) + NVIDIA GPU + Container Toolkit

### Esecuzione

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/resnet18

# Avvia il container (backend + frontend su porta 5000)
./podman-start.sh
```

### Sviluppo (con hot-reload)

```bash
# Terminale 1 — Backend
cd backend
python server.py --reload

# Terminale 2 — Frontend (Vite dev server su porta 5173)
cd frontend
npm install
npm run dev
```

Frontend su http://localhost:5173, proxy API → backend :5000.

### CLI

```bash
cd backend

python cli.py --status                                          # Verifica stato

# K-fold cross-validation su clean split (exp6)
python cli.py --cross-validate --exp 6 --model resnet18 \
  --epochs 100 --patience 15 --accumulation 4

python cli.py --cross-validate --exp 6 --model convnext_tiny \
  --epochs 100 --patience 15 --accumulation 4

# Fold singolo (per resume)
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0 \
  --epochs 100 --patience 15 --accumulation 4

# Aggregazione fold completati
python cli.py --cross-validate --exp 6 --model resnet50 \
  --epochs 100 --patience 15 --accumulation 4 --aggregate-only
```

---

## 🏗️ Struttura Progetto

```
resnet18/
├── backend/                      # Codice applicazione
│   ├── app/                      # Codice sorgente (modulare)
│   │   ├── core/                 # Moduli ML core
│   │   │   ├── models/           # Multi-model registry + classificatori (ResNet, ConvNeXt, DINOv2)
│   │   │   ├── training.py       # Loop di training
│   │   │   ├── evaluation.py     # Metriche e valutazione
│   │   │   └── kfold/            # K-fold cross-validation
│   │   ├── cli/                  # Interfaccia CLI
│   │   ├── api/                  # Endpoint FastAPI
│   │   │   └── routes/           # experiments, classify, species, dataset
│   │   ├── preprocessing/        # Preprocessing dati
│   │   ├── explainability/       # Captum explainability
│   │   └── visualization/        # Generazione grafici
│   ├── experiments/              # Risultati esperimenti
│   ├── cli.py                    # Entry point CLI
│   ├── server.py                 # Entry point API server
│   └── requirements.txt
├── frontend/                     # React SPA
│   ├── src/
│   │   ├── components/           # Layout + Shadcn/ui componenti
│   │   ├── features/
│   │   │   ├── explorer/         # Grafici, hook, config esperimenti
│   │   │   ├── classify/         # Hook classificazione
│   │   │   └── dataset/          # Galleria dataset explorer e grafici
│   │   ├── pages/                # Dashboard, Experiments, Detail, Classify, Dataset, Ricerca
│   │   ├── stores/               # Zustand (tema, sidebar)
│   │   └── lib/                  # Client API, React Query, tipi
│   └── package.json
├── Dockerfile                    # Multi-stage (Node.js build + PyTorch)
├── podman-start.sh
├── entrypoint.sh
├── docs/
└── assets/
```

---

## 🔧 Stack Tecnologico

| | Componente |
|---|------------|
| 🧠 | PyTorch 2.6.0 + CUDA 12.4 |
| 🏛️ | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base + LoRA (pre-addestrati ImageNet) |
| 🔍 | Captum (explainability) |
| 🖥️ | React 19 + TypeScript + Vite + Tailwind v4 + Shadcn/ui |
| 📊 | Recharts (grafici interattivi), React Query, Zustand |
| 📦 | Podman + NVIDIA Container Toolkit |

---

## 📡 API

```bash
cd backend
python server.py                    # Avvia API su porta 5000
```

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify/upload` | POST | Classifica immagine caricata (multipart, `model_id` opzionale) |
| `/classify/explain` | POST | Heatmap occlusione per immagine caricata |
| `/classify/models` | GET | Lista modelli addestrati con metriche |
| `/api/experiments` | GET | Lista tutti gli esperimenti |
| `/api/experiments/{id}` | GET | Dettaglio esperimento con fold |
| `/api/experiments/{id}/folds/{n}` | GET | Dati singolo fold (0-based) |
| `/api/species` | GET | Lista specie con conteggio immagini |
| `/api/dataset/classes` | GET | Classi dataset con conteggi train/test |
| `/api/dataset/classes/{name}/images` | GET | Lista immagini paginata per classe |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Serve immagine dataset |
| `/docs` | GET | Documentazione Swagger UI |

---

## 📄 Licenza

Vedi [LICENSE](LICENSE).
