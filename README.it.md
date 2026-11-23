<p align="center">
  <img src="assets/images/banner.webp" alt="OrchID OSR Banner" width="100%"/>
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white" alt="SQLite"/>
  <br/>
  <em>Classificazione ultra-fine-grained e riconoscimento open-set per orchidee <b>Ophrys</b> tramite apprendimento contrastivo profondo</em>
</p>

<p align="center">
  <strong>Tesi magistrale — Università degli Studi di Napoli Federico II (DIETI)</strong>
  <br/>
  <a href="https://github.com/squidslab/OrchID">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">Articolo SpHOR</a>
</p>

---

## Abstract

Sei specie di *Ophrys* morfologicamente simili, una classe novel nascosta (*O. garganica*), 2.192 immagini di training + 300 di test. Confrontiamo 6 architetture (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 3 paradigmi di apprendimento (CE, SupCon, SpHOR). **Campione closed-set:** ResNet‑50 + CE (0,884 F1 test). **Campione open-set:** DINOv2‑Base + SpHOR (AUROC 0,875 sulla novel class nascosta). L'open-set recognition rimane dipendente dall'architettura e lontano dall'essere risolto in questo dominio ultra-fine-grained.

---

## Risultati (dalla tesi)

### Studio I-A (Exp 1) — Baseline Cross-Entropy

| Modello | F1 Test |
|---------|:-------:|
| **ResNet‑50** | **0,884** |
| ResNet‑18 | 0,847 |
| DINOv2‑Base | 0,764 |
| DINOv2‑Small | 0,695 |
| ConvNeXt‑Tiny | 0,686 |
| ConvNeXt‑Small | 0,513 |

### Studio I-B (Exp 2) — Supervised Contrastive Learning

| Modello | F1 Test |
|---------|:-------:|
| **DINOv2‑Base** | **0,808** |
| DINOv2‑Small | 0,710 |
| ConvNeXt‑Tiny | 0,573 |
| ConvNeXt‑Small | 0,450 |
| ResNet‑50 | 0,355 |
| ResNet‑18 | 0,287 |

### Studio I-C (Exp 3) — SpHOR (Open-World)

| Modello | F1 Test |
|---------|:-------:|
| **DINOv2‑Base** | **0,833** |
| DINOv2‑Small | 0,772 |
| ResNet‑50 | 0,719 |
| ConvNeXt‑Tiny | 0,710 |
| ConvNeXt‑Small | 0,677 |
| ResNet‑18 | 0,442 |

### Studio II-A (Exp 4) — NCD (Open-Set Recognition)

| Modello | F1 Test | AUROC (miglior rivelatore) |
|---------|:-------:|:--------------------------:|
| DINOv2‑Base | 0,928 | **0,875** (Ensemble) |
| **ConvNeXt‑Small** | **0,930** | 0,839 (Ensemble) |
| DINOv2‑Small | 0,924 | 0,758 (KNN) |
| ConvNeXt‑Tiny | 0,929 | 0,850 (Ensemble) |
| ResNet‑50 | 0,784 | 0,776 (KNN) |
| ResNet‑18 | 0,525 | 0,507 (Mahalanobis) |

*Studio II-A (Exp 4) esclude O. garganica dal training, lasciandola come novel class nascosta per la valutazione open-set.*

### Studio II-B (Exp 5) — NCD con 2 Novel Nascoste

| Modello | F1 Test (4 cls) | AUROC (Ensemble) |
|---------|:---------------:|:----------------:|
| **DINOv2‑Base** | **0,930** | 0,805 |
| ConvNeXt‑Tiny | 0,922 | **0,843** |
| DINOv2‑Small | 0,923 | 0,734 |
| ConvNeXt‑Small | 0,916 | 0,718 |
| ResNet‑50 | 0,878 | 0,634 |
| ResNet‑18 | 0,810 | 0,666 |

*Studio II-B (Exp 5) esclude O. garganica + O. majellensis (4 classi note). OSR scala (ConvNeXt-Tiny 0,843); la GCD discovery crolla (SS-KMeans ACC new 0,12–0,38, NMI 0,92–0,95 stabile). Vedi tesi Sec. 4.5.*

---

## Risultati Principali

1. **Campione closed-set:** ResNet‑50 + CE (0,884 F1 test) — migliore accuratezza complessiva; miglior contrastivo DINOv2‑Base + SupCon (0,808)
2. **Campione open-set:** DINOv2‑Base + SpHOR (AUROC 0,875 sulla novel class nascosta) — migliore OSR in ultra-FGVC
3. **OSR dipendente dall'architettura:** l'AUROC col miglior rivelatore varia da 0,507 (ResNet‑18) a 0,875 (DINOv2‑Base); le ResNet crollano in SupCon (F1 0,287/0,355) — ipotesi BatchNorm × batch 8, senza ablazione
4. **GCD divisa in due:** SS-KMeans ritrova le note (ACC old fino a 0,93) ma la scoperta fallisce (ACC new ≈ 0, eccezione ResNet‑50 0,44); il K-means non informato isola la novel solo su DINOv2‑Base (0,83 ± 0,03)
5. **Significativo dove conta:** DeLong KNN contro Energy su DINOv2-Base +0,110, p = 0,0015

---

## Avvio Rapido

```bash
# Backend
cd backend
pip install -e ".[dev]"
python server.py            # → http://localhost:5000

# Frontend
cd frontend
npm install
npm run dev                 # → http://localhost:5175
```

### Training

```bash
cd backend
python cli.py --cross-validate --exp 1 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 2 --model resnet18    # SupCon
python cli.py --cross-validate --exp 3 --model resnet18 --sphor   # SpHOR
python cli.py --cross-validate --exp 4 --model dinov2_small --sphor   # NCD (novel class nascosta)
python cli.py --ncd --exp 4 --model dinov2_small                  # valutazione OSR + clustering
python cli.py --cross-validate --exp 3 --model resnet18 --novel garganica,majellensis  # generico: 2 novel override
```

### Orchid CLI (Typer)

```bash
orchid --help
orchid train --help
orchid train --exp 4 --model dinov2_small --sphor
orchid figures --all  # rigenera le figure in documents/_figures/ (§3.5 riproducibile)
```

> `python cli.py` rimane disponibile come alias deprecato di `orchid`.

Vedi [AGENTS.md](AGENTS.md) per tutti i comandi.

---

## Dataset

| Specie | Train | Test | Novel |
|--------|:-----:|:----:|:-----:|
| O. exaltata | 490 | 50 | |
| O. garganica | 563 | 50 | |
| O. incubacea | 352 | 50 | |
| O. majellensis | 152 | 50 | |
| O. sphegodes | 425 | 50 | |
| O. sphegodes Palena | 210 | 50 | |

*Conteggi train post deduplicazione MD5 (2.192 totali). O. garganica (riga precedente) è usata come novel class nascosta in Exp 4 (Studio II-A): esclusa dal training, riservata solo al test.*

Download: `snapshot_download("marcellorusso/orchid-osr-dataset", repo_type="dataset")`

---

## Deploy

Una sola immagine (`app`) su GHCR, costruita in CI e deployata con uno script:

- **Tag `v*`** → workflow GitHub → `ghcr.io/squidslab/orchid-app:<tag>` (con il `GITHUB_TOKEN` del repo), **GitHub Release con le note generate dai commit** e prune GHCR che tiene le 3 versioni più recenti
- **`scripts/deploy-vps.sh [tag]`** (default: ultimo tag) → `compose.yml` + `edge.yml` presi **dal tag**, `pull` + `up -d` in `/srv/orchid` → healthcheck in loopback (rollback al tag precedente se fallisce)
- **Nessun database:** tutto su filesystem (JSON, .pt, .npy); dataset e pesi su HuggingFace
- **Live demo:** https://osr.marrusso.dev

Si tagga quando una versione deve andare online (patch = fix, minor =
funzionalità, major = cambi incompatibili): i tag sono il registro delle
versioni pubblicate, non la cronologia dello sviluppo. Il deploy parte dalla
macchina di sviluppo perché l'SSH della VPS è aperto solo sulla tailnet;
l'host non ha il sorgente, prende solo l'immagine pinnata.

---

## Licenza

[MIT](LICENSE)
