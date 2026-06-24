# Draft Capitolo 5: Esperimenti

## Setup Sperimentale

### Hardware
- **ResNet (18/50)**: Quadro T2000, 4GB VRAM, batch=4, accum=2 (eff. 8, training locale exp 6) / Tesla V100S, 32GB VRAM, batch=8, accum=4 (eff. 32, OVH exp 7-8-10)
- **ConvNeXt (T/S)**: Tesla V100S, 32GB VRAM, batch=8, accum=4 (eff. 32) — OVH Cloud
  - ⚠️ Su V100S, ConvNeXt + SupCon produce loss instabile (50 invece di 2) per problemi cuDNN su SM 7.0
  - Risultati vecchi (da HF `old/`) sono validi, ottenuti su hardware diverso con stesse recipe
- **DINOv2 (S/B)**: Tesla V100S, 32GB VRAM, batch=8, accum=1 (eff. 8) — OVH Cloud

### Dataset
- 2.232 training, 300 test, 6 specie
- 5-fold stratified CV, seed 57
- Letterbox resize, normalizzazione ImageNet

### Recipe
| Modello | Input | LR | Opt | WD | Batch | Accum | Eff. Batch |
|---------|:-----:|:--:|:---:|:--:|:-----:|:-----:|:----------:|
| ResNet-18 | 256×512 | 1e-3 | Adam | 0.0 | 8 | 1 | 8 |
| ResNet-50 | 256×512 | 1e-3 | Adam | 0.0 | 8 | 1 | 8 |
| ConvNeXt-T | 224×224 | 3e-4 | AdamW | 5e-2 | 8 | 4 | 32 |
| ConvNeXt-S | 224×224 | 3e-4 | AdamW | 5e-2 | 8 | 4 | 32 |
| DINOv2-S | 224×224 | 2e-4 | AdamW | 1e-2 | 8 | 1 | 8 |
| DINOv2-B | 224×224 | 2e-4 | AdamW | 1e-2 | 8 | 1 | 8 |

### Esp 7: Two-Phase Training
- Phase 1: 100 ep SupCon, KNN validation, τ=0.07
- Phase 2: 50 ep CE head-only, LR dal registry, backbone frozen

### Esp 8: SpHOR
- Come Esp 7 + prototipi ortogonali L2, loss repulsiva λ=0.01

## Tabella Riassuntiva (Test F1)

### Exp 6 — CE Baseline
| Modello | CV F1 | Test F1 | Fonte |
|---------|:-----:|:-------:|-------|
| ResNet-18 | — | — | locale |
| ResNet-50 | — | — | locale |
| ConvNeXt-T | 0.674 | — | OVH |
| ConvNeXt-S | 0.688 | — | OVH |
| DINOv2-S | — | — | OVH |
| DINOv2-B | 0.927 | — | OVH |

*(dati da inserire quando training completi)*

### Exp 7 — SupCon
| Modello | CV F1 | Test F1 | CV-Test Gap | Note |
|---------|:-----:|:-------:|:-----------:|------|
| ResNet-18 | — | — | — | locale |
| ResNet-50 | — | — | — | locale |
| ConvNeXt-T | 0.871* | 0.681* | 0.190 | *da HF old/ |
| ConvNeXt-S | 0.884* | 0.689* | 0.195 | *da HF old/ |
| DINOv2-S (old) | 0.906* | 0.896* | 0.010 | *da HF old/ |
| **DINOv2-S (new)** | **0.889** | **0.888** | **0.001** | ✅ OVH |
| DINOv2-B (old) | 0.914* | 0.687* | 0.227 | *da HF old/ |
| **DINOv2-B (new)** | **0.930** | **0.936** | **-0.006** | ✅ **OVH (+25%)** |

### Exp 8 — SpHOR
| Modello | CV F1 | Test F1 | CV-Test Gap | SpHOR Probe | Note |
|---------|:-----:|:-------:|:-----------:|:-----------:|------|
| ResNet-18 | — | — | — | — | locale |
| ResNet-50 | — | — | — | — | locale |
| ConvNeXt-T | — | — | — | — | *da HF old/ |
| ConvNeXt-S | — | — | — | — | *da HF old/ |
| **DINOv2-S (new)** | — | **0.886** | — | — | ✅ OVH |
| **DINOv2-B (new)** | **0.930** | **0.911** | **0.019** | — | ✅ OVH |

## Risultati Principali (da Analizzare)

### Finding 1: DINOv2-Small è il modello più robusto
- CV-test gap = 0.001 (praticamente zero, nessun overfitting)
- Non migliora con più parametri (DINOv2-Base overfitta)

### Finding 1b: Solo DINOv2 beneficia di SupCon
- **ResNet-18**: CE F1~0.88 → SupCon F1~0.41 (collasso)
- **ConvNeXt-T**: CE F1~0.89 → SupCon F1~0.35-0.40 (stesso collasso)
- **DINOv2-B**: CE F1~0.93 → SupCon F1~0.94 (leggero miglioramento)
- Causa: **allineamento del pretraining**. DINOv2 è addestrato con self-supervision contrastiva su 142M immagini → le sue features sono già organizzate per similarità visiva. SupCon si limita a raffinare i confini tra specie. ResNet18/ConvNeXt arrivano da supervised learning su ImageNet (classi generiche) — devono reimparare lo spazio da zero, ma con solo 2232 immagini il segnale contrastivo è troppo debole.
- Implicazione: **non è questione di dimensione del modello, ma di allineamento pretraining/task**. DINOv2 parte già "allineato" perché la sua rappresentazione è intrinsecamente contrastiva.

### Finding 2: DINOv2-Base beneficia del LR fix
- Vecchio: test F1=0.687 (con Phase 2 LR=0.01 buggato)
- Nuovo: test F1=0.936 (con LR corretto dal registry)
- **Miglioramento: +25%**

### Finding 3: ConvNeXt soffre di overfitting
- CV-test gap ~0.19 (grande divario)
- Performance modeste (~0.68 test F1)
- Meglio di ResNet ma peggio di DINOv2

### Finding 4: SupCon > CE
- DINOv2-B: 0.936 (SupCon) vs 0.927 (CE)
- Il miglioramento è modesto ma consistente
- Spazio embedding più strutturato (visibile da t-SNE)

### Finding 5: SpHOR non migliora SupCon
- DINOv2-B: 0.911 (SpHOR) vs 0.936 (SupCon)
- La loss repulsiva compete con l'obiettivo contrastivo
- Utile per OSR ma danneggia la classificazione

### Finding 6: SupCon non funziona affatto per backbones deboli
- ResNet-18 e ConvNeXt-T/S: KNN validation accuracy resta a ~0.35-0.40 per tutte le epoche, equivalente a embeddings casuali
- La loss SupCon cala (da 25 a ~11) ma le embeddings non diventano più discriminanti → il gradiente contrastivo non è abbastanza informativo per rimodellare uno spazio features ImageNet-adatto verso le sottili differenze delle orchidee
- La Phase 2 (CE head-only su backbone congelato) eredità embeddings povere → F1 ~0.41
- Questo è un **risultato negativo** rilevante: dimostra che SupCon per UFGVC è efficace solo con backbones già sufficientemente potenti e allineati al dominio visivo

## SpHOR Probe Metrics

Bug fixato nel codice (nearest centroid). Per i già completati:
- ✅ **DINOv2-Small Exp 8**: probe calcolata e caricata su HF (Acc=0.920, F1=0.920)
- ❌ **DINOv2-Base Exp 8**: probe non disponibile (checkpoint non su HF)
- ❌ **ConvNeXt old Exp 8**: probe non disponibile (nessun embedding)
- 🔄 **Locale (ResNet/ConvNeXt)**: fix attivo, probe sarà calcolata

## Note per la Scrittura

1. **Confronto con letteratura**: I nostri risultati (~0.93 test F1 per DINOv2 su SupCon) sono competitivi con lo stato dell'arte UFGVC nonostante il dataset molto più piccolo e specializzato.
2. **Risultato negativo su OSR**: Da citare come contributo — lo spazio SupCon non separa sufficientemente classi ultra-fine-grained per NCD.
3. **Problema ConvNeXt su V100S**: Da documentare come limitazione tecnica.
4. **Vecchi vs nuovi**: Specificare sempre quali risultati sono aggiornati e quali no.
5. **ResNet/ConvNeXt SupCon debole**: Risultato negativo importante — dimostra che SupCon non è universalmente vantaggioso. Richiede backbone con pretraining allineato (contrastivo/self-supervised). Implicazione pratica: per UFGVC con pochi dati, meglio CE classico se il backbone non è DINOv2 o simile.
6. **DINOv2 è il backbone ideale per SupCon**: Il suo pretraining contrastivo lo rende naturalmente adatto. Questo spiega perché DINOv2-Small (solo 21M params) performa come DINOv2-Base (86M) — entrambi partono già con embeddings di alta qualità, SupCon fornisce solo un affinamento minimo.
