# OrchID — Training Status & Results

## Stato Generale

| Modello | Hardware | Exp 6 (CE) | Exp 7 (SupCon) | Exp 8 (SpHOR) |
|---------|:--------:|:----------:|:--------------:|:-------------:|
| ResNet-18 | Quadro T2000 (4GB) | ✅ **locale** | ✅ locale | ✅ locale |
| ResNet-50 | Quadro T2000 (4GB) | ✅ locale | ✅ locale | ✅ locale |
| ConvNeXt-Tiny | Tesla V100S (32GB) | ✅ **OVH** | ⚠️ **vecchi** (HF old/) | ⚠️ **vecchi** (HF old/) |
| ConvNeXt-Small | Tesla V100S (32GB) | ✅ OVH | ⚠️ vecchi (HF old/) | ⚠️ vecchi (HF old/) |
| DINOv2-Small | Tesla V100S (32GB) | ✅ OVH | ✅ **nuovi OVH** | ✅ **nuovi OVH** |
| DINOv2-Base | Tesla V100S (32GB) | ✅ OVH | ✅ **nuovi OVH** | ✅ **nuovi OVH** |

## Dove trovare i risultati

### Nuovi risultati (OVH v13 — da preferire)
**HF**: `marcellorusso/orchid-ncd-models/exp7_supcon/` e `exp8_sphor/`
Contiene: DINOv2 S/B exp 7-8, ConvNeXt exp 6 (parte)

### Vecchi risultati (locale — ConvNeXt)
**HF**: `marcellorusso/orchid-ncd-models/old/exp7_supcon/` e `old/exp8_sphor/`
Contiene: ConvNeXt T/S exp 7-8

### In corso (locale)
**Path locale**: `backend/experiments/`
ResNet18/50 exp 6-7-8 (script: `/tmp/local_training.sh`)

## Architetture e Recipe

| Modello | Input | Params | LR | Ottimizzatore | Weight Decay | Training |
|---------|:-----:|:------:|:--:|:-------------:|:------------:|:--------:|
| ResNet-18 | 256×512 | 11M | 1e-3 | Adam | 0.0 | Full |
| ResNet-50 | 256×512 | 25M | 1e-3 | Adam | 0.0 | Full |
| ConvNeXt-Tiny | 224×224 | 28M | 3e-4 | AdamW | 5e-2 | Full |
| ConvNeXt-Small | 224×224 | 50M | 3e-4 | AdamW | 5e-2 | Full |
| DINOv2-Small | 224×224 | 21M | 2e-4 | AdamW | 1e-2 | LoRA (r=8) |
| DINOv2-Base | 224×224 | 86M | 2e-4 | AdamW | 1e-2 | LoRA (r=8) |

Tutti i modelli con **multi_grain=True**, tranne ConvNeXt su OVH (fallato per instabilità V100S).

## Problema ConvNeXt su V100S

ConvNeXt produce loss SupCon ~51 e KNN ~0.32 su V100S (vs loss ~2 e KNN ~0.73 su locale T2000). Il problema:
- **Non** è il multi-grain (testato con multi_grain=False → stessa loss)
- **Non** è il pretrained loading (testato con entrambi)
- È specifico di ConvNeXt + SupCon su V100S (Exp 6 CE funziona, DINOv2 funziona)
- Probabile instabilità numerica di cuDNN su SM 7.0 con operazioni ConvNeXt-specifiche

**Soluzione**: usare i risultati vecchi (da HF `old/`), validi perché ottenuti con lo stesso codice su hardware diverso.

## Risultati Numerici

### Exp 6 — Cross-Entropy Baseline (mean CV F1)
| Modello | Vecchio (26ep) | Nuovo (100ep) | Fonte |
|---------|:--------------:|:-------------:|:-----:|
| ResNet-18 | 0.860 | 0.880 | locale |
| ResNet-50 | 0.882 | — | locale (in corso) |
| ConvNeXt-Tiny | 0.892 | 0.674 | OVH |
| ConvNeXt-Small | 0.863 | 0.688 | OVH |
| DINOv2-Small | 0.888 | — | OVH |
| DINOv2-Base | 0.927 | — | OVH |

### Exp 7 — SupCon (test F1)
| Modello | Vecchio | Nuovo | Fonte |
|---------|:-------:|:-----:|:-----:|
| ResNet-18 | — | — | locale (in corso) |
| ResNet-50 | — | — | locale (in corso) |
| ConvNeXt-Tiny | 0.681 | ❌ fallato | **vecchio** (HF old/) |
| ConvNeXt-Small | 0.689 | ❌ fallato | **vecchio** (HF old/) |
| DINOv2-Small | 0.896 | 0.888 | nuovo OVH |
| DINOv2-Base | 0.687 | **0.936** | **nuovo OVH** |

### Exp 8 — SpHOR (test F1)
| Modello | Vecchio | Nuovo | Fonte |
|---------|:-------:|:-----:|:-----:|
| ResNet-18 | — | — | locale (in corso) |
| ResNet-50 | — | — | locale (in corso) |
| ConvNeXt-Tiny | ❌ | — | **vecchio** (HF old/) |
| ConvNeXt-Small | ❌ | — | **vecchio** (HF old/) |
| DINOv2-Small | — | 0.886 | nuovo OVH |
| DINOv2-Base | 0.907 | **0.929** | **nuovo OVH** |

## Costi OVH Stimati
- Speso finora: ~€45
- Job ConvNeXt+DINOv2 (v13): ~€15
- **Totale stimato**: ~€60 su €200 disponibili

## Bug Fix Applicati (PR #50)
1. Phase 2 LR: da hardcoded 0.01 a registry LR
2. Gradient accumulation + clipping in Phase 2
3. Torcheval → sklearn metrics
4. PyTorch 2.1.0 pinnato per compatibilità cuDNN
5. Transformers 4.43.0 pinnato per compatibilità PyTorch
6. SupCon early stopping (patience=15)
7. NCD mode (skip Phase 2 per exp 9-10)
