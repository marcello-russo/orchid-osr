# Classificazione Orchidee con ResNet18

Pipeline di deep learning per la classificazione di specie di orchidee del genere *Ophrys* utilizzando transfer learning con ResNet18.

📄 **Paper correlato:** [ScienceDirect - S2590005625001079](https://www.sciencedirect.com/science/article/pii/S2590005625001079)

---

## Crediti

| Ruolo | Nome |
|-------|------|
| Ricerca e implementazione originale | **Davide Di Pierro** (UNINA) |
| Manutentore repository | Marcello Russo |

Questo progetto estende il lavoro di tesi originale di Davide Di Pierro presso l'Università degli Studi di Napoli Federico II.

---

## Specie Classificate

| Classe | Nome Scientifico |
|--------|------------------|
| O. exaltata | *Ophrys exaltata* |
| O. garganica | *Ophrys garganica* |
| O. incubacea | *Ophrys incubacea* |
| O. majellensis | *Ophrys majellensis* |
| O. sphegodes | *Ophrys sphegodes* |
| O. sphegodes_Palena | *Ophrys sphegodes* (variante Palena) |

---

## Prestazioni

### Implementazione Originale (Davide Di Pierro)

| Fase | Durata |
|------|--------|
| Augmentation | ~118 min |
| Normalizzazione | ~20 sec |
| Preprocessing | ~77 sec |
| Training (26 epoche) | ~30 min |
| Testing | ~27 min |
| **Totale** | **~3 ore** |

| Metrica | Valore |
|---------|--------|
| Accuracy | TBD |
| F1 Score (Macro) | TBD |
| F1 Score (Micro) | TBD |
| Precision | TBD |
| Recall | TBD |

### Implementazione Corrente

*I risultati verranno aggiunti dopo aver eseguito nuovamente la pipeline.*

---

## Changelog

| Data | Autore | Descrizione |
|------|--------|-------------|
| 2026-01-29 | Davide Di Pierro | Implementazione iniziale e discussione tesi |
| 2026-02-23 | Marcello Russo | Setup repository, fix sistema di timing, aggiunto notebook pulito |
| | | |

---

## Avvio Rapido

### Prerequisiti
- Linux (testato su Ubuntu/Pop!_OS)
- Podman (o Docker)
- GPU NVIDIA + Driver + Container Toolkit

### Esecuzione

```bash
# Clona la repository
git clone https://github.com/marcellorussox/orchid-dipierro.git
cd orchid-dipierro

# Avvia il container
./run_container.sh

# Apri Jupyter
# http://localhost:8888
```

### Notebook

| File | Descrizione |
|------|-------------|
| `notebook.ipynb` | Notebook originale (timing corretto) |
| `notebook_clean.ipynb` | Versione ristrutturata e pulita |

---

## Pipeline

```
Immagini Raw → Augmentation → Normalizzazione → Preprocessing → Training → Testing
```

1. **Augmentation**: Bilancia le classi con flip, rotazioni
2. **Normalizzazione**: Calcola mean/std per il dataset
3. **Preprocessing**: Converti immagini in tensori
4. **Training**: Fine-tuning ResNet18 (pre-addestrata su ImageNet)
5. **Testing**: Valuta sul test set con matrice di confusione

---

## Struttura Progetto

```
├── app/                    # Codice sorgente
│   ├── preprocessing_tools/
│   ├── explenability_tools/
│   ├── train_model.py
│   ├── test_model.py
│   └── timing.py
├── dataset/
│   ├── raw/               # Immagini originali per classe
│   ├── augmented/         # Train/val/test augmentati
│   └── processed/         # Tensori salvati (.pt)
├── models/                # Modelli addestrati
├── logs/                  # Log dei tempi
├── results/               # Output test
├── notebook.ipynb         # Notebook originale
├── notebook_clean.ipynb   # Notebook pulito
├── Dockerfile
└── docker-compose.yaml
```

---

## Stack Tecnologico

- **PyTorch 2.1.0** - Framework deep learning
- **ResNet18** - Architettura CNN (pre-addestrata su ImageNet)
- **Captum** - Explainability del modello (Integrated Gradients, Occlusion)
- **Podman** - Container runtime con supporto GPU
- **CUDA 11.8** - Accelerazione GPU

---

## Licenza

Vedi [LICENSE](LICENSE) per i dettagli.
