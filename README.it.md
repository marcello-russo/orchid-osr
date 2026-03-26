# Classificazione Orchidee con ResNet18

Pipeline di deep learning per la classificazione di specie di orchidee del genere *Ophrys* utilizzando transfer learning con ResNet18.

**Paper correlato:** [ScienceDirect - S2590005625001079](https://www.sciencedirect.com/science/article/pii/S2590005625001079)

---

## Crediti

| Ruolo | Nome |
|-------|------|
| Ricerca e implementazione originale | **Davide Di Pierro** (UNINA) |
| Manutentore repository | Marcello Russo |

Questo progetto estende il lavoro di tesi originale di Davide Di Pierro presso l'Universita degli Studi di Napoli Federico II.

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

## Galleria

### Immagini Raw

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/raw/O_exaltata.png) | ![](docs/images/raw/O_garganica.png) | ![](docs/images/raw/O_incubacea.png) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/raw/O_majellensis.jpg) | ![](docs/images/raw/O_sphegodes.png) | ![](docs/images/raw/O_sphegodes_Palena.png) |

### Immagini Augmentate

Dopo augmentation (flip, rotazioni) per bilanciamento classi:

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/augmented/O_exaltata.jpg) | ![](docs/images/augmented/O_garganica.jpg) | ![](docs/images/augmented/O_incubacea.jpg) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/augmented/O_majellensis.jpg) | ![](docs/images/augmented/O_sphegodes.jpg) | ![](docs/images/augmented/O_sphegodes_Palena.jpg) |

### Output del Modello

Risultati della classificazione con visualizzazione sliding window:

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/results/O_exaltata.png) | ![](docs/images/results/O_garganica.png) | ![](docs/images/results/O_incubacea.png) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/results/O_majellensis.png) | ![](docs/images/results/O_sphegodes.png) | ![](docs/images/results/O_sphegodes_Palena.png) |

---

## Esperimenti

### Esperimento Originale (Davide Di Pierro)

Classificazione sul dataset privato con 6 specie. Risultati ottenuti con augmentation tradizionale (classi sbilanciate).

**Configurazione:**
- Epochs: 26
- Batch size: 8
- Learning rate: 0.001
- Class weights: No

**Risultati:**

| Metrica (Macro) | Valore |
|-----------------|--------|
| Accuracy | 89.19% |
| F1 Score | 88.78% |
| Precision | 88.52% |
| Recall | 89.19% |

| Metrica (Micro) | Valore |
|-----------------|--------|
| Accuracy | 91.16% |
| F1 Score | 91.16% |
| Precision | 91.16% |
| Recall | 91.16% |

### Risultati Per-Classe (Originale)

| Classe | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| O. exaltata | 92.86% | 92.86% | 92.86% | 92.86% |
| O. garganica | 92.00% | 92.93% | 93.88% | 92.00% |
| O. incubacea | 93.55% | 87.88% | 82.86% | 93.55% |
| O. majellensis | 92.50% | 93.67% | 94.87% | 92.50% |
| O. sphegodes | 70.00% | 68.29% | 66.67% | 70.00% |
| O. sphegodes_Palena | 94.23% | 97.03% | 100.00% | 94.23% |

### Matrice di Confusione (Originale)

```
[[52,  0,  0,  0,  4,  0],
 [ 0, 46,  4,  0,  0,  0],
 [ 0,  2, 29,  0,  0,  0],
 [ 0,  0,  0, 37,  3,  0],
 [ 4,  1,  0,  1, 14,  0],
 [ 0,  0,  2,  1,  0, 49]]
```

---

### Esperimenti di Domain Shift (Marcello Russo)

Cinque esperimenti per valutare l'impatto del domain shift, l'efficacia del data augmentation e il catastrophic forgetting.

**Configurazione (stessa dell'originale Di Pierro):**
- Epochs: 26
- Batch size: 8
- Learning rate: 0.001
- Class weights: No
- Augmentation bilanciata: 280 immagini/classe

**Dataset utilizzati:**

| Dataset | Fonte | Classi | Immagini |
|---------|-------|--------|----------|
| Privato | Raccolta personale | 6 | 1.252 |
| Internet Raw | Web (non ritagliate) | 4 | 1.080 |
| Internet Cropped | Web (ritagliate) | 4 | 1.080 |
| Combined | Privato + Internet Cropped | 6 | 2.332 |

**Nota:** Il dataset internet contiene solo 4 classi (O. exaltata, O. garganica, O. incubacea, O. sphegodes), mentre il dataset privato ha tutte le 6 classi.

**Risultati complessivi:**

| Esperimento | Train | Test | Accuracy | F1 Macro | Descrizione |
|-------------|-------|------|----------|----------|-------------|
| Exp1 | Privato | Privato | 94.78% | 93.72% | Baseline (6 classi) |
| Exp2 | Privato | Internet Raw | 35.09% | 37.57% | Domain shift forte |
| Exp3 | Privato | Internet Cropped | 48.15% | 50.43% | Cropping aiuta |
| **Exp4** | **Combined** | **Internet Cropped** | **95.19%** | **95.37%** | **Miglior su internet** |
| **Exp5** | **Combined** | **Privato** | **96.79%** | **95.83%** | **Nessun forgetting** |

### Risultati Per-Classe

**Exp4 - Modello Combined su Internet Cropped (Miglior risultato su dati internet):**

| Classe | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| O. exaltata | 96.26% | 94.50% | 95.37% |
| O. garganica | 95.62% | 95.92% | 95.77% |
| O. incubacea | 95.75% | 96.67% | 96.21% |
| O. sphegodes | 94.28% | 93.99% | 94.14% |

**Exp5 - Modello Combined su Privato (Test catastrophic forgetting):**

| Classe | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| O. exaltata | 100.00% | 98.21% | 99.10% |
| O. garganica | 96.00% | 96.00% | 96.00% |
| O. incubacea | 93.55% | 93.55% | 93.55% |
| O. majellensis | 97.56% | 100.00% | 98.77% |
| O. sphegodes | 94.44% | 85.00% | 89.47% |
| O. sphegodes_Palena | 96.30% | 100.00% | 98.11% |

### Conclusioni Principali

1. **Il domain shift impatta significativamente le performance**
   - F1 scende da 93.72% (Exp1) a 37.57% (Exp2) testando su immagini internet
   - Dimostra la difficolta di generalizzare su nuove fonti dati

2. **Il cropping migliora i risultati**
   - Immagini raw: F1 = 37.57% (Exp2)
   - Immagini croppate: F1 = 50.43% (Exp3)
   - Miglioramento: +12.9%

3. **Il data augmentation e molto efficace**
   - Modello solo privato su internet: F1 = 50.43% (Exp3)
   - Modello combined su internet: F1 = 95.37% (Exp4)
   - Miglioramento: +44.9%

4. **Nessun catastrophic forgetting**
   - Exp1 (modello privato su privato): F1 = 93.72%
   - Exp5 (modello combined su privato): F1 = 95.83%
   - Il modello combined performa **meglio** sul dataset privato
   - Aggiungere dati internet al training non degrada le performance sulle classi originali

5. **Tutte le classi beneficiano del training combined**
   - Tutte le classi raggiungono >94% F1 su dati internet (Exp4)
   - Tutte le classi raggiungono >89% F1 su dati privati (Exp5)

---

## Avvio Rapido

### Prerequisiti
- Linux (testato su Ubuntu/Pop!_OS)
- Podman (o Docker)
- GPU NVIDIA + Driver + Container Toolkit

### Esecuzione

```bash
# Clona la repository
git clone https://github.com/marcellorusso-x/orchid-classification.git
cd orchid-classification/resnet18

# Avvia il container
podman compose up -d --build

# Apri Jupyter
# http://localhost:8888
```

### Eseguire gli Esperimenti

```bash
# Esegui tutti gli esperimenti (nel container)
python run_experiments.py --all

# Oppure esegui fasi singole
python run_experiments.py --status     # Verifica stato
python run_experiments.py --setup     # Setup dataset
python run_experiments.py --preprocess # Preprocessing dati
python run_experiments.py --train     # Training modelli
python run_experiments.py --test      # Esegui test
```

### Notebook

| File | Descrizione |
|------|-------------|
| `notebook_dipierro.ipynb` | Notebook originale di Di Pierro |
| `notebook.ipynb` | Versione ristrutturata e pulita (replica run_pipeline.py) |

---

## Pipeline

```
Immagini Raw -> Augmentation -> Normalizzazione -> Preprocessing -> Training -> Testing
```

1. **Augmentation**: Bilancia le classi con flip, rotazioni (target: 280 img/classe in train)
2. **Normalizzazione**: Calcola mean/std per il dataset
3. **Preprocessing**: Converti immagini in tensori PyTorch
4. **Training**: Fine-tuning ResNet18 (pre-addestrata su ImageNet)
5. **Testing**: Valuta sul test set con matrice di confusione e visualizzazione explainability

---

## Struttura Progetto

```
resnet18/
├── app/                        # Codice sorgente
│   ├── preprocessing_tools/    # Utility preprocessing dati
│   ├── explenability_tools/    # Explainability modello (Captum)
│   ├── train_model.py          # Loop di training
│   ├── test_model.py           # Testing e valutazione
│   └── timing.py               # Utility di timing
├── dataset/
│   ├── raw/                    # Immagini private per classe
│   ├── internet_raw/           # Immagini internet (non ritagliate)
│   ├── internet_cropped/       # Immagini internet (ritagliate)
│   ├── combined/               # Dataset combinato (privato + internet)
│   └── *.pt                    # Tensori preprocessati
├── models/                     # Modelli addestrati
│   ├── train_privato/          # Modello solo privato
│   └── train_combined/         # Modello combined
├── results/                    # Output test e visualizzazione
│   ├── exp1_privato_privato/   # Immagini output Exp1
│   ├── exp2_privato_internet_raw/
│   ├── exp3_privato_internet_cropped/
│   ├── exp4_combined_internet_cropped/
│   └── exp5_combined_privato/
├── logs/                       # Log di timing
├── docs/                       # Documentazione e immagini
├── run_experiments.py          # Esegui tutti gli esperimenti
├── run_pipeline.py             # CLI per pipeline singola
├── generate_report.py           # Genera report finale
├── notebook_dipierro.ipynb     # Notebook originale
├── notebook.ipynb              # Notebook pulito
├── Dockerfile
└── docker-compose.yaml
```

---

## Stack Tecnologico

- **PyTorch 2.1.0** - Framework deep learning
- **ResNet18** - Architettura CNN (pre-addestrata su ImageNet)
- **Captum** - Explainability del modello (Integrated Gradients, Occlusion)
- **Torcheval** - Metriche di valutazione
- **Podman** - Container runtime con supporto GPU
- **CUDA 11.8** - Accelerazione GPU

---

## Note Tecniche

### Gestione Class Mismatch

Per Exp2-4, il test set ha 4 classi mentre il modello ne output 6. Quando ImageFolder carica il test set internet, assegna le label in ordine alfabetico:
- O. sphegodes nel test set: indice 3
- O. sphegodes nel modello: indice 4

Le metriche sono state ricalcolate con il mapping corretto degli indici.

### Configurazione Training

Stessa configurazione della tesi originale Di Pierro:
- ResNet18 pre-addestrata su ImageNet
- 26 epochs
- Batch size: 8
- Learning rate: 0.001
- Class weights: No
- Augmentation bilanciata: 280 immagini/classe

---

## Licenza

Vedi [LICENSE](LICENSE) per i dettagli.
