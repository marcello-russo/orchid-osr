# Analisi: O. sphegodes

**Data:** 2026-02-25
**Autore:** Marcello Russo

> [!WARNING]
> Questa analisi è basata sugli esperimenti legacy (exp1-5) che presentavano data leakage tra training e test set. I risultati e le metriche mostrati in questo documento sono gonfiati e non riflettono le reali performance del modello. Il documento viene mantenuto esclusivamente come riferimento storico per tracciare il processo decisionale.

---

## Problema

La classe *O. sphegodes* mostra performance significativamente inferiori rispetto alle altre classi:

| Classe | F1 Score |
|--------|----------|
| O. sphegodes_Palena | 97.03% |
| O. majellensis | 93.67% |
| O. exaltata | 92.86% |
| O. garganica | 92.93% |
| O. incubacea | 87.88% |
| **O. sphegodes** | **68.29%** ⚠️ |

---

## Analisi del Dataset

### Dimensioni per Classe

| Classe | Immagini Raw | Immagini Augmented (Train) |
|--------|--------------|---------------------------|
| O. exaltata | 283 | 793 |
| O. garganica | 251 | 704 |
| O. incubacea | 156 | 440 |
| O. majellensis | 202 | 568 |
| **O. sphegodes** | **100** ⚠️ | **280** ⚠️ |
| O. sphegodes_Palena | 260 | 728 |

**Osservazione:** O. sphegodes ha il dataset più piccolo:
- Solo **100 immagini raw** (vs 156-283 delle altre)
- Solo **280 immagini augmented** (vs 440-793 delle altre)

### Rapporto con altre classi

| Confronto | Rapporto immagini |
|-----------|-------------------|
| O. sphegodes vs O. exaltata | 1 : 2.8 |
| O. sphegodes vs O. sphegodes_Palena | 1 : 2.6 |
| O. sphegodes vs media altre classi | 1 : 2.0 |

---

## Analisi della Confusion Matrix

```
         ex  ga  in  ma  sp  sP    ← Predetto
exaltata 52   0   0   0   4   0
garganic  0  46   4   0   0   0
incubace  0   2  29   0   0   0
majellen  0   0   0  37   3   0
sphegode  4   1   0   1  14   0    ← O. sphegodes: solo 14/20 corrette
spheg_Pa  0   0   2   1   0  49
```

### Errori di O. sphegodes (6 su 20 campioni di test)

| Confusa con | Count | Possibile causa |
|-------------|-------|-----------------|
| O. exaltata | 4 | Somiglianza morfologica |
| O. garganica | 1 | Somiglianza morfologica |
| O. majellensis | 1 | Somiglianza morfologica |

### Chi confonde O. sphegodes?

| Classe | Confusa con O. sphegodes |
|--------|-------------------------|
| O. exaltata | 4 |
| O. majellensis | 3 |

---

## Ipotesi

1. **Dataset sbilanciato**: O. sphegodes ha meno della metà delle immagini rispetto alle altre classi
2. **Somiglianza morfologica**: O. sphegodes è simile ad O. exaltata (stesso genere), causando confusione
3. **Variabilità intrinseca**: Poche immagini potrebbero non catturare la variabilità della specie

---

## Possibili Soluzioni

| Soluzione | Descrizione | Difficoltà |
|-----------|-------------|------------|
| **Raccogliere più immagini** | Aumentare dataset O. sphegodes | Media |
| **Class weighting** | Dare più peso alla classe minoritaria nel loss | Bassa |
| **Oversampling** | Duplicare campioni O. sphegodes | Bassa |
| **Data augmentation aggiuntiva** | Più trasformazioni per O. sphegodes | Bassa |
| **Focal Loss** | Loss function che penalizza classi difficili | Media |

---

## Esperimenti Suggeriti

1. **Exp 3**: Ripetere con class weighting
2. **Exp 4**: Oversampling di O. sphegodes
3. **Exp 5**: Focal Loss invece di CrossEntropy

---

## Conclusioni

La bassa performance di *O. sphegodes* è probabilmente dovuta principalmente alla **scarsa rappresentazione nel dataset** (100 immagini vs 150-280 delle altre classi). La confusione con *O. exaltata* suggerisce anche una possibile somiglianza morfologica tra le due specie.

Per migliorare, sarebbe necessario:
1. Raccogliere più immagini di O. sphegodes
2. Implementare tecniche per gestire lo sbilanciamento delle classi
