# Redesign slide 10 (OSR) — design spec

Data: 2026-09-23 · Stato: approvato dall'utente · Mockup: `http://localhost:8931/slide10.html` (file in `/tmp/opencode/mockup10/`)

## 1. Problema

La slide 10 ("Open-Set Recognition è possibile", 10/15) è affollata e il messaggio è confuso:
titolo + hero FPR@95 + blocco "due vincitori" + heatmap AUROC + callout 0,875 + caption con
estremi/IC95%. L'heatmap non comunica a colpo d'occhio; segnale, costo e onestà si mescolano.
Gli hero numerici (0,34) funzionano e restano.

## 2. Decisione messaggio (condivisa, aggiornata: metrica operating point)

La metrica protagonista è il **richiamo a specificità fissata al 95%** (tengo il 95% delle
note, leggo quante novel trovo): TPR@5%FPR, ricalcolato per tutti i modelli e scenari
il 2026-09-23. Tabella completa (Ensemble):

| Scenario | Tiny | Small | Base | D-Small | RN-50 | RN-18 |
|---|---|---|---|---|---|---|
| II-A (novel/50) | **0,44 (22)** | 0,24 (12) | 0,22 (11) | 0,16 (8) | 0,08 (4) | 0,00 (0) |
| II-B (novel/100) | 0,28 (28) | 0,09 (9) | 0,30–0,32* (30–32) | 0,11 (11) | 0,16 (16) | 0,13 (13) |

\* II-B Base: Mahalanobis 0,32 (32), KNN 0,31, Ensemble 0,30 — hero di slide 12: Mahalanobis 0,32.
Miglior singolo II-A: Tiny-Energy 0,34 (17), sotto l'Ensemble 0,44.

Filo: **specificità fissata → richiamo letto → onestà sui due punti**.
Hero slide 10: Tiny 0,44 ("tenendo 237 note su 250, becco 22 novel su 50").
Hero slide 12: Base-Mahalanobis 0,32 ("tenendo 190 note su 200, becco 32 novel su 100").
Secondo punto operativo (richiamo fissato al 95%, FPR@95): Base 0,34 vs Tiny 0,54 —
due punti operativi, due vincitori diversi, nessun cherry-picking.
Linguaggio: neutro e impersonale.Dire che a specificità fissata il richiamo resta
contenuto per tutti i modelli.NIENTE frasi editoriali tipo "prezzo della severità".

## 3. Layout (due colonne, invariata la struttura split della slide)

- **Sinistra**: titolo h2 invariato ("Open-Set Recognition / a granularità ultra-fine / è possibile.")
  + hero "RICHIAMO A SPECIFICITÀ 95% · TINY + ENSEMBLE" con numero 0,44 e riga
  "tenendo 237 note su 250, becco 22 novel su 50"
  + riga onestà (bordo sinistro accent): "A richiamo fissato al 95% vince Base
  (FPR@95 0,34 vs 0,54) — due punti operativi, due vincitori: nessun cherry-picking."
- **Destra**: UN solo grafico — curve ROC DINOv2-Base + ConvNeXt-Tiny (Ensemble, II-A),
  diagonale del caso, ★ al punto operativo a specificità 95% (FPR=0,05) di ciascuna
  curva, legenda con AUROC (0,875 / 0,850). Caption: "Il segnale c'è: AUROC 0,875 (Base)
  · 0,850 (Tiny). Le ★ marcano il richiamo a specificità 95%."
- **Rimossi dalla slide**: heatmap AUROC, callout con IC95%, caption con estremo RN-18
  chance-level. (IC95% e DeLong restano solo nel parlato, dove sono già coperti.)

## 4. Grafico: specifiche tecniche

- Sorgente dati: `backend/experiments/exp4_sphor_ncd_{dinov2_base,convnext_tiny}/osr_scores.npz`,
  colonna `Ensemble`, `y_true==1` = novel. Generato con matplotlib (script da scrivere
  in fase di implementazione, riusabile): curve `roc_curve`, stelle al punto con
  FPR=0,05 (specificità 95%) di ciascuna curva, diagonale tratteggiata, griglia leggera, font ≥12pt, palette deck
  (Base `#002FA7`, Tiny `#5b8def`).
- File: `documents/images/roc_osr.png` (+ copia in `presentazione_consegna/images/`),
  risoluzione adeguata a box ~600px di larghezza (dpi ≥150).
- Valori da ricalcolare a implementazione (attesi, già verificati): Base FPR@95 0,34,
  Tiny FPR@95 0,54, DR@10% 0,54 / 0,64, AUROC 0,875 / 0,850.

## 5. Discorso: unica aggiunta

Nel parlato della slide OSR (`documents/defense_speech.md`,
`documents/presentazione_consegna/discorso_per_slide.md`, note relatore in
`thesis_defense.html`), la frase operativa diventa: fissata la specificità al 95%
(tengo 237 note su 250), Tiny con Ensemble trova 22 novel su 50; a richiamo fissato
al 95% il confronto si inverte (Base 0,34 vs Tiny 0,54). Subito dopo, la frase ROC:

> "Le stelle sul grafico marcano il punto a specificità 95%: stare sopra la diagonale
> significa meglio del caso, e più la curva sta in alto a sinistra, meglio separa
> note e novel."

Solo lettura del grafico, nessuna teoria aggiuntiva. Stessa regola per la slide II-B
(hero 0,32: tenendo 190 note su 200, 32 novel su 100 con Base-Mahalanobis). Durata totale deve restare
19–21 min (oggi ~19,1): compensare con micro-tagli nel resto del blocco OSR se serve.

## 6. Validazione (criteri di accettazione)

- Screenshot headless 1920×1080: nessun overflow, hero e grafico leggibili, ★ visibili.
- `scripts/audit_defense_numbers.py`: verde (142/142 o superiore se aggiunge claim).
- Le due copie `thesis_defense.html` byte-identiche; discorso riallineato nei 3 punti
  (md ×2 + note HTML).
- Ogni parola della slide deve avere corrispondenza nel parlato (regola anti-brutte-figure).

## 7. Fuori scopo

- Rigenerazione `.pptx` (export manuale a deck definitivo).
- Modifiche alle altre 14 slide, al resto del discorso, ai numeri (solo layout+testi).
- Stile globale deck (palette, font, fondo bianco: intoccati).
