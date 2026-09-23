# Redesign slide 10 (OSR) — design spec

Data: 2026-09-23 · Stato: approvato dall'utente · Mockup: `http://localhost:8931/slide10.html` (file in `/tmp/opencode/mockup10/`)

## 1. Problema

La slide 10 ("Open-Set Recognition è possibile", 10/15) è affollata e il messaggio è confuso:
titolo + hero FPR@95 + blocco "due vincitori" + heatmap AUROC + callout 0,875 + caption con
estremi/IC95%. L'heatmap non comunica a colpo d'occhio; segnale, costo e onestà si mescolano.
Gli hero numerici (0,34) funzionano e restano.

## 2. Decisione messaggio (condivisa)

Un solo filo: **segnale → costo → onestà**.
- Il segnale c'è (AUROC: solo ordinamento, nessuna soglia).
- Il costo si paga in falsi allarmi (FPR@95 = 0,34: 85 note sacrificate per 47–48 novel su 50).
- Onestà: due punti operativi, due vincitori diversi (a FPR fissato al 10% vince Tiny 0,64 vs 0,54).
Nessun concetto nuovo rispetto al discorso già approvato: solo riordino.

## 3. Layout (due colonne, invariata la struttura split della slide)

- **Sinistra**: titolo h2 invariato ("Open-Set Recognition / a granularità ultra-fine / è possibile.")
  + hero "IL COSTO DELLA SOGLIA · FPR@95 · DV-B + ENSEMBLE" con numero 0,34 e riga
  "per prendere 47–48 novel su 50 ne sacrifico 85 su 250 note"
  + riga onestà (bordo sinistro accent): "A falsi allarmi fissi (10%) vince Tiny,
  0,64 vs 0,54 — due punti operativi, due vincitori: nessun cherry-picking."
- **Destra**: UN solo grafico — curve ROC DINOv2-Base + ConvNeXt-Tiny (Ensemble, II-A),
  diagonale del caso, ★ sul punto operativo FPR@95 di ciascuna curva, legenda con AUROC
  (0,875 / 0,850). Caption: "Il segnale c'è: AUROC 0,875 (Base) · 0,850 (Tiny).
  Le ★ marcano il costo operativo."
- **Rimossi dalla slide**: heatmap AUROC, callout con IC95%, caption con estremo RN-18
  chance-level. (IC95% e DeLong restano solo nel parlato, dove sono già coperti.)

## 4. Grafico: specifiche tecniche

- Sorgente dati: `backend/experiments/exp4_sphor_ncd_{dinov2_base,convnext_tiny}/osr_scores.npz`,
  colonna `Ensemble`, `y_true==1` = novel. Generato con matplotlib (script da scrivere
  in fase di implementazione, riusabile): curve `roc_curve`, stelle al primo punto con
  TPR≥0.95, diagonale tratteggiata, griglia leggera, font ≥12pt, palette deck
  (Base `#002FA7`, Tiny `#5b8def`).
- File: `documents/images/roc_osr.png` (+ copia in `presentazione_consegna/images/`),
  risoluzione adeguata a box ~600px di larghezza (dpi ≥150).
- Valori da ricalcolare a implementazione (attesi, già verificati): Base FPR@95 0,34,
  Tiny FPR@95 0,54, DR@10% 0,54 / 0,64, AUROC 0,875 / 0,850.

## 5. Discorso: unica aggiunta

Nel parlato della slide OSR (`documents/defense_speech.md`,
`documents/presentazione_consegna/discorso_per_slide.md`, note relatore in
`thesis_defense.html`), subito dopo la frase 85-vs-47, aggiungere:

> "Le stelle sul grafico marcano proprio quel punto: stare sopra la diagonale
> significa meglio del caso, e più la curva sta in alto a sinistra, meglio separa
> note e novel."

Solo lettura del grafico, nessuna teoria aggiuntiva. Durata totale deve restare
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
