# OrchID OSR — Riassunto sessione & stato

Ultimo aggiornamento: 25/08/2026

## OBIETTIVO
Tesi magistrale "Classificazione Ultra-Fine-Grained e Riconoscimento Open-Set delle Orchidee Ophrys tramite Apprendimento Contrastivo Profondo". Verificare/correggere data leakage e incoerenze numeriche; consegnare tesi + frontend + HF tutti coerenti.

---

## 1. DATA LEAKAGE — RISOLTO (verificato, impatto minimo)

### Cosa c'è
- 39/300 immagini test (13%) hanno duplicato percettivo (pHash≤2) nel train. Palena 28, incubacea 6, exaltata 4, sphegodes 1.
- 1 solo leak esatto MD5 e **cross-specie**: `train/O. incubacea/I (35).jpg` = `test/O. sphegodes/S (257).jpg` (MD5 e83784ba...). Ma il test è puro per fonte (I=internet, S=internet) — nessun leak cross-fonte reale.
- Causa: `split.py` deduplica solo per MD5 + split per filename → burst dello stesso fiore divisi train/test.

### Impatto misurato (F1 macro, test corretto per architettura)
- Exp 8: delta da +0,001 a −0,015 (convnext_tiny 0,911→0,912)
- Exp 9: delta da +0,002 a −0,034 (resnet18 0,590→0,556)
- **Conclusione: impatto NEGLIGIBILE (≤3,5%). NON serve riaddestrare.**

### Verifica mini-training (fatta, poi pulita)
- Addestrato convnext_tiny exp8 su train pulito (2192, senza i 40 duplicati): F1=0,882 vs 0,912 attuale → delta −0,03 confermato.
- **Tutto ripulito**: `leak_fix/`, `exp8_clean_convnext_tiny/`, `train_pulito/`, `test_pulito/`, cache — eliminati. `git status` pulito.

### Verifica results.json vs checkpoint
- **CONFERMATO VALIDO**: le metriche coincidono coi checkpoint se si usa il test **corretto per architettura**:
  - ResNet (256×512) → `cropped_256x512/test`
  - ConvNeXt/DINOv2 (224×224) → `cropped_224x224/test`
- Errori precedenti erano solo dataset sbagliato usato nella verifica.

---

## 2. NUOVO PROBLEMA SCOPERTO: 2192 vs 2232 (duplicati intra-train)

### Il fatto
- Train **256×512 = 2192** immagini; train **224×224 = 2232** immagini. Test identico (300) in entrambe.
- La pipeline 224 ha **40 immagini in più**, tutte nelle 4 specie a fonti miste (exaltata +11, garganica +7, incubacea +14, sphegodes +8). Majellensis e Palena identiche.
- **CAUSA VERIFICATA**: le 40 sono **duplicati intra-train** (stessa foto salvata in `.jpg` e `.jpeg`/`.png`, es. `E (16).jpg` = `E (16).jpeg`). Train 224: 2232 file ma **2150 foto uniche** (82 duplicati totali). Train 256: 2192 file, ~2150 uniche (42 duplicati).
- **NON è data leakage**: 0 delle 40 hanno duplicato nel test (verificato con pHash). Impatto training = neutro (modello vede foto già viste 2 volte).

### Decisione utente (opzione 3 scelta)
**Sommario riscritto con numero pulito + pulizia HF. NO riaddestramento.**
- Obiettivo: **2192 e 300 in TUTTA la tesi/frontend/README** (un solo numero coerente)
- Rimuovere i 40 duplicati dal train 224 (locale + HF)
- Riscrivere il sommario senza frasi allarmanti tipo "differiscono di 40 immagini"
- Onestà scientifica: i checkpoint 224 sono stati addestrati su 2232, ma essendo duplicati ridondanti, riaddestrare su 2192 darebbe risultati identici → riportare 2192 è riproducibile.

### Punti che citano i numeri (DA CORREGGERE)
- `documents/_chapters/0-abstract.tex:8` — "2 192... (2 232 per quelle a 224×224)"
- `documents/_chapters/3-dataset.tex:45` — caption distribuzione (verifica contenuto)
- `documents/_chapters/4-experiments.tex:39` — "2192... 2232... differiscono di 40 immagini"
- `README.md:30` / `README.it.md:30` — "2,232 training + 300"
- `frontend/src/locales/it.json` contrib_2 / `en.json` contrib_2 — "2.192... (2.232 per quelle a 224×224)"

---

## 3. COSA È STATO FATTO IN QUESTA SESSIONE

### Frontespizio (tesi)
- Relatori: **Ch.mo Prof. Sergio DI MARTINO, Ch.mo Prof. Luigi Libero Lucio STARACE, Ch.ma Prof.ssa Mara SANGIOVANNI** (corretto cognome SANGIOVANI→SANGIOVANNI)
- Correlatore: **Ch.mo Prof. Alessandro DE LUCA**
- Font ridotto `\large`→`\normalsize` in `uninafrontespizio.sty` → nomi su 1 riga (verificato col layout)
- Titolo italiano ✓, label "Relatori" plurale

### Figure tesi (14 aggiunte, tutte aggiornate 23-24 ago)
Aggiunte a `_chapters/4-experiments.tex`:
- Exp 6: `loss_curves_exp6_dinov2b`, `cv_test_gap`, `loss_curves_aggregated`
- Exp 7: `metric_curves_dinov2s`
- Exp 8: `tsne_exp8_multimodel`, `per_species_f1`
- Exp 9: `osr_detection_rate`, `osr_detection_heatmap`, `spotlight_convnext_osr`
- Sintesi: `confusion_matrices`, `closed_vs_open_scatter`, `benchmark_heatmap`, `benchmark_bar`, `summary_dual_heatmap`, `spotlight_species_evolution`
- **Tesi compilata: 75 pagine, 0 errori** (container docker `orchid-latex` con immagine `thesis-latex:latest`)

### README (EN + IT) — già fatto
- Banner: `banner.png`(7MB,vecchio)→`banner.webp`(nuovo frontend,336KB)
- Rimosso Exp 10/MGH-NCD (obsoleto), aggiunta sezione Exp 9
- Numeri corretti coi dati reali: Exp6 DINOv2 0.927→ResNet-50 0.980, Exp7 DINOv2-Small 0.896→DINOv2-Base 0.936, Exp8 0.759→0.911, Exp9 nuova (0.932/0.875)
- Novel class Napoli→garganica, NMI 0.16-0.19→0.84

### Hook post-commit `.git/hooks/post-commit`
- **RIPARATO**: puntava a path vecchi (`orchid-ncd/.env`, `orchid-ncd/`) e a HF Space inesistente
- Ora: solo `git push origin main` (il push automatico funziona)

### Tesi (verifica numeri)
- Tutti i F1/AUROC verificati 1:1 coi results.json reali — COERENTI
- Corretto "clean split MD5"→"deduplicazione MD5" (4-experiments:331)
- Corretto "limiti degli esperimenti multi-granulari"→"limiti della discussione" (5-conclusions:43)
- Terminologia leakage: "garantito/zero overlap"→"deduplicazione MD5/escludendo duplicati binari"

---

## 4. CONTESTO CRITICO / GOTCHAS

- **Tesi (documents/) è un repo git SEPARATO** — i commit della tesi vanno lì, NON su /OrchID. `documents/.git` esiste.
- Hook post-commit su /OrchID pusha automaticamente su `origin/main` (quindi niente PR/merge manuali — il push diretto è il flusso). CI con job `deploy` su main (VPS) — sempre success finora.
- `exp10_mgh_ncd` resta nel CLI (`definitions.py`) come legacy **inattivo** — nascosto da API+frontend+test. NON rimuovere.
- La cartella `orchid-ncd/` (path /home/user/Documenti/orchid/) è un **residuo senza .git** — DA ELIMINARE (utente ha dato ok).
- Container latex: `orchid-latex` (fermo dopo build), immagine `thesis-latex:latest`. Compilare con `docker start orchid-latex && docker exec orchid-latex latexmk -g -pdf thesis.tex`.
- Test set 224: `E (35).jpg` non esiste in 224 (i nomi differiscono leggermente tra pipeline).
- HF token è in `orchid-osr/.env` (non `orchid-ncd/.env`).

---

## 5. PROSSIME COSE DA FARE (in ordine)

### URGENTE (coerenza numeri — opzione 3)
1. **Rimuovere i 40 duplicati dal train 224** (in `backend/dataset/cropped_224x224/train/`) — tenere il file "canonico", scartare il duplicato. Dopo: 2192 file.
2. **Aggiornare HF dataset** (`marcellorusso/orchid-osr-dataset`) con il train pulito (manifest incluso).
3. **Riscrivere il sommario** (`0-abstract.tex`) → "2 192 immagini di addestramento e 300 di test" senza la parentesi 2232.
4. **Correggere** `3-dataset.tex:45`, `4-experiments.tex:39`, `README.md/it:30`, `locales it/en contrib_2` → tutti a 2192/300.
5. **Commit della tesi nel repo separato** (documents/) — NON su /OrchID.

### In sospeso / da decidere
6. **Eliminare `orchid-ncd/`** (residuo, ok utente).
7. **Backup tesi** se serve (gitignored dal /OrchID ma ha repo proprio).

### Verifiche utili prima di consegnare
8. Ricompilare tesi dopo modifiche sommario/numeri.
9. `npm run build` frontend + commit.
10. Verificare deploy CI su main.

---

## 6. DATI CHIAVE (per riferimento rapido)

| Exp | Miglior F1 test | Campione |
|-----|----------------|----------|
| 6 (CE) | 0,980 | ResNet-50 |
| 7 (SupCon) | 0,936 | DINOv2-Base |
| 8 (SpHOR) | 0,911 | ConvNeXt-Tiny |
| 9 (NCD) | 0,932 (5 cls) | DINOv2-Base; AUROC 0,875 |

- Novel class: **O. garganica** (mai Napoli)
- NMI clustering fino a 0,84 (convnext_tiny exp9)
- AUROC OSR best: DINOv2-Base 0,875 (Ensemble), ConvNeXt-Tiny 0,850
- Train/test: 2192/300 (uniformare da 2232) · 6 architetture · seed 57 · 5-fold
- Container: thesis-latex, orchid-latex