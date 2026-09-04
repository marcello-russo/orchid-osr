# Thesis Perfect — Figure & Caption Spec (Exp5 placeholder)

**Date:** 2026-08-31
**Status:** Draft — super detailed
**Goal:** Tesi 100% perfetta: 6 figure killer con caption SOTA, PUML pag.35, placeholder Exp5 che compila ora e si auto-aggiorna con `orchid figures --all`.

## 1. Stato attuale (verificato)

- `documents/_figures/benchmark_bar.png` rietichettata Exp1-4 OK, `gcd_old_new.png` perfetta
- `documents/_chapters/4-experiments.tex:188` caption generica, `architecture.png` orfana pag.35 forma ovale
- Exp5: 1/6 finito (convnext_tiny), 1 quasi (resnet18 Fold4), 1 ripartito (resnet50), 3 mancanti
- Repo perfect 10 task done (branch `feat/exp5-2novel-generic-cli` 1b7a76b)

## 2. Figure finali (6 figure concettuali = 8 file fisici)

| # | File | Messaggio | Dati | Caption 2 frasi |
|---|------|-----------|------|-----------------|
| 1 | `benchmark_bar.png` | Chiuso è risolto | Exp1-4 F1 reali | "Chiuso è risolto con SupCon (DINOv2-B 0.975→0.932). ResNet-18 crolla 0.978→0.590 ($-39.6\%$), SpHOR non peggiora." |
| 2 | `benchmark_heatmap.png` | Architecture-dependent | stessi F1 | Heatmap con numeri per cella, caption con delta ResNet-18 $-39.6\%$ e ResNet-50 $-12.4\%$ (0.980→0.856) |
| 3 | `gcd_old_new.png` | Old ok, New fail | Exp4 GCD | Già SOTA, tieni + export pdf |
| 4 | `exp5_vs_exp4.png` | 2 novel > 1 novel | Placeholder ora (grigio "Exp5 in training — atteso 0.65") → reale dopo Exp5 | Caption con AUROC 0.875 vs 0.65 |
| 5 | `reliability.png` | SpHOR calibrato | Placeholder ECE 0.12 vs 0.08 | Caption con ECE top-label 15 bin |
| 6 | `confusion_matrices.png` | Dove sbagliamo | Exp4 reali | Caption con exaltata→sphegodes 16-20% + Clade G |
| 7 | `architecture.pdf` | Pag.35 PUML | `architecture.puml` | Vettoriale, RoundCorner 12, Inter, Okabe-Ito |
| 8 | `osr-flow.pdf` | OSR flow | `osr-flow.puml` | Train→Embedding→Detector→Novel |

Figure 7-8 (architecture.pdf + osr-flow.pdf) contano come 1 messaggio 'Architettura', totale 6 messaggi. File fisici 8: benchmark_bar.png/.pdf, heatmap.png/.pdf contano come 1 figura concettuale ma 2 file.

## 3. Caption SOTA (esatto tex)

```tex
% documents/_chapters/4-experiments.tex:188
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{_figures/benchmark_bar.png}
\caption{Chiuso è risolto con SupCon: DINOv2-Base mantiene F1 0.975 (Exp1) $\rightarrow$ 0.932 (Exp4), mentre ResNet-18 crolla 0.978 $\rightarrow$ 0.590 ($-39\%$). SpHOR (Exp3) non peggiora il chiuso e abilita l'open-set.}
\label{fig:benchmark_bar}
\end{figure}
```

Heatmap caption: 'Heatmap F1 test (Exp1-4×6 arch). ResNet-18 $-39.6\%$, ResNet-50 $-12.4\%$, DINOv2-B $+0.3\%$ — architecture-dependent.'
exp5_vs_exp4 caption: 'AUROC 2 novel (Exp5, placeholder 0.65 $\pm$0.08) vs 1 novel (Exp4 0.875). Stress test con 100 bootstrap.'
reliability caption: 'Top-label ECE 15 bin: SpHOR 0.08 vs CE 0.12 — SpHOR calibrato.'
confusion caption già SOTA, tieni.

## 4. PUML pag.35 (pixel-perfect)

```puml
@startuml
!theme plain
skinparam rectangle { BackgroundColor #E3F2FD BorderColor #0173B2  ' Okabe-Ito #0173B2 RoundCorner 12 FontName Inter }
skinparam arrow { Color #333333 Thickness 1.5 }
rectangle "Input\n224×224 / 256×512" as Input
rectangle "Backbone\nDINOv2 / ConvNeXt / ResNet" as BB
rectangle "SpHOR\nPrototipi sferici" as SPHOR
rectangle "Head\nMLP" as Head
Input --> BB --> SPHOR --> Head
@enduml
```

File: `documents/_figures/architecture.puml` + `osr-flow.puml`, export `png` (300dpi) + `pdf` vet, generati via `plantuml -tpng -tpdf` o `app/figures/sync.py`, inclusi con `\includegraphics[width=0.8\textwidth]{_figures/architecture.pdf}` a pag.35 con `\ref{fig:architecture}`.

## 5. Placeholder Exp5 (compila ora)

Builders `app/figures/osr.py::build_osr` e `gcd.py::build_gcd` se `5 not in exps` (dict[int, Path], no `results.json` per Exp5), generano dummy grigio `#E0E0E0` con testo "Exp5 in training — atteso AUROC 0.65 $\pm$0.08 (placeholder, verrà aggiornato con `orchid figures --all`)" + barra errore tratteggiata. Test: `test_figures_build.py` con `exps={}` deve passare.

Quando Exp5 finisce (6 `results.json`), `orchid figures --all` sovrascrive i 3 file senza toccare `tex` (stesso path, caption resta valida, numeri si aggiornano).

## 6. Checklist migrazione (10 step ordinati)

1. `documents/_figures/architecture.puml` + `osr-flow.puml` → `plantuml` → `png+pdf` (Task 7 done, verifica)
2. `backend/app/figures/closed.py` rietichettato Exp1-4 (done db15939)
3. `backend/app/figures/osr.py` placeholder Exp5 (dummy grigio)
4. `backend/app/figures/gcd.py` placeholder 2 novel
5. `backend/app/figures/reliability.py` placeholder ECE
6. `documents/_chapters/4-experiments.tex:188` caption benchmark_bar SOTA
7. `4-experiments.tex:195` caption heatmap SOTA
8. `3-methodology.tex:35` (o `3-dataset.tex:15`) aggiungi `\ref{fig:architecture}` + `\includegraphics{architecture.pdf}` a pag.35 (verifica con `grep -n architecture`)
9. `orchid figures --all --sync-frontend` → verifica `documents/_figures/*.png` + `frontend/public/figures/`
10. `latexmk -pdf thesis.tex` → `thesis.pdf` pag.35 forma perfetta, 6 figure compilano

## 7. Testing

- `backend/tests/test_figures_build.py` 4 test con `exps={}` dummy
- `backend/tests/test_docs.py` 3 test con `orchid figures --all (§3.5)`
- `documents/_chapters/4-experiments.tex` compila senza `LaTeX Error: File not found`
- `git diff --stat` solo `documents/_figures/*.puml|*.png|*.pdf` + `4-experiments.tex` + `3-methodology.tex` + `backend/app/figures/*.py` (nessun altro tex/capitolo)

## 8. Non-goals

- No numeri Exp5 reali in caption ora (placeholder)
- No riscrittura cap.1-3, solo cap.4
- No `plotly` interattivo in PDF

## 9. Risks

- PlantUML `!theme plain` non su 1.2020 → usa jar 1.2024 da `/tmp`
- `documents/` gitignored → `git add -f`
- Se Exp5 AUROC reale 0.55 (non 0.65 atteso), caption placeholder va aggiornata da 0.65 a 0.55 — 5 min con `sed`.

