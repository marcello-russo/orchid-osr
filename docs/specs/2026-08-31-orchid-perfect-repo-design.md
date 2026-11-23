# OrchID Perfect Repo — Design Spec

**Date:** 2026-08-31
**Status:** Draft — awaiting user review
**Goal:** Repo da tesi magistrale "perfetto" (no compromessi), approccio B full-modern. 6 figure killer (1 messaggio = 1 figura), CLI Typer, quality gate strict, scripts consolidati.

## 1. CLI `orchid` con Typer

**Current:** `backend/cli.py` + `app/cli/experiments.py` (728 righe argparse), help piatto.

**Design:**
- `backend/app/cli/main.py` — `typer.Typer(rich_markup_mode="rich", add_completion=True)` con comandi: `train`, `ncd`, `figures`, `hf`, `status`, `embeddings`.
- Ogni comando tipizzato: `ExpId = Annotated[int, typer.Argument(help="1-5")]`, `ModelVariant`, `NovelArg` validato via `parse_novel_arg` (pydantic).
- `orchid train --exp 4 --model dinov2_small --sphor` + `--novel garganica,majellensis` override (precedenza: `--novel` > `EXPERIMENTS[5]`; se `--novel` passato, `resolve_experiment` sovrascrive `exclude_classes`/`num_classes`/`ncd_mode` e logga `override: Exp 5 (2 novel) → 1 novel via --novel`); errori in `rich.Panel` non traceback.
- Retrocompat: `backend/cli.py` shim chiama `typer` con `UserWarning: deprecated, usa orchid train`.
- `runner.run_experiment()` prende `ExperimentConfig(BaseModel)` non `dict`.

**Success:** `orchid --help` presentabile in discussione; `python cli.py --exp 4` ancora funziona.

## 2. Consolidamento `backend/scripts/` → `app/figures/` + `app/cli/commands/`

**Current:** 15 file in `backend/scripts/` (5 generatori figure sovrapposti + 3 train wrappers) + `EXPERIMENTS[10]` (MGH, 6 cartelle in `backend/experiments/exp10_*` ancora presenti) + `EXPERIMENTS[5]` con `ncd_mode: False` incoerente con GCD.

**Design:**
- `backend/app/figures/` package (4 moduli, non 5):
  - `closed.py` → `benchmark_bar` + `heatmap` (1 funzione `build_closed(fig_dir, exps)`)
  - `osr.py` → `grouped_auroc` (6 arch × 4 detector, Exp4+Exp5) + `per_species_auroc`
  - `gcd.py` → `gcd_old_new` (unico, `ss_kmeans` + `kmeans_true_k`)
  - `reliability.py` → `reliability_diagram` + `confusion_matrices` (solo DINOv2-Base)
  - Ogni modulo: `def build(fig_dir: Path, experiments: dict[int, Path]) -> Path` — se `experiments` manca, `raise FileNotFoundError` con messaggio `orchid figures` suggerito; non genera figure vuote.
- `backend/app/cli/commands/figures.py` → `orchid figures --all` (default) | `--only gcd,osr` (Typer `List[str]` comma-separated, `case-insensitive`) | `--check` (regressione su `*.png` sorgente con `Pillow` perceptual hash `phash` Hamming ≤5, non `sha256` su webp lossy; snapshot in `tests/__snapshots__/figures/*.png.phash`).
- `scripts/archive/thesis-figures/` con `README.md` "deprecated 2026-08-31, vedi `app/figures/` — replicabile con `orchid figures --all`".
- Train: `orchid train --exp 4 --model dinov2_small --profile gpu|cpu` (unico, sostituisce 3 `.sh`); `--profile` setta `CUDA_VISIBLE_DEVICES` + `batch_size` profile.
- **Fix Exp5/Exp10:** `definitions.py` → `EXPERIMENTS[5]` `ncd_mode: True` (per far scattare GCD con `n_novel=2`), `EXPERIMENTS[10]` spostato in `EXPERIMENTS_ARCHIVED = {10: {...}}` con commento `# MGH escluso dalla narrazione 1-5, mantenuto per comparabilità` + `backend/experiments/exp10_*` spostate in `backend/experiments/_archived/` (gitignored, non contate in `orchid status`).

**Data flow:** `backend/experiments/{exp*_*}*/results.json` + `ncd_results.json` → `app/figures/*.py` (matplotlib + seaborn, style `seaborn-v0_8-whitegrid`, font `Inter`, palette `colorblind`, `savefig(dpi=300, bbox_inches="tight")`) → `documents/_figures/*.png` (sorgente vettoriale) + `documents/_figures/*.webp` (export) + `frontend/public/figures/` **copia** via `python -m app.figures.sync` (non symlink, `shutil.copy2` in `orchid figures --sync-frontend`, CI verifica `diff` non symlink).

**Testing:** `tests/test_figures.py` genera in `/tmp` e assert `exists + width>800 + AUROC in [0.5,1.0] + top-label ECE in [0,0.3] + per-species n=50`; `--check` fallisce se `phash` Hamming >5.

**Error handling:** se `experiments/` vuoto, `orchid figures` esce con `rich` panel giallo "Nessun esperimento trovato, lancia `orchid train` prima" (exit 0, non 1) per non rompere CI su branch vuoti.

## 3. Quality gate

**Current:** ruff `E,F,I,W` + `cov-fail-under=30` + per-file-ignores larghi.

**Design:**
- `ruff`: add `UP,D,PL,SIM`, `line-length=120` (invariato), `target-version=py311`, rimuovi per-file-ignores tranne `E402` per `preprocessing`.
- `mypy --strict` su `app/cli/` + `app/core/` (`[tool.mypy] disallow_untyped_defs=true`, `ignore_missing_imports=true` per `captum`/`torch`).
- `pre-commit` (`.pre-commit-config.yaml`): `ruff check --fix`, `ruff format`, `mypy`, `pytest -q` — hook `pre-push` non `pre-commit` per non bloccare `git commit --amend`.
- `pytest --cov-fail-under=60` su `app/` (`--cov=app --cov-append --cov-report=term`) **escludendo** `app/figures/` (`[tool.coverage.run] omit = ["app/figures/*"]`) + badge `coverage.svg` in README generato da `gen-badge` (solo `app/core`+`app/cli` contano, `figures` testato via `test_figures.py` separato).
- CI 3 job invariato, threshold 60; `mypy` in CI come 4° job separato per non bloccare `backend` job.

## 4. Audit figure + nuove figure

**Keep (4):** benchmark_bar/heatmap, confusion_matrices (solo DINOv2-Base), exp9_osr_auroc (rifatta grouped), gcd_old_new.

**Remove/refactor (4):** closed_vs_open_scatter, exp9_tsne_novel, loss_curves triplicate, architecture/osr-flow manual.

**New (3):**
1. reliability_diagram (top-label ECE, 15 bin `numpy.histogram`, `max_prob` vs `accuracy`, ECE in caption con `ECE = Σ |acc - conf| * bin_size`, non `calibration_curve` binaria)
2. per_species_auroc (6 barre, 1 per specie novel vs rest, AUROC 0.5 baseline tratteggiata, 95% CI via 1000 bootstrap stratificato `sklearn.utils.resample`, barra errore in figura + `n=50` annotato)
3. exp5_vs_exp4 (AUROC Exp4 0.875 vs Exp5 atteso 0.65-0.75, grouped bar per detector)

**Total 6 figure concettuali = 8 file** (closed: 2 file bar+heatmap contano come 1 messaggio; osr: 2 file grouped+per-species contano come 1 messaggio) — in tesi main 6 figure (heatmap in appendice), caption stile CVPR in italiano (2 frasi: cosa mostra + takeaway), ogni figura 1 messaggio forte (vedi §5). `loss_curves` e `t-SNE` spostate in `appendix/` non in tesi main.

## 5. Messaggi forti + migrazione

**6 messaggi visivi (1:1 con 6 figure):**
1. Closed-set è risolto (benchmark bar Exp1→3)
2. Open-set è architecture-dependent (grouped AUROC)
3. GCD: old ok, new fail tranne DINOv2 (gcd_old_new)
4. 2 novel > 1 novel stress test (exp5_vs_exp4)
5. SpHOR è calibrato (reliability)
6. Dove sbagliamo (confusion 5×5 + novel)

**Migrazione:** shim `cli.py`, `scripts/` archiviato, `orchid figures --all` citabile in tesi §3.5.

## 6. Non-goals

- No config file `orchid.toml` / man page (future work, citato in tesi §5.4).
- No monorepo `nx` / Rust CLI.
- No riscrittura `app/api/` legacy in questo sprint (solo `app/cli/` + `app/core/`).
- No figure interattive `plotly` in tesi PDF (solo `webp` statiche; interattive restano in frontend `explorer/`).

## 7. Docs & migrazione

- `README.md` / `README.it.md` + `AGENTS.md` aggiornati con `orchid --help` + `orchid figures --all` (1 comando riproducibile citato in tesi §3.5).
- `docs/superpowers/specs/` + `docs/specs/` deprecato — source of truth resta `docs/specs/` (git `add -f` per bypass `.gitignore`).
- `backend/cli.py` shim con `warnings.warn(..., DeprecationWarning)` + `typer` forward, rimosso in `v1.0`.

## 8. Risks & mitigazioni

- Typer aggiunge `typer[all]` + `rich` (~5MB) — accettabile, già in `huggingface_hub` deps.
- `mypy --strict` richiede 1 giorno annotazioni — mitigato: `mypy` solo su `app/cli/`+`app/core/` nuovo, legacy `app/api/` escluso via `mypy.ini` `exclude`.
- **Nome comando:** `orchid` confligge con `pip orchid` (genomica) — comando resta `orchid` in tesi ma `pyproject.toml [project.scripts]` espone `orchid-osr = app.cli.main:app` come alias ufficiale, documentato in README come `orchid-osr` per installazioni pip.
- `orchid figures` senza `experiments/` → exit 0 con warning (non rompe CI su PR vuote).
- `line-length 120` invariato → diff minimo, non 100.

## 9. Dettaglio super — file tree esatto

```
backend/
  app/
    cli/
      main.py                 # Typer app
      commands/
        train.py              # def train(exp, model, sphor, novel, profile)
        ncd.py                # def ncd(exp, model)
        figures.py            # def figures(all, only, check, sync_frontend)
        hf.py                 # def hf_upload/download
        status.py             # def status()
      definitions.py          # EXPERIMENTS[1-5] + EXPERIMENTS_ARCHIVED[10] + resolve_experiment()
      runner.py               # run_experiment(config: ExperimentConfig)
    figures/
      __init__.py
      closed.py               # build_closed(fig_dir, exps) -> Path(s)
      osr.py                  # build_osr(fig_dir, exps) -> Path(s)
      gcd.py                  # build_gcd(fig_dir, exps) -> Path
      reliability.py          # build_reliability(fig_dir, exps) -> Path(s)
      sync.py                 # def sync_frontend(fig_dir, frontend_dir)
      style.py                # MATPLOTLIB_RC, PALETTE, FONT
    core/...                  # invariato
  scripts/
    archive/
      thesis-figures/         # 5 vecchi + README deprecated
  cli.py                      # shim deprecated
  pyproject.toml              # [project.scripts] orchid + orchid-osr
```

## 10. Dettaglio super — firme Typer

```python
# backend/app/cli/main.py
app = typer.Typer(rich_markup_mode="rich", add_completion=True)

@app.command()
def train(
    exp: Annotated[int, typer.Argument(help="Exp 1-5", min=1, max=5)],
    model: Annotated[str, typer.Option(help="resnet18|resnet50|convnext_tiny|convnext_small|dinov2_small|dinov2_base")] = "resnet18",
    sphor: Annotated[bool, typer.Option(help="Abilita SpHOR (solo Exp 3-5)")] = False,
    novel: Annotated[str | None, typer.Option(help="Override novel: garganica,majellensis")] = None,
    profile: Annotated[str, typer.Option(help="cpu|gpu")] = "gpu",
): ...

@app.command()
def figures(
    all: bool = typer.Option(True, help="Tutte le 6 figure"),
    only: list[str] | None = typer.Option(None, help="gcd,osr,closed,reliability"),
    check: bool = typer.Option(False, help="Regressione phash"),
    sync_frontend: bool = typer.Option(False, help="Copia in frontend/public/figures"),
): ...
```

Help esempio: `orchid train --help` mostra `Examples: orchid train --exp 4 --model dinov2_small --sphor`.

## 11. Dettaglio super — checklist migrazione ordinata

1. `definitions.py` → `EXPERIMENTS[5] ncd_mode=True`, `EXPERIMENTS_ARCHIVED[10]`
2. `backend/experiments/exp10_*` → `backend/experiments/_archived/` + `.gitignore` + `orchid status` esclude `_archived`
3. `backend/app/figures/` package + `style.py` (palette colorblind `#0173B2,#DE8F05,#029E73,#CC78BC,#CA9161,#F49DAA`)
4. `backend/app/cli/main.py` + `commands/*.py` + `pyproject.toml [project.scripts]`
5. `backend/cli.py` shim con `warnings.warn`
6. `ruff` + `mypy.ini` + `.pre-commit-config.yaml` + `pytest --cov` omit
7. `tests/test_figures.py` + `tests/__snapshots__/figures/*.png.phash`
8. `README`/`AGENTS` con `orchid --help` + `orchid figures --all` citazione tesi §3.5
9. `scripts/archive/thesis-figures/README.md`
10. CI 4° job `mypy`

Ordine con dipendenze: 1→2→3→4→5→6→7→8→9→10.

## 12. Dettaglio super — figure pixel-perfect

- `dpi=300`, `bbox_inches="tight"`, `format="png"` sorgente + `webp` export `quality=90`
- Font `Inter` 10pt (titoli 12pt bold), `seaborn-v0_8-whitegrid`, `axes.titlesize=12`
- Palette colorblind Okabe-Ito 6 colori sopra, `hatch` per stampa B/W
- Dimensioni: `benchmark_bar` 8×4 in, `grouped_auroc` 10×5 in, `gcd_old_new` 6×4 in, `reliability` 6×6 in, `confusion` 6×5 in, `per_species` 8×4 in
- Ogni figura salva `*.png` + `*.webp` + `*.pdf` (vettoriale per tesi)

## 13. Dettaglio super — snippet pre-commit / mypy

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: mypy
        name: mypy strict
        entry: mypy --strict app/cli app/core --ignore-missing-imports
        language: system
        pass_filenames: false
```

```ini
# pyproject.toml [tool.mypy]
[[tool.mypy.overrides]]
module = "app.api.*"
ignore_errors = true
```

```ini
# pyproject.toml [tool.coverage.run]
omit = ["app/figures/*", "app/api/*"]
```

