# OrchID Perfect Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trasformare il repo in artifact da tesi "perfetto" — CLI Typer `orchid`, 6 figure killer pixel-perfect, quality gate strict, scripts consolidati, senza rompere training Exp 1-5.

**Architecture:** Typer + Rich su `app/cli/main.py` con `ExperimentConfig` pydantic; `app/figures/` con 4 moduli testabili; `pre-commit` + `mypy --strict` + `cov 60`; `backend/cli.py` shim deprecated; `orchid figures` con phash.

**Tech Stack:** Typer 0.12, Rich 13, Pydantic 2, Ruff 0.5, mypy 1.11, pytest-cov, Pillow phash, matplotlib 3.8 + seaborn

---

## File Structure

**Create:**
- `backend/app/cli/main.py` — Typer app
- `backend/app/cli/commands/train.py`
- `backend/app/cli/commands/ncd.py`
- `backend/app/cli/commands/figures.py`
- `backend/app/cli/commands/hf.py`
- `backend/app/cli/commands/status.py`
- `backend/app/figures/__init__.py`
- `backend/app/figures/style.py`
- `backend/app/figures/closed.py`
- `backend/app/figures/osr.py`
- `backend/app/figures/gcd.py`
- `backend/app/figures/reliability.py`
- `backend/app/figures/sync.py`
- `backend/scripts/archive/thesis-figures/README.md`
- `tests/test_figures.py`
- `tests/__snapshots__/figures/.gitkeep`
- `.pre-commit-config.yaml`
- `mypy.ini` (o `[tool.mypy]` in pyproject.toml)

**Modify:**
- `backend/app/cli/definitions.py:118-145` — Exp5 ncd_mode True + EXPERIMENTS_ARCHIVED[10]
- `backend/app/cli/runner.py:70-110` — ExperimentConfig
- `backend/pyproject.toml:16,45,51` — scripts, ruff, mypy, coverage
- `backend/cli.py` — shim
- `backend/.gitignore` — _archived/
- `README.md`, `README.it.md`, `AGENTS.md`
- `.github/workflows/ci.yml` — 4° job mypy

---

### Task 1: Fix definitions.py (Exp5 + Exp10)

**Files:**
- Modify: `backend/app/cli/definitions.py:118-145`
- Test: `backend/tests/test_definitions.py` (new)

- [ ] **Step 1: Write failing test for Exp5 ncd_mode**

```python
# backend/tests/test_definitions.py
def test_exp5_ncd_mode_true():
    from app.cli.definitions import EXPERIMENTS, EXPERIMENTS_ARCHIVED
    assert EXPERIMENTS[5]["ncd_mode"] is True
    assert EXPERIMENTS[5]["num_classes"] == 4
    assert EXPERIMENTS[5]["exclude_classes"] == ["O. garganica", "O. majellensis"]
    assert 10 not in EXPERIMENTS
    assert 10 in EXPERIMENTS_ARCHIVED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_definitions.py::test_exp5_ncd_mode_true -v`
Expected: FAIL `assert False is True` (ncd_mode False) + `assert 10 not in EXPERIMENTS` fail

- [ ] **Step 3: Fix definitions.py**

```python
# backend/app/cli/definitions.py:118
    5: {
        "name": "exp5_sphor_ncd",
        "ncd_mode": True,  # fix
        ...
    },
EXPERIMENTS_ARCHIVED = {
    10: {
        "name": "exp10_mgh_ncd",
        # ... copy from old EXPERIMENTS[10]
    }
}
# Remove 10 from EXPERIMENTS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_definitions.py::test_exp5_ncd_mode_true -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/cli/definitions.py backend/tests/test_definitions.py
git commit -m "fix: Exp5 ncd_mode True + archive Exp10"
```

---

### Task 2: Create app/figures package (style + stubs)

**Files:**
- Create: `backend/app/figures/__init__.py`
- Create: `backend/app/figures/style.py`
- Create: `backend/app/figures/closed.py`
- Create: `backend/app/figures/osr.py`
- Create: `backend/app/figures/gcd.py`
- Create: `backend/app/figures/reliability.py`
- Create: `backend/app/figures/sync.py`
- Test: `backend/tests/test_figures.py`

- [ ] **Step 1: Write failing test**

```python
def test_figures_style_exists():
    from app.figures.style import PALETTE, MATPLOTLIB_RC
    assert len(PALETTE) == 6
    assert "figure.dpi" in MATPLOTLIB_RC
```

- [ ] **Step 2: Run test fails**

Run: `pytest tests/test_figures.py::test_figures_style_exists -v` → ModuleNotFoundError

- [ ] **Step 3: Implement style.py**

```python
# backend/app/figures/style.py
PALETTE = ["#0173B2","#DE8F05","#029E73","#CC78BC","#CA9161","#F49DAA"]
MATPLOTLIB_RC = {"figure.dpi": 300, "font.family": "Inter", "axes.titlesize": 12, ...}
```

- [ ] **Step 4: Run test passes**

- [ ] **Step 5: Commit**

---

### Task 3: Typer CLI main.py + commands

**Files:**
- Create: `backend/app/cli/main.py`
- Create: `backend/app/cli/commands/train.py`
- Create: `backend/app/cli/commands/ncd.py`
- Create: `backend/app/cli/commands/figures.py`
- Create: `backend/app/cli/commands/hf.py`
- Create: `backend/app/cli/commands/status.py`
- Modify: `backend/pyproject.toml:16` — add `[project.scripts] orchid = "app.cli.main:app"` + `orchid-osr` alias

- [ ] **Step 1: Write failing test for CLI help**

```python
def test_orchid_help():
    from typer.testing import CliRunner
    from app.cli.main import app
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
```

- [ ] **Step 2: Run fails**

- [ ] **Step 3: Implement main.py with signatures from spec §10**

- [ ] **Step 4: Run passes + `orchid --help` shows rich panels**

- [ ] **Step 5: Commit**

---

### Task 4: Shim cli.py deprecated

**Files:**
- Modify: `backend/cli.py`

- [ ] **Step 1: Test shim warns**

```python
def test_cli_shim_deprecated():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import cli
        assert any("deprecated" in str(x.message) for x in w)
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement shim**

```python
# backend/cli.py
import warnings, sys
warnings.warn("deprecated: usa 'orchid train'", DeprecationWarning, stacklevel=2)
from app.cli.main import app
if __name__ == "__main__":
    sys.exit(app())
# import side-effect: no app() call, test can import safely
```

- [ ] **Step 4: Pass**

- [ ] **Step 5: Commit**

---

### Task 5: Quality gate (ruff, mypy, pre-commit, coverage)

**Files:**
- Modify: `backend/pyproject.toml:45-55` — ruff select + coverage omit
- Create: `.pre-commit-config.yaml` (root)
- Modify: `.github/workflows/ci.yml:60-80` — add mypy job
- Test: `backend/tests/test_quality_gate.py`

- [ ] **Step 1: Write failing test for ruff + coverage config**

```python
# backend/tests/test_quality_gate.py
def test_ruff_strict_and_coverage_omit():
    import tomllib
    cfg = tomllib.load(open("pyproject.toml","rb"))
    sel = cfg["tool"]["ruff"]["lint"]["select"]
    assert "UP" in sel and "D" in sel and "SIM" in sel
    omit = cfg["tool"]["coverage"]["run"]["omit"]
    assert "app/figures/*" in omit
    assert cfg["tool"]["coverage"]["report"]["fail_under"] == 60
```

- [ ] **Step 2: Run test fails**

Run: `cd backend && pytest tests/test_quality_gate.py -v`
Expected: FAIL `UP not in ['E','F','I','W']`

- [ ] **Step 3: Implement pyproject.toml + pre-commit**

```toml
# backend/pyproject.toml
[tool.ruff.lint]
select = ["E","F","I","W","UP","D","PL","SIM"]
[tool.coverage.run]
omit = ["app/figures/*", "app/api/*"]
[tool.coverage.report]
fail_under = 60
```

```yaml
# .pre-commit-config.yaml (root)
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
        entry: mypy --strict backend/app/cli backend/app/core --ignore-missing-imports
        language: system
        pass_filenames: false
```

- [ ] **Step 4: Run test passes + `ruff check` passes + `pre-commit run --all` passes**

Run: `cd backend && pytest tests/test_quality_gate.py -v` → PASS
Run: `ruff check app/ tests/ --select UP,D,PL,SIM` → 0 errors (or fix)

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml .pre-commit-config.yaml backend/tests/test_quality_gate.py
git commit -m "chore: ruff strict + mypy + coverage 60 + pre-commit"
```

---

### Task 6: Implement figure builders (pixel-perfect, 4 moduli)

**Files:**
- Modify: `backend/app/figures/closed.py` — `build_closed(fig_dir, exps) -> list[Path]`
- Modify: `backend/app/figures/osr.py` — `build_osr(fig_dir, exps) -> list[Path]`
- Modify: `backend/app/figures/gcd.py` — `build_gcd(fig_dir, exps) -> Path`
- Modify: `backend/app/figures/reliability.py` — `build_reliability(fig_dir, exps) -> list[Path]`
- Modify: `backend/app/figures/sync.py` — `sync_frontend(src, dst)` + `style.py`
- Test: `backend/tests/test_figures_build.py`

- [ ] **Step 1: Write failing test for closed builder**

```python
def test_closed_builds_png(tmp_path):
    from app.figures.closed import build_closed
    exps = {1: Path("backend/experiments/exp1_ce_resnet18/results.json")}
    # fake minimal results.json in tmp_path
    out = build_closed(tmp_path, exps)
    assert len(out) == 2
    assert all(p.suffix == ".png" for p in out)
    assert all(p.stat().st_size > 1000 for p in out)
```

- [ ] **Step 2: Run fails** → ModuleNotFoundError or NotImplemented

- [ ] **Step 3: Implement closed.py**

```python
# backend/app/figures/closed.py
import matplotlib.pyplot as plt
from app.figures.style import PALETTE, MATPLOTLIB_RC
plt.rcParams.update(MATPLOTLIB_RC)
def build_closed(fig_dir: Path, exps: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8,4))
    # ... grouped bar Exp1-5 vs F1, color PALETTE ...
    out_png = fig_dir / "benchmark_bar.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".webp"), dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    return [out_png, out_png.with_suffix(".webp")]
```

- [ ] **Step 4: Run passes + `orchid figures --only closed --check` phash Hamming ≤5**

Run: `pytest tests/test_figures_build.py::test_closed_builds_png -v` → PASS

- [ ] **Step 5: Commit per modulo (closed, osr, gcd, reliability) — 4 commits**

```bash
git add backend/app/figures/closed.py backend/tests/test_figures_build.py
git commit -m "feat(figures): closed builder pixel-perfect"
```

---

### Task 7: Move Exp10 + archive scripts

**Files:**
- Modify: `backend/.gitignore`
- Create: `backend/scripts/archive/thesis-figures/README.md`

```bash
mkdir -p backend/experiments/_archived
# use git mv to keep history
for d in backend/experiments/exp10_*; do [ -d "$d" ] && git mv "$d" backend/experiments/_archived/; done
echo "_archived/" >> backend/.gitignore
git add backend/.gitignore
```

- [ ] **Step 2: Verify `orchid status` excludes _archived**

Run: `orchid status | grep -v _archived` → no exp10

---

### Task 8: Docs + README

**Files:**
- Modify: `README.md:114-120`
- Modify: `AGENTS.md:27-35`
- Test: `backend/tests/test_docs.py`

- [ ] **Step 1: Write failing test**

```python
def test_readme_has_orchid_help():
    txt = open("README.md").read()
    assert "orchid train --help" in txt
    assert "orchid figures --all" in txt
    assert "§3.5" in txt
```

- [ ] **Step 2: Run fails**
- [ ] **Step 3: Update docs**
- [ ] **Step 4: Pass**
- [ ] **Step 5: Commit**

---

### Task 9: CI 4° job mypy

**Files:**
- Modify: `.github/workflows/ci.yml:70-90`

- [ ] **Step 1: Write failing CI check** `grep -q "mypy" .github/workflows/ci.yml || exit 1`
- [ ] **Step 2: Add job**

```yaml
  mypy:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install mypy
      - run: mypy --strict app/cli app/core --ignore-missing-imports
```

- [ ] **Step 3: Verify `act` local or `gh workflow view`**
- [ ] **Step 4: Commit**

---

### Task 10: Final verification

- [ ] `orchid --help` rich
- [ ] `orchid figures --all` genera 8 file in /tmp
- [ ] `pytest --cov` 60 pass
- [ ] `mypy --strict app/cli app/core` pass
- [ ] `pre-commit run --all` pass

