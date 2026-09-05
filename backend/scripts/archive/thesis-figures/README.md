# Archived one-shot scripts (frozen 2026-09-05)

These scripts served their purpose during thesis work and are frozen here for
provenance. Do not run them: live logic lives under `app/` and is reachable
via `orchid --help`.

| Script | Produced | Lives on as |
|---|---|---|
| `dedup_train_224.py` | MD5 leak cleanup | `app/preprocessing/split.py` |
| `domain_shift_analysis.py` | field-vs-internet numbers (§Limitazioni) | thesis text only |
| `extract_all_novel.py`, `extract_novel_embeddings.py` | novel embeddings (old layout) | `app/core/ncd/extract.py` |
| `generate_cv_test_gap.py` | `cv_test_gap.png` | `app/figures/` |
| `generate_gcd_figure.py` | `gcd_old_new.png` (old) | `app/figures/gcd.py` |
| `regenerate_exp9_embeddings.py` | Exp 4 (ex-9) embeddings refresh | `orchid ncd` |
| `upload_exp89_results.py` | old bulk upload | `orchid hf upload-models` |
| `audit_hf.py` | one-off HF repo audit | — |
