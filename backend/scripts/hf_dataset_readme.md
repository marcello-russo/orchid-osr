---
license: mit
task_categories:
  - image-classification
tags:
  - biology
  - flowers
  - orchid
  - ophrys
size_categories:
  - 1K<n<10K
---

# OrchID NCD — Orchid Classification Dataset

6-class image dataset for **Ophrys** (orchid) species classification, focused on taxa from the **National Center for Biodiversity (NCD)** research area in Italy.

## Classes

| Class | Train | Test | Source |
|-------|-------|------|--------|
| *O. exaltata* | 446 | 50 | private + internet |
| *O. garganica* | 517 | 50 | private + internet |
| *O. incubacea* | 312 | 50 | private + internet |
| *O. majellensis* | 152 | 50 | private only |
| *O. sphegodes* | 380 | 50 | private + internet |
| *O. sphegodes* Palena | 210 | 50 | private only |
| **Total** | **2017** | **300** | |

## Split details

- **Seed**: 42 (reproducible)
- **Test set**: 50 images per class, balanced, held out from all sources
- **Zero data leakage**: verified via MD5 hash comparison between train and test
- **15 duplicate images** (same content, different filenames) found between private and internet sources, removed from training set
- Internet-sourced classes test from `internet_cropped`, private-only classes test from `raw` (private collection)

## Why this split?

Previous experiments (exp 1–5) suffered from data leakage: training sets included augmented versions of test images, or combined datasets that overlapped with test sources. This split was created with `create_clean_split.py` to ensure zero content overlap.

## Structure

```
train_clean/
  O. exaltata/
  O. garganica/
  O. incubacea/
  O. majellensis/
  O. sphegodes/
  O. sphegodes_Palena/
test_clean/
  O. exaltata/
  ...
split_manifest.json
create_clean_split.py
```

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="marcellorusso/orchid-ncd-dataset",
    repo_type="dataset",
    local_dir="dataset",
)
```

## Citation

If you use this dataset, please cite the accompanying thesis (details TBD).
