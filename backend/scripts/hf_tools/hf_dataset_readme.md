---
license: mit
task_categories:
  - image-classification
tags:
  - biology
  - flowers
  - orchid
  - ophrys
  - fine-grained-classification
  - novel-class-discovery
size_categories:
  - 1K<n<10K
viewer: true
---

# OrchID OSR-NCD — Ophrys Orchid Classification Dataset

6-class ultra-fine-grained image dataset for **Ophrys** orchid species classification from the [*Ophrys sphegodes*](https://en.wikipedia.org/wiki/Ophrys_sphegodes) complex — a group of Mediterranean orchids known for extreme morphological similarity and frequent hybridization.

Part of the **OrchID OSR-NCD** project: representation learning and novel class discovery applied to biodiversity data.

## Links

- **Live demo:** [marcellorusso-orchid-osr.hf.space](https://marcellorusso-orchid-osr.hf.space)
- **Code:** [github.com/squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Models:** [marcellorusso/orchid-osr-models](https://huggingface.co/marcellorusso/orchid-osr-models)

---

## Classes

Six taxa from the *Ophrys sphegodes* complex:

| Taxon | Train | Test | Source |
|-------|------:|-----:|--------|
| *O. exaltata* | 501 | 50 | private + internet |
| *O. garganica* | 570 | 50 | private + internet |
| *O. incubacea* | 366 | 50 | private + internet |
| *O. majellensis* | 152 | 50 | private only |
| *O. sphegodes* | 433 | 50 | private + internet |
| *O. sphegodes* (Palena) | 210 | 50 | private only |
| **Total** | **2232** | **300** | |

These species are visually similar — even expert botanists struggle to distinguish them — making this a challenging ultra-fine-grained classification benchmark.

---

## Structure

```
raw/
  private/                     Field photos (6 classes, 1252 images)
    O. exaltata/
    O. garganica/
    O. incubacea/
    O. majellensis/
    O. sphegodes/
    O. sphegodes_Palena/
  internet/                    Internet-sourced images (4 classes, 1080 images)
    O.exaltata/
    O.garganica/
    O.incubacea/
    O.sphegodes/

cropped/
  256x512/                     Letterbox 256×512 (ResNet18, ResNet50)
    train/{class}/
    test/{class}/
  224x224/                     Letterbox 224×224 (ConvNeXt, DINOv2)
    train/{class}/
    test/{class}/
```

### What each folder contains

| Folder | Content |
|--------|---------|
| `raw/private/` | Original field photos from private collection — 6 classes |
| `raw/internet/` | Raw internet-sourced images — 4 classes (no *O. majellensis* or Palena online) |
| `cropped/256x512/` | Faster R-CNN cropped + letterbox resized for ResNet architectures |
| `cropped/224x224/` | Same pipeline, sized for ConvNeXt-Tiny/Small and DINOv2-Small/Base |

---

## Preprocessing Pipeline

```
Raw images ──► Faster R-CNN ──► Clean split ──► Letterbox resize
               (flower crop)   (MD5 dedup)     (per-architecture)
```

1. **Crop** — `fasterrcnn_resnet50_fpn` (2 classes: background + flower) detects and crops the orchid flower from each photo
2. **Split** — Clean train/test split with MD5 content deduplication. Zero overlap verified
3. **Resize** — Letterbox resize: images scale to fit target dimensions preserving aspect ratio, black padding fills the rest

### Split rules

- **Seed:** 42 (fully reproducible)
- **Test:** 50 images per class, balanced
  - 4 internet-sourced classes: test drawn from internet cropped, train from all private cropped + remaining internet cropped
  - 2 private-only classes (*O. majellensis*, *O. sphegodes* Palena): test from private cropped, train from remaining private
- **Zero data leakage:** MD5 content hashing guarantees no overlap between train and test
- **15 duplicate images** (identical content, different filenames across sources) removed from training set

---

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="marcellorusso/orchid-osr-dataset",
    repo_type="dataset",
    local_dir="dataset",
)
```

### Load with torchvision

```python
from torchvision import datasets, transforms

# Example: load 224×224 train split
train_dataset = datasets.ImageFolder(
    root="dataset/cropped/224x224/train",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
)
print(train_dataset.classes)  # ['O. exaltata', 'O. garganica', ...]
```

---

## Why this dataset?

Previous experiments suffered from **data leakage** — the same photo, downloaded multiple times under different filenames, appeared in both training and test sets. This dataset was built from scratch with MD5-verified zero overlap, and includes both raw and preprocessed versions so researchers can apply their own pipeline or use ours directly.

---

## Links

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID) · [marcello-russo](https://github.com/marcello-russo)
- **Live demo:** [marcellorusso-orchid-osr.hf.space](https://marcellorusso-orchid-osr.hf.space)
- **Models:** [huggingface.co/marcellorusso/orchid-osr-models](https://huggingface.co/marcellorusso/orchid-osr-models)

## License

[MIT](https://choosealicense.com/licenses/mit/)
