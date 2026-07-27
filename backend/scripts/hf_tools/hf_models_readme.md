---
license: mit
tags:
  - image-classification
  - pytorch
  - resnet
  - convnext
  - dinov2
  - orchid
  - ophrys
---

# OrchID OSR — Model Zoo

Trained models for **Ophrys** orchid classification (6 species), part of the **Novel Class Discovery (NCD)** research.

**Live Demo (Embed Version):** [https://marcellorusso-orchid-osr.hf.space](https://marcellorusso-orchid-osr.hf.space)

All models trained on the [orchid-osr-dataset](https://huggingface.co/datasets/marcellorusso/orchid-osr-dataset) (exp6 clean split, zero data leakage).

## Exp 6 — Cross-Entropy Loss

Fair comparison of 6 architectures with standard cross-entropy loss. All models share identical training configuration for comparability.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 100 |
| Early stopping | Patience=15 on val_loss |
| Batch size | 8 (effective 32 with gradient accumulation ×4) |
| Optimizer | Per-architecture (see below) |
| Scheduler | ReduceLROnPlateau |
| Validation | 5-fold StratifiedKFold (seed=57) |
| Pretrained | ImageNet |

**Per-architecture settings:**

| Model | Optimizer | LR | Weight Decay |
|-------|-----------|----|-------------|
| ResNet18/50 | Adam | 0.001 | 0 |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 |

### Models

| Model | Parameters | Type | Status |
|-------|-----------|------|--------|
| ResNet18 | 11.7M | CNN | Complete |
| ResNet50 | 25.6M | CNN | Complete |
| ConvNeXt-Tiny | 28.6M | Modern CNN | Complete |
| ConvNeXt-Small | 50.2M | Modern CNN | Pending |
| DINOv2-Small | 21M | ViT+LoRA | Complete |
| DINOv2-Base | 86M | ViT+LoRA | Pending |

### Results

| Model | Val F1 Macro | Test Accuracy | Test F1 Macro |
|-------|-------------|---------------|---------------|
| ResNet18 | 0.8680 ± 0.016 | 0.8500 | 0.8496 |
| ResNet50 | 0.9051 ± 0.027 | 0.8800 | 0.8801 |
| ConvNeXt-Tiny | 0.8620 ± 0.020 | 0.8400 | 0.8391 |
| DINOv2-Small | 0.8930 ± 0.006 | 0.8533 | 0.8550 |

### Structure

Each model directory contains:
```
exp6_ce/exp6_clean_split_resnet18/
  fold_0.json          # Per-fold training history (loss, F1, accuracy per epoch)
  fold_1.json
  fold_2.json
  fold_3.json
  fold_4.json
  best_fold_0.pt       # Best model weights per fold
  best_fold_1.pt
  ...
  results.json          # Aggregated metrics (mean±std across folds)
  config.json           # Training configuration
```

## Exp 7 — Supervised Contrastive Learning

Same 6 architectures with SupCon loss (Khosla et al., 2020). Trains same-class proximity + different-class separation in embedding space.

## Hardware

All models trained on NVIDIA Quadro T2000 (4GB VRAM) in a Podman container with GPU passthrough.

## Download

```python
from huggingface_hub import snapshot_download

# Download a specific model
snapshot_download(
    repo_id="marcellorusso/orchid-osr-models",
    allow_patterns="exp6_ce/exp6_clean_split_resnet18/*",
    local_dir="models",
)

# Download everything
snapshot_download(
    repo_id="marcellorusso/orchid-osr-models",
    local_dir="models",
)
```

## Citation

If you use these models, please cite the accompanying thesis (details TBD).
