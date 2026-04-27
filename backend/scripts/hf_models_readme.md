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

# OrchID NCD — Model Zoo

Trained models for **Ophrys** orchid classification (6 species), part of the National Center for Biodiversity (NCD) research.

All models trained on the [orchid-ncd-dataset](https://huggingface.co/datasets/marcellorusso/orchid-ncd-dataset) (exp6 clean split, zero data leakage).

## Serie 1 — Cross-Entropy Loss

Fair comparison of 6 architectures with standard cross-entropy loss. All models share identical training configuration for comparability.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 100 |
| Early stopping | Patience=15 on val_loss |
| Batch size | 8 (effective 32 with gradient accumulation ×4) |
| Optimizer | Adam, lr=0.001 |
| Scheduler | ReduceLROnPlateau |
| Validation | 5-fold StratifiedKFold (seed=57) |
| Pretrained | ImageNet |

### Models

| Model | Parameters | Type | Status |
|-------|-----------|------|--------|
| ResNet18 | 11.7M | CNN | Complete |
| ResNet50 | 25.6M | CNN | Complete |
| ConvNeXt-Tiny | 28.6M | Modern CNN | Pending |
| ConvNeXt-Small | 50.2M | Modern CNN | Pending |
| DINOv2-Small | 21M | ViT+LoRA | Pending |
| DINOv2-Base | 86M | ViT+LoRA | Pending |

### Results

| Model | Val F1 Macro | Test Accuracy | Test F1 Macro |
|-------|-------------|---------------|---------------|
| ResNet18 | 0.8623 ± 0.023 | 0.8500 | 0.8496 |
| ResNet50 | 0.9051 ± 0.027 | 0.8800 | 0.8801 |

### Structure

Each model directory contains:
```
serie1_ce/exp6_clean_split_resnet18/
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

## Serie 2 — Supervised Contrastive Loss (planned)

Same 6 architectures with SupCon loss (Khosla et al., 2020). Trains same-class proximity + different-class separation in embedding space.

## Hardware

All models trained on NVIDIA Quadro T2000 (4GB VRAM) in a Podman container with GPU passthrough.

## Download

```python
from huggingface_hub import snapshot_download

# Download a specific model
snapshot_download(
    repo_id="marcellorusso/orchid-ncd-models",
    allow_patterns="serie1_ce/exp6_clean_split_resnet18/*",
    local_dir="models",
)

# Download everything
snapshot_download(
    repo_id="marcellorusso/orchid-ncd-models",
    local_dir="models",
)
```

## Citation

If you use these models, please cite the accompanying thesis (details TBD).
