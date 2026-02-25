# Orchid Classification with ResNet18

A deep learning pipeline for classifying orchid species of the genus *Ophrys* using transfer learning with ResNet18.

📄 **Related Paper:** [ScienceDirect - S2590005625001079](https://www.sciencedirect.com/science/article/pii/S2590005625001079)

---

## Credits

| Role | Name |
|------|------|
| Original Research & Implementation | **Davide Di Pierro** (UNINA) |
| Repository Maintainer | Marcello Russo |

This project extends the original thesis work by Davide Di Pierro at Università degli Studi di Napoli Federico II.

---

## Species Classified

| Class | Scientific Name |
|-------|-----------------|
| O. exaltata | *Ophrys exaltata* |
| O. garganica | *Ophrys garganica* |
| O. incubacea | *Ophrys incubacea* |
| O. majellensis | *Ophrys majellensis* |
| O. sphegodes | *Ophrys sphegodes* |
| O. sphegodes_Palena | *Ophrys sphegodes* (Palena variant) |

---

## Performance

### Original Implementation (Davide Di Pierro)

| Phase | Duration |
|-------|----------|
| Augmentation | ~118 min |
| Normalization | ~20 sec |
| Preprocessing | ~77 sec |
| Training (26 epochs) | ~30 min |
| Testing | ~27 min |
| **Total** | **~3 hours** |

### Overall Metrics

| Metric (Macro) | Score |
|----------------|-------|
| Accuracy | 89.19% |
| F1 Score | 88.78% |
| Precision | 88.52% |
| Recall | 89.19% |

| Metric (Micro) | Score |
|----------------|-------|
| Accuracy | 91.16% |
| F1 Score | 91.16% |
| Precision | 91.16% |
| Recall | 91.16% |

### Per-Class Results

| Class | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| O. exaltata | 92.86% | 92.86% | 92.86% | 92.86% |
| O. garganica | 92.00% | 92.93% | 93.88% | 92.00% |
| O. incubacea | 93.55% | 87.88% | 82.86% | 93.55% |
| O. majellensis | 92.50% | 93.67% | 94.87% | 92.50% |
| O. sphegodes | 70.00% | 68.29% | 66.67% | 70.00% |
| O. sphegodes_Palena | 94.23% | 97.03% | 100.00% | 94.23% |

### Confusion Matrix

```
[[52,  0,  0,  0,  4,  0],
 [ 0, 46,  4,  0,  0,  0],
 [ 0,  2, 29,  0,  0,  0],
 [ 0,  0,  0, 37,  3,  0],
 [ 4,  1,  0,  1, 14,  0],
 [ 0,  0,  2,  1,  0, 49]]
```

### Current Implementation

*Results will be added after re-running the pipeline.*

---

## Changelog

| Date | Author | Description |
|------|--------|-------------|
| 2026-01-29 | Davide Di Pierro | Initial implementation and thesis defense |
| 2026-02-23 | Marcello Russo | Repository setup, fixed timing system, added clean notebook |
| | | |

---

## Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Podman (or Docker)
- NVIDIA GPU + Drivers + Container Toolkit

### Run

```bash
# Clone the repository
git clone https://github.com/marcello-russo/orchid-dipierro.git
cd orchid-dipierro

# Start the container
podman compose up -d --build

# Open Jupyter
# http://localhost:8888
```

### Notebooks

| File | Description |
|------|-------------|
| `notebook.ipynb` | Original notebook (fixed timing) |
| `notebook_clean.ipynb` | Clean restructured version |

---

## Pipeline

```
Raw Images → Augmentation → Normalization → Preprocessing → Training → Testing
```

1. **Augmentation**: Balance classes with flips, rotations
2. **Normalization**: Calculate mean/std for the dataset
3. **Preprocessing**: Convert images to tensors
4. **Training**: Fine-tune ResNet18 (pretrained on ImageNet)
5. **Testing**: Evaluate on test set with confusion matrix

---

## Project Structure

```
├── app/                    # Source code
│   ├── preprocessing_tools/
│   ├── explenability_tools/
│   ├── train_model.py
│   ├── test_model.py
│   └── timing.py
├── dataset/
│   ├── raw/               # Original images by class
│   ├── augmented/         # Augmented train/val/test
│   └── processed/         # Saved tensors (.pt)
├── models/                # Trained models
├── logs/                  # Timing logs
├── results/               # Test outputs
├── notebook.ipynb         # Original notebook
├── notebook_clean.ipynb   # Clean notebook
├── Dockerfile
└── docker-compose.yaml
```

---

## Tech Stack

- **PyTorch 2.1.0** - Deep learning framework
- **ResNet18** - CNN architecture (pretrained on ImageNet)
- **Captum** - Model explainability (Integrated Gradients, Occlusion)
- **Podman** - Container runtime with GPU support
- **CUDA 11.8** - GPU acceleration

---

## License

See [LICENSE](LICENSE) for details.
