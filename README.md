# Orchid Classification with ResNet18

Deep learning pipeline for classifying *Ophrys* orchid species using transfer learning with ResNet18.

**Related Paper:** [ScienceDirect - S2590005625001079](https://www.sciencedirect.com/science/article/pii/S2590005625001079)

---

## Credits

| Role | Name |
|------|------|
| Original research and implementation | **Davide Di Pierro** (UNINA) |
| Repository maintainer | Marcello Russo |

This project extends the original thesis work of Davide Di Pierro at the University of Naples Federico II.

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

## Gallery

### Raw Images

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/raw/O_exaltata.png) | ![](docs/images/raw/O_garganica.png) | ![](docs/images/raw/O_incubacea.png) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/raw/O_majellensis.jpg) | ![](docs/images/raw/O_sphegodes.png) | ![](docs/images/raw/O_sphegodes_Palena.png) |

### Augmented Images

After augmentation (flips, rotations) for class balancing:

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/augmented/O_exaltata.jpg) | ![](docs/images/augmented/O_garganica.jpg) | ![](docs/images/augmented/O_incubacea.jpg) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/augmented/O_majellensis.jpg) | ![](docs/images/augmented/O_sphegodes.jpg) | ![](docs/images/augmented/O_sphegodes_Palena.jpg) |

### Model Output

Classification results with sliding window visualization:

| O. exaltata | O. garganica | O. incubacea |
|:-----------:|:------------:|:------------:|
| ![](docs/images/results/O_exaltata.png) | ![](docs/images/results/O_garganica.png) | ![](docs/images/results/O_incubacea.png) |

| O. majellensis | O. sphegodes | O. sphegodes_Palena |
|:--------------:|:------------:|:-------------------:|
| ![](docs/images/results/O_majellensis.png) | ![](docs/images/results/O_sphegodes.png) | ![](docs/images/results/O_sphegodes_Palena.png) |

---

## Experiments

### Original Experiment (Davide Di Pierro)

Classification on private dataset with 6 species. Results obtained with traditional augmentation (unbalanced classes).

**Configuration:**
- Epochs: 26
- Batch size: 8
- Learning rate: 0.001
- Class weights: No

**Results:**

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

### Per-Class Results (Original)

| Class | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| O. exaltata | 92.86% | 92.86% | 92.86% | 92.86% |
| O. garganica | 92.00% | 92.93% | 93.88% | 92.00% |
| O. incubacea | 93.55% | 87.88% | 82.86% | 93.55% |
| O. majellensis | 92.50% | 93.67% | 94.87% | 92.50% |
| O. sphegodes | 70.00% | 68.29% | 66.67% | 70.00% |
| O. sphegodes_Palena | 94.23% | 97.03% | 100.00% | 94.23% |

### Confusion Matrix (Original)

```
[[52,  0,  0,  0,  4,  0],
 [ 0, 46,  4,  0,  0,  0],
 [ 0,  2, 29,  0,  0,  0],
 [ 0,  0,  0, 37,  3,  0],
 [ 4,  1,  0,  1, 14,  0],
 [ 0,  0,  2,  1,  0, 49]]
```

---

### Domain Shift Experiments (Marcello Russo)

Five experiments to evaluate domain shift impact, data augmentation effectiveness, and catastrophic forgetting.

**Configuration (same as original Di Pierro):**
- Epochs: 26
- Batch size: 8
- Learning rate: 0.001
- Class weights: No
- Balanced augmentation: 280 images/class

**Datasets Used:**

| Dataset | Source | Classes | Images |
|---------|--------|---------|--------|
| Private | Personal collection | 6 | 1,252 |
| Internet Raw | Web (uncropped) | 4 | 1,080 |
| Internet Cropped | Web (cropped) | 4 | 1,080 |
| Combined | Private + Internet Cropped | 6 | 2,332 |

**Note:** The internet dataset contains only 4 classes (O. exaltata, O. garganica, O. incubacea, O. sphegodes), while the private dataset has all 6 classes.

**Overall Results:**

| Experiment | Train | Test | Accuracy | F1 Macro | Description |
|------------|-------|------|----------|----------|-------------|
| Exp1 | Private | Private | 94.78% | 93.72% | Baseline (6 classes) |
| Exp2 | Private | Internet Raw | 35.09% | 37.57% | Strong domain shift |
| Exp3 | Private | Internet Cropped | 48.15% | 50.43% | Cropping helps |
| **Exp4** | **Combined** | **Internet Cropped** | **95.19%** | **95.37%** | **Best on internet** |
| **Exp5** | **Combined** | **Private** | **96.79%** | **95.83%** | **No forgetting** |

### Per-class Results

**Exp4 - Combined Model on Internet Cropped (Best result on internet data):**

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| O. exaltata | 96.26% | 94.50% | 95.37% |
| O. garganica | 95.62% | 95.92% | 95.77% |
| O. incubacea | 95.75% | 96.67% | 96.21% |
| O. sphegodes | 94.28% | 93.99% | 94.14% |

**Exp5 - Combined Model on Private (Catastrophic forgetting test):**

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| O. exaltata | 100.00% | 98.21% | 99.10% |
| O. garganica | 96.00% | 96.00% | 96.00% |
| O. incubacea | 93.55% | 93.55% | 93.55% |
| O. majellensis | 97.56% | 100.00% | 98.77% |
| O. sphegodes | 94.44% | 85.00% | 89.47% |
| O. sphegodes_Palena | 96.30% | 100.00% | 98.11% |

### Key Findings

1. **Domain shift significantly impacts performance**
   - F1 drops from 93.72% (Exp1) to 37.57% (Exp2) when testing on internet images
   - This demonstrates the challenge of generalizing to new data sources

2. **Cropping improves results**
   - Raw images: F1 = 37.57% (Exp2)
   - Cropped images: F1 = 50.43% (Exp3)
   - Improvement: +12.9%

3. **Data augmentation is highly effective**
   - Private-only model on internet: F1 = 50.43% (Exp3)
   - Combined model on internet: F1 = 95.37% (Exp4)
   - Improvement: +44.9%

4. **No catastrophic forgetting**
   - Exp1 (private model on private): F1 = 93.72%
   - Exp5 (combined model on private): F1 = 95.83%
   - The combined model actually performs **better** on the private dataset
   - Adding internet data to training does not degrade performance on original classes

5. **All classes benefit from combined training**
   - Every class achieves >94% F1 on internet data (Exp4)
   - Every class achieves >89% F1 on private data (Exp5)

---

## Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Podman (or Docker)
- NVIDIA GPU + Drivers + Container Toolkit

### Run

```bash
# Clone the repository
git clone https://github.com/marcellorusso-x/orchid-classification.git
cd orchid-classification/resnet18

# Start the container
podman compose up -d --build

# Open Jupyter
# http://localhost:8888
```

### Running Experiments

```bash
# Run all experiments (inside container)
python run_experiments.py --all

# Or run individual phases
python run_experiments.py --status     # Check status
python run_experiments.py --setup     # Setup datasets
python run_experiments.py --preprocess # Preprocess data
python run_experiments.py --train     # Train models
python run_experiments.py --test      # Run tests
```

### Notebooks

| File | Description |
|------|-------------|
| `notebook_dipierro.ipynb` | Original notebook by Di Pierro |
| `notebook.ipynb` | Clean restructured version (replicates run_pipeline.py) |

---

## Pipeline

```
Raw Images -> Augmentation -> Normalization -> Preprocessing -> Training -> Testing
```

1. **Augmentation**: Balance classes with flips, rotations (target: 280 images/class in train)
2. **Normalization**: Calculate mean/std for the dataset
3. **Preprocessing**: Convert images to PyTorch tensors
4. **Training**: Fine-tune ResNet18 (pretrained on ImageNet)
5. **Testing**: Evaluate on test set with confusion matrix and explainability visualization

---

## Project Structure

```
resnet18/
├── app/                        # Source code
│   ├── preprocessing_tools/    # Data preprocessing utilities
│   ├── explenability_tools/    # Model explainability (Captum)
│   ├── train_model.py          # Training loop
│   ├── test_model.py           # Testing and evaluation
│   └── timing.py               # Timing utilities
├── dataset/
│   ├── raw/                    # Private images by class
│   ├── internet_raw/           # Internet images (uncropped)
│   ├── internet_cropped/       # Internet images (cropped)
│   ├── combined/               # Combined dataset (private + internet)
│   └── *.pt                    # Preprocessed tensors
├── models/                     # Trained models
│   ├── train_privato/          # Private-only model
│   └── train_combined/         # Combined model
├── results/                    # Test outputs and visualization
│   ├── exp1_privato_privato/   # Exp1 output images
│   ├── exp2_privato_internet_raw/
│   ├── exp3_privato_internet_cropped/
│   ├── exp4_combined_internet_cropped/
│   └── exp5_combined_privato/
├── logs/                       # Timing logs
├── docs/                       # Documentation and images
├── run_experiments.py          # Run all experiments
├── run_pipeline.py             # CLI for single pipeline
├── generate_report.py           # Generate final report
├── notebook_dipierro.ipynb     # Original notebook
├── notebook.ipynb              # Clean notebook
├── Dockerfile
└── docker-compose.yaml
```

---

## Tech Stack

- **PyTorch 2.1.0** - Deep learning framework
- **ResNet18** - CNN architecture (pretrained on ImageNet)
- **Captum** - Model explainability (Integrated Gradients, Occlusion)
- **Torcheval** - Evaluation metrics
- **Podman** - Container runtime with GPU support
- **CUDA 11.8** - GPU acceleration

---

## Technical Notes

### Class Mismatch Handling

For Exp2-4, the test set has 4 classes while the model outputs 6. When ImageFolder loads the internet test set, it assigns labels alphabetically:
- O. sphegodes in test set: index 3
- O. sphegodes in model: index 4

Metrics were recalculated with proper class index mapping.

### Training Configuration

Same configuration as original Di Pierro thesis:
- ResNet18 pretrained on ImageNet
- 26 epochs
- Batch size: 8
- Learning rate: 0.001
- Class weights: No
- Balanced augmentation: 280 images/class

---

## License

See [LICENSE](LICENSE) for details.
