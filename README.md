[🇮🇹 Italiano](README.it.md)

# 🌸 OrchID — Orchid Representation Learning

> Learning structured vector spaces of *Ophrys* orchid species through contrastive learning and metric embedding techniques.

---

## 🎯 Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **P1 — Classification** | ResNet18/50 baseline, domain shift experiments, k-fold CV | ✅ Done |
| **P2 — Embedding Space** | Contrastive learning (SimCLR, Triplet Loss, SupCon) to build a structured orchid vector space | 🔜 Next |
| **P3 — Retrieval & Similarity** | Nearest-neighbor search, species clustering, unknown class detection | 📋 Planned |

---

## 🌿 Species

| | Species |
|---|---------|
| 🌸 | *Ophrys exaltata* |
| 🌺 | *Ophrys garganica* |
| 🌻 | *Ophrys incubacea* |
| 🌼 | *Ophrys majellensis* |
| 🌷 | *Ophrys sphegodes* |
| 🌱 | *Ophrys sphegodes* (Palena) |

---

## 📸 Gallery

### *Ophrys exaltata*
<p float="left">
<img src="assets/images/raw/O_exaltata.png" height="220">
<img src="assets/images/augmented/O_exaltata.jpg" height="220">
<img src="assets/images/results/O_exaltata.png" height="220">
</p>

### *Ophrys garganica*
<p float="left">
<img src="assets/images/raw/O_garganica.png" height="220">
<img src="assets/images/augmented/O_garganica.jpg" height="220">
<img src="assets/images/results/O_garganica.png" height="220">
</p>

### *Ophrys incubacea*
<p float="left">
<img src="assets/images/raw/O_incubacea.png" height="220">
<img src="assets/images/augmented/O_incubacea.jpg" height="220">
<img src="assets/images/results/O_incubacea.png" height="220">
</p>

<details>
<summary>See all species</summary>

### *Ophrys majellensis*
<p float="left">
<img src="assets/images/raw/O_majellensis.jpg" height="220">
<img src="assets/images/augmented/O_majellensis.jpg" height="220">
<img src="assets/images/results/O_majellensis.png" height="220">
</p>

### *Ophrys sphegodes*
<p float="left">
<img src="assets/images/raw/O_sphegodes.png" height="220">
<img src="assets/images/augmented/O_sphegodes.jpg" height="220">
<img src="assets/images/results/O_sphegodes.png" height="220">
</p>

### *Ophrys sphegodes* (Palena)
<p float="left">
<img src="assets/images/raw/O_sphegodes_Palena.png" height="220">
<img src="assets/images/augmented/O_sphegodes_Palena.jpg" height="220">
<img src="assets/images/results/O_sphegodes_Palena.png" height="220">
</p>

</details>

---

## 📊 Results — Phase 1 (Classification)

### K-Fold Cross-Validation (5-fold, combined dataset)

| Model | Val Accuracy | Val F1 | Test Accuracy | Test F1 |
|-------|-------------|--------|---------------|---------|
| ResNet18 | 86.97% ± 1.43% | 84.85% ± 1.72% | 97.69% | 97.74% |
| ResNet50 | 85.18% ± 1.89% | 84.85% ± 1.72% | 97.41% | 97.67% |

<details>
<summary>Domain shift experiments</summary>

| Experiment | Train | Test | Accuracy | F1 Macro | Notes |
|------------|-------|------|----------|----------|-------|
| Exp1 | Private | Private | 94.78% | 93.72% | Baseline (6 classes) |
| Exp2 | Private | Internet Raw | 35.09% | 37.57% | Strong domain shift |
| Exp3 | Private | Internet Cropped | 48.15% | 50.43% | Cropping helps (+12.9%) |
| **Exp4** | **Combined** | **Internet Cropped** | **95.19%** | **95.37%** | **Best on internet (+44.9%)** |
| **Exp5** | **Combined** | **Private** | **96.79%** | **95.83%** | **No catastrophic forgetting** |

**Key findings:**
1. Domain shift drops F1 from 93.72% → 37.57%
2. Cropping recovers +12.9%
3. Combined training recovers +44.9% (F1 → 95.37%)
4. No catastrophic forgetting (F1 93.72% → 95.83%)

</details>

---

## 🚀 Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Podman (or Docker) + NVIDIA GPU + Container Toolkit

### Run

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/resnet18

# Start container
podman compose up -d --build

# Jupyter at http://localhost:8888
```

### CLI

```bash
python cli.py --status                              # Check status
python cli.py --all                                  # Run all experiments

# K-fold cross-validation
python cli.py --cross-validate --exp 4 --model resnet50 --epochs 100

# Single fold (for long runs)
python cli.py --cross-validate --exp 4 --model resnet50 --fold 1
python cli.py --cross-validate --exp 4 --model resnet50 --aggregate-only
```

---

## 🏗️ Project Structure

```
resnet18/
├── app/                          # Source code (modular)
│   ├── core/                     # ML core
│   │   ├── models/               # ResNet classifier (ABC pattern)
│   │   ├── training.py           # Training loop
│   │   ├── evaluation.py         # Metrics & evaluation
│   │   └── kfold.py              # K-fold cross-validation
│   ├── cli/                      # CLI interface
│   ├── api/                      # FastAPI endpoints
│   ├── preprocessing/            # Data preprocessing
│   └── explainability/           # Captum explainability
├── tests/                        # Unit tests (61 tests)
├── experiments/                  # Experiment results (one dir each)
├── dataset/                      # Datasets (gitignored)
├── models/                       # Trained models (gitignored)
├── cli.py                        # CLI entrypoint
├── server.py                     # API server entrypoint
└── Dockerfile
```

---

## 🔧 Tech Stack

| | Component |
|---|-----------|
| 🧠 | PyTorch 2.1.0 + CUDA 11.8 |
| 🏛️ | ResNet18/50 (ImageNet pretrained) |
| 🔍 | Captum (explainability) |
| 📦 | Podman + NVIDIA Container Toolkit |

---

## 📡 API

```bash
python server.py                    # Start API on port 5000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Classify image (base64) |
| `/docs` | GET | OpenAPI docs |

---

## 📄 License

See [LICENSE](LICENSE).
