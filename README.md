
# ResNet18 Orchid Classification Project

This project implements a ResNet18-based image classification pipeline for orchid species. It is containerized using Podman (compatible with Docker) and includes a complete workflow from data augmentation to training and inference.

## 🚀 Getting Started

### Prerequisites
- **Linux Environment** (Tested on Ubuntu)
- **Podman** (System installation via APT recommended)
- **NVIDIA GPU** & Drivers (for training)
- **NVIDIA Container Toolkit**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marcellorussox/orchid-dipierro.git
    cd orchid-dipierro
    ```

2.  **Start the environment:**
    The project uses `podman-compose` to manage the container.
    ```bash
    podman compose up -d --build
    ```
    This command will:
    *   Build the Docker image (optimised with PyTorch & CUDA 12.4).
    *   Mount the necessary volumes (`dataset`, `models`, `logs`, `results`).
    *   Start the Jupyter Notebook server on port **8888**.

3.  **Access the Notebook:**
    Open your browser or VS Code and connect to:
    `http://localhost:8888`

## 📊 Project Workflow

The core logic is contained in `notebook.ipynb`. The workflow consists of:

1.  **Configuration**: Setup of paths, hyperparameters, and device consistency.
2.  **Augmentation**: Dynamic data augmentation to balance classes (saved to `dataset/augmented`).
3.  **Normalization**: Calculation of Mean and STD for the dataset.
4.  **Preprocessing**: Conversion of images to tensors (saved to `dataset/processed`).
5.  **Training**: Fine-tuning of the ResNet18 model.
6.  **Testing**: Evaluation of the model on the test set.

## ⏱️ Performance Logging

The notebook is instrumented to measure execution time for critical blocks.
The timing data is printed to the console and automatically saved to a CSV file on the host:

*   **Log File**: `logs/timing_log.csv`
*   **Format**: `Timestamp, Block Name, Duration (s)`

## 📂 Directory Structure

*   `app/`: Source code for model, training, and preprocessing logic.
    *   `timing.py`: Utility functions for execution timing and logging.
*   `dataset/`: Contains raw, augmented, and processed data.
*   `models/`: Stores trained model checkpoints (`.pt` files).
*   `logs/`: persistent storage for execution logs.
*   `results/`: Output images and inference results.
*   `ProjectDocker/`: Original docker context (legacy).
*   `docker-compose.yaml`: Service definition.
*   `dockerfile`: Container build definition.

## 🛠️ Tech Stack
*   **PyTorch**: Deep Learning framework.
*   **Podman**: Container runtime.
*   **Jupyter**: Interactive development environment.
*   **Python 3.10**: Programming language.
