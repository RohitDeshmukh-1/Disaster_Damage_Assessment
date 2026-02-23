#  Building Damage Assessment (xBD Dataset)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance deep learning pipeline for **Satellite Imagery Building Damage Assessment** using the xBD dataset. This project implements a **Siamese U-Net** architecture to perform change detection and multi-class damage segmentation on pre- and post-disaster imagery.

---

## 🚀 Key Features

*   **Siamese U-Net Architecture**: Concurrent processing of pre- and post-disaster satellite imagery for precise change detection.
*   **Mixed Precision Training**: Optimized for speed and memory efficiency using `mixed_float16`.
*   **Custom Sparse IoU Metric**: Robust evaluation metric tailored for multi-class building damage segmentation.
*   **Automated Data Pipeline**: Scripts for mask generation, polygon-to-pixel conversion, and real-time augmentation.
*   **Intelligence Inferences**: Professional assessment reports with severity scoring and side-by-side visualizations.

---

## 📊 Dataset: xBD

The [xBD dataset](https://xview2.org/dataset) is a large-scale dataset for building damage assessment. 

### Conceptual Design
*   **Pre-disaster data** defines **where** buildings exist.
*   **Post-disaster data** defines **what changed** (damage severity).
*   The project uses **Tier 1** imagery (highest quality annotations) by default.

### Damage Classes
1.  **No Damage** (Green)
2.  **Minor Damage** (Yellow)
3.  **Major Damage** (Orange)
4.  **Destroyed** (Red)

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/RohitDeshmukh-1/Disaster_Damage_Assessment.git
    cd Building-Damage-Assessment
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Download Model Weights

Since the trained model file is large (~1GB), it is not included in the repository.
1. **Download** the `xbd_tier1_best.keras` file from [GitHub Releases](https://github.com/RohitDeshmukh-1/Disaster_Damage_Assessment/releases/tag/MODEL).
2. **Place** the file in: `results/models/xbd_tier1_best.keras`

---

## 🏃 Usage

### 1. Data Preprocessing
Generate semantic masks from the raw xBD JSON annotations.
```bash
python preprocess.py --data_dir path/to/xbd/tier1 --out_model data/processed/train_masks
```

### 2. Training
Train the Siamese U-Net model with mixed precision and early stopping.
```bash
python train.py
```
*Weights are automatically saved to `results/models/xbd_tier1_best.keras`.*

## 📈 Web Dashboard & API

This project includes a professional-grade web interface for real-time disaster assessment.

### 1. Start the Inference Engine (Backend)
The backend is powered by FastAPI and handles the neural fusion of image pairs.
```bash
python server.py
```
*Port: 8000*

### 2. Launch the Dashboard (Frontend)
Serve the glassmorphism UI to interact with the model.
```bash
cd web
python -m http.server 8081
```
*Access via: http://localhost:8081*

---

## 📈 Sample Assessment Report

The `inference.py` and Web UI generate comprehensive visualizations including:
*   **Side-by-side comparison** (Pre vs Post imagery).
*   **AI-Generated Damage Mask** (Color-coded by severity).
*   **Damage Distribution Analytics** (Percentage breakdown of structural states).
*   **Cyclone Severity Score (0-10)** (Immediate impact metric).

---

## 📂 Project Structure

```text
Building-Damage-Assessment/
├── src/
│   ├── dataloader.py      # Data generation and augmentation
│   ├── model.py           # Siamese U-Net implementation
│   └── __init__.py
├── results/
│   ├── models/            # Place trained .keras files here
│   └── logs/              # Training logs
├── web/                   # Glassmorphism Frontend Dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train.py               # Main training entry point
├── server.py              # FastAPI Inference Server
├── inference.py           # Optimized CLI evaluation script
├── preprocess.py          # Mask generation script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
