# Smart Workspace Monitor
### Context-aware object detection for productivity

A computer vision system that detects desk objects using YOLOv8 and classifies workspaces as **Focused** or **Distracted** using machine learning.

---

## Project Overview

This project combines:
- **YOLOv8** for real-time object detection
- **Feature extraction** from detected objects
- **Context classification** (Focused vs Distracted)
- **OpenCV** for visualization

## Features

- Real-time desk object detection
- Workspace context classification
- Webcam integration with live overlay
- Productivity scoring with smoothed predictions
- Feature extraction and analysis pipeline

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)

| Tool | Role |
|------|------|
| YOLOv8n (Ultralytics) | Object detection |
| TensorFlow | Context classifier |
| PyTorch / NumPy | Feature extraction utilities |
| OpenCV | Visualization & webcam capture |
| Pandas / scikit-learn | Data handling & evaluation |

---

## Project Structure

```
smart-workspace-monitor/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── raw_images/          # original desk images
│   ├── labels/              # YOLO label files (.txt)
│   └── processed/
│       └── images.csv       # feature-extracted dataset
│
├── models/
│   ├── yolo/                # yolov8n.pt (auto-downloaded)
│   └── context_classifier/
│       ├── context_model.keras
│       ├── scaler.pkl
│       └── feature_cols.json
│
├── src/
│   ├── detect_objects.py        # Phase 1 — YOLO detection
│   ├── extract_features.py      # Phase 2 — detections → CSV
│   ├── train_context_model.py   # Phase 3 — train TF classifier
│   ├── predict_context.py       # Phase 3 — single image prediction
│   └── webcam_demo.py           # Phase 4 — real-time demo
│
├── results/
│   ├── metrics/             # confusion matrix, ROC, training plots
│   └── sample_outputs/      # annotated frame screenshots
│
└── notebooks/
    └── experiments.ipynb    # EDA, evaluation, charts
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<aleesha-10>/smart-workspace-monitor.git
cd smart-workspace-monitor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

YOLOv8n weights (~6 MB) are downloaded automatically on first run.

---

## Usage

### Phase 1 — Object detection on images

```bash
python src/detect_objects.py --image path/to/desk.jpg
```

### Phase 2 — Extract features from a folder of images

```bash
python src/extract_features.py --input dataset/raw_images --output dataset/processed/images.csv
```

### Phase 3 — Train the context classifier

```bash
python src/train_context_model.py
```

Reads `dataset/processed/images.csv`, trains a TensorFlow Dense network, and saves:
- `models/context_classifier/context_model.keras`
- `results/metrics/confusion_matrix.png`
- `results/metrics/training_history.png`
- `results/metrics/metrics.json`

### Phase 3 — Predict on a single image

```bash
python src/predict_context.py --image path/to/desk.jpg --save
```

### Phase 4 — Real-time webcam demo

```bash
python src/webcam_demo.py                     # default webcam
python src/webcam_demo.py --source 1          # second camera
python src/webcam_demo.py --source desk.mp4   # video file
```

**Controls:** `Q` / `ESC` quit &nbsp;|&nbsp; `S` save frame &nbsp;|&nbsp; `P` pause

---

## Pipeline

```
Image / Webcam
      ↓
YOLOv8 Object Detection
      ↓
Feature Extraction (object counts, ratios, area coverage)
      ↓
TensorFlow Context Classifier
      ↓
OpenCV Visualization + Result
```

---

## Model Architecture

```
Input (15 features)
      ↓
Dense(64, relu) → BatchNorm → Dropout(0.3)
      ↓
Dense(32, relu) → BatchNorm → Dropout(0.2)
      ↓
Dense(16, relu)
      ↓
Dense(1, sigmoid)  →  Focused probability
```

**Features used:**

| Feature | Description |
|---------|-------------|
| `total_objects` | Total YOLO detections |
| `productive_count` | Laptops, books, keyboards |
| `distracting_count` | Phones, TVs, remotes |
| `focused_score` | `productive_ratio − distracting_ratio` |
| `clutter_score` | Normalised object count |
| `avg_confidence` | Mean YOLO detection confidence |
| `has_laptop/phone/book/person` | Binary presence flags |

---

## Evaluation Metrics

> Results saved to `results/metrics/` after running `train_context_model.py`

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| AUC-ROC | Ability to distinguish classes across thresholds |
| Confusion matrix | TP / FP / TN / FN breakdown |
| mAP@0.5 | YOLO detection quality (IoU threshold 0.5) |

---

## What This Project Demonstrates

-  Multi-stage CV pipeline (detect → extract → classify → visualise)
- Transfer learning with YOLOv8 pretrained on COCO
- TensorFlow classifier with early stopping and learning rate scheduling
- Feature engineering from raw bounding-box detections
- Real-time OpenCV overlay with smoothed predictions
- Reproducible ML: saved scalers, feature lists, model checkpoints
- Experiment tracking notebook with EDA, ROC, confusion matrix

---

## License

MIT
