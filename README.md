# 🌿 Plant Disease Prediction using CNN and Transfer Learning

An end-to-end deep learning web application that accurately diagnoses plant diseases from leaf imagery using **Convolutional Neural Networks (CNN)**, **Transfer Learning with MobileNetV2**, and **Explainable AI (Grad-CAM)**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.20%2B-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

Plant diseases severely degrade agricultural yields and threaten food security worldwide. Early and accurate diagnosis is critical for containment. This project delivers an automated, high-precision computer vision pipeline capable of detecting and distinguishing 15 distinct crop conditions across Bell Pepper, Potato, and Tomato crops.

### Key Features
- **Accurate Diagnosis:** Achieved **87.34% test accuracy** and **0.8670 Macro F1-score** across 15 categories on unseen test imagery.
- **Two-Stage Transfer Learning:** Combines ImageNet feature extraction with targeted upper-layer fine-tuning of MobileNetV2.
- **Explainable AI (Grad-CAM):** Visualizes the exact spatial regions (lesions, blights, discoloration) driving model classifications.
- **Interactive Streamlit Web App:** Supports drag-and-drop leaf uploads, built-in demo samples, top-3 confidence bar charts, and interactive Grad-CAM heatmap overlay controls.
- **Zero Dataset Dependency for Inference:** Self-contained for cloud deployment via a lightweight (~24.5 MB) production model and bundled demo samples.

---

## 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
           │
           ▼
MobileNetV2 Base (Pretrained on ImageNet, 155 Layers)
- Depthwise Separable Convolutions
- Inverted Residuals & Linear Bottlenecks
           │
           ▼
GlobalAveragePooling2D (Collapses 7×7×1280 to 1280-d vector)
           │
           ▼
Dropout (p = 0.3)
           │
           ▼
Dense Feature Layer (256 units, ReLU activation)
           │
           ▼
BatchNormalization
           │
           ▼
Dropout (p = 0.2)
           │
           ▼
Output Layer: Dense (15 units, Softmax activation)
```

### Staged Training Strategy
1. **Stage 1 (Feature Extraction):** MobileNetV2 base is frozen (`trainable = False`). Only the custom classification head is trained with `Adam(lr = 1e-3)` and balanced class weights.
2. **Stage 2 (Fine-Tuning):** The top 30 layers (`block_14`, `block_15`, `block_16`, `Conv_1`, `out_relu`) are unfrozen and gently optimized with `Adam(lr = 1e-5)` while keeping `BatchNormalization` frozen to preserve running statistics.

---

## 📊 Dataset & Verified Statistics

The model was developed and evaluated using the PlantVillage dataset present in the workspace:
- **Total Valid Images:** `20,638`
- **Total Classes:** `15`
- **Resolution:** Uniformly `256 × 256` RGB pixels
- **Class Imbalance Handling:** Addressed via Scikit-Learn balanced class weighting ($w_j = \frac{N}{K \cdot n_j}$).

| # | Class Name | Crop | Condition | Support (Test Set) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| 1 | `Pepper__bell___Bacterial_spot` | Pepper | Bacterial Spot | 37 | 0.9000 | 0.9730 | 0.9351 |
| 2 | `Pepper__bell___healthy` | Pepper | Healthy | 56 | 0.9655 | 1.0000 | 0.9825 |
| 3 | `Potato___Early_blight` | Potato | Early Blight | 37 | 0.9231 | 0.9730 | 0.9474 |
| 4 | `Potato___healthy` | Potato | Healthy | 5 | 0.7143 | 1.0000 | 0.8333 |
| 5 | `Potato___Late_blight` | Potato | Late Blight | 37 | 0.8919 | 0.8919 | 0.8919 |
| 6 | `Tomato__Target_Spot` | Tomato | Target Spot | 53 | 0.6897 | 0.7547 | 0.7207 |
| 7 | `Tomato__Tomato_mosaic_virus` | Tomato | Mosaic Virus | 14 | 0.9286 | 0.9286 | 0.9286 |
| 8 | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato | Yellow Leaf Curl | 121 | 1.0000 | 0.8678 | 0.9292 |
| 9 | `Tomato_Bacterial_spot` | Tomato | Bacterial Spot | 80 | 0.8132 | 0.9250 | 0.8655 |
| 10 | `Tomato_Early_blight` | Tomato | Early Blight | 37 | 0.9091 | 0.5405 | 0.6780 |
| 11 | `Tomato_healthy` | Tomato | Healthy | 60 | 0.8676 | 0.9833 | 0.9219 |
| 12 | `Tomato_Late_blight` | Tomato | Late Blight | 72 | 0.8571 | 0.9167 | 0.8859 |
| 13 | `Tomato_Leaf_Mold` | Tomato | Leaf Mold | 35 | 0.8378 | 0.8857 | 0.8611 |
| 14 | `Tomato_Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | 67 | 0.8571 | 0.7164 | 0.7805 |
| 15 | `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomato | Spider Mites | 63 | 0.8308 | 0.8571 | 0.8438 |
| **—** | **Overall / Macro Average** | | | **774** | **0.8657** | **0.8809** | **0.8734 Acc** |

---

## 📁 Repository Structure

```
plant_disease_CNN/
│
├── models/
│   ├── best_plant_disease_model.keras  # Final production model (24.5 MB, tracked in Git)
│   └── class_names.json                # Canonical 15-class mapping index
│
├── sample_images/                      # Lightweight demo images for standalone deployment
│   ├── sample_pepper_bacterial_spot.jpg
│   ├── sample_pepper_healthy.jpg
│   ├── sample_tomato_early_blight.jpg
│   └── sample_tomato_healthy.jpg
│
├── src/                                # Reusable modules
│   ├── __init__.py
│   ├── config.py                       # Cross-platform paths and hyperparameters
│   ├── data_loader.py                  # Stratified splitting and tf.data streaming
│   ├── preprocessing.py                # MobileNetV2 normalization & augmentation
│   ├── model.py                        # MobileNetV2 definition & fine-tuning logic
│   ├── train.py                        # Two-stage training pipeline
│   ├── evaluate.py                     # Unseen test set evaluation & report generator
│   ├── predict.py                      # Predictor class & Grad-CAM generator
│   └── utils.py                        # History plotting, class weights, seeding
│
├── results/                            # Authentic evaluation artifacts
│   ├── experiment_02/                  # Experiment 02 metrics & plots (87.34% accuracy)
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── classification_report.txt
│   └── metrics.json
│
├── notebooks/
│   └── plant_disease_training.ipynb    # 18-step interactive training notebook
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation
├── INTERVIEW_NOTES.md                  # Viva/interview preparation guide (25 Q&As)
├── dataset_info.txt                    # Verified dataset inspection report
└── .gitignore                          # Excludes raw dataset, large checkpoints, venv

## ⚠️ Limitations & Future Work

- **Background Variations:** The model was trained on imagery captured against relatively uniform laboratory backdrops. Real-world field photographs with heavy soil clutter, direct sunlight glare, or weed interference may introduce noise.
- **Single-Leaf Focus:** Designed for individual leaf images rather than full field canopy drone captures.
- **Future Scope:** Adding upstream leaf detection with YOLOv8, quantizing the model to TensorFlow Lite (INT8) for offline mobile deployment, and integrating agronomic treatment recommendations.
