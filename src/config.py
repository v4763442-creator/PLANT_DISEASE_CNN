"""
Central configuration module for Plant Disease Prediction project.
All paths, hyperparameters, and dataset settings are defined here.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory detection
def detect_dataset_path() -> Path:
    """
    Dynamically locate the directory containing the plant disease class folders.
    Checks nested PlantVillage/PlantVillage first, then PlantVillage, then dataset.
    """
    candidates = [
        BASE_DIR / "PlantVillage" / "PlantVillage",
        BASE_DIR / "PlantVillage",
        BASE_DIR / "dataset",
        BASE_DIR / "dataset" / "PlantVillage",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it contains expected class folders
            subdirs = [d.name for d in candidate.iterdir() if d.is_dir()]
            # A valid dataset folder should contain typical plant disease subfolders
            if any("Tomato" in s or "Potato" in s or "Pepper" in s for s in subdirs):
                # Ensure it's not the outer folder that contains a nested 'PlantVillage' folder
                # unless that's all there is
                if "PlantVillage" in subdirs and (candidate / "PlantVillage").is_dir():
                    nested = candidate / "PlantVillage"
                    nested_subdirs = [d.name for d in nested.iterdir() if d.is_dir()]
                    if any("Tomato" in s for s in nested_subdirs):
                        return nested
                return candidate
    return BASE_DIR / "PlantVillage" / "PlantVillage"

DATASET_PATH = detect_dataset_path()

# Model and artifact directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
BEST_MODEL_PATH = MODELS_DIR / "best_plant_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Results and visualization paths
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"

# Image specifications for MobileNetV2
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CHANNELS = 3

# Data splitting ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

# Early stopping & ReduceLROnPlateau settings
PATIENCE_EARLY_STOP = 5
PATIENCE_REDUCE_LR = 3
REDUCE_LR_FACTOR = 0.2

# Transfer learning fine-tuning settings
# Number of top layers to unfreeze in stage 2
UNFREEZE_LAYERS = 30
