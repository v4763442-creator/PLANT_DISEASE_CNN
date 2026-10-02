"""
Utility functions for Plant Disease Prediction.
Handles class mapping, history plotting, metrics reporting, and reproducible seeding.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_class_names(class_names: List[str], filepath: Union[str, Path]) -> None:
    """Save the list of class names to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"[INFO] Saved {len(class_names)} classes to {filepath}")


def load_class_names(filepath: Union[str, Path]) -> List[str]:
    """Load class names from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Class names file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_class_name(raw_name: str) -> str:
    """
    Convert raw folder name (e.g., 'Tomato__Tomato_YellowLeaf__Curl_Virus')
    into a clean, readable name (e.g., 'Tomato: Tomato Yellow Leaf Curl Virus').
    """
    clean = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    # Clean up double dashes or double spaces
    clean = " ".join(clean.split())
    clean = clean.replace(" - - ", " - ")
    return clean


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate balanced class weights to address dataset imbalance.
    Returns a dictionary mapping class integer index to weight.
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    return class_weights


def plot_training_history(history_dict: dict, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot and save training & validation accuracy and loss curves.
    Accepts history dictionary or History object's .history attribute.
    """
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, "r-s", label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, "r-s", label="Validation Loss", linewidth=2)
    plt.title("Model Loss across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved training history plot to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = False
) -> None:
    """
    Compute, plot, and save the confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    clean_labels = [format_class_name(c) for c in class_names]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=True
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved confusion matrix heatmap to {save_path}")
    plt.close()


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path]
) -> str:
    """
    Generate and save scikit-learn classification report (precision, recall, f1-score).
    """
    clean_labels = [format_class_name(c) for c in class_names]
    report = classification_report(
        y_true,
        y_pred,
        target_names=clean_labels,
        digits=4,
        zero_division=0
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("PLANT DISEASE PREDICTION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"[INFO] Saved classification report to {save_path}")
    return report
