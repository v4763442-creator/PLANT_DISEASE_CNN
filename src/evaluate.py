"""
Evaluation module for Plant Disease Prediction.
Evaluates model on unseen test set, generates confusion matrix and classification report.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_JSON_PATH,
    MODEL_PATH,
)
from src.data_loader import build_tf_dataset, create_train_val_test_splits, scan_dataset
from src.utils import load_class_names, plot_confusion_matrix, save_classification_report


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_true: np.ndarray,
    class_names: List[str],
    confusion_matrix_path: Optional[Union[str, Path]] = CONFUSION_MATRIX_PATH,
    report_path: Optional[Union[str, Path]] = CLASSIFICATION_REPORT_PATH,
    metrics_json_path: Optional[Union[str, Path]] = METRICS_JSON_PATH
) -> Dict[str, float]:
    """
    Perform rigorous evaluation of the trained model on the unseen test set.
    Computes loss, accuracy, precision, recall, and F1-score.
    Saves confusion matrix plot and classification report text file.
    """
    print("\n" + "=" * 60)
    print("RUNNING MODEL EVALUATION ON UNSEEN TEST SET")
    print("=" * 60)

    # Evaluate loss and accuracy
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = float(eval_results[0])
    test_acc = float(eval_results[1])

    # Predict probabilities and predicted classes
    print("[INFO] Computing test predictions...")
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    # Ensure y_true matches length of y_pred
    y_true = y_true[:len(y_pred)]

    # Compute macro and weighted metrics
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "total_test_samples": int(len(y_true)),
        "num_classes": len(class_names),
    }

    # Save Confusion Matrix
    if confusion_matrix_path:
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=confusion_matrix_path
        )

    # Save Classification Report
    if report_path:
        report_text = save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=report_path
        )
        print("\nClassification Report:\n" + report_text)

    # Save metrics JSON
    if metrics_json_path:
        metrics_json_path = Path(metrics_json_path)
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"[INFO] Saved evaluation metrics JSON to {metrics_json_path}")

    print("\n" + "-" * 40)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Macro F1:      {f1_macro:.4f}")
    print(f"Weighted F1:   {f1_weighted:.4f}")
    print("-" * 40 + "\n")

    return metrics


if __name__ == "__main__":
    # Standalone evaluation entrypoint
    target_model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
    if not target_model_path.exists():
        print(f"[ERROR] No trained model found at {target_model_path}. Train the model first.")
        exit(1)

    print(f"[INFO] Loading model from: {target_model_path}")
    model = tf.keras.models.load_model(target_model_path)
    class_names = load_class_names(CLASS_NAMES_PATH)

    filepaths, labels, detected_classes, _ = scan_dataset()
    _, _, (test_files, test_labels) = create_train_val_test_splits(filepaths, labels)
    test_ds = build_tf_dataset(test_files, test_labels, is_training=False)

    evaluate_model(
        model=model,
        test_ds=test_ds,
        y_true=test_labels,
        class_names=class_names
    )
