"""
Two-stage training pipeline for Plant Disease Prediction using Transfer Learning.
Executes Stage 1 (frozen base) and Stage 2 (fine-tuning) with callbacks,
class weighting for imbalance handling, evaluation on test set, and metrics generation.
Supports multi-experiment isolation and automatic batch-size memory fallback.
"""

import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    DATASET_PATH,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    INITIAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    PATIENCE_EARLY_STOP,
    PATIENCE_REDUCE_LR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    RESULTS_DIR,
    TRAINING_HISTORY_PATH,
    UNFREEZE_LAYERS,
)
from src.data_loader import (
    build_tf_dataset,
    create_train_val_test_splits,
    scan_dataset,
)
from src.evaluate import evaluate_model
from src.model import build_model, configure_fine_tuning
from src.utils import (
    calculate_class_weights,
    plot_training_history,
    save_class_names,
    set_seed,
)


def train_pipeline(
    dataset_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    initial_epochs: int = INITIAL_EPOCHS,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    initial_lr: float = INITIAL_LEARNING_RATE,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
    unfreeze_layers: int = UNFREEZE_LAYERS,
    sample_fraction: Optional[float] = None,
    use_class_weights: bool = True,
    seed: int = RANDOM_SEED,
    experiment_name: Optional[str] = None,
) -> Dict[str, any]:
    """
    Execute the complete end-to-end training and evaluation pipeline.
    """
    set_seed(seed)
    dataset_dir = dataset_path or DATASET_PATH

    # Set up experiment-specific directories
    if experiment_name:
        exp_results_dir = RESULTS_DIR / experiment_name
        exp_models_dir = MODELS_DIR / experiment_name
    else:
        exp_results_dir = RESULTS_DIR
        exp_models_dir = MODELS_DIR

    exp_results_dir.mkdir(parents=True, exist_ok=True)
    exp_models_dir.mkdir(parents=True, exist_ok=True)

    stage1_ckpt_path = exp_models_dir / "stage1_best_model.keras"
    best_ckpt_path = exp_models_dir / "best_plant_disease_model.keras"
    final_model_path = exp_models_dir / "plant_disease_model.keras"
    exp_class_names_path = exp_models_dir / "class_names.json"

    exp_history_path = exp_results_dir / "training_history.png"
    exp_cm_path = exp_results_dir / "confusion_matrix.png"
    exp_report_path = exp_results_dir / "classification_report.txt"
    exp_metrics_path = exp_results_dir / "metrics.json"

    print("=" * 65)
    print("PLANT DISEASE PREDICTION - TRAINING PIPELINE")
    if experiment_name:
        print(f"Experiment:         {experiment_name}")
    print("=" * 65)
    print(f"Dataset Path:       {dataset_dir}")
    print(f"Batch Size:         {batch_size}")
    print(f"Initial Epochs:     {initial_epochs}")
    print(f"Fine-tune Epochs:   {fine_tune_epochs}")
    print(f"Initial LR:         {initial_lr}")
    print(f"Fine-tune LR:       {fine_tune_lr}")
    print(f"Sample Fraction:    {sample_fraction or 'Full Dataset (1.0)'}")
    print(f"Use Class Weights:  {use_class_weights}")
    print(f"Results Dir:        {exp_results_dir}")
    print(f"Models Dir:         {exp_models_dir}")
    print("=" * 65)

    # 1. Scan Dataset
    print("\n[STEP 1/6] Scanning dataset directory...")
    filepaths, labels, class_names, class_counts = scan_dataset(dataset_dir)
    num_classes = len(class_names)
    print(f"[INFO] Found {len(filepaths)} valid images across {num_classes} classes.")

    # Save class names mapping to both global and experiment directories
    save_class_names(class_names, CLASS_NAMES_PATH)
    if experiment_name:
        save_class_names(class_names, exp_class_names_path)

    # 2. Stratified Data Split
    print("\n[STEP 2/6] Splitting dataset into stratified Train/Val/Test partitions...")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = create_train_val_test_splits(
        filepaths=filepaths,
        labels=labels,
        random_seed=seed,
        sample_fraction=sample_fraction
    )
    total_samples = len(train_files) + len(val_files) + len(test_files)
    print(f"[INFO] Total partition samples: {total_samples:,}")
    print(f"  - Training Set:   {len(train_files):,} samples ({len(train_files)/total_samples*100:.1f}%)")
    print(f"  - Validation Set: {len(val_files):,} samples ({len(val_files)/total_samples*100:.1f}%)")
    print(f"  - Test Set:       {len(test_files):,} samples ({len(test_files)/total_samples*100:.1f}%)")

    # 3. Class Imbalance Handling
    class_weights_dict = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(train_labels)
        print(f"[INFO] Computed balanced class weights for {len(class_weights_dict)} classes.")

    # 4. Construct tf.data Datasets (with batch size fallback)
    print(f"\n[STEP 3/6] Building tf.data pipelines with batch_size={batch_size}...")
    try:
        train_ds = build_tf_dataset(train_files, train_labels, batch_size=batch_size, is_training=True)
        val_ds = build_tf_dataset(val_files, val_labels, batch_size=batch_size, is_training=False)
        test_ds = build_tf_dataset(test_files, test_labels, batch_size=batch_size, is_training=False)
    except (tf.errors.ResourceExhaustedError, MemoryError) as e:
        print(f"[WARNING] Memory error with batch_size={batch_size}: {e}. Falling back to batch_size=16.")
        batch_size = 16
        train_ds = build_tf_dataset(train_files, train_labels, batch_size=batch_size, is_training=True)
        val_ds = build_tf_dataset(val_files, val_labels, batch_size=batch_size, is_training=False)
        test_ds = build_tf_dataset(test_files, test_labels, batch_size=batch_size, is_training=False)

    # 5. Build Model (Stage 1)
    print("\n[STEP 4/6] Building MobileNetV2 Transfer Learning model...")
    model, base_model = build_model(
        num_classes=num_classes,
        learning_rate=initial_lr
    )

    # Stage 1 Callbacks
    stage1_callbacks = [
        ModelCheckpoint(
            filepath=str(stage1_ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # STAGE 1: Train Classification Head
    print("\n" + "=" * 50)
    print(f"STAGE 1: TRAINING CLASSIFICATION HEAD ({initial_epochs} EPOCHS)")
    print("=" * 50)
    try:
        history_stage1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs,
            callbacks=stage1_callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
    except (tf.errors.ResourceExhaustedError, MemoryError) as oom_err:
        print(f"[ERROR] OOM during Stage 1 with batch_size={batch_size}: {oom_err}")
        if batch_size > 16:
            print("[INFO] Halving batch size to 16 and restarting Stage 1...")
            batch_size = 16
            train_ds = build_tf_dataset(train_files, train_labels, batch_size=batch_size, is_training=True)
            val_ds = build_tf_dataset(val_files, val_labels, batch_size=batch_size, is_training=False)
            test_ds = build_tf_dataset(test_files, test_labels, batch_size=batch_size, is_training=False)
            history_stage1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=initial_epochs,
                callbacks=stage1_callbacks,
                class_weight=class_weights_dict,
                verbose=1
            )
        else:
            raise oom_err

    combined_history = {
        "accuracy": list(history_stage1.history.get("accuracy", [])),
        "val_accuracy": list(history_stage1.history.get("val_accuracy", [])),
        "loss": list(history_stage1.history.get("loss", [])),
        "val_loss": list(history_stage1.history.get("val_loss", [])),
    }

    # STAGE 2: Fine-Tuning
    if fine_tune_epochs > 0:
        print("\n" + "=" * 50)
        print(f"STAGE 2: FINE-TUNING UPPER {unfreeze_layers} LAYERS ({fine_tune_epochs} EPOCHS)")
        print("=" * 50)
        model = configure_fine_tuning(
            model=model,
            base_model=base_model,
            unfreeze_layers=unfreeze_layers,
            fine_tune_lr=fine_tune_lr
        )

        stage2_callbacks = [
            ModelCheckpoint(
                filepath=str(best_ckpt_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=PATIENCE_EARLY_STOP,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=REDUCE_LR_FACTOR,
                patience=PATIENCE_REDUCE_LR,
                min_lr=1e-7,
                verbose=1
            ),
        ]

        initial_epoch_stage2 = len(history_stage1.epoch)
        total_epochs = initial_epoch_stage2 + fine_tune_epochs

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch_stage2,
            epochs=total_epochs,
            callbacks=stage2_callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )

        combined_history["accuracy"].extend(history_stage2.history.get("accuracy", []))
        combined_history["val_accuracy"].extend(history_stage2.history.get("val_accuracy", []))
        combined_history["loss"].extend(history_stage2.history.get("loss", []))
        combined_history["val_loss"].extend(history_stage2.history.get("val_loss", []))

    # Save final model
    model.save(str(final_model_path))
    print(f"[INFO] Final model saved to: {final_model_path}")

    # Plot training history
    plot_training_history(combined_history, save_path=exp_history_path)

    # 6. Evaluation on Unseen Test Set
    print("\n[STEP 6/6] Evaluating best model on unseen test set...")
    eval_model = model
    if best_ckpt_path.exists():
        try:
            print(f"[INFO] Loading best Stage 2 model checkpoint from: {best_ckpt_path}")
            eval_model = tf.keras.models.load_model(str(best_ckpt_path))
        except Exception as e:
            print(f"[WARNING] Could not load Stage 2 checkpoint: {e}")
    elif stage1_ckpt_path.exists():
        try:
            print(f"[INFO] Loading best Stage 1 checkpoint from: {stage1_ckpt_path}")
            eval_model = tf.keras.models.load_model(str(stage1_ckpt_path))
        except Exception as e:
            print(f"[WARNING] Could not load Stage 1 checkpoint: {e}")

    test_metrics = evaluate_model(
        model=eval_model,
        test_ds=test_ds,
        y_true=test_labels,
        class_names=class_names,
        confusion_matrix_path=exp_cm_path,
        report_path=exp_report_path,
        metrics_json_path=exp_metrics_path
    )

    # If this experiment yielded improved accuracy, also update the main serving checkpoint
    if experiment_name and best_ckpt_path.exists():
        prev_best_acc = 0.0
        if (RESULTS_DIR / "metrics.json").exists():
            import json
            try:
                with open(RESULTS_DIR / "metrics.json", "r") as f:
                    prev_best_acc = json.load(f).get("test_accuracy", 0.0)
            except Exception:
                pass
        
        if test_metrics["test_accuracy"] >= prev_best_acc:
            print(f"[INFO] Experiment improved accuracy ({test_metrics['test_accuracy']:.4f} >= {prev_best_acc:.4f}). Updating main models/ directory.")
            shutil.copy2(best_ckpt_path, BEST_MODEL_PATH)
            shutil.copy2(final_model_path, MODEL_PATH)

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return {
        "model": eval_model,
        "class_names": class_names,
        "history": combined_history,
        "metrics": test_metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant Disease MobileNetV2 model.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS, help="Stage 1 epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Stage 2 epochs")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Fraction of data for quick run (e.g. 0.25)")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment subfolder name (e.g. experiment_02)")
    args = parser.parse_args()

    train_pipeline(
        dataset_path=Path(args.dataset) if args.dataset else None,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        sample_fraction=args.sample_fraction,
        experiment_name=args.experiment
    )
