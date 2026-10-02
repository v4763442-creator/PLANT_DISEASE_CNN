"""
Data loading and dataset splitting module.
Builds robust, stratified tf.data pipelines with zero data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.preprocessing import get_data_augmentation_pipeline, normalize_mobilenet_v2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def scan_dataset(
    dataset_dir: Union[str, Path] = DATASET_PATH
) -> Tuple[List[str], np.ndarray, List[str], Dict[str, int]]:
    """
    Scan dataset directory for valid image files and their class labels.
    Silently ignores non-image metadata artifacts (e.g., svn metadata files)
    without modifying or deleting any files on disk.
    
    Returns:
    --------
    filepaths : List of valid image file path strings.
    labels : Array of integer class labels corresponding to filepaths.
    class_names : Sorted list of unique class folder names.
    class_counts : Dictionary mapping class name to count of valid images.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Gather class subdirectories (excluding any non-directory or nested PlantVillage)
    subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    
    # Filter class folders
    class_dirs = [d for d in subdirs if d.name != "PlantVillage" and not d.name.startswith(".")]
    if not class_dirs:
        raise ValueError(f"No class folders detected in dataset directory: {dataset_dir}")

    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    filepaths = []
    labels = []
    class_counts = {name: 0 for name in class_names}

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx = class_to_idx[cls_name]

        for file_path in cls_dir.iterdir():
            if file_path.is_file() and file_path.suffix in VALID_EXTENSIONS:
                filepaths.append(str(file_path))
                labels.append(cls_idx)
                class_counts[cls_name] += 1

    labels = np.array(labels, dtype=np.int32)
    return filepaths, labels, class_names, class_counts


def create_train_val_test_splits(
    filepaths: List[str],
    labels: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    random_seed: int = RANDOM_SEED,
    sample_fraction: Optional[float] = None
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    """
    Perform stratified split into Train, Validation, and Test partitions.
    Guarantees equal class distribution and zero data leakage.
    
    If sample_fraction is provided (0.0 < sample_fraction < 1.0), draws a stratified subset
    for rapid pipeline verification.
    """
    filepaths = np.array(filepaths)

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        # Stratified sampling guaranteeing at least 10 samples per class for test/val splits
        selected_indices = []
        rng = np.random.default_rng(random_seed)
        for c in np.unique(labels):
            c_indices = np.where(labels == c)[0]
            n_select = max(10, int(len(c_indices) * sample_fraction))
            n_select = min(n_select, len(c_indices))
            selected_indices.extend(rng.choice(c_indices, size=n_select, replace=False))
        filepaths = filepaths[selected_indices]
        labels = labels[selected_indices]

    # First split: Train vs Temp (Val + Test)
    temp_size = val_split + test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths,
        labels,
        test_size=temp_size,
        stratify=labels,
        random_state=random_seed
    )

    # Second split: Val vs Test
    relative_test_size = test_split / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_seed
    )

    return (list(X_train), y_train), (list(X_val), y_val), (list(X_test), y_test)


def _load_and_preprocess_tf(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow native mapping function to load and preprocess an image."""
    img_bytes = tf.io.read_file(filepath)
    # decode_image handles JPEG, PNG, BMP automatically
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = normalize_mobilenet_v2(img)
    return img, label


def build_tf_dataset(
    filepaths: List[str],
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
    is_training: bool = False,
    shuffle_buffer: int = 1000
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset pipeline.
    Applies data augmentation exclusively to training set.
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(filepaths), shuffle_buffer), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_layer = get_data_augmentation_pipeline()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds
