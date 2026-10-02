"""
Image preprocessing and data augmentation module for MobileNetV2.
Ensures identical preprocessing across training, evaluation, and inference.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE


def get_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Construct a realistic, non-destructive data augmentation pipeline.
    Leaves the test and validation sets unaltered.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
            tf.keras.layers.RandomRotation(0.08, fill_mode="nearest", name="aug_small_rotation"), # ~15 degrees
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest", name="aug_small_zoom"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", name="aug_small_translation"),
        ],
        name="data_augmentation",
    )


def normalize_mobilenet_v2(image_tensor: tf.Tensor) -> tf.Tensor:
    """
    Preprocess image tensor using standard MobileNetV2 scaling:
    Maps pixel values in [0, 255] to [-1, 1].
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_tensor)


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, Image.Image]:
    """
    Load and preprocess a single image for inference.
    
    Parameters:
    -----------
    image_input : File path (str/Path), PIL Image, or numpy array.
    target_size : Target tuple (height, width), default (224, 224).
    
    Returns:
    --------
    preprocessed_array : Batch array of shape (1, height, width, 3) normalized to [-1, 1].
    display_image : PIL Image in RGB format suitable for UI display.
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Convert OpenCV BGR or raw numpy array to PIL
        if image_input.dtype != np.uint8:
            image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize image with high-quality Lanczos resampling
    resized_pil = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized_pil, dtype=np.float32)

    # Add batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # Normalize to [-1, 1] using MobileNetV2 standard
    preprocessed_batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch_array)

    return preprocessed_batch, pil_img
