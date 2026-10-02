"""
Model architecture and transfer learning module using MobileNetV2.
Supports frozen feature-extraction (Stage 1) and upper-layer fine-tuning (Stage 2).
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

from src.config import (
    FINE_TUNE_LEARNING_RATE,
    INITIAL_LEARNING_RATE,
    INPUT_SHAPE,
    UNFREEZE_LAYERS,
)


def build_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    weights: Optional[str] = "imagenet",
    learning_rate: float = INITIAL_LEARNING_RATE,
    dropout_rate: float = 0.3
) -> Tuple[Model, Model]:
    """
    Construct the Transfer Learning model using MobileNetV2 base + custom classification head.
    
    Parameters:
    -----------
    num_classes : Number of target classes detected in the dataset.
    input_shape : Input image dimensions, default (224, 224, 3).
    weights : Pretrained weights ('imagenet' or None).
    learning_rate : Initial learning rate for Stage 1.
    dropout_rate : Dropout rate for regularization.
    
    Returns:
    --------
    model : Compiled tf.keras.Model ready for Stage 1 training.
    base_model : Underlying MobileNetV2 base model.
    """
    # Instantiate pretrained base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    # STAGE 1: Freeze entire base model
    base_model.trainable = False

    # Classification Head
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(dropout_rate, name="head_dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_features")(x)
    x = BatchNormalization(name="head_batch_norm")(x)
    x = Dropout(dropout_rate * 0.7, name="head_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PlantDisease_MobileNetV2")

    # Compile for Stage 1
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model, base_model


def configure_fine_tuning(
    model: Model,
    base_model: Model,
    unfreeze_layers: int = UNFREEZE_LAYERS,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE
) -> Model:
    """
    STAGE 2: Unfreeze top layers of MobileNetV2 for gentle fine-tuning.
    Freezes all BatchNormalization layers to preserve population statistics.
    Re-compiles with a reduced learning rate.
    """
    base_model.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - unfreeze_layers)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Freeze BatchNormalization layers inside the unfreezed block
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Re-compile with small fine-tuning learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"[INFO] Fine-tuning configured: {unfreeze_layers} upper layers unfrozen.")
    print(f"[INFO] Trainable parameters: {trainable_count:,}")

    return model
