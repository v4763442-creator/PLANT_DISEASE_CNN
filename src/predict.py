"""
Inference and Explainable AI (Grad-CAM) module for Plant Disease Prediction.
Loads the trained model and class mappings to produce predictions, confidence scores,
and visual Grad-CAM attribution heatmaps.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import BEST_MODEL_PATH, CLASS_NAMES_PATH, MODEL_PATH
from src.preprocessing import load_and_preprocess_image
from src.utils import format_class_name, load_class_names


class PlantDiseasePredictor:
    """
    Production-grade predictor for Plant Disease classification.
    Maintains loaded model and class names in memory for fast inference.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
    ):
        if model_path is None:
            model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
        self.model_path = Path(model_path)

        if class_names_path is None:
            class_names_path = CLASS_NAMES_PATH
        self.class_names_path = Path(class_names_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found at: {self.class_names_path}")

        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        print(f"[INFO] Loading class mappings from: {self.class_names_path}")
        self.class_names = load_class_names(self.class_names_path)
        self.num_classes = len(self.class_names)

        # Locate the final conv/activation layer for Grad-CAM
        self.gradcam_layer_name = self._find_gradcam_layer()

    def _find_gradcam_layer(self) -> str:
        """Find the last suitable 4D feature map layer for Grad-CAM."""
        candidates = ["out_relu", "Conv_1", "block_16_project"]
        for name in candidates:
            try:
                self.model.get_layer(name)
                return name
            except ValueError:
                continue

        # Fallback: find any layer producing 4D tensor before pooling
        for layer in reversed(self.model.layers):
            if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
                return layer.name
        return "out_relu"

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Run inference on an input image.
        
        Returns:
        --------
        dict containing:
          - predicted_class: Raw class name
          - formatted_class: Clean human-readable name
          - confidence: Float confidence (0.0 to 1.0)
          - top_predictions: List of dicts with top-k predictions and probabilities
          - all_probabilities: Full probability distribution array
        """
        preprocessed_batch, original_pil = load_and_preprocess_image(image_input)

        raw_probs = self.model.predict(preprocessed_batch, verbose=0)[0]
        pred_idx = int(np.argmax(raw_probs))
        top_k = min(top_k, self.num_classes)
        top_indices = np.argsort(raw_probs)[::-1][:top_k]

        top_predictions = [
            {
                "rank": rank + 1,
                "class_name": self.class_names[idx],
                "formatted_name": format_class_name(self.class_names[idx]),
                "probability": float(raw_probs[idx]),
                "percentage": float(raw_probs[idx] * 100),
            }
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "predicted_class": self.class_names[pred_idx],
            "formatted_class": format_class_name(self.class_names[pred_idx]),
            "confidence": float(raw_probs[pred_idx]),
            "confidence_percentage": float(raw_probs[pred_idx] * 100),
            "top_predictions": top_predictions,
            "all_probabilities": raw_probs.tolist(),
            "class_index": pred_idx,
            "preprocessed_batch": preprocessed_batch,
            "original_pil": original_pil,
        }

    def generate_gradcam(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        pred_index: Optional[int] = None,
        alpha: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
        """
        Generate authentic Grad-CAM heatmap visualization.
        
        Parameters:
        -----------
        image_input : Input leaf image
        pred_index : Optional class index to compute gradients for (defaults to top predicted class)
        alpha : Blending transparency (0.0 = original only, 1.0 = heatmap only)
        
        Returns:
        --------
        superimposed_image : RGB numpy array (H, W, 3) with heatmap blended onto leaf image.
        heatmap : Grayscale 2D numpy array [0, 1] representing activation weights.
        original_pil : Original PIL Image in RGB format.
        """
        preprocessed_batch, original_pil = load_and_preprocess_image(image_input)

        if pred_index is None:
            preds = self.model.predict(preprocessed_batch, verbose=0)
            pred_index = int(np.argmax(preds[0]))

        # Build submodel extracting target conv layer outputs and model predictions
        target_layer = self.model.get_layer(self.gradcam_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_batch)
            loss = predictions[:, pred_index]

        # Extract gradients of target class output w.r.t. conv feature maps
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients: weight of each feature channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv feature maps by channel importances
        conv_outputs_val = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_val), axis=-1)

        # Apply ReLU to retain only features that positively influence the target class
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Resize heatmap to original image dimensions
        orig_w, orig_h = original_pil.size
        heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap for intuitive temperature visualization
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        orig_np = np.array(original_pil)
        # Superimpose heatmap onto original image
        superimposed = cv2.addWeighted(orig_np, 1.0 - alpha, colormap_rgb, alpha, 0)

        return superimposed, heatmap_resized, original_pil


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict plant disease from leaf image.")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image.")
    args = parser.parse_args()

    predictor = PlantDiseasePredictor()
    result = predictor.predict(args.image)

    print("\n" + "=" * 50)
    print(f"PREDICTION: {result['formatted_class']}")
    print(f"CONFIDENCE: {result['confidence_percentage']:.2f}%")
    print("=" * 50)
    print("Top 3 Predictions:")
    for item in result["top_predictions"]:
        print(f"  {item['rank']}. {item['formatted_name']} ({item['percentage']:.2f}%)")
