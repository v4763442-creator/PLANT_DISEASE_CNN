"""
Streamlit Web Application for Plant Disease Prediction.
Integrates MobileNetV2 Transfer Learning and Explainable AI (Grad-CAM).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    DATASET_PATH,
    IMAGE_SIZE,
    MODEL_PATH,
)
from src.predict import PlantDiseasePredictor
from src.utils import format_class_name, load_class_names

# Configure page layout and theme
st.set_page_config(
    page_title="Plant Disease Prediction | CNN & Transfer Learning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F1F8E9;
        border-left: 5px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .disease-badge-healthy {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    .disease-badge-diseased {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> Optional[PlantDiseasePredictor]:
    """Cache and return the predictor instance."""
    target_model = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
    if not target_model.exists() or not CLASS_NAMES_PATH.exists():
        return None
    try:
        return PlantDiseasePredictor(model_path=target_model, class_names_path=CLASS_NAMES_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_sample_images(dataset_dir: Path, max_samples: int = 15) -> List[Path]:
    """Find a selection of sample leaf images for quick demonstration."""
    sample_paths = []
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG"}
    
    # Check bundled deployment demo images first
    bundled_dir = Path(__file__).resolve().parent / "sample_images"
    if bundled_dir.exists():
        for f in sorted(bundled_dir.iterdir()):
            if f.is_file() and f.suffix in valid_exts:
                sample_paths.append(f)

    # Check local dataset directory if present
    if dataset_dir.exists():
        for cls_dir in sorted(dataset_dir.iterdir()):
            if cls_dir.is_dir() and cls_dir.name != "PlantVillage" and not cls_dir.name.startswith("."):
                for f in cls_dir.iterdir():
                    if f.suffix in valid_exts:
                        sample_paths.append(f)
                        break
            if len(sample_paths) >= max_samples:
                break
    return sample_paths


def main():
    st.markdown('<div class="main-title">🌿 Plant Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">CNN + Transfer Learning (MobileNetV2) based plant disease classification</div>',
        unsafe_allow_html=True,
    )

    predictor = get_predictor()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration & Info")
        st.markdown(
            """
            **Model:** MobileNetV2 (ImageNet Pretrained)  
            **Architecture:** Transfer Learning + Fine-Tuning  
            **Input Resolution:** 224 × 224 × 3  
            **Classes Detected:** 15 Plant Disease Classes  
            **Framework:** TensorFlow / Keras  
            """
        )
        st.divider()

        # Input Selection Mode
        input_mode = st.radio(
            "Select Input Source:",
            ["Upload Your Own Leaf Image", "Use Sample Leaf from Dataset"],
            index=0,
        )

        selected_image_path = None
        uploaded_file = None

        if input_mode == "Upload Your Own Leaf Image":
            uploaded_file = st.file_uploader(
                "Upload a leaf image (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                help="Select a clear photograph of a single plant leaf.",
            )
        else:
            sample_images = get_sample_images(DATASET_PATH)
            if sample_images:
                sample_options = {}
                for p in sample_images:
                    if p.parent.name == "sample_images":
                        clean_name = p.stem.replace("sample_", "").replace("_", " ").title()
                        sample_options[f"Demo: {clean_name}"] = p
                    else:
                        sample_options[f"{format_class_name(p.parent.name)} ({p.name})"] = p
                chosen_sample = st.selectbox("Choose sample image:", list(sample_options.keys()))
                selected_image_path = sample_options[chosen_sample]
            else:
                st.info("No sample images detected at configured dataset path.")

        st.divider()
        st.subheader("🔍 Explainable AI")
        enable_gradcam = st.checkbox("Generate Grad-CAM Heatmap", value=True)
        gradcam_alpha = st.slider("Heatmap Intensity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    # Main UI Layout
    if predictor is None:
        st.warning("⚠️ Model weights not found in `models/`.")
        st.markdown(
            """
            To train the model using your existing dataset, run the following command in PowerShell:
            ```bash
            python -m src.train
            ```
            Or for a fast pipeline verification run:
            ```bash
            python -m src.train --sample-fraction 0.05 --initial-epochs 1 --fine-tune-epochs 1
            ```
            """
        )
        return

    # Image Acquisition
    image_to_process = None
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
    elif selected_image_path is not None:
        image_to_process = Image.open(selected_image_path).convert("RGB")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📸 Leaf Image Preview")
        if image_to_process is not None:
            st.image(image_to_process, caption="Uploaded Leaf Image", use_container_width=True)
            predict_button = st.button("🔍 Diagnose Leaf Disease", type="primary", use_container_width=True)
        else:
            st.info("👈 Please upload a leaf image or select a sample from the sidebar.")
            predict_button = False

    with col2:
        st.subheader("📊 Diagnostic Results")
        if image_to_process is not None and predict_button:
            with st.spinner("Analyzing leaf with MobileNetV2..."):
                prediction_result = predictor.predict(image_to_process, top_k=3)

                pred_class = prediction_result["predicted_class"]
                formatted_name = prediction_result["formatted_class"]
                confidence = prediction_result["confidence_percentage"]
                is_healthy = "healthy" in pred_class.lower()

                # Diagnosis Card
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 0.95rem; color: #555; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Diagnosis</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #1B5E20; margin: 4px 0 8px 0;">{formatted_name}</div>
                        <div>
                            <span class="{'disease-badge-healthy' if is_healthy else 'disease-badge-diseased'}">
                                {'✅ Healthy Crop' if is_healthy else '⚠️ Disease Detected'}
                            </span>
                            <span style="font-weight: 600; font-size: 1.05rem; margin-left: 12px;">Confidence: {confidence:.2f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top-3 Predictions Bar Chart
                st.markdown("##### 🏆 Top 3 Predictions")
                top_preds = prediction_result["top_predictions"]
                df_top = pd.DataFrame(
                    {
                        "Class": [item["formatted_name"] for item in top_preds],
                        "Probability (%)": [item["percentage"] for item in top_preds],
                    }
                )

                # Render horizontal probability bars
                for item in top_preds:
                    st.write(f"**{item['formatted_name']}**")
                    st.progress(float(item["probability"]))
                    st.caption(f"Probability: {item['percentage']:.2f}%")

                # Grad-CAM Explainability Section
                if enable_gradcam:
                    st.markdown("---")
                    st.subheader("🔥 Grad-CAM Visual Attribution")
                    st.caption(
                        "Grad-CAM highlights the exact regions of the leaf that led MobileNetV2 to its prediction. "
                        "Warmer red and yellow regions represent strong positive influence."
                    )
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        try:
                            superimposed, heatmap, orig = predictor.generate_gradcam(
                                image_to_process, alpha=gradcam_alpha
                            )
                            cam_col1, cam_col2 = st.columns(2)
                            with cam_col1:
                                st.image(orig, caption="Original Input", use_container_width=True)
                            with cam_col2:
                                st.image(superimposed, caption="Grad-CAM Overlay", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Grad-CAM could not be computed for this layer: {e}")
        elif image_to_process is not None:
            st.write("Click **Diagnose Leaf Disease** to analyze the leaf image.")
        else:
            st.write("Results and predictions will appear here.")


if __name__ == "__main__":
    main()
