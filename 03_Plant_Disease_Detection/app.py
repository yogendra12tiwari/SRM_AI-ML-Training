"""
app.py
Streamlit web application for the Plant Disease Detection System.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub and deploy via Streamlit Community Cloud,
    pointing it at app.py. Make sure models/best_model.keras and
    models/class_indices.json are committed (or use Git LFS for large
    model files).
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from config import (
    APP_TITLE, APP_ICON, BEST_MODEL_PATH, CLASS_INDICES_PATH,
    CUSTOM_CNN_MODEL_PATH, MOBILENET_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    RESNET_MODEL_PATH, MODEL_COMPARISON_PATH, IMG_SIZE, CONFIDENCE_THRESHOLD,
)
from disease_info import get_disease_info
from utils.image_utils import load_image_from_bytes, preprocess_for_model, is_probably_leaf_image

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_class_names():
    if not os.path.exists(CLASS_INDICES_PATH):
        return None
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # index -> class name


@st.cache_data(show_spinner=False)
def load_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        return None
    with open(MODEL_COMPARISON_PATH, "r") as f:
        return json.load(f)


MODEL_OPTIONS = {
    "Best Model (Auto-selected)": BEST_MODEL_PATH,
    "Custom CNN": CUSTOM_CNN_MODEL_PATH,
    "MobileNetV2": MOBILENET_MODEL_PATH,
    "EfficientNetB0": EFFICIENTNET_MODEL_PATH,
    "ResNet50": RESNET_MODEL_PATH,
}


# ---------------------------------------------------------------------
# PREDICTION HELPER
# ---------------------------------------------------------------------
def predict(model, image: Image.Image, index_to_class: dict, top_k: int = 3):
    """Run the model on a preprocessed image and return top-k predictions."""
    batch = preprocess_for_model(image, target_size=IMG_SIZE)
    preds = model.predict(batch, verbose=0)[0]

    top_indices = preds.argsort()[-top_k:][::-1]
    results = [
        {"class_name": index_to_class[int(i)], "confidence": float(preds[i])}
        for i in top_indices
    ]
    return results


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.title(f"{APP_ICON} Plant Disease Detector")
    st.markdown(
        "AI-powered leaf disease detection using **CNN** and **Transfer "
        "Learning** (MobileNetV2, EfficientNetB0, ResNet50)."
    )
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Detect Disease", "📊 Model Comparison", "ℹ️ About the Project"],
    )

    st.divider()
    selected_model_label = st.selectbox("Model to use for prediction", list(MODEL_OPTIONS.keys()))
    st.caption(
        "Choose which trained model runs the prediction. "
        "'Best Model' is automatically the top performer from training."
    )

    st.divider()
    st.markdown("**Supported crops:** 🍅 Tomato · 🥔 Potato · 🌽 Corn · 🍎 Apple · "
                "🍇 Grape · 🌶 Pepper · 🍑 Peach · 🍓 Strawberry · 🍊 Orange · "
                "🫐 Blueberry · 🌿 Soybean · 🍒 Cherry")


# ---------------------------------------------------------------------
# PAGE 1: DETECT DISEASE
# ---------------------------------------------------------------------
if page == "🔍 Detect Disease":
    st.title(f"{APP_ICON} Plant Disease Detection")
    st.write("Upload a clear photo of a single plant leaf to get an instant diagnosis, "
             "confidence score, and treatment recommendations.")

    model_path = MODEL_OPTIONS[selected_model_label]
    model = load_model(model_path)
    index_to_class = load_class_names()

    if model is None:
        st.warning(
            f"⚠️ No trained model found at `{model_path}`.\n\n"
            "Train a model first by running `python train.py --model all`, "
            "then re-run this app. Once training finishes, `models/best_model.keras` "
            "and `models/class_indices.json` will be created automatically."
        )
    elif index_to_class is None:
        st.warning("⚠️ No class_indices.json found. Please run training first "
                   "so class labels can be decoded.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload a leaf image", type=["jpg", "jpeg", "png"]
            )
            camera_image = st.camera_input("...or take a photo")

        file_bytes = None
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
        elif camera_image is not None:
            file_bytes = camera_image.read()

        if file_bytes is not None:
            image = load_image_from_bytes(file_bytes)

            with col1:
                st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

            with col2:
                with st.spinner("Analyzing leaf image..."):
                    if not is_probably_leaf_image(image):
                        st.info(
                            "🤔 This image doesn't look like a typical leaf photo. "
                            "The model will still attempt a prediction, but please "
                            "double check you've uploaded a close-up leaf image for "
                            "best accuracy."
                        )

                    results = predict(model, image, index_to_class, top_k=3)
                    top_result = results[0]
                    class_name = top_result["class_name"]
                    confidence = top_result["confidence"]
                    info = get_disease_info(class_name)

                st.subheader("🔬 Prediction Result")

                if info["is_healthy"]:
                    st.success(f"✅ **{info['plant']} — Healthy Leaf**")
                else:
                    st.error(f"🦠 **{info['plant']} — {info['disease']}**")

                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.progress(min(confidence, 1.0))

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        "⚠️ Confidence is relatively low. Consider retaking the photo "
                        "with better lighting and a closer, single-leaf frame, or "
                        "consult an expert to confirm this diagnosis."
                    )

                with st.expander("📋 Top 3 Predictions"):
                    for r in results:
                        r_info = get_disease_info(r["class_name"])
                        st.write(f"- **{r_info['plant']} — {r_info['disease']}**: "
                                 f"{r['confidence'] * 100:.2f}%")

        # Info panel below columns, full width
        if file_bytes is not None:
            st.divider()
            st.subheader("📖 Disease Information")
            st.write(info["description"])

            if info["symptoms"]:
                st.markdown("**🩺 Symptoms**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")

            if info["treatment"]:
                st.markdown("**💊 Treatment**")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")

            if info["prevention"]:
                st.markdown("**🛡️ Prevention**")
                for p in info["prevention"]:
                    st.markdown(f"- {p}")


# ---------------------------------------------------------------------
# PAGE 2: MODEL COMPARISON
# ---------------------------------------------------------------------
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.write("Comparison of the Custom CNN against transfer learning models "
             "(MobileNetV2, EfficientNetB0, ResNet50) on the validation set.")

    comparison = load_model_comparison()

    if comparison is None:
        st.info("No comparison data yet. Run `python train.py --model all` "
                "to train all models and generate `models/model_comparison.json`.")
    else:
        df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "model"})
        df["val_accuracy_pct"] = (df["val_accuracy"] * 100).round(2)
        df["val_top3_accuracy_pct"] = (df["val_top3_accuracy"] * 100).round(2)

        st.dataframe(
            df[["model", "val_accuracy_pct", "val_top3_accuracy_pct",
                "val_loss", "params", "training_time_sec"]]
            .rename(columns={
                "model": "Model",
                "val_accuracy_pct": "Val Accuracy (%)",
                "val_top3_accuracy_pct": "Top-3 Accuracy (%)",
                "val_loss": "Val Loss",
                "params": "Parameters",
                "training_time_sec": "Train Time (s)",
            }),
            use_container_width=True,
        )

        fig = go.Figure(
            data=[go.Bar(x=df["model"], y=df["val_accuracy_pct"],
                         text=df["val_accuracy_pct"], textposition="auto")]
        )
        fig.update_layout(
            title="Validation Accuracy by Model",
            yaxis_title="Accuracy (%)",
            xaxis_title="Model",
        )
        st.plotly_chart(fig, use_container_width=True)

        best_model_name = df.loc[df["val_accuracy_pct"].idxmax(), "model"]
        st.success(f"🏆 Best performing model: **{best_model_name}**")


# ---------------------------------------------------------------------
# PAGE 3: ABOUT
# ---------------------------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown("""
### 🌿 Plant Disease Detection using Deep Learning

This system uses **Convolutional Neural Networks (CNN)** and **Transfer Learning**
to automatically detect plant diseases from leaf images, helping farmers get
fast, accurate diagnoses without needing an agricultural expert on-site.

**Problem it solves:**
- Lack of agricultural experts in remote areas
- Delays in disease diagnosis
- Incorrect pesticide usage from misdiagnosis
- Reduced crop yield and increased costs

**Approach:**
1. A **Custom CNN** is built and trained from scratch as a baseline.
2. Three **pretrained transfer learning models** — MobileNetV2, EfficientNetB0,
   and ResNet50 — are fine-tuned on the same dataset.
3. All models are compared on validation accuracy, loss, and inference cost.
4. The best-performing model is deployed in this app.

**Tech stack:** Python · TensorFlow/Keras · OpenCV · Pillow · NumPy · Pandas ·
Matplotlib · Plotly · Streamlit

**Dataset:** Kaggle — *New Plant Diseases Dataset (Augmented)*, covering 12+
crop types and 35+ disease/healthy classes.

**Target users:** Farmers, agricultural researchers, students, universities,
NGOs, government agriculture departments, and smart farming companies.
    """)

    st.info(
        "⚠️ **Disclaimer:** This tool provides AI-assisted suggestions and is "
        "not a substitute for professional agricultural consultation. Always "
        "confirm severe or unfamiliar cases with a local expert before applying "
        "chemical treatments."
    )