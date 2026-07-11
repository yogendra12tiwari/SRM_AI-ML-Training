"""
Plant Disease Detection - Streamlit App
-----------------------------------------
Upload a leaf image and get a disease prediction from either a custom CNN
or a fine-tuned MobileNetV2 model.

Expected files in the same folder as this script:
    - plant_disease_cnn.keras
    - plant_disease_mobilenetv2.keras
    - class_names.pkl
    - history.pkl   (optional, used for the "Training History" tab)
"""

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "MobileNetV2 (Transfer Learning)": APP_DIR / "plant_disease_mobilenetv2.keras",
    "Custom CNN": APP_DIR / "plant_disease_cnn.keras",
}
CLASS_NAMES_PATH = APP_DIR / "class_names.pkl"
HISTORY_PATH = APP_DIR / "history.pkl"

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_class_names(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_history(path: str):
    if not Path(path).exists():
        return None
    return joblib.load(path)


def prettify_label(raw_label: str) -> str:
    """Turn 'Tomato___Late_blight' into 'Tomato - Late blight'."""
    plant, _, disease = raw_label.partition("___")
    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy":
        return f"{plant} - Healthy ✅"
    return f"{plant} - {disease}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, scale to [0, 1], and add batch dimension (matches training pipeline)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🌿 Settings")
model_choice = st.sidebar.radio("Choose a model", list(MODEL_PATHS.keys()))
top_k = st.sidebar.slider("Number of predictions to show", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "This app classifies plant leaf images into 38 categories covering "
    "14 crop species and their common diseases (or healthy status)."
)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a photo of a plant leaf and this app will predict the disease "
    "(or confirm it's healthy) using a deep learning model trained on the "
    "PlantVillage dataset."
)

tab_predict, tab_history = st.tabs(["🔍 Predict", "📊 Training History"])

# ---- Predict tab ----------------------------------------------------------
with tab_predict:
    uploaded_file = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        # Load model / classes only when needed
        model_path = MODEL_PATHS[model_choice]
        if not model_path.exists():
            st.error(f"Model file not found: {model_path.name}")
        elif not CLASS_NAMES_PATH.exists():
            st.error("class_names.pkl not found.")
        else:
            with st.spinner(f"Loading {model_choice} and running prediction..."):
                
                model = load_model(str(model_path))
                class_names = load_class_names(str(CLASS_NAMES_PATH))
                input_arr = preprocess_image(image)
            
                preds = model.predict(input_arr, verbose=0)[0]
            
                predicted_index = np.argmax(preds)
            
                st.write("Predicted Index:", predicted_index)
                st.write("Raw Class:", class_names[predicted_index])
                st.write("Confidence:", float(preds[predicted_index]))

            top_indices = preds.argsort()[::-1][:top_k]
            top_labels = [prettify_label(class_names[i]) for i in top_indices]
            top_scores = [float(preds[i]) * 100 for i in top_indices]

            with col2:
                st.subheader("Prediction")
                st.markdown(f"### {top_labels[0]}")
                st.metric("Confidence", f"{top_scores[0]:.2f}%")
                st.caption(f"Model used: {model_choice}")

            st.markdown("---")
            st.subheader(f"Top {top_k} predictions")
            results_df = pd.DataFrame(
                {"Class": top_labels, "Confidence (%)": top_scores}
            ).set_index("Class")
            st.bar_chart(results_df)
            st.dataframe(
                results_df.style.format({"Confidence (%)": "{:.2f}"}),
                use_container_width=True,
            )
    else:
        st.info("👆 Upload an image to get a prediction.")

# ---- History tab ------------------------------------------------------------
with tab_history:
    history = load_history(str(HISTORY_PATH))
    if history is None:
        st.info("No training history file (history.pkl) found.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df.index = range(1, len(hist_df) + 1)
        hist_df.index.name = "Epoch"

        st.subheader("Accuracy")
        acc_cols = [c for c in ["accuracy", "val_accuracy"] if c in hist_df.columns]
        if acc_cols:
            st.line_chart(hist_df[acc_cols])

        st.subheader("Loss")
        loss_cols = [c for c in ["loss", "val_loss"] if c in hist_df.columns]
        if loss_cols:
            st.line_chart(hist_df[loss_cols])

        st.subheader("Raw values")
        st.dataframe(hist_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit & TensorFlow • Plant Disease Detection Project")
