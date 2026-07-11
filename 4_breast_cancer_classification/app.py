"""
Breast Cancer Classification — Streamlit App
==============================================
Interactive web app that trains the MLP neural network on the Wisconsin
Diagnostic Breast Cancer dataset and lets users adjust feature values with
sliders to get a live malignant/benign prediction with probability.

Run locally:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load data & train model (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    return model, scaler, data, feature_names, target_names, X, test_acc, test_auc


model, scaler, data, feature_names, target_names, X_full, test_acc, test_auc = load_and_train()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🩺 Breast Cancer Classification — Neural Network")
st.markdown(
    "Predicts whether a tumor is **malignant** or **benign** from cell-nuclei "
    "measurements, using a Multi-Layer Perceptron neural network trained on the "
    "Wisconsin Diagnostic Breast Cancer dataset."
)

col_a, col_b = st.columns(2)
col_a.metric("Model Test Accuracy", f"{test_acc:.1%}")
col_b.metric("Model Test ROC-AUC", f"{test_auc:.3f}")

st.divider()

# ----------------------------------------------------------------------
# Sidebar: feature inputs
# ----------------------------------------------------------------------
st.sidebar.header("Input Features")
st.sidebar.caption(
    "Adjust the sliders to match a patient's measurements, or click a preset below."
)

preset = st.sidebar.radio(
    "Quick presets",
    ["Custom", "Sample: malignant case", "Sample: benign case"],
    index=0,
)

df_full = pd.DataFrame(X_full, columns=feature_names)
df_full["target"] = data.target  # 0 = malignant, 1 = benign

if preset == "Sample: malignant case":
    default_row = df_full[df_full["target"] == 0].iloc[0]
elif preset == "Sample: benign case":
    default_row = df_full[df_full["target"] == 1].iloc[0]
else:
    default_row = df_full.mean()

user_input = {}
# Group features into the 3 natural categories (mean / error / worst)
groups = {
    "Mean values": [f for f in feature_names if f.startswith("mean")],
    "Standard error values": [f for f in feature_names if "error" in f],
    "Worst (largest) values": [f for f in feature_names if f.startswith("worst")],
}

for group_name, feats in groups.items():
    with st.sidebar.expander(group_name, expanded=(group_name == "Mean values")):
        for feat in feats:
            col_min = float(df_full[feat].min())
            col_max = float(df_full[feat].max())
            col_default = float(default_row[feat])
            user_input[feat] = st.slider(
                feat, min_value=col_min, max_value=col_max,
                value=col_default, step=(col_max - col_min) / 200,
                key=feat,
            )

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
input_array = np.array([[user_input[f] for f in feature_names]])
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

st.subheader("Prediction")

pred_label = target_names[prediction]
confidence = proba[prediction]

if pred_label == "malignant":
    st.error(f"⚠️ **Malignant** — confidence: {confidence:.1%}")
else:
    st.success(f"✅ **Benign** — confidence: {confidence:.1%}")

col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Class probabilities**")
    prob_df = pd.DataFrame({
        "Class": target_names,
        "Probability": proba,
    }).set_index("Class")
    st.bar_chart(prob_df)

with col2:
    st.write("**Where this patient falls vs. the dataset** (mean radius vs. mean texture)")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = df_full["target"].map({0: "#c53030", 1: "#2b6cb0"})
    ax.scatter(df_full["mean radius"], df_full["mean texture"], c=colors, alpha=0.4, s=20)
    ax.scatter(
        user_input["mean radius"], user_input["mean texture"],
        c="black", s=150, marker="*", label="Your input", edgecolors="white"
    )
    ax.set_xlabel("Mean radius")
    ax.set_ylabel("Mean texture")
    ax.legend(loc="upper right")
    st.pyplot(fig)

st.divider()
st.caption(
    "⚕️ **Disclaimer**: This app is for educational/demonstration purposes only "
    "and is not a substitute for professional medical diagnosis."
)
