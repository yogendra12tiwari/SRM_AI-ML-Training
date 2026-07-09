# ============================================================
# 🍷 VinoInsight
# AI Powered Wine Quality Prediction Dashboard
# ============================================================

# -----------------------------
# Import Libraries
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="VinoInsight",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Models
# -----------------------------

@st.cache_resource
def load_models():

    logistic_model = joblib.load("logistic_model.pkl")

    neural_network = tf.keras.models.load_model(
        "neural_network.keras"
    )

    scaler = joblib.load("scaler.pkl")

    label_encoder = joblib.load(
        "label_encoder.pkl"
    )

    return (
        logistic_model,
        neural_network,
        scaler,
        label_encoder
    )


# Load everything once

logistic_model, neural_network, scaler, label_encoder = load_models()
# ============================================================
# Premium CSS
# ============================================================

st.markdown("""
<style>

/* ---------------- Background ---------------- */

.stApp{
    background:#0E1117;
}

/* Hide Streamlit Branding */

#MainMenu{
    visibility:hidden;
}

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

/* ---------------- Sidebar ---------------- */

section[data-testid="stSidebar"]{
    background:#161B22;
    border-right:1px solid #2B313C;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* ---------------- Title ---------------- */

.title{

    text-align:center;

    font-size:58px;

    font-weight:800;

    color:#C2185B;

    margin-top:15px;

}

.subtitle{

    text-align:center;

    font-size:20px;

    color:#B0B6C3;

    margin-bottom:30px;

}

/* ---------------- Cards ---------------- */

.card{

    background:#1B212C;

    border:1px solid #2E3642;

    border-radius:22px;

    padding:25px;

    margin-bottom:20px;

    transition:.3s;

}

.card:hover{

    transform:translateY(-4px);

    box-shadow:0px 15px 35px rgba(0,0,0,.35);

}

/* ---------------- Metric Cards ---------------- */

.metric-card{

    background:#1F2633;

    padding:18px;

    border-radius:16px;

    border:1px solid #343B48;

    text-align:center;

}

/* ---------------- Prediction Card ---------------- */

.result-card{

    background:linear-gradient(135deg,#5B0F4D,#8E244D);

    border-radius:25px;

    padding:35px;

    color:white;

    text-align:center;

}

/* ---------------- Buttons ---------------- */

div.stButton>button{

    width:100%;

    height:60px;

    background:#C2185B;

    color:white;

    font-size:20px;

    font-weight:bold;

    border:none;

    border-radius:14px;

}

div.stButton>button:hover{

    background:#E91E63;

}

/* ---------------- Inputs ---------------- */

.stNumberInput input{

    background:#232A36 !important;

    color:white !important;

    border-radius:12px;

    border:1px solid #394150;

}

/* ---------------- Divider ---------------- */

hr{

    border:1px solid #262D38;

}

/* ---------------- Footer ---------------- */

.footer{

    text-align:center;

    color:#888;

    margin-top:30px;

    padding:20px;

}

/* ---------------- Success ---------------- */

.success{

    color:#4CAF50;

    font-weight:bold;

}

.warning{

    color:#FFC107;

    font-weight:bold;

}

.danger{

    color:#FF5252;

    font-weight:bold;

}

</style>

""", unsafe_allow_html=True)

# ============================================================
# Hero Section
# ============================================================

st.markdown("""

<div class="title">

🍷 VinoInsight

</div>

<div class="subtitle">

AI Powered Wine Quality Prediction Dashboard

</div>

""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("⚙️ Dashboard")

    st.write("")

    selected_model = st.radio(

        "Choose Prediction Model",

        [

            "Logistic Regression",

            "Neural Network"

        ]

    )

    st.divider()

    st.subheader("📊 Dataset")

    st.metric(

        "Samples",

        "1599"

    )

    st.metric(

        "Features",

        "11"

    )

    st.metric(

        "Classes",

        "6"

    )

    st.divider()

    st.subheader("🏆 Best Model")

    st.success("Neural Network")

    st.metric(

        "Accuracy",

        "67%"

    )

    st.divider()

    st.info(

        """
This dashboard predicts wine quality using

• Logistic Regression

• Neural Network

Switch models anytime using the selector above.
"""

    )

# ============================================================
# 🍷 Wine Input Section
# ============================================================

st.markdown("""
<div class="card">

<h2 style="text-align:center;color:white;">
🍷 Enter Wine Properties
</h2>

<p style="text-align:center;color:#B0B6C3;font-size:16px;">
Enter the physicochemical properties of the wine for prediction.
</p>

</div>
""", unsafe_allow_html=True)


# Create Two Columns
left, right = st.columns(2)

# ============================================================
# Left Column
# ============================================================

with left:

    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        value=7.4,
        step=0.1
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.70,
        step=0.01
    )

    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        value=0.00,
        step=0.01
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.0,
        value=1.9,
        step=0.1
    )

    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.076,
        step=0.001,
        format="%.3f"
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=0,
        value=11
    )

# ============================================================
# Right Column
# ============================================================

with right:

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=0,
        value=34
    )

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.9978,
        step=0.0001,
        format="%.4f"
    )

    ph = st.number_input(
        "pH",
        min_value=0.0,
        value=3.51,
        step=0.01
    )

    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.56,
        step=0.01
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=9.4,
        step=0.1
    )
# ============================================================
# Predict Button
# ============================================================

st.write("")

predict = st.button(
    "🍷 Predict Wine Quality",
    key="predict_btn"
)

# Input Array

input_data = np.array([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol
]])

scaled_data = scaler.transform(input_data)

# ============================================================
# Predict Button
# ============================================================

st.write("")



# Input Array

input_data = np.array([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol
]])

scaled_data = scaler.transform(input_data)

   

# ============================================================
# Prediction Logic
# ============================================================

if predict:

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    if selected_model == "Logistic Regression":

        prediction = logistic_model.predict(scaled_data)

        quality = prediction[0]

        confidence = max(
            logistic_model.predict_proba(scaled_data)[0]
        ) * 100

# ============================================================
# Prediction Logic
# ============================================================

if predict:

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    if selected_model == "Logistic Regression":

        prediction = logistic_model.predict(scaled_data)

        quality = prediction[0]

        confidence = max(
            logistic_model.predict_proba(scaled_data)[0]
        ) * 100

    # -----------------------------
    # Neural Network
    # -----------------------------

    else:

        prediction = neural_network.predict(
            scaled_data,
            verbose=0
        )

        predicted_class = np.argmax(prediction)

        quality = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = float(
            np.max(prediction) * 100
        )
    # ============================================================
    # Prediction Result Card
    # ============================================================

    st.write("")

    st.markdown(f"""
    <div class="result-card">

    <h2>🍷 Wine Quality Prediction</h2>

    <h1 style="font-size:70px;">
        {quality}
    </h1>

    <h3>
        Confidence : {confidence:.2f}%
    </h3>

    <h4>
        Model : {selected_model}
    </h4>

    </div>

    """, unsafe_allow_html=True)
    # ============================================================
    # Star Rating
    # ============================================================

    stars = "⭐" * int(quality)

    st.markdown(
        f"""
        <h2 style='text-align:center;color:#FFD700;'>
        {stars}
        </h2>
        """,
        unsafe_allow_html=True
    )
    # ============================================================
    # Quality Badge
    # ============================================================

    if quality >= 7:

        st.success("🍷 Premium Quality Wine")

    elif quality >= 6:

        st.info("🍷 Good Quality Wine")

    elif quality >= 5:

        st.warning("🍷 Average Quality Wine")

    else:

        st.error("🍷 Low Quality Wine")
    # ============================================================
    # AI Insights
    # ============================================================

    st.markdown("---")

    st.subheader("🧠 AI Insights")

    insights = []

    # Alcohol Analysis
    if alcohol >= 11:
        insights.append("🍷 High alcohol content contributes positively to wine quality.")
    elif alcohol >= 9:
        insights.append("🍷 Alcohol level is within the normal range.")
    else:
        insights.append("⚠️ Low alcohol content may reduce wine quality.")

    # Volatile Acidity
    if volatile_acidity < 0.4:
        insights.append("✅ Low volatile acidity is generally associated with better quality.")
    else:
        insights.append("⚠️ High volatile acidity may negatively affect taste.")

    # Sulphates
    if sulphates >= 0.65:
        insights.append("✅ Sulphates indicate good preservation characteristics.")
    else:
        insights.append("ℹ️ Sulphate level is relatively low.")

    # pH
    if 3.2 <= ph <= 3.5:
        insights.append("✅ pH is in the optimal range.")
    else:
        insights.append("ℹ️ pH is outside the common optimal range.")

    # Density
    if density < 0.996:
        insights.append("🍇 Lower density often indicates higher alcohol concentration.")
    else:
        insights.append("🍷 Density is within the expected range.")

    # Display Insights
    for item in insights:
        st.success(item)

    # ============================================================
    # Model Comparison
    # ============================================================

    st.markdown("---")

    st.subheader("📊 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Logistic Regression")

        st.metric(
            "Accuracy",
            "61%"
        )

        st.metric(
            "Precision",
            "60%"
        )

        st.metric(
            "Recall",
            "61%"
        )

        st.metric(
            "F1 Score",
            "60%"
        )

    with col2:

        st.markdown("### Neural Network")

        st.metric(
            "Accuracy",
            "67%"
        )

        st.metric(
            "Precision",
            "66%"
        )

        st.metric(
            "Recall",
            "67%"
        )

        st.metric(
            "F1 Score",
            "66%"
        )

    st.success("🏆 Best Model : Neural Network")

    # ============================================================
    # Accuracy Comparison Chart
    # ============================================================

    st.markdown("---")

    st.subheader("📈 Accuracy Comparison")

    chart = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Neural Network"
            ],
            "Accuracy": [
                61,
                67
            ]
        }
    )

    st.bar_chart(
        chart,
        x="Model",
        y="Accuracy"
    )
    # ============================================================
    # Confidence Meter
    # ============================================================

    st.markdown("---")

    st.subheader("🎯 Prediction Confidence")

    st.progress(min(confidence / 100, 1.0))

    st.write(f"Model Confidence: **{confidence:.2f}%**")

    # ============================================================
    # Prediction Probability
    # ============================================================

    if selected_model == "Neural Network":

        st.markdown("---")

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame({
            "Quality": label_encoder.classes_,
            "Probability": prediction[0] * 100
        })

        st.bar_chart(
            probability_df,
            x="Quality",
            y="Probability"
        )
    # ============================================================
    # Download Prediction
    # ============================================================

    st.markdown("---")

    result_df = pd.DataFrame({

        "Model":[selected_model],

        "Predicted Quality":[quality],

        "Confidence (%)":[round(confidence,2)]

    })

    st.download_button(

        label="📥 Download Prediction",

        data=result_df.to_csv(index=False),

        file_name="wine_prediction.csv",

        mime="text/csv"

    )

    # ============================================================
    # Footer
    # ============================================================

    st.markdown("---")

    st.markdown(
    """
    <div style='text-align:center; color:#9CA3AF; padding:20px;'>

    <h4>🍷 VinoInsight</h4>

    <p>
    Built with ❤️ using
    <strong>Python</strong>,
    <strong>TensorFlow</strong>,
    <strong>Scikit-Learn</strong> &
    <strong>Streamlit</strong>
    </p>

    <p>
    Developed by <strong>Yogendra Tiwari</strong>
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )
# ============================================================
# Model Comparison Dashboard
# ============================================================

st.markdown("---")
st.header("📊 Model Comparison Dashboard")

try:
    comparison_df = pd.read_csv("model_comparison.csv")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    winner = comparison_df.loc[
        comparison_df["Accuracy"].idxmax(),
        "Model"
    ]

    winner_acc = comparison_df["Accuracy"].max()

    st.success(
        f"🏆 Best Performing Model: **{winner}** ({winner_acc:.2%})"
    )

except FileNotFoundError:
    st.warning("⚠️ model_comparison.csv not found. Run compare_models.py first.")

# ============================================================
# Accuracy Chart
# ============================================================

try:

    chart_data = comparison_df.set_index("Model")

    st.subheader("📈 Accuracy Comparison")

    st.bar_chart(chart_data["Accuracy"])

except:
    pass

# ============================================================
# About Project
# ============================================================

with st.expander("ℹ️ About This Project"):

    st.markdown("""

### 🍷 Wine Quality Prediction

This application predicts the quality of red wine using Machine Learning and Deep Learning.

### Models Used

- Logistic Regression
- Neural Network (TensorFlow)

### Features

- 11 Physicochemical Features
- Real-time Prediction
- AI Insights
- Model Comparison
- Confidence Score

### Tech Stack

- Python
- Streamlit
- Scikit-Learn
- TensorFlow
- Pandas
- NumPy

""")

st.sidebar.markdown("## 📊 Dashboard Stats")

st.sidebar.metric("🍷 Dataset", "1599 Samples")
st.sidebar.metric("📑 Features", "11")
st.sidebar.metric("🤖 Models", "2")
st.sidebar.metric("🎯 Target", "Wine Quality")

st.sidebar.divider()

st.sidebar.markdown("### 🚀 Technologies")

st.sidebar.markdown("""
- 🐍 Python
- 🤖 TensorFlow
- 📊 Scikit-Learn
- 🎨 Streamlit
""")





