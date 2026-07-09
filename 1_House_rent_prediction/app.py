# ============================================
# House Rent Prediction
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

DATASET_DIR = BASE_DIR / "dataset"
st.markdown("""
<style>

.block-container{

    padding-top:2rem;

    padding-bottom:2rem;

}

[data-testid="stMetricValue"]{

    font-size:28px;

    color:green;

}

</style>
""", unsafe_allow_html=True)


# ============================================
# Load Models
# ============================================

@st.cache_resource
def load_models():

    rf_model = joblib.load(
        MODELS_DIR / "house_rent_model.pkl"
    )

    nn_model = tf.keras.models.load_model(
        MODELS_DIR / "house_rent_nn.keras"
    )

    encoders = joblib.load(
        MODELS_DIR / "encoders.pkl"
    )

    scaler = joblib.load(
        MODELS_DIR / "scaler.pkl"
    )

    return rf_model, nn_model, encoders, scaler

rf_model, nn_model, encoders, scaler = load_models()


# ============================================
# Load Dataset
# ============================================

@st.cache_data
def load_data():

    csv_path = DATASET_DIR / "House_rent_Dataset.csv"

    return pd.read_csv(csv_path)

df = load_data()
# ============================================
# Sidebar
# ============================================

st.sidebar.title("🏠 House Rent Prediction")

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("### Models")

st.sidebar.write("🌳 Random Forest")
st.sidebar.write("🧠 Neural Network")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Developer**

    Yogendra Tiwari

    AI • ML • GenAI
    """
)

# ============================================
# Main Header
# ============================================

st.title("🏠 House Rent Prediction Dashboard")

st.markdown(
    """
Predict monthly house rent using **Machine Learning** and
**Deep Learning** models.
"""
)

st.divider()

# ============================================
# Metrics
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", len(df))

with col2:
    st.metric("Features", df.shape[1]-1)

with col3:
    st.metric("ML Model", "Random Forest")

with col4:
    st.metric("DL Model", "Neural Network")

st.divider()

# ============================================
# Dataset Preview
# ============================================

st.subheader("📊 Dataset Preview")

col1, col2 = st.columns([3,1])

with col1:
    st.dataframe(df.head(), use_container_width=True)

with col2:

    st.write("### Dataset Information")

    st.write(f"Rows : {df.shape[0]}")
    st.write(f"Columns : {df.shape[1]}")

    st.write("### Missing Values")

    st.write(df.isnull().sum().sum())

st.divider()

# ============================================
# Prediction Section
# ============================================

st.header("🤖 House Rent Prediction")

col1, col2 = st.columns(2)

with col1:

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    size = st.number_input(
        "Size (Sq. Ft.)",
        min_value=100,
        max_value=10000,
        value=1000
    )

    bathroom = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    floor = st.selectbox(
        "Floor",
        encoders["Floor"].classes_
    )

    area_type = st.selectbox(
        "Area Type",
        encoders["Area Type"].classes_
    )


with col2:

    locality = st.selectbox(
        "Area Locality",
        encoders["Area Locality"].classes_
    )

    city = st.selectbox(
        "City",
        encoders["City"].classes_
    )

    furnishing = st.selectbox(
        "Furnishing Status",
        encoders["Furnishing Status"].classes_
    )

    tenant = st.selectbox(
        "Tenant Preferred",
        encoders["Tenant Preferred"].classes_
    )

    contact = st.selectbox(
        "Point of Contact",
        encoders["Point of Contact"].classes_
    )

st.divider()

# ============================================
# Analytics Dashboard
# ============================================

st.divider()

st.header("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(
    [
        "Rent Distribution",
        "City Wise",
        "BHK Wise"
    ]
)

# ============================================
# Model Comparison
# ============================================

st.divider()

st.header("📈 Model Comparison")

comparison_df = pd.DataFrame({

    "Model": ["Random Forest", "Neural Network"],

    "MAE": [12474.35, 15194.33],

    "R² Score": [0.6576, 0.6119]

})

st.dataframe(
    comparison_df,
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("MAE Comparison")

    fig = px.bar(
        comparison_df,
        x="Model",
        y="MAE",
        color="Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    st.subheader("R² Comparison")

    fig = px.bar(
        comparison_df,
        x="Model",
        y="R² Score",
        color="Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------
# Rent Distribution
# ---------------------------------------------

with tab1:

    fig = px.histogram(
        df,
        x="Rent",
        nbins=50,
        title="Rent Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------
# City Wise
# ---------------------------------------------

with tab2:

    city_avg = (
        df.groupby("City")["Rent"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(

        city_avg,

        x="City",

        y="Rent",

        title="Average Rent by City"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------
# BHK Wise
# ---------------------------------------------

with tab3:

    bhk_avg = (

        df.groupby("BHK")["Rent"]

        .mean()

        .reset_index()

    )

    fig = px.bar(

        bhk_avg,

        x="BHK",

        y="Rent",

        title="Average Rent by BHK"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================
# Prediction Button
# ============================================

if st.button("Predict Rent", use_container_width=True):

    input_data = pd.DataFrame({

        "BHK":[bhk],

        "Size":[size],

        "Floor":[encoders["Floor"].transform([floor])[0]],

        "Area Type":[encoders["Area Type"].transform([area_type])[0]],

        "Area Locality":[encoders["Area Locality"].transform([locality])[0]],

        "City":[encoders["City"].transform([city])[0]],

        "Furnishing Status":[encoders["Furnishing Status"].transform([furnishing])[0]],

        "Tenant Preferred":[encoders["Tenant Preferred"].transform([tenant])[0]],

        "Bathroom":[bathroom],

        "Point of Contact":[encoders["Point of Contact"].transform([contact])[0]]

    })

    # Random Forest Prediction

    rf_prediction = rf_model.predict(input_data)[0]

    # Neural Network Prediction

    scaled = scaler.transform(input_data)

    nn_prediction = nn_model.predict(scaled, verbose=0)[0][0]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.success("🌳 Random Forest Prediction")

        st.metric(
            "Estimated Rent",
            f"₹ {rf_prediction:,.0f}"
        )

    with col2:

        st.info("🧠 Neural Network Prediction")

        st.metric(
            "Estimated Rent",
            f"₹ {nn_prediction:,.0f}"
        )

# ============================================
# Footer
# ============================================

st.divider()

st.markdown(
    """
### 👨‍💻 Developed By

**Yogendra Tiwari**

AI • Machine Learning • Deep Learning • GenAI

---

⭐ If you like this project, don't forget to star the repository on GitHub.
"""
)