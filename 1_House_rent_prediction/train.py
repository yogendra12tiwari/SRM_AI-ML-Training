# ===========================================
# House Rent Prediction - Training Script
# ===========================================

import os
import joblib
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ===========================================
# Load Dataset
# ===========================================

DATA_PATH = "dataset/House_rent_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(f"Shape : {df.shape}")

print(df.head())

# ===========================================
# Data Cleaning
# ===========================================

print("\nChecking Missing Values...\n")

print(df.isnull().sum())

# Drop unnecessary column
if "Posted On" in df.columns:
    df.drop(columns=["Posted On"], inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ===========================================
# Label Encoding
# ===========================================

encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(categorical_columns)

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

print("\nLabel Encoding Completed ✅")

# Save encoders
os.makedirs("models", exist_ok=True)

joblib.dump(encoders, "models/encoders.pkl")

print("Encoders Saved Successfully ✅")

# ===========================================
# Feature & Target
# ===========================================

X = df.drop("Rent", axis=1)
y = df["Rent"]

print("\nFeature Shape :", X.shape)
print("Target Shape  :", y.shape)


# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Test Split Completed ✅")

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")


# ===========================================
# Standard Scaling (Only for Neural Network)
# ===========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler Saved Successfully ✅")

# ===========================================
# Random Forest Training
# ===========================================

print("\n" + "=" * 50)
print("Training Random Forest Model...")
print("=" * 50)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest Training Completed ✅")

# -------------------------------------------
# Prediction
# -------------------------------------------

rf_predictions = rf_model.predict(X_test)

# -------------------------------------------
# Evaluation
# -------------------------------------------

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Performance")
print("-" * 35)

print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")

# -------------------------------------------
# Save Model
# -------------------------------------------

joblib.dump(rf_model, "models/house_rent_model.pkl")

print("\nRandom Forest Model Saved Successfully ✅")

# ===========================================
# Neural Network Training
# ===========================================

print("\n" + "=" * 50)
print("Training Neural Network...")
print("=" * 50)

nn_model = Sequential([

    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu"),

    Dense(32, activation="relu"),

    Dense(1)

])

nn_model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

history = nn_model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

print("\nNeural Network Training Completed ✅")

# ===========================================
# Save Neural Network
# ===========================================

nn_model.save("models/house_rent_nn.keras")

print("Neural Network Saved Successfully ✅")

# ===========================================
# Neural Network Evaluation
# ===========================================

nn_predictions = nn_model.predict(X_test_scaled)

nn_mae = mean_absolute_error(y_test, nn_predictions)

nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))

nn_r2 = r2_score(y_test, nn_predictions)

print("\nNeural Network Performance")
print("-" * 35)

print(f"MAE  : {nn_mae:.2f}")
print(f"RMSE : {nn_rmse:.2f}")
print(f"R²   : {nn_r2:.4f}")

# ===========================================
# Model Comparison
# ===========================================

comparison = pd.DataFrame({
    "Model": ["Random Forest", "Neural Network"],
    "MAE": [rf_mae, nn_mae],
    "RMSE": [rf_rmse, nn_rmse],
    "R2 Score": [rf_r2, nn_r2]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)

# ===========================================
# Best Model
# ===========================================

if rf_mae < nn_mae:
    best_model = "Random Forest"
else:
    best_model = "Neural Network"

print("\n" + "=" * 60)
print(f"🏆 Best Model : {best_model}")
print("=" * 60)

# ===========================================
# Training Completed
# ===========================================

print("\n✅ Training Pipeline Completed Successfully!")

print("\nSaved Files:")
print("✔ models/house_rent_model.pkl")
print("✔ models/house_rent_nn.keras")
print("✔ models/encoders.pkl")
print("✔ models/scaler.pkl")