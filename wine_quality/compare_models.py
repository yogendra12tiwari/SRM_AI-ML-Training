# ==========================================
# Compare Logistic Regression vs Neural Network
# ==========================================

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("Lab  winequality-red.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

# Encode labels

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Scale Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load Logistic Regression

lr_model = joblib.load("logistic_model.pkl")

# Logistic Regression Prediction

lr_prediction = lr_model.predict(X_test)

# Load Neural Network

nn_model = tf.keras.models.load_model(
    "neural_network.keras"
)

# Neural Network Prediction

nn_prediction = nn_model.predict(X_test)

nn_prediction = np.argmax(nn_prediction, axis=1)

# ----------------------------
# Model Comparison
# ----------------------------

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Neural Network"
    ],

    "Accuracy": [

        accuracy_score(y_test, lr_prediction),

        accuracy_score(y_test, nn_prediction)

    ],

    "Precision": [

        precision_score(
            y_test,
            lr_prediction,
            average="weighted"
        ),

        precision_score(
            y_test,
            nn_prediction,
            average="weighted"
        )

    ],

    "Recall": [

        recall_score(
            y_test,
            lr_prediction,
            average="weighted"
        ),

        recall_score(
            y_test,
            nn_prediction,
            average="weighted"
        )

    ],

    "F1 Score": [

        f1_score(
            y_test,
            lr_prediction,
            average="weighted"
        ),

        f1_score(
            y_test,
            nn_prediction,
            average="weighted"
        )

    ]

})

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(comparison)

comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison Saved Successfully!")