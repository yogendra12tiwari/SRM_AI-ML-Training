# ==========================================================
# Wine Quality Prediction using Neural Network (TensorFlow)
# ==========================================================

# -----------------------------
# Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Load Dataset
# -----------------------------
# Change the file name/path if required
df = pd.read_csv("Lab  winequality-red.csv")

print("Dataset Loaded Successfully!\n")

# -----------------------------
# Display Basic Information
# -----------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Separate Features and Target
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# Encode Target Labels
# Example:
# 3 -> 0
# 4 -> 1
# 5 -> 2
# 6 -> 3
# 7 -> 4
# 8 -> 5
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save Scaler and Encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")

# -----------------------------
# Number of Classes
# -----------------------------
num_classes = len(np.unique(y))

# -----------------------------
# Build Neural Network
# -----------------------------
model = Sequential([

    # Input Layer
    tf.keras.Input(shape=(11,)),

    # Hidden Layer 1
    Dense(128, activation="relu"),

    # Randomly drops 30% neurons to reduce overfitting
    Dropout(0.3),

    # Hidden Layer 2
    Dense(64, activation="relu"),

    # Again drop 30% neurons
    Dropout(0.3),

    # Hidden Layer 3
    Dense(32, activation="relu"),

    # Output Layer
    Dense(num_classes, activation="softmax")

])


# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display Model Summary
print("\nModel Summary:")
model.summary()

# -----------------------------
# Early Stopping
# Stops training if validation
# accuracy does not improve.
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Save the Best Model Automatically
# -----------------------------
checkpoint = ModelCheckpoint(
    "best_neural_network.keras",   # File name to save
    monitor="val_accuracy",        # Watch validation accuracy
    save_best_only=True,           # Save only the best model
    mode="max",                    # Higher accuracy is better
    verbose=1
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\n==============================")
print("Test Accuracy :", accuracy)
print("Test Loss     :", loss)
print("==============================")

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

# -----------------------------
# Accuracy
# -----------------------------
print("\nAccuracy Score")
print(accuracy_score(y_test, predicted_classes))

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predicted_classes))

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report")
print(classification_report(y_test, predicted_classes))

# -----------------------------
# Save Neural Network
# -----------------------------
model.save("neural_network.keras")

print("\nNeural Network Model Saved Successfully!")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()