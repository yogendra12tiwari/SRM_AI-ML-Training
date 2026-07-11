"""
Breast Cancer Classification using a Neural Network (MLP)
============================================================
Dataset : Wisconsin Diagnostic Breast Cancer (built into scikit-learn)
Model   : Multi-Layer Perceptron (feedforward neural network)

Pipeline:
1. Load & explore data
2. Split into train/test sets
3. Scale features (critical for neural networks)
4. Build & train an MLP classifier
5. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
6. Visualize training loss curve, confusion matrix, and ROC curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # y: 0 = malignant, 1 = benign
feature_names = data.feature_names
target_names = data.target_names

print("Dataset shape :", X.shape)
print("Classes       :", dict(zip(target_names, np.bincount(y))))
print("Features      :", list(feature_names[:5]), "... (30 total)")

# ----------------------------------------------------------------------
# 2. Train / test split (stratified to preserve class balance)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ----------------------------------------------------------------------
# 3. Feature scaling (neural nets are sensitive to feature magnitude)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 4. Build & train the neural network
# ----------------------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),   # 2 hidden layers
    activation='relu',
    solver='adam',
    alpha=1e-4,                    # L2 regularization
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=15,
    validation_fraction=0.1,
    random_state=RANDOM_STATE,
)

model.fit(X_train_scaled, y_train)
print(f"\nTraining stopped after {model.n_iter_} iterations "
      f"(best validation score: {model.best_validation_score_:.4f})")

# ----------------------------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n===== Test Set Performance =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))

# ----------------------------------------------------------------------
# 6. Visualizations
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Training loss curve
axes[0].plot(model.loss_curve_, color='#2b6cb0')
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

# (b) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# (c) ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[2].plot(fpr, tpr, color='#c53030', label=f"AUC = {auc:.3f}")
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title("ROC Curve")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(loc='lower right')
axes[2].grid(alpha=0.3)

plt.tight_layout()

# Save results relative to project root (src/../results/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "breast_cancer_nn_results.png")

plt.savefig(output_path, dpi=150)
print(f"\nSaved visualizations to {output_path}")
