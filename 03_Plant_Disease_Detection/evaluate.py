"""
evaluate.py
Generates evaluation artifacts after training:
  - Accuracy / loss curves per model
  - Confusion matrix for the best model
  - Classification report (precision/recall/F1 per class)
  - Model comparison bar chart (custom CNN vs transfer learning models)

Run:
    python evaluate.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TRAINING_HISTORY_PATH, MODEL_COMPARISON_PATH, BEST_MODEL_PATH,
    MODELS_DIR,
)
from data_preprocessing import build_data_generators, get_index_to_class


def plot_training_curves():
    if not os.path.exists(TRAINING_HISTORY_PATH):
        print("No training history found. Run train.py first.")
        return

    with open(TRAINING_HISTORY_PATH, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, hist in history.items():
        if "accuracy" in hist:
            axes[0].plot(hist["accuracy"], label=f"{model_name} (train)")
        if "val_accuracy" in hist:
            axes[0].plot(hist["val_accuracy"], linestyle="--", label=f"{model_name} (val)")
        if "loss" in hist:
            axes[1].plot(hist["loss"], label=f"{model_name} (train)")
        if "val_loss" in hist:
            axes[1].plot(hist["val_loss"], linestyle="--", label=f"{model_name} (val)")

    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")
    plt.close()


def plot_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        print("No model comparison file found. Run train.py first.")
        return

    with open(MODEL_COMPARISON_PATH, "r") as f:
        results = json.load(f)

    names = list(results.keys())
    accuracies = [results[n]["val_accuracy"] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)])
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison: Custom CNN vs Transfer Learning")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}",
                 ha="center", fontsize=10)

    out_path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved model comparison chart to {out_path}")
    plt.close()


def evaluate_best_model():
    if not os.path.exists(BEST_MODEL_PATH):
        print("No best model found. Run train.py first.")
        return

    print(f"Loading best model from {BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    _, valid_gen = build_data_generators(augment=False)
    index_to_class = get_index_to_class()
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    print("Running predictions on validation set...")
    valid_gen.reset()
    preds = model.predict(valid_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = valid_gen.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                     zero_division=0)
    report_path = os.path.join(MODELS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
    print(report)

    # Confusion matrix (can be large with 35+ classes; saved as a big figure)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Best Model")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to {cm_path}")
    plt.close()


if __name__ == "__main__":
    plot_training_curves()
    plot_model_comparison()
    evaluate_best_model()