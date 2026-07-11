"""
train.py
Main training script. Trains the Custom CNN and all three transfer
learning models (MobileNetV2, EfficientNetB0, ResNet50), evaluates
each on the validation set, and saves:
  - Each model's weights to models/
  - Training history (accuracy/loss curves) to models/training_history.json
  - A model comparison table to models/model_comparison.json
  - The single best-performing model copied to models/best_model.keras

Run:
    python train.py --model all
    python train.py --model custom_cnn
    python train.py --model mobilenetv2
"""

import argparse
import json
import os
import shutil
import time

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from config import (
    EPOCHS_CUSTOM_CNN, EPOCHS_TRANSFER_LEARNING,
    CUSTOM_CNN_MODEL_PATH, MOBILENET_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH, RESNET_MODEL_PATH,
    BEST_MODEL_PATH, TRAINING_HISTORY_PATH, MODEL_COMPARISON_PATH,
)
from data_preprocessing import build_data_generators, build_test_generator
from custom_cnn_model import build_custom_cnn
from transfer_learning_model import build_transfer_model, unfreeze_for_fine_tuning


def get_callbacks(checkpoint_path, patience=6):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                         save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                           min_lr=1e-7, verbose=1),
    ]


def train_custom_cnn(train_gen, valid_gen, history_store, results_store):
    print("\n" + "=" * 60)
    print("TRAINING: Custom CNN")
    print("=" * 60)

    model = build_custom_cnn()
    callbacks = get_callbacks(CUSTOM_CNN_MODEL_PATH)

    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_CUSTOM_CNN,
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    val_loss, val_acc, val_top3 = model.evaluate(valid_gen, verbose=0)
    history_store["custom_cnn"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    results_store["custom_cnn"] = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "val_top3_accuracy": float(val_top3),
        "training_time_sec": round(elapsed, 2),
        "params": model.count_params(),
    }
    print(f"Custom CNN -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
    return model


def train_transfer_model(model_name, path, train_gen, valid_gen,
                          history_store, results_store, fine_tune=True):
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name} (Transfer Learning)")
    print("=" * 60)

    model, base_model = build_transfer_model(model_name)
    callbacks = get_callbacks(path)

    start = time.time()
    # Phase 1: train classifier head only (frozen backbone)
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_TRANSFER_LEARNING,
        callbacks=callbacks,
    )

    combined_history = {k: list(v) for k, v in history1.history.items()}

    if fine_tune:
        # Phase 2: unfreeze top layers and fine-tune with a low LR
        print(f"\nFine-tuning {model_name} (unfreezing top layers)...")
        model = unfreeze_for_fine_tuning(model, base_model)
        history2 = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS_TRANSFER_LEARNING,
            callbacks=callbacks,
        )
        for k, v in history2.history.items():
            combined_history.setdefault(k, []).extend(v)

    elapsed = time.time() - start

    val_loss, val_acc, val_top3 = model.evaluate(valid_gen, verbose=0)
    history_store[model_name.lower()] = {k: [float(x) for x in vals] for k, vals in combined_history.items()}
    results_store[model_name.lower()] = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "val_top3_accuracy": float(val_top3),
        "training_time_sec": round(elapsed, 2),
        "params": model.count_params(),
    }
    print(f"{model_name} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

    model.save(path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train plant disease detection models.")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "custom_cnn", "mobilenetv2", "efficientnetb0", "resnet50"],
        help="Which model(s) to train."
    )
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip the fine-tuning phase for transfer learning models.")
    args = parser.parse_args()

    print("Building data generators...")
    train_gen, valid_gen = build_data_generators(augment=True)

    history_store = {}
    results_store = {}
    model_paths = {
        "custom_cnn": CUSTOM_CNN_MODEL_PATH,
        "mobilenetv2": MOBILENET_MODEL_PATH,
        "efficientnetb0": EFFICIENTNET_MODEL_PATH,
        "resnet50": RESNET_MODEL_PATH,
    }

    to_run = list(model_paths.keys()) if args.model == "all" else [args.model]

    for name in to_run:
        if name == "custom_cnn":
            train_custom_cnn(train_gen, valid_gen, history_store, results_store)
        else:
            display_name = {"mobilenetv2": "MobileNetV2",
                             "efficientnetb0": "EfficientNetB0",
                             "resnet50": "ResNet50"}[name]
            train_transfer_model(display_name, model_paths[name], train_gen, valid_gen,
                                 history_store, results_store,
                                 fine_tune=not args.no_finetune)

    # Save training history for later plotting in evaluate.py / the app
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_store, f, indent=2)

    # Save comparison table
    with open(MODEL_COMPARISON_PATH, "w") as f:
        json.dump(results_store, f, indent=2)

    # Pick the best model by validation accuracy and copy it to best_model.keras
    if results_store:
        best_name = max(results_store, key=lambda k: results_store[k]["val_accuracy"])
        best_path = model_paths[best_name]
        if os.path.exists(best_path):
            shutil.copy(best_path, BEST_MODEL_PATH)
            print(f"\n🏆 Best model: {best_name} "
                  f"(val_accuracy={results_store[best_name]['val_accuracy']:.4f}) "
                  f"-> copied to {BEST_MODEL_PATH}")

    print("\nAll requested models trained. Comparison saved to:", MODEL_COMPARISON_PATH)


if __name__ == "__main__":
    main()