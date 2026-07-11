"""
data_preprocessing.py
Handles loading the "New Plant Diseases Dataset (Augmented)" from Kaggle,
building tf.data / ImageDataGenerator pipelines, and saving the resulting
class-index mapping so the Streamlit app can decode predictions later.

Expected folder layout (standard Kaggle download):

dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
├── valid/
│   ├── Apple___Apple_scab/
│   └── ...
└── test/            (optional, flat folder of images for final holdout)
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, SEED, CLASS_INDICES_PATH
)


def build_data_generators(augment: bool = True):
    """
    Build train/validation ImageDataGenerators using directory-based flow.

    Returns:
        train_generator, valid_generator
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

    # Persist class index mapping: {"Apple___Apple_scab": 0, ...}
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(train_generator.class_indices, f, indent=2)

    print(f"Found {train_generator.num_classes} classes.")
    print(f"Class indices saved to {CLASS_INDICES_PATH}")

    return train_generator, valid_generator


def build_test_generator():
    """
    Build a generator for the held-out test set (if it exists as a
    directory of class subfolders, same structure as train/valid).
    """
    if not os.path.isdir(TEST_DIR):
        print(f"No test directory found at {TEST_DIR}. Skipping.")
        return None

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return test_generator


def load_class_indices() -> dict:
    """Load the saved class_indices.json as {class_name: index}."""
    with open(CLASS_INDICES_PATH, "r") as f:
        return json.load(f)


def get_index_to_class() -> dict:
    """Inverse mapping: {index: class_name} — used to decode predictions."""
    class_indices = load_class_indices()
    return {v: k for k, v in class_indices.items()}


def compute_class_weights(train_generator) -> dict:
    """
    Compute class weights to counter any class imbalance, useful even
    though the Kaggle dataset is largely balanced.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(train_generator.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_generator.classes,
    )
    return dict(zip(classes, weights))


if __name__ == "__main__":
    # Quick sanity check when run directly: prints dataset stats.
    train_gen, valid_gen = build_data_generators()
    print("Train samples:", train_gen.samples)
    print("Valid samples:", valid_gen.samples)
    print("Classes:", list(train_gen.class_indices.keys())[:5], "...")