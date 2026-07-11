"""
custom_cnn_model.py
Defines a Custom Convolutional Neural Network built from scratch
(no pretrained weights) as a baseline to compare against transfer
learning models.

Architecture:
    [Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout] x 4
    -> Flatten -> Dense -> BatchNorm -> Dropout -> Dense(softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                      learning_rate=LEARNING_RATE):
    """
    Build and compile a custom CNN from scratch.

    Returns:
        A compiled tf.keras.Model
    """
    model = models.Sequential(name="Custom_CNN")

    # ---------------- Block 1 ----------------
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                             input_shape=input_shape,
                             kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ---------------- Block 2 ----------------
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ---------------- Block 3 ----------------
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # ---------------- Block 4 ----------------
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # ---------------- Classifier head ----------------
    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


if __name__ == "__main__":
    m = build_custom_cnn()
    m.summary()