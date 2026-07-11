"""
transfer_learning_model.py
Builds transfer-learning models on top of ImageNet-pretrained backbones:
MobileNetV2, EfficientNetB0, and ResNet50. Each backbone is frozen
initially (feature extraction), with an option to unfreeze the top
layers for fine-tuning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, FINE_TUNE_LEARNING_RATE


def _get_base_model(model_name: str):
    """Instantiate the requested pretrained backbone (frozen, no top)."""
    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        base = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    elif model_name == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    elif model_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base = ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                          f"Choose from mobilenetv2, efficientnetb0, resnet50.")

    base.trainable = False
    return base


def build_transfer_model(model_name: str, num_classes=NUM_CLASSES,
                          learning_rate=LEARNING_RATE):
    """
    Build and compile a transfer-learning model with a frozen backbone.

    Args:
        model_name: One of "MobileNetV2", "EfficientNetB0", "ResNet50"
        num_classes: Number of output classes
        learning_rate: Initial learning rate for the classifier head

    Returns:
        Compiled tf.keras.Model, and the base_model reference (for fine-tuning later)
    """
    base_model = _get_base_model(model_name)

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{model_name}_TransferLearning")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, num_layers_to_unfreeze=30,
                              learning_rate=FINE_TUNE_LEARNING_RATE):
    """
    Unfreeze the top N layers of the base model for fine-tuning and
    re-compile with a much lower learning rate.
    """
    base_model.trainable = True

    # Freeze all layers except the last `num_layers_to_unfreeze`
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


def get_preprocessing_function(model_name: str):
    """
    Each backbone expects its own preprocessing (different pixel scaling).
    Use this instead of the generic 1/255 rescale for best transfer results.
    """
    model_name = model_name.lower()
    if model_name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return preprocess_input


if __name__ == "__main__":
    for name in ["MobileNetV2", "EfficientNetB0", "ResNet50"]:
        model, base = build_transfer_model(name)
        print(f"\n===== {name} =====")
        model.summary()