"""
config.py
Central configuration file for the Plant Disease Detection project.
All paths, hyperparameters, and constants live here so every other
script can just do `from config import *`.
"""

import os

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CUSTOM_CNN_MODEL_PATH = os.path.join(MODELS_DIR, "custom_cnn_model.keras")
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_model.keras")
EFFICIENTNET_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetb0_model.keras")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "resnet50_model.keras")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
MODEL_COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

# ---------------------------------------------------------------------
# IMAGE / TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 32
EPOCHS_CUSTOM_CNN = 25
EPOCHS_TRANSFER_LEARNING = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

VALIDATION_SPLIT = 0.2
SEED = 42

# ---------------------------------------------------------------------
# CLASS LABELS
# (Matches the Kaggle "New Plant Diseases Dataset (Augmented)" naming
#  convention: "<Plant>___<Disease>". Update this list automatically
#  from the training directory at runtime if it differs.)
# ---------------------------------------------------------------------
DEFAULT_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Soybean___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(DEFAULT_CLASS_NAMES)

# ---------------------------------------------------------------------
# TRANSFER LEARNING MODEL CHOICES
# ---------------------------------------------------------------------
TRANSFER_MODELS = ["MobileNetV2", "EfficientNetB0", "ResNet50"]

# ---------------------------------------------------------------------
# APP SETTINGS
# ---------------------------------------------------------------------
APP_TITLE = "🌿 Plant Disease Detection System"
APP_ICON = "🌿"
CONFIDENCE_THRESHOLD = 0.60  # below this, warn the user prediction is uncertain