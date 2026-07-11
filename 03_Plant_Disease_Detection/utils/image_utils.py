"""
utils/image_utils.py
Image preprocessing helpers shared between training/evaluation and the
Streamlit app (e.g. reading an uploaded file, resizing, normalizing).
"""

import io

import numpy as np
from PIL import Image


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load a PIL Image from raw uploaded file bytes, ensuring RGB mode."""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_for_model(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Resize and normalize a PIL image into a batch-ready numpy array
    suitable for model.predict(). Matches the 1/255 rescale used
    during training in data_preprocessing.py.
    """
    image = image.resize(target_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension -> (1, H, W, 3)
    return arr


def is_probably_leaf_image(image: Image.Image, green_threshold: float = 0.05) -> bool:
    """
    Very lightweight heuristic sanity-check: verifies the image contains
    a meaningful proportion of green/brown pixels typical of leaves,
    to help catch obviously wrong uploads (e.g. a photo of a car).
    This is NOT a substitute for the model — just a soft warning signal.
    """
    small = image.resize((64, 64))
    arr = np.array(small).astype("float32") / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # green-ish or brown-ish (leaf/soil) pixel mask
    green_mask = (g > r * 0.9) & (g > b * 0.9)
    brown_mask = (r > 0.3) & (g > 0.2) & (b < 0.35) & (r > b)
    plant_mask = green_mask | brown_mask

    proportion = float(np.mean(plant_mask))
    return proportion >= green_threshold