from __future__ import annotations

from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "waste_model.h5"
CLASS_NAMES_PATH = ROOT_DIR / "model" / "class_names.json"
IMAGE_SIZE = (224, 224)

DEFAULT_CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

KEYWORD_MAP = {
    "battery": "battery",
    "cell": "battery",
    "paper": "paper",
    "newspaper": "paper",
    "book": "paper",
    "cardboard": "cardboard",
    "carton": "cardboard",
    "plastic": "plastic",
    "bottle": "plastic",
    "wrapper": "plastic",
    "poly": "plastic",
    "metal": "metal",
    "can": "metal",
    "tin": "metal",
    "glass": "white-glass",
    "brown": "brown-glass",
    "green": "green-glass",
    "cloth": "clothes",
    "shirt": "clothes",
    "dress": "clothes",
    "shoe": "shoes",
    "slipper": "shoes",
    "food": "food",
    "banana": "biological",
    "apple": "biological",
    "leaf": "biological",
    "trash": "trash",
    "garbage": "trash",
}


def get_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and names:
                return [str(name).lower().strip() for name in names]
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


@st.cache_resource(show_spinner=False)
def load_model_if_available():
    """Load a Keras model only when model/waste_model.h5 exists.

    The app is designed to run safely even before the model is trained.
    """
    if not MODEL_PATH.exists():
        return None
    try:
        from tensorflow.keras.models import load_model

        return load_model(MODEL_PATH)
    except Exception as exc:
        st.warning(f"Model file was found, but it could not be loaded: {exc}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def _keyword_prediction(filename: str | None) -> str | None:
    if not filename:
        return None
    lowered = filename.lower()
    for keyword, class_name in KEYWORD_MAP.items():
        if keyword in lowered:
            return class_name
    return None


def _image_signature_prediction(image: Image.Image, class_names: list[str], filename: str | None = None) -> tuple[str, float, dict[str, float]]:
    """Deterministic demo prediction for portfolios before the real model is added.

    This is not a real trained model. It keeps the product demo working until
    model/waste_model.h5 is created using train_model.py.
    """
    keyword_class = _keyword_prediction(filename)
    if keyword_class:
        top_class = keyword_class
        confidence = 0.91
    else:
        small = image.convert("RGB").resize((32, 32))
        arr = np.asarray(small).astype("float32")
        avg = arr.mean(axis=(0, 1))
        brightness = float(avg.mean())
        red, green, blue = avg.tolist()

        # Simple visual heuristics to make demo behavior feel sensible.
        if brightness > 205:
            top_class = "paper"
            confidence = 0.74
        elif green > red + 20 and green > blue + 10:
            top_class = "biological"
            confidence = 0.73
        elif blue > red + 15 and blue > green + 10:
            top_class = "plastic"
            confidence = 0.71
        elif abs(red - green) < 10 and abs(green - blue) < 10 and brightness < 100:
            top_class = "metal"
            confidence = 0.69
        else:
            digest = hashlib.md5(np.asarray(small).tobytes()).hexdigest()
            index = int(digest[:8], 16) % len(class_names)
            top_class = class_names[index]
            confidence = 0.66 + (int(digest[8:10], 16) % 20) / 100

    if top_class not in class_names:
        class_names.append(top_class)

    scores = {name: round((1.0 - confidence) / max(len(class_names) - 1, 1), 4) for name in class_names}
    scores[top_class] = round(confidence, 4)
    return top_class, confidence, scores


def predict_waste(image: Image.Image, filename: str | None = None) -> dict:
    """Predict waste class.

    Returns a dictionary with class_name, confidence, top_scores, and mode.
    mode = "trained_model" when a Keras model is available.
    mode = "demo_mode" when no model is available.
    """
    class_names = get_class_names()
    model = load_model_if_available()

    if model is None:
        class_name, confidence, top_scores = _image_signature_prediction(image, class_names, filename)
        return {
            "class_name": class_name,
            "confidence": float(confidence),
            "top_scores": dict(sorted(top_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
            "mode": "demo_mode",
        }

    x = preprocess_image(image)
    prediction = model.predict(x, verbose=0)[0]
    top_index = int(np.argmax(prediction))
    confidence = float(prediction[top_index])
    class_name = class_names[top_index] if top_index < len(class_names) else str(top_index)

    top_indices = prediction.argsort()[-5:][::-1]
    top_scores = {}
    for idx in top_indices:
        label = class_names[int(idx)] if int(idx) < len(class_names) else str(idx)
        top_scores[label] = round(float(prediction[int(idx)]), 4)

    return {
        "class_name": class_name,
        "confidence": confidence,
        "top_scores": top_scores,
        "mode": "trained_model",
    }
