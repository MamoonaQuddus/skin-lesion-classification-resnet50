"""
Helper functions for the Skin Lesion Classifier app.
Keeps app.py focused on UI; all model/image logic lives here.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model_and_config(model_path="skin_lesion_resnet50.keras",
                           config_path="inference_config.json"):
    """Load the trained model and its inference config (threshold, img size, classes)."""
    model = keras.models.load_model(model_path)
    with open(config_path) as f:
        config = json.load(f)
    return model, config


# ---------------------------------------------------------------------------
# Preprocessing / prediction
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """Resize + apply the same preprocessing used during training (NOT /255)."""
    resized = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(resized, dtype="float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, img_size: int, threshold: float, classes: list):
    """Run the model on one image. Returns (predicted_label, prob_malignant)."""
    batch = preprocess_image(image, img_size)
    prob_malignant = float(model.predict(batch, verbose=0)[0, 0])
    predicted_label = classes[int(prob_malignant >= threshold)]
    return predicted_label, prob_malignant


# ---------------------------------------------------------------------------
# Grad-CAM (same logic/layer names as the training notebook)
# ---------------------------------------------------------------------------
def gradcam_overlay(model, image: Image.Image, img_size: int):
    """
    Returns (original_array, heatmap_0_to_1, prob_malignant).
    Uses the same named layers as the notebook: resnet50_base, gap, drop, out.
    """
    base = model.get_layer("resnet50_base")
    gap = model.get_layer("gap")
    drop = model.get_layer("drop")
    out = model.get_layer("out")

    resized = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(resized, dtype="float32")
    x = tf.convert_to_tensor(preprocess_input(arr[None].copy()))

    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        p = out(drop(gap(conv), training=False))

    grads = tape.gradient(p, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype("uint8")).resize((img_size, img_size), Image.BILINEAR)
    ) / 255.0

    return arr.astype("uint8"), cam_resized, float(p.numpy()[0, 0])


# ---------------------------------------------------------------------------
# Basic input guardrails
# ---------------------------------------------------------------------------
def blur_score(image: Image.Image) -> float:
    """
    Simple blur estimate using Laplacian variance (no OpenCV dependency).
    Lower score = blurrier image. Rough guide: below ~50 is quite blurry.
    """
    gray = np.array(image.convert("L"), dtype="float64")
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.signal import convolve2d
    lap = convolve2d(gray, laplacian_kernel, mode="valid")
    return float(lap.var())


def image_quality_warnings(image: Image.Image) -> list:
    """Return a list of human-readable warnings about the uploaded image, if any."""
    warnings = []
    w, h = image.size

    if w < 100 or h < 100:
        warnings.append(f"Image is very small ({w}x{h}px). Predictions on tiny images are less reliable.")

    try:
        score = blur_score(image)
        if score < 30:
            warnings.append("Image looks quite blurry. Try a sharper, well-focused photo.")
    except Exception:
        pass  # blur check is a nice-to-have, never block prediction if it fails

    return warnings
