import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")

# ---- Load model + config once, cached across reruns ----
@st.cache_resource
def load_model_and_config():
    model = keras.models.load_model("skin_lesion_resnet50.keras")
    with open("inference_config.json") as f:
        config = json.load(f)
    return model, config

model, config = load_model_and_config()
IMG_SIZE = config["img_size"]
THRESHOLD = config["threshold"]
CLASSES = config["classes"]  # ["Benign", "Malignant"]

st.title("🔬 Skin Lesion Classifier")

uploaded_file = st.file_uploader(
    "Upload a skin lesion image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess exactly like training: resize -> resnet50 preprocess_input
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_resized, dtype="float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Running model..."):
        prob_malignant = float(model.predict(arr, verbose=0)[0, 0])

    predicted_idx = int(prob_malignant >= THRESHOLD)
    predicted_label = CLASSES[predicted_idx]

    st.subheader(f"Prediction: **{predicted_label}**")
    st.write(f"P(Malignant) = `{prob_malignant:.3f}`  (decision threshold = `{THRESHOLD}`)")

    if predicted_label == "Malignant":
        st.warning("Model flags this as potentially malignant. This is NOT a diagnosis — please consult a dermatologist.")
    else:
        st.success("Model flags this as benign. Still, consult a dermatologist for any real concern.")

    with st.expander("Raw probability details"):
        st.json({
            "prob_malignant": prob_malignant,
            "threshold_used": THRESHOLD,
            "predicted_class": predicted_label,
            "classes": CLASSES,
        })
else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.caption(
    "Model: ResNet50 fine-tuned on the melanoma-cancer-dataset (Kaggle). "
    "Preprocessing must match training (resnet50.preprocess_input)."
)
