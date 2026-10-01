import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_model_and_config, predict, gradcam_overlay, image_quality_warnings

st.set_page_config(page_title="Melanoma Classifier", page_icon="\U0001FA78", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1f2b52 0%, #3a1f52 100%);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.9rem; }
.hero p { margin: 0.35rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

.result-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-top: 0.4rem;
    border: 1px solid rgba(128,128,128,0.25);
}
.result-malignant { background: rgba(220, 53, 69, 0.10); border-color: rgba(220, 53, 69, 0.35); }
.result-benign { background: rgba(40, 167, 69, 0.10); border-color: rgba(40, 167, 69, 0.35); }

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
.badge-malignant { background: #dc3545; color: white; }
.badge-benign { background: #28a745; color: white; }

.conf-bar-bg {
    width: 100%; height: 10px; border-radius: 6px;
    background: rgba(128,128,128,0.2); margin-top: 0.5rem; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_and_config():
    return load_model_and_config()


model, config = get_model_and_config()
IMG_SIZE = config["img_size"]
THRESHOLD = config["threshold"]
CLASSES = config["classes"]  # ["Benign", "Malignant"]

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: file, prediction, prob_malignant
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()  # tracks file_ids already added to history


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001FA78 About")
    st.write(
        "ResNet50, fine-tuned on the Melanoma Cancer Image Dataset (Kaggle). "
        "Classifies a lesion photo as **Benign** or **Malignant**."
    )
    st.markdown("**Test-set performance**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "92.4%")
    c2.metric("ROC-AUC", "0.968")
    c1.metric("Sensitivity", "88.4%")
    c2.metric("Specificity", "94.6%")
    st.caption(f"Decision threshold: {THRESHOLD}")
    st.divider()
    st.caption(
        "\u26a0\ufe0f Educational / research project only. Not a medical diagnostic tool. "
        "Always consult a dermatologist for real concerns."
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>\U0001FA78 Melanoma Classifier</h1>
  <p>Upload a lesion photo to get a Benign / Malignant prediction, with an explainability heatmap.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: render one prediction (used for both single and batch mode)
# ---------------------------------------------------------------------------
def render_prediction(image: Image.Image, filename: str, show_gradcam: bool, file_id: str):
    for warning in image_quality_warnings(image):
        st.warning(warning)

    label, prob_malignant = predict(model, image, IMG_SIZE, THRESHOLD, CLASSES)

    # Only add to history the first time we see this exact uploaded file.
    # Streamlit re-runs the whole script on every interaction (e.g. toggling
    # Grad-CAM), so without this check the same upload gets appended again
    # and again, filling history with duplicate rows.
    if file_id not in st.session_state.processed_ids:
        st.session_state.processed_ids.add(file_id)
        st.session_state.history.append({"file": filename, "prediction": label, "prob_malignant": prob_malignant})

    is_malignant = (label == "Malignant")
    badge_class = "badge-malignant" if is_malignant else "badge-benign"
    card_class = "result-malignant" if is_malignant else "result-benign"
    bar_color = "#dc3545" if is_malignant else "#28a745"
    pct = min(max(prob_malignant, 0.0), 1.0) * 100

    col1, col2 = st.columns([1, 1.1], gap="large")
    with col1:
        st.image(image, caption=filename, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="result-card {card_class}">
            <span class="badge {badge_class}">{label.upper()}</span>
            <div style="margin-top:0.6rem; font-size:0.9rem; opacity:0.85;">
                P(Malignant) = <b>{prob_malignant:.3f}</b> &nbsp;|&nbsp; threshold = {THRESHOLD}
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if is_malignant:
            st.warning("Flagged as potentially malignant. This is **not** a diagnosis, consult a dermatologist.")
        else:
            st.success("Flagged as benign. Still consult a dermatologist for any real concern.")

    if show_gradcam:
        with st.spinner("Generating Grad-CAM heatmap..."):
            arr, cam, _ = gradcam_overlay(model, image, IMG_SIZE)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.patch.set_alpha(0)
        axes[0].imshow(arr); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(arr); axes[1].imshow(cam, cmap="jet", alpha=0.4)
        axes[1].set_title("Grad-CAM (model focus)"); axes[1].axis("off")
        st.pyplot(fig)
        st.caption("Grad-CAM is an interpretability aid only, it does not prove the model reasons like a clinician.")

    with st.expander("Raw probability details"):
        st.json({"prob_malignant": prob_malignant, "threshold_used": THRESHOLD,
                  "predicted_class": label, "classes": CLASSES})


# ---------------------------------------------------------------------------
# Upload UI
# ---------------------------------------------------------------------------
top_col1, top_col2 = st.columns([2, 1])
with top_col1:
    tab_single, tab_batch = st.tabs(["\U0001F4F7 Single image", "\U0001F4DA Batch upload"])
with top_col2:
    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)

with tab_single:
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        render_prediction(Image.open(uploaded_file), uploaded_file.name, show_gradcam, uploaded_file.file_id)
    else:
        st.info("Upload an image to get a prediction.")

with tab_batch:
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="batch")
    if uploaded_files:
        for f in uploaded_files:
            st.divider()
            render_prediction(Image.open(f), f.name, show_gradcam, f.file_id)
    else:
        st.info("Upload one or more images to get predictions.")

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    with st.expander(f"\U0001F4CB Prediction history ({len(st.session_state.history)} this session)", expanded=False):
        st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.processed_ids = set()
            st.rerun()

st.divider()
st.caption(
    "Model: ResNet50 fine-tuned on the melanoma-cancer-dataset. "
    "Preprocessing must match training (resnet50.preprocess_input)."
)
