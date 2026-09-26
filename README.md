# Melanoma Classification Using ResNet50

Binary image classifier that predicts whether a skin lesion is **Benign** or **Malignant melanoma**, built with transfer learning on ResNet50 and served through a Streamlit web app.

> **Disclaimer:** This is an educational / research project. 

---

## Problem Statement

Can a ResNet50-based classifier distinguish benign lesions from malignant melanoma, and how reliably does it generalize to unseen images? The project prioritizes careful data handling and honest evaluation over chasing the highest possible accuracy number.

## Dataset

- **Source:** [Melanoma Cancer Image Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) 
- **Training:** 11,879 images (6,289 Benign / 5,590 Malignant)
- **Test:** 2,000 images (1,000 / 1,000), reduced to a leakage-cleaned set of 931 images after duplicate removal (see below)
- Images are RGB, resized to 224x224 for the model

## Methodology

### 1. Data leakage prevention
The dataset has no patient IDs, so a perceptual hash (dHash) was used to detect near-duplicate images. 1,069 of 2,000 test images were found to be near-duplicates of training images and removed. Training data was grouped by near-duplicate clusters so no group crosses the train/validation split. This is a heuristic safeguard, not a guarantee.

### 2. Preprocessing
224x224 resize, RGB conversion, and `resnet50.preprocess_input` (matches the normalization ResNet50's ImageNet weights were trained with) — applied identically during training and inference.

### 3. Data augmentation (training set only)
Rotation (20°), width/height shift (10%), zoom (10%), horizontal + vertical flip, brightness variation (0.85x–1.15x).

### 4. Model architecture
ResNet50 (ImageNet pretrained) → Global Average Pooling → Dropout(0.5) → Dense(1, sigmoid).

### 5. Training (two phases)
- **Phase 1 — frozen backbone:** only the new classification head trained (Adam, lr=1e-3, 15 epochs).
- **Phase 2 — fine-tuning:** last 30 backbone layers unfrozen, BatchNorm layers kept frozen, lr dropped to 1e-5 (20 epochs) to adapt features without catastrophic forgetting.
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint used throughout.

### 6. Threshold selection
Swept on the validation set only. Chosen threshold (0.60) maximizes specificity while keeping sensitivity ≥ 90%, rather than defaulting to 0.5 or optimizing on the test set.

## Results (on the leakage-cleaned test set)

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline CNN @0.5 | 0.834 | 0.732 | 0.848 | 0.826 | 0.786 | 0.914 |
| ResNet50 frozen @0.5 | 0.886 | 0.838 | 0.848 | 0.908 | 0.843 | 0.950 |
| ResNet50 fine-tuned @0.5 | 0.910 | 0.860 | 0.896 | 0.918 | 0.877 | 0.968 |
| **ResNet50 fine-tuned @0.60 (final)** | **0.924** | **0.902** | **0.884** | **0.946** | **0.893** | **0.968** |

### Interpretability & robustness
- **Grad-CAM** confirms the model generally focuses on the lesion region rather than background artifacts.
- **Robustness testing** revealed a real weakness: specificity collapses from 0.946 to 0.250 under 15° image rotation, while brightness, JPEG compression, and resizing have minimal effect.

## Limitations

- Trained on a single dataset source — no diversity across cameras, lighting conditions, or skin tones has been verified.
- No true patient-wise split (no patient IDs available); near-duplicate detection is a heuristic, not a guarantee.
- Not rotation-invariant — a known, unresolved weakness (see Robustness above).
- Not validated on any external/out-of-distribution dataset.
- Not a substitute for professional medical diagnosis.

## Project Structure

```
├── app.py                      # Streamlit UI
├── utils.py                    # Model loading, preprocessing, Grad-CAM, input checks
├── requirements.txt            # Python dependencies
├── inference_config.json       # Saved threshold, image size, class names
└── melanoma-cancer-image-classification.ipynb # Trained notebook
```

## Ethical Considerations

This model was trained on a limited, single-source dataset and has known weaknesses (e.g. rotation sensitivity). It has not been clinically validated and must never be used to make or support real medical decisions. Any real diagnostic use of melanoma imagery should rely on qualified dermatologists and clinically validated tools.

## Future Improvements

- Combine multiple datasets (e.g. ISIC archive) for broader real-world coverage
- Fix rotation sensitivity with wider augmentation ranges and targeted re-evaluation
- Obtain a dataset with patient IDs for true patient-wise splitting
- Cross-validation to confirm result stability across different splits
- External/out-of-distribution validation
- Confidence calibration analysis
