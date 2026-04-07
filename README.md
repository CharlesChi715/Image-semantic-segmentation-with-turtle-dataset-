# Sea Turtle Semantic Segmentation

> UNet-based pixel-level segmentation of sea turtle body parts (turtle body, flippers, head) using TensorFlow/Keras and the SeaTurtleID2022 dataset.

---

## Overview

This project implements a custom **UNet** architecture in TensorFlow/Keras to perform semantic segmentation on sea turtle photographs. The model classifies each pixel into one of four categories: background, turtle body, flipper, or head. Annotations are loaded from COCO-format JSON and the dataset is split chronologically by timestamp.

---

## Dataset

**SeaTurtleID2022** — 8,729 annotated photographs of sea turtles with COCO-format segmentation masks.

| Split | Date Range | Images |
|-------|-----------|--------|
| Train | Before 2021-01-01 | 7,502 |
| Validation | 2021-01-01 onwards | 1,227 |

**Segmentation classes:**

| Class ID | Label |
|----------|-------|
| 0 | Background |
| 1 | Turtle body |
| 2 | Flipper |
| 3 | Head |

Images are resized to `128×128` for training. The original max resolution is `2000×2000`.

---

## Model Architecture

A lightweight **UNet** with 2 encoder levels and 2 decoder levels, implemented in Keras functional API.

```
Input (128×128×3)
    │
    ├─ Encoder Block 1 → [Conv2D×2 + BN + ReLU] → MaxPool  (64 filters)
    ├─ Encoder Block 2 → [Conv2D×2 + BN + ReLU] → MaxPool  (128 filters)
    │
    └─ Bottleneck      → [Conv2D×2 + BN + ReLU]             (256 filters)
    │
    ├─ Decoder Block 2 → Conv2DTranspose + Skip Connect + Conv2D×2  (128 filters)
    ├─ Decoder Block 1 → Conv2DTranspose + Skip Connect + Conv2D×2  (64 filters)
    │
Output → Conv2D (1×1, softmax, 4 classes)
```

Each conv block consists of two `Conv2D(3×3, same padding)` layers with BatchNormalization and ReLU activation. Skip connections concatenate encoder feature maps to their corresponding decoder levels.

---

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| Epochs | 20 |
| Batch size | 4 |
| Input size | 128 × 128 |

### Training Log (Summary)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.2406 | 0.8845 | 0.1839 | 0.8573 |
| 5 | 0.0779 | 0.8640 | 0.0880 | 0.8591 |
| 10 | 0.0498 | 0.8619 | 0.0612 | 0.8635 |
| 17 | 0.0314 | 0.8607 | 0.0562 | 0.8628 |
| 20 | 0.0271 | 0.8604 | 0.0565 | 0.8628 |

Training loss decreased steadily from 0.24 → 0.027 over 20 epochs. Validation loss stabilised around 0.056–0.062, indicating good generalisation without significant overfitting.

---

## Data Pipeline

The `SeaTurtleDataset` class extends `tf.keras.utils.Sequence` and handles:

- **Chronological splitting** by EXIF timestamp parsed from COCO metadata
- **COCO annotation loading** via `pycocotools` for multi-class mask generation
- **Mask compositing** — flipper and head labels (IDs 2, 3) take priority over turtle body (ID 1) when overlapping
- **Batch generation** with per-epoch shuffling
- **Graceful fallback** — images with no annotations receive an all-zero background mask

---

## Project Structure

```
├── segmentation.ipynb         # Full pipeline: data loading, training, prediction
├── annotations.json           # COCO-format annotations
├── UnetModel/                 # Saved TensorFlow model
│   ├── assets/
│   ├── saved_model.pb
│   └── variables/
├── images/                    # Raw sea turtle photographs
└── README.md
```

---

## Usage

### Install dependencies

```bash
pip install tensorflow pycocotools opencv-python pillow matplotlib numpy
```

### Train the model

```bash
jupyter notebook segmentation.ipynb
```

Run all cells through the **Build Model, train Model** section. The trained model is saved automatically to `UnetModel/`.

### Load and run inference

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("UnetModel")
result = model.predict(images)           # shape: (B, 128, 128, 4)
predictions = np.argmax(result, axis=-1) # shape: (B, 128, 128)
```

### Visualise predictions

The notebook renders a 3-row grid per batch:
- Row 1: Original image
- Row 2: Ground truth mask
- Row 3: Predicted mask

---

## Key Concepts Demonstrated

- **UNet encoder-decoder architecture** with skip connections in TensorFlow/Keras
- **COCO annotation parsing** — multi-class mask generation from polygon annotations
- **Chronological train/val splitting** — using image timestamps rather than random splits to simulate real deployment conditions
- **Mask compositing priority** — handling overlapping segmentation regions across body part categories
- **Model persistence** — saving and reloading TensorFlow SavedModel format for inference
