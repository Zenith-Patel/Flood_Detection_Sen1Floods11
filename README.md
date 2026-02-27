# 🌊 Flood Extent Mapping from Sentinel-1 SAR
### DeepLabV3+ · EfficientNet-B5 · Sen1Floods11 Benchmark

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)


---

## 📊 Results — Official Sen1Floods11 Test Split (90 chips)

| Metric | Score |
|--------|-------|
| **IoU** | **0.5907** |
| **Recall** | **0.8080** |

> Finds **8 out of 10 flooded areas** in the official benchmark test set.

### Benchmark Context

| Method | IoU | Notes |
|--------|-----|-------|
| Prithvi-CAFE (WACV 2026) | 0.8341 | NASA/IBM 100M-param foundation model |
| Baseline U-Net (literature) | 0.7057 | Standard architecture |
| **This model** | **0.5907** | Single T4 GPU, ~2 hours training |
| Simple threshold | ~0.35 | No deep learning |

---

## 🛰️ Why SAR?

Floods happen during storms. Storms mean clouds. Clouds block optical satellites completely.

Sentinel-1 SAR radar works **through clouds, through rain, at night** — exactly when you need it most.

```
Flood water  →  smooth surface  →  radar bounces away  →  DARK pixel
Dry land     →  rough surface   →  radar scatters back →  BRIGHT pixel
```

The model reads this pattern and draws a flood map.

---

## 🏗️ Architecture

```
Input: Sentinel-1 SAR — 2 channels (VV + VH), 512×512 pixels
                    ↓
        EfficientNet-B5 Encoder
        pretrained: noisy-student (300M images)
                    ↓
        ASPP — Atrous Spatial Pyramid Pooling
        (4 scales simultaneously)
                    ↓
        DeepLabV3+ Decoder
                    ↓
Output: Flood probability map → binary mask
        0 = dry land  |  1 = flooded
```

**Why DeepLabV3+ over U-Net?**

ASPP looks at the scene at 4 different scales simultaneously. This is critical for SAR flood detection — it helps distinguish dark water from dark shadows, roads, and urban areas that all look similar in radar.

---

## 🔧 Training Strategy

### Two-Phase Pipeline

**Phase 1 — Weak Label Pre-training (5 epochs)**

| Setting | Value |
|---------|-------|
| Data | 4,385 weakly-labeled + 357 hand-labeled SAR chips |
| Labels | Otsu algorithm (automated, noisy) |
| Purpose | Teach the model what water looks like in SAR globally |
| LR | 3e-4 with CosineAnnealingWarmRestarts |
| Loss | Tversky(α=0.3, β=0.7) + 0.5×BCE |

*Only 5 epochs — weak labels are noisy, model peaks early. More = overfitting on noise.*

**Phase 2 — Fine-tuning on Human Labels (up to 30 epochs)**

| Setting | Value |
|---------|-------|
| Data | 357 expert hand-labeled chips |
| Labels | Human annotated — precise flood boundaries |
| Purpose | Sharpen precision, correct weak label noise |
| LR | 1e-4 → decays via ReduceLROnPlateau (patience=5) |
| Early stop | Patience = 10 epochs |

### Key Techniques

**Tversky Loss (α=0.3, β=0.7)**
Penalizes missed floods 2.3× more than false alarms. In disaster response, missing a flood is more dangerous than a false alarm.

**Test-Time Augmentation (TTA)**
3 predictions averaged: normal + horizontal flip + vertical flip. Free +2–3% IoU.

**Gradient Clipping (max_norm=1.0)**
Prevents exploding gradients when fine-tuning a large encoder on a small dataset.

---

## 📦 Dataset — Sen1Floods11

Globally distributed flood benchmark across 11 events on 6 continents.

```
Hand-labeled:     446 chips — expert human flood annotations
Weakly-labeled: 4,385 chips — Otsu algorithm labels
Official test:     90 chips — never seen during training
Resolution:        10m × 10m per pixel
Events:  Bolivia, Cambodia, Ghana, India, Mekong, Nigeria,
         Pakistan, Paraguay, Somalia, Spain, USA
```

Download:
```bash
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/ .
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/ .
gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv .
```

---

## 🚀 Quick Start

### Requirements
```bash
pip install segmentation-models-pytorch albumentations rasterio \
            rioxarray torchmetrics scikit-learn torch torchvision
```

### Run in Google Colab
Open `Flood_Detection_Sen1Floods11.ipynb` with GPU runtime (T4 sufficient).  
Run all cells top to bottom. Total runtime ~2 hours.

### Run Inference on Any Sentinel-1 Image
```python
mask = run_flood_inference(
    model=model,
    input_tif="your_sentinel1_scene.tif",  # needs VV + VH bands
    output_tif="flood_map.tif"             # GeoTIFF, coordinates preserved
)
```

### Calculate Flooded Area
```python
import numpy as np

flood_pixels = np.count_nonzero(mask)
area_km2     = flood_pixels * 100 / 1_000_000  # 10m × 10m = 100 m² per pixel
print(f"Flooded area: {area_km2:.2f} km²")
```

---

## 📁 Repository Structure

```
├── Flood_Detection_Sen1Floods11.ipynb   # Complete pipeline
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
└── .gitignore
```

---

## 🌍 Real-World Applications

- **Copernicus Emergency Management Service (CEMS)** — EU rapid flood mapping
- **EU Floods Directive** — member states require updated flood hazard maps
- **Humanitarian response** — UNOSAT rapid flood extent for disaster relief
- **Insurance** — EU Solvency II flood exposure quantification
- **Climate monitoring** — tracking flood frequency under climate change

---

## 📚 References

```
Sen1Floods11:
Bonafilia et al. (2020) — Sen1Floods11: A Georeferenced Dataset
to Train and Test Deep Learning Flood Algorithms.
CVPR EarthVision Workshop.

Current SOTA:
Kaushik, Maurya, Tellman (2026) — Prithvi-CAFE: Unlocking
Full-Potential for Flood Inundation Mapping.
CV4EO Workshop @ WACV 2026. arXiv:2601.02315

DeepLabV3+:
Chen et al. (2018) — Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation. ECCV 2018.
```

---

## 👤 Author

**Zenith Patel**  
MSc Environmental and Resource Management  
Brandenburg University of Technology (BTU) Cottbus-Senftenberg, Germany

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
