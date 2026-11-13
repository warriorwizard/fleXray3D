# FleXray Models & Selection Guide

> **Reference Repository**: [VictorButoi/flexray on Hugging Face](https://huggingface.co/VictorButoi/flexray)  
> **Official Paper**: *FleXray: Universal Clinical X-ray Segmentation* (MIT CSAIL, MGH, Harvard Medical School — [arXiv:2609.26756](https://arxiv.org/abs/2609.26756))  
> **Interactive Demo**: [flexray.csail.mit.edu](https://flexray.csail.mit.edu/)

---

## 1. Architectural Foundation

All FleXray models share the **identical deep 2D Residual U-Net architecture**, the same 61-channel output space (background + 60 anatomical structures), and the same pre/post-processing pipelines:

- **Input Dimension**: Single-channel grayscale ($1 \times 256 \times 256$), normalized using percentile min-max ($0.5\text{th} \text{ to } 99.5\text{th}$ percentiles) and padded to square.
- **Network Backbone**: `fxr.models.UNet` with 7 encoder/decoder stages:
  - Channels per level: `[64, 128, 256, 512, 512, 720, 1024]`
  - Convolutions per block: 3
  - Instance Normalization with residual connections and skip paths.
- **Output Vocabulary**: 61 independent sigmoid channels ($256 \times 256 \times 61$ multilabel probabilities, thresholded at $\ge 0.5$):
  - **Soft Tissue (5)**: Lungs, Heart, Liver, Kidneys, Spleen.
  - **Spine (25)**: Cervical (C1–C7), Thoracic (T1–T12), Lumbar (L1–L5), Sacrum.
  - **Thoracic Cage (15)**: Ribs 1 through 12 individually, Sternum, Clavicles, Scapulae.
  - **Pelvis & Lower Extremities (7)**: Hips, Femurs, Patellae, Tibiae, Fibulae, Tarsals, Metatarsals, Toes.
  - **Upper Extremities (8)**: Skull, Humeri, Radii, Ulnae, Carpals, Metacarpals, Phalanges.
- **Weights & Formats**: Available in PyTorch EMA safetensors (`model.safetensors`), PyTorch checkpoints (`.pt`), and quantized/FP16 ONNX (`flexray-flux*-256-fp16.onnx`, ~203 MB each).

---

## 2. Understanding the 5 Model Variants

The five ensemble models differ **exclusively in the FluXray Proportion** used in their synthetic training data mix.

### What is "FluXray"?
To avoid relying solely on scarce manual 2D annotations, FleXray trains primarily on 3D CT volumes ray-cast into 2D Digitally Reconstructed Radiographs (DRRs):
1. **Raw DRRs**: Ray-cast from labeled CTs (like TotalSegmentator). They have **flawless anatomical labels**, but look artificially clean (lack detector grain, scatter, collimation shutters, and hospital artifacts).
2. **FluXray (Generatively Enhanced DRRs)**: Raw DRRs passed through Black Forest Labs' **FLUX.2** diffusion model. FLUX.2 injects sensor noise, beam hardening, collimation borders, lead markers, and exposure anomalies while **strictly preserving underlying organ and bone boundaries**.

### The 5 Checkpoints

| Checkpoint Subfolder | FluXray % | Training Data Composition | Clinical Behavioral Profile |
| :--- | :---: | :--- | :--- |
| `members/flux000` | **0.0%** | Pure physics DRRs + real radiographs | **Strict physical attenuation**. Extremely faithful to anatomical geometry; zero generative artifacts. More sensitive to messy clinical noise, lead tags, and burned-in text. |
| `members/flux025` | **25.0%** | 25% FLUX.2 / 75% raw DRR + real | Mild exposure and sensor noise tolerance while preserving rigid physical priors. |
| `members/flux0375`<br>*(Flagship / Default)* | **37.5%** | 37.5% FLUX.2 / 62.5% raw DRR + real | **The "Goldilocks" Optimum**. Empirically validated across 8 held-out clinical datasets as the single highest-performing checkpoint. Strong artifact invariance without generative drift. |
| `members/flux050` | **50.0%** | 50% FLUX.2 / 50% raw DRR + real | Balanced hybrid. High tolerance to patient habitus variations, contrast anomalies, and non-standard beam angles. |
| `members/flux075` | **75.0%** | 75% FLUX.2 / 25% raw DRR + real | **Heavy generative noise conditioning**. Best suited for severely degraded acquisitions (low-dose fluoroscopy, heavy scatter, portable bedside units). |

---

## 3. Inference Modes & Test-Time Augmentation (TTA)

FleXray provides four inference modes balancing speed against segmentation precision:

| Mode | Models Executed | TTA Passes | Forward Passes | Approx. Latency (GPU) | Approx. Latency (WebGPU) | Target Environment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low** | Flagship (`flux0375`) | 1 (none) | 1 | ~15–30 ms | ~2.5–4.0 s | Real-time viewers, interactive GUIs, edge devices. |
| **Normal** | Flagship (`flux0375`) | 8 or 16 | 8–16 | ~120–250 ms | ~12–20 s | High-accuracy single-model setups with limited VRAM. |
| **High** | All 5 Ensemble Members | 1 (none) | 5 | ~75–120 ms | ~10–15 s | Fast consensus multi-model segmentation without TTA overhead. |
| **X-High** | All 5 Ensemble Members | 8 or 16 | 40–80 | ~600–1200 ms | ~40–70 s | Gold-standard quantitative biometrics and surgical planning. |

### How TTA Works (`tta_v3` Chain)
When running with TTA ($N$ passes), the image runs through 1 un-augmented pass plus $N - 1$ stochastic transforms:
1. **Horizontal Flip** ($p=0.5$): Inverted back geometrically on the output probability before averaging.
2. **Intensity Inversion** ($p=0.5$): Inverts pixel values (`1 - image`).
3. **Contrast / Sharpness / Gaussian Noise**: Random scaling (0.7–1.3) and noise ($\sigma=0.01$).
4. **CLAHE / Gamma Jitter**: Contrast limited adaptive histogram equalization ($p=0.1$) or gamma modulation ($p=0.25$).

The sigmoid probabilities across all passes and ensemble models are averaged:
$$\bar{P}_{c}(x, y) = \frac{1}{M \times N} \sum_{m=1}^{M} \sum_{n=1}^{N} \sigma\left(z_{m, n, c}(x, y)\right)$$

---

## 4. When to Use Which Configuration

```
                                  What is your primary constraint / objective?
                                                      │
             ┌────────────────────────────────────────┼────────────────────────────────────────┐
             ▼                                        ▼                                        ▼
    Latency / Interactive                   Diagnostic Precision / Pre-op              Clinical Setting / Noise
             │                                        │                                        │
    ┌────────┴────────┐                      ┌────────┴────────┐                      ┌────────┴────────┐
    ▼                 ▼                      ▼                 ▼                      ▼                 ▼
Real-time UI      PACS Auto-tag       Surgical Planning   2D/3D Registration   Clean Hospital DR    Noisy Bedside / Fluoro
(Browser/Desktop) (10k images/hr)     (Cobb angle / JSN)  (CT to X-ray)        (Standard PA/AP)     (Scatter, Hardware, C-arm)
    │                 │                      │                 │                      │                 │
 flux0375          flux0375               Ensemble          flux000 or             flux0375          flux050 or
 (1-pass)          (1-pass)               + 16 TTA          Ensemble               (1-pass)          flux075
```

### Scenario Breakdown

#### 1. Real-Time Radiologist PACS Viewer & Browser Demos
- **Choice**: **`flux0375` (Low Mode)**
- **Why**: Downloading one ONNX file takes ~200 MB rather than ~1 GB. A single forward pass takes ~20 ms on modern GPUs, enabling instant pan-and-zoom mask rendering.

#### 2. Orthopedic Biometrics: Cobb Angle (Scoliosis) & Knee Joint Space (JSN)
- **Choice**: **5-Model Ensemble + 16 TTA (`--ensemble --tta-samples 16`)**
- **Why**: Cobb angle calculation relies on finding the exact slope of vertebral endplates (C1–L5). Individual models can have slight pixel jitter at bone boundaries; the ensemble smooths boundary contours, bringing error down to **5.13° MAE** (below human inter-observer variability).

#### 3. 2D/3D Registration & Intraoperative Guidance
- **Choice**: **`flux000`** or **Full Ensemble**
- **Why**: In 2D/3D registration (such as aligning intraoperative fluoroscopy to preoperative CT volumes via XVR), predictions are compared against simulated CT ray-casts using Chamfer distance. `flux000` provides pure physical projection boundaries with no generative distortion.

#### 4. Automated PACS Archiving & Metadata Healing
- **Choice**: **`flux0375` (1 pass)**
- **Why**: When indexing incoming DICOM studies to verify or restore missing `BodyPartExamined` or `ViewPosition` tags, you only need organ presence/absence confidence (e.g., lungs+heart $\rightarrow$ Chest; pelvis+femurs $\rightarrow$ Pelvis). Single-pass flagship provides 100% classification confidence with maximum throughput.

#### 5. Portable Bedside X-rays & Low-Dose Fluoroscopy
- **Choice**: **`flux075`** or **Ensemble**
- **Why**: Bedside chest radiographs frequently have severe collimation, rotated patient posture, external leads/wires, and low exposure contrast. `flux075` was specifically trained on high generative variance, preventing wire shadows or tags from breaking rib and lung segmentations.

---

## 5. Quick Usage Recipes

### Single-Pass Flagship (Python)
```python
from fxr.inference import FleXraySegmenter

# Loads flagship flux0375 by default
segmenter = FleXraySegmenter.from_pretrained("VictorButoi/flexray")

prediction = segmenter.predict("radiograph.png", threshold=0.5)
masks = prediction.masks          # uint8 tensor [1, 61, 256, 256]
probabilities = prediction.probabilities  # float32 tensor [1, 61, 256, 256]
```

### Five-Model Ensemble with 16 TTA Passes (CLI)
```bash
# Maximum accuracy mode for clinical studies or benchmarks
flexify --ensemble --tta-samples 16 --input ./radiographs/ --output-dir ./results/
```

### Loading a Specific Member (e.g., pure physics for CT registration)
```python
segmenter_physics = FleXraySegmenter.from_pretrained(
    "VictorButoi/flexray", subfolder="members/flux000"
)
prediction = segmenter_physics.predict("intraop_fluoro.png")
```

### Extracting a Single Anatomy Channel
```python
# Extract only the femurs channel (shape: [1, 1, 256, 256])
femur_pred = segmenter.predict("knee.dcm", label="femurs")
```

