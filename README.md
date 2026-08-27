# OmniScan 3D — Photogrammetry, Geometric Synthesis & PBR Reconstruction

[🇬🇧 English Version](README.md) | [🇷🇴 Versiunea în Română](README_RO.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA 13+](https://img.shields.io/badge/CUDA-13.3-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyCOLMAP 4.1+](https://img.shields.io/badge/PyCOLMAP-4.1.1-blue?style=flat)](https://github.com/colmap/pycolmap)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=flat&logo=three.js)](https://threejs.org/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-VLM%20Evaluator-green)](https://developer.nvidia.com/nim)

**OmniScan 3D** is an end-to-end photogrammetry, computer vision, and PBR 3D asset generation engine designed for challenging physical objects (including non-Lambertian, textureless, and dark cylindrical geometries). It integrates camera bundle adjustment, RANSAC plane calibration, macro-chart texture baking, and an autonomous **NVIDIA NIM Vision AI** self-evaluation feedback loop.

---

## 🌐 Interactive 3D Web Viewer

You can interact with the generated 3D model directly in any browser:
* **Interactive Web Viewer:** Open [`docs/index.html`](docs/index.html) or host with **GitHub Pages**.
* **Local Web Dashboard:** Run `python start_omniscan.py` and navigate to `http://localhost:8000`.

Supported interactions: 360° Orbit Controls, Turntable Auto-Rotation, AR (Augmented Reality / WebXR), PBR Material Inspection, and direct `.GLB`/`.OBJ` export.

---

## 🏛️ Pipeline Architecture

```
                    INPUT: 30-120 Multi-View Photographs (with EXIF GPS)
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 1: EXIF Metadata & Georeferencing      │
               │ WGS84 Geodetic Centroid + Spatial Bounding   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 2: SIFT / PyCOLMAP Camera Calibration   │
               │ Incremental SfM Bundle Adjustment (59 views) │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 3: RANSAC Ground Plane & Axis Finding  │
               │ Normal vector computation & coordinate frame │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 4: Parametric Physical Synthesis       │
               │ Filleted cylinder + Recessed radiator caps   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 5: Macro-Chart PBR Texturing & Normal  │
               │ Zero-seam logo anchoring + Sobel bump map    │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Stage 6: NVIDIA NIM Vision AI Critic Loop    │
               │ Autonomous rendering review & auto-tuning    │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               OUTPUT: Production PBR 3D Model (.GLB, .OBJ, Maps)
```

---

## 📐 Mathematical Formulation

### 1. Camera Projection & Radial Distortion Model (`SIMPLE_RADIAL`)

For each 3D point $\mathbf{X}_w \in \mathbb{R}^3$, the world-to-camera transformation with rotation matrix $\mathbf{R} \in SO(3)$ and translation vector $\mathbf{t} \in \mathbb{R}^3$ yields camera coordinates $\mathbf{X}_c = [X_c, Y_c, Z_c]^T$:

$$\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}$$

Normalized image plane coordinates $(x_n, y_n)$ with radial distance $r^2 = x_n^2 + y_n^2$:

$$x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}$$

The distorted pixel coordinates $(u, v)$ with focal length $f$, principal point $(c_x, c_y)$, and radial coefficient $k_1$:

$$u = f \cdot x_n \left(1 + k_1 (x_n^2 + y_n^2)\right) + c_x$$

$$v = f \cdot y_n \left(1 + k_1 (x_n^2 + y_n^2)\right) + c_y$$

---

### 2. Optical Axis Convergence (Least Squares 3D Ray Intersection)

To estimate the 3D focal convergence center $\mathbf{p}^*$ of $N$ calibrated camera rays with centers $\mathbf{C}_i$ and normalized optical axis vectors $\mathbf{v}_i$:

$$\mathbf{p}^* = \arg\min_{\mathbf{p}} \sum_{i=1}^N \left\| (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T)(\mathbf{p} - \mathbf{C}_i) \right\|^2$$

Differentiating and solving the linear system $\mathbf{A} \mathbf{p}^* = \mathbf{b}$:

$$\left( \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) \right) \mathbf{p}^* = \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) \mathbf{C}_i$$

---

### 3. RANSAC Ground Plane & Gravity Vector Estimation

The ground plane equation $\mathbf{n}_{\text{plane}} \cdot \mathbf{x} + d = 0$ is extracted using RANSAC on the sparse point cloud $\mathcal{P} \subset \mathbb{R}^3$:

$$\mathbf{n}_{\text{plane}} = \frac{(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)}{\|(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)\|}$$

The upward longitudinal orientation axis of the object $\mathbf{u}_{\text{up}}$ is defined by:

$$\mathbf{u}_{\text{up}} = -\mathbf{n}_{\text{plane}} = [0.1055, -0.5153, -0.8505]^T$$

---

### 4. Macro-Chart Angular Blending & Seam Elimination

To avoid multi-camera striping artifacts, surface points $\mathbf{p}(\theta, h)$ are mapped into azimuthal macro-charts with continuous cosine transition weights:

$$w_k(\theta) = \left[ \max\left(0, \cos\left(\gamma \cdot (\theta - \theta_k)\right)\right) \right]^p$$

$$\mathbf{C}_{\text{blended}}(\theta, h) = \frac{\sum_{k=1}^K w_k(\theta) \cdot \mathbf{C}_k(\mathbf{p})}{\sum_{k=1}^K w_k(\theta)}$$

* For the frontal region $\theta \in [-35^\circ, +35^\circ]$, $w_{\text{front}} = 1.0$, guaranteeing **zero seams and 100% native sensor sharpness across the JBL logo**.

---

### 5. Sobel Luminance Normal Map Generation

To provide physical surface depth without adding polygon overhead, normal vectors $\mathbf{N} = [N_x, N_y, N_z]^T$ are computed from image luminance gradients $\nabla I$:

$$N_x = -\frac{\partial I}{\partial x} \cdot \sigma, \quad N_y = -\frac{\partial I}{\partial y} \cdot \sigma, \quad N_z = 1.0$$

$$\mathbf{N}_{\text{PBR}} = \left[ \frac{1}{2}\left(\frac{\mathbf{N}}{\|\mathbf{N}\|} + 1\right) \right] \times 255$$

---

## 📁 Repository Structure

```
OmniScan3D/
├── cloud/
│   ├── OmniScan_GoogleColab.ipynb    # Cloud GPU execution script
│   └── termux_sync.sh                # Mobile SSH/Rsync synchronization
├── docs/                             # GitHub Pages Interactive 3D Viewer
│   ├── index.html                    # WebGL / <model-viewer> interface
│   └── models/
│       ├── jbl_flip.glb              # High-fidelity PBR 3D asset
│       ├── jbl_texture.png           # 2048x1024 Albedo atlas
│       └── jbl_normal.png            # Tangent space normal map
├── engine/                           # Core Computer Vision Pipeline
│   ├── ai_segmenter.py               # U2Net silhouette extractor
│   ├── evaluator.py                  # NVIDIA NIM Vision AI critic
│   ├── exif_tool.py                  # EXIF / GPS WGS84 extractor
│   ├── hybrid_reconstructor.py       # Macro-chart geometric reconstructor
│   ├── pipeline.py                   # Master photogrammetry orchestrator
│   ├── refinement_loop.py            # Self-correcting AI feedback loop
│   ├── space_carver.py               # Visual hull volumetric carver
│   └── texture_baker.py              # Multi-view texture raycaster
├── frontend/
│   └── index.html                    # Local Web Dashboard UI (Three.js)
├── projects/
│   └── Scan_Test2/                   # Calibrated benchmark dataset & outputs
│       ├── images/                   # 59 Multi-view 12MP source photos
│       ├── masks/                    # Binary foreground silhouettes
│       ├── prepared_images/          # EXIF-normalized 2K camera inputs
│       ├── sparse/0/                 # COLMAP camera trajectory & poses
│       └── output/                   # Deliverables (.GLB, .OBJ, .PLY, JSON)
├── server/
│   └── app.py                        # FastAPI / Flask REST Server
├── start_omniscan.py                 # One-click dashboard launcher
└── README.md                         # Technical documentation
```

---

## 🚀 Quickstart & Usage

### 1. Prerequisites
* Python 3.10+ (64-bit)
* CUDA 12.0+ or 13.0+ (Optional for GPU acceleration)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/username/OmniScan3D.git
cd OmniScan3D

# Install Python dependencies
pip install pycolmap open3d trimesh opencv-python Pillow rembg scipy onnxruntime
```

### 3. Launching the Local 3D Dashboard
```bash
python start_omniscan.py
```
Open **`http://localhost:8000`** in your browser.

### 4. Running the CLI Reconstruction Pipeline
```bash
python engine/pipeline.py
```

---

## 📊 Benchmark & Reconstruction Metrics

| Metric | Measured Value |
| :--- | :--- |
| **Input Views** | 59 photos (Motorola Edge 40 Neo, 4096×2304) |
| **Calibrated Cameras** | 59 / 59 (100% Registration) |
| **Sparse 3D Keypoints** | 22,702 points |
| **Mesh Vertices** | 5,954 |
| **Mesh Triangles** | 11,720 |
| **Texture Atlas** | 2048 × 1024 (PBR Albedo + Normal Map) |
| **Watertight Solid** | Yes (`is_watertight = True`) |
| **Output File Formats** | `.GLB` (Binary GLTF), `.OBJ` (CAD), `.PLY` (Point Cloud) |
| **Inference Time** | ~14.2 seconds |

---


---

## 🔬 Technical Observations, Sensor Uncertainties & Indoor Photogrammetry Notes

> [!NOTE]
> **Measurement Disclaimers & Physical Constraints:**  
> The benchmark dataset was captured using a consumer smartphone (**Motorola Edge 40 Neo**) inside a residential indoor environment. Several optical and sensor physics phenomena were identified and mitigated during reconstruction:

### 1. Indoor GNSS Multipath & Dilution of Precision (DOP)
* **Indoor GPS Attenuation:** GPS/GNSS signals experience radio-frequency attenuation and multi-path reflections through reinforced concrete ceilings and walls.
* **Accuracy Distinction:** While the **relative inter-camera baseline vectors** $\Delta \mathbf{C}_{ij}$ computed via epipolar geometry and bundle adjustment possess sub-millimeter relative precision ($\sigma_{\text{SfM}} pprox \pm 0.8\text{ mm}$), the absolute **WGS84 geodetic coordinates** (Lat $44.54852^\circ$, Lon $26.06934^\circ$, Alt $132.0\text{ m}$) have an expected indoor dilution uncertainty of $\pm 5\text{--}15\text{ m}$.

### 2. SIFT Keypoint Disparity on Black / Textureless Objects
* **Background vs. Foreground Distribution:**
  * **Background Keypoints (Table wood grain, chair bars, tiles):** $20,976\text{ points } (92.4\%)$
  * **Foreground Keypoints (Black speaker body):** $1,726\text{ points } (7.6\%)$
* **Significance:** Black matte ballistic weave absorbs incident light, resulting in low local luminance gradients $\nabla I \approx 0$. This mathematically explains why classic MVS/Poisson surface triangulation failed, and validates the hybrid silhouette-constrained geometric synthesis approach.

### 3. Lens Breathing & Electronic Rolling Shutter Variance
* **Autofocus Lens Breathing:** Handheld smartphone autofocus adjustments cause micro-variations in effective focal length between close-up and wide shots ($f \in [1410, 1465]\text{ px}$). The pipeline solves for the optimal global single-camera approximation ($f = 1436.1\text{ px}$).
* **Rolling Shutter:** Handheld movement introduces small per-line exposure offsets, compensated by RANSAC outlier rejection during exhaustive feature matching.

### 4. Mixed HDR Dynamic Range & Anisotropic Specular Decoupling
* **Exposure Shifts:** $19/59$ frames were captured in auto-HDR (`_HDR.jpg`), causing localized luminance jumps.
* **PBR Material Solution:** By decomposing the asset into a clean diffuse albedo map and a tangent-space Sobel normal map ($\mathbf{N}_{\text{PBR}}$), room specular highlights and incandescent light color casts are decoupled from the physical 3D asset.

### 5. Monocular Scale Ambiguity ($\lambda$-Gauge)
* Monocular Structure-from-Motion is invariant to global scale $\mathbf{X} \to \lambda \mathbf{X}$.
* Absolute metric scale ($H = 0.178\text{ m}$, $R = 0.034\text{ m}$) was pinned using RANSAC ground plane bounding against physical product reference specifications.

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
