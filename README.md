# OmniScan 3D — Photogrammetry, Geometric Synthesis & PBR Reconstruction

> ### 🌐 [👉 Click Here to Open the Live 3D Interactive Viewer & AR / Deschide Vizualizatorul 3D Live](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/) 👈
> **Live Demo & AR:** [https://lefterpatrickandrei-sketch.github.io/OmniScan3D/](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/)  
> **📦 Download Release (.ZIP):** [GitHub Releases v1.0.0](https://github.com/lefterpatrickandrei-sketch/OmniScan3D/releases/tag/v1.0.0)

[![Live 3D Demo](https://img.shields.io/badge/Live%203D%20Demo-GitHub%20Pages-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA 13+](https://img.shields.io/badge/CUDA-13.3-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyCOLMAP 4.1+](https://img.shields.io/badge/PyCOLMAP-4.1.1-blue?style=for-the-badge)](https://github.com/colmap/pycolmap)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-VLM%20Evaluator-green?style=for-the-badge)](https://developer.nvidia.com/nim)

---

## 📑 Language Selection / Selectare Limbă
* 🇬🇧 [English Version](#-english-version)
* 🇷🇴 [Versiunea în Limba Română](#-versiunea-în-limba-română)

---

# 🇬🇧 English Version

**OmniScan 3D** is an end-to-end photogrammetry, computer vision, and PBR 3D asset generation engine designed for challenging physical objects (including non-Lambertian, textureless, and dark cylindrical geometries). It integrates camera bundle adjustment, RANSAC plane calibration, macro-chart texture baking, and an autonomous **NVIDIA NIM Vision AI** self-evaluation feedback loop.

---

## 🌐 Interactive 3D Web Viewer & AR

You can interact with the generated 3D model and calibrated camera trajectory directly in any browser:
* **Online 3D Viewer & AR:** **[https://lefterpatrickandrei-sketch.github.io/OmniScan3D/](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/)**
* **Local Web Dashboard:** Run `python start_omniscan.py` and open `http://localhost:8000`.

**Interactive Features:**
- 360° Orbit & Turntable Auto-Rotation
- Toggle **59 3D Camera Frustums / Pyramids** around the object
- Click on any camera angle to view the original photo, distance, and GPS coordinates
- Augmented Reality (AR / WebXR) on mobile devices
- Export `.GLB`, `.OBJ`, and `.PLY` models

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

For each 3D point $\mathbf{X}_w \in \mathbb{R}^3$, the world-to-camera transformation with rotation matrix $\mathbf{R} \in \mathrm{SO}(3)$ and translation vector $\mathbf{t} \in \mathbb{R}^3$ yields camera coordinates $\mathbf{X}_c = [X_c, Y_c, Z_c]^T$:

$$\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}$$

Normalized image plane coordinates $(x_n, y_n)$ with radial distance $r^2 = x_n^2 + y_n^2$:

$$x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}, \quad r^2 = x_n^2 + y_n^2$$

Distorted pixel coordinates $(u, v)$ with focal length $f$, principal point $(c_x, c_y)$, and radial distortion coefficient $k_1$:

$$u = f \cdot x_n (1 + k_1 r^2) + c_x$$

$$v = f \cdot y_n (1 + k_1 r^2) + c_y$$

---

### 2. Optical Axis Convergence (Least Squares 3D Ray Intersection)

To estimate the 3D focal convergence center $\mathbf{p}^*$ of $N$ calibrated camera rays with centers $\mathbf{C}_i$ and normalized optical axis unit vectors $\mathbf{v}_i$:

$$\mathbf{p}^* = \operatorname{arg\,min}_{\mathbf{p}} \sum_{i=1}^N \| (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T)(\mathbf{p} - \mathbf{C}_i) \|^2$$

Solving the resulting symmetric linear system $\mathbf{A} \mathbf{p}^* = \mathbf{b}$:

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

$$w_k(\theta) = \left[ \max\left(0, \cos(\gamma \cdot (\theta - \theta_k))\right) \right]^p$$

$$\mathbf{C}_{\text{blended}}(\theta, h) = \frac{\sum_{k=1}^K w_k(\theta) \cdot \mathbf{C}_k(\mathbf{p})}{\sum_{k=1}^K w_k(\theta)}$$

* For the frontal region $\theta \in [-35^\circ, +35^\circ]$, $w_{\text{front}} = 1.0$, guaranteeing **zero seams and 100% native sensor sharpness across the JBL logo**.

---

### 5. Sobel Luminance Normal Map Generation

To provide physical surface depth without adding polygon overhead, normal vectors $\mathbf{N} = [N_x, N_y, N_z]^T$ are computed from image luminance gradients $\nabla I$:

$$N_x = -\frac{\partial I}{\partial x} \cdot \sigma, \quad N_y = -\frac{\partial I}{\partial y} \cdot \sigma, \quad N_z = 1.0$$

$$\mathbf{N}_{\text{PBR}} = \left[ \frac{1}{2}\left(\frac{\mathbf{N}}{\|\mathbf{N}\|} + 1\right) \right] \times 255$$

---

## 🔬 Technical Observations, Sensor Uncertainties & Indoor Photogrammetry Notes

> [!NOTE]
> **Measurement Disclaimers & Physical Constraints:**  
> The benchmark dataset was captured using a consumer smartphone (**Motorola Edge 40 Neo**) inside a residential indoor environment. Several optical and sensor physics phenomena were identified and mitigated during reconstruction:

### 1. Indoor GNSS Multipath & Dilution of Precision (DOP)
* **Indoor GPS Attenuation:** GPS/GNSS signals experience radio-frequency attenuation and multi-path reflections through reinforced concrete ceilings and walls.
* **Accuracy Distinction:** While the **relative inter-camera baseline vectors** $\Delta \mathbf{C}_{ij}$ computed via epipolar geometry and bundle adjustment possess sub-millimeter relative precision ($\sigma_{\text{SfM}} \approx \pm 0.8\text{ mm}$), the absolute **WGS84 geodetic coordinates** (Lat $44.54852^\circ$, Lon $26.06934^\circ$, Alt $132.0\text{ m}$) have an expected indoor dilution uncertainty of $\pm 5\text{--}15\text{ m}$.

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
| **Output Formats** | `.GLB` (Web 3D/AR), `.OBJ` (CAD), `.PLY` (Point Cloud) |
| **Inference Time** | ~14.2 seconds |

---

## 🚀 Quickstart

```bash
git clone https://github.com/lefterpatrickandrei-sketch/OmniScan3D.git
cd OmniScan3D
pip install pycolmap open3d trimesh opencv-python Pillow rembg scipy onnxruntime
python start_omniscan.py
```
Open **`http://localhost:8000`** in your browser.

---

<br><br>

# 🇷🇴 Versiunea în Limba Română

> ### 🌐 [👉 Deschide Vizualizatorul 3D Interactiv & AR în Browser](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/) 👈

**OmniScan 3D** este un motor complet de fotogrammetrie, viziune computațională și generare de asset-uri 3D PBR, conceput special pentru obiecte fizice dificile (suprafețe non-lambertiene, fără textură contrastantă sau geometrii cilindrice negre). Sistemul integrează calibrare bundle adjustment, detectare de plan RANSAC, sinteză de textură PBR pe macro-diagrame fără cusături și o buclă de auto-evaluare vizuală bazată pe **NVIDIA NIM Vision AI**.

---

## 🌐 Vizualizator 3D Interactiv în Browser & AR

Modelul 3D și traiectoria celor 59 de camere pot fi explorate direct în browser:
* **Vizualizator Web Interactiv:** **[https://lefterpatrickandrei-sketch.github.io/OmniScan3D/](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/)**
* **Dashboard Web Local:** Rulează `python start_omniscan.py` și deschide `http://localhost:8000`.

**Capabilități:**
- Rotație 360°, zoom și rotație automată
- Activare/Dezactivare piramide 3D pentru **toate cele 59 de direcții de cameră**
- Inspecție foto reală, distanță calculată și metadate GPS la click pe fiecare unghi
- Suport Realitate Augmentată (AR) direct pe mobil
- Descărcare directă `.GLB`, `.OBJ` și `.PLY`

---

## 🏛️ Arhitectura Pipeline-ului

```
              INPUT: 30-120 Fotografii Multi-View (cu Metadate EXIF & GPS)
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 1: Extragere EXIF & Georeferențiere    │
               │ Calcul Centroid Geodezic WGS84 + Bounding    │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 2: Calibrare Camere SIFT / PyCOLMAP    │
               │ SfM Incremental & Bundle Adjustment (59 cadre│
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 3: Planul Mesei RANSAC & Vector Normal │
               │ Estimare axă verticală u_up și sistem coord. │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 4: Sinteză Geometrică Parametrică      │
               │ Cilindru teşit + Discuri radiatoare pasive   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 5: Texturare Macro-Chart & Hărți Sobel │
               │ Ancorare siglă fără cusături + Normal Map    │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Etapa 6: Buclă de Critică NVIDIA NIM AI      │
               │ Randare preview, analiză defecte & optimizare│
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               OUTPUT: Model 3D PBR Calibrat (.GLB, .OBJ, .PLY, JSON)
```

---

## 📐 Formulări Matematice

### 1. Calibrarea Camerei și Distorsiunea Radială (`SIMPLE_RADIAL`)

Pentru fiecare punct 3D din spațiul lumii $\mathbf{X}_w \in \mathbb{R}^3$, transformarea în coordonatele camerei $\mathbf{X}_c = [X_c, Y_c, Z_c]^T$ folosind matricea de rotație $\mathbf{R} \in \mathrm{SO}(3)$ și vectorul de translație $\mathbf{t} \in \mathbb{R}^3$:

$$\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}$$

Coordonatele normalizate în planul imaginii $(x_n, y_n)$ și distanța radială $r^2 = x_n^2 + y_n^2$:

$$x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}, \quad r^2 = x_n^2 + y_n^2$$

Coordonatele finale în pixeli $(u, v)$ cu distanța focală $f$, punctul principal $(c_x, c_y)$ și coeficientul de distorsiune radială $k_1$:

$$u = f \cdot x_n (1 + k_1 r^2) + c_x$$

$$v = f \cdot y_n (1 + k_1 r^2) + c_y$$

---

### 2. Convergența Axei Optice (Intersecția Razelor prin Cele Mai Mici Pătrate)

Pentru a estima centrul de focalizare 3D $\mathbf{p}^*$ al celor $N$ raze de cameră calibrate cu centrele $\mathbf{C}_i$ și vectorii direcționali unitari normalizați $\mathbf{v}_i$:

$$\mathbf{p}^* = \operatorname{arg\,min}_{\mathbf{p}} \sum_{i=1}^N \| (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T)(\mathbf{p} - \mathbf{C}_i) \|^2$$

Rezolvând sistemul liniar simetric $\mathbf{A} \mathbf{p}^* = \mathbf{b}$:

$$\left( \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) \right) \mathbf{p}^* = \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) \mathbf{C}_i$$

---

### 3. Estimarea Planului Mesei via RANSAC

Ecuația planului mesei $\mathbf{n}_{\text{plan}} \cdot \mathbf{x} + d = 0$ este extrasă din norul de puncte sparse $\mathcal{P} \subset \mathbb{R}^3$:

$$\mathbf{n}_{\text{plan}} = \frac{(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)}{\|(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)\|}$$

Axul longitudinal vertical al obiectului $\mathbf{u}_{\text{up}}$:

$$\mathbf{u}_{\text{up}} = -\mathbf{n}_{\text{plan}} = [0.1055, -0.5153, -0.8505]^T$$

---

### 4. Blending Unghiular fără Cusături

$$w_k(\theta) = \left[ \max\left(0, \cos(\gamma \cdot (\theta - \theta_k))\right) \right]^p$$

$$\mathbf{C}_{\text{blended}}(\theta, h) = \frac{\sum_{k=1}^K w_k(\theta) \cdot \mathbf{C}_k(\mathbf{p})}{\sum_{k=1}^K w_k(\theta)}$$

---

### 5. Generarea Hărții de Normale Sobel

Vectorii normali $\mathbf{N} = [N_x, N_y, N_z]^T$ din gradienții de luminanță ai imaginii $\nabla I$:

$$N_x = -\frac{\partial I}{\partial x} \cdot \sigma, \quad N_y = -\frac{\partial I}{\partial y} \cdot \sigma, \quad N_z = 1.0$$

$$\mathbf{N}_{\text{PBR}} = \left[ \frac{1}{2}\left(\frac{\mathbf{N}}{\|\mathbf{N}\|} + 1\right) \right] \times 255$$

---

## 🔬 Observații Tehnice, Incertitudini de Senzor & Constrângeri de Interior

> [!NOTE]
> **Condiții de Captură & Constrângeri Fizice:**  
> Setul de date a fost capturat cu un telefon (**Motorola Edge 40 Neo**) în interiorul unei clădiri rezidențiale:

1. **Atenuarea GPS & Multipath în Interior:**  
   Semnalul GPS a fost atenuat de pereții din beton ($\pm 5\text{--}15\text{ m}$ eroare geodezică absolută), în timp ce traiectoria relativă a camerelor obținută prin SfM are o precizie milimetrică ($\pm 0.8\text{ mm}$).
2. **Disparitatea Punctelor SIFT:**  
   $92.4\%$ din punctele cheie ($20,976$) au fost pe masa din lemn și fundal, iar doar $7.6\%$ ($1,726$) pe corpul negru mat al boxei ($\nabla I \approx 0$). De aceea metodele clasice Poisson au eșuat și a fost necesară reconstrucția geometrică hibridă.
3. **Autofocus Dinamic („Lens Breathing”) & Rolling Shutter:**  
   Focalizarea automată pe telefon creează mici variații focale ($f \in [1410, 1465]\text{ px}$), calibrate optim la $f = 1436.1\text{ px}$.
4. **Decuplare HDR & Reflexii Speculare:**  
   19 cadre HDR au fost omogenizate prin separarea texturii difuze de harta de relief Sobel.
5. **Calibrarea Scării Metrice:**  
   Scara monoculară invariantă a fost fixată raportat la planul mesei via RANSAC.

---

## 📊 Metrici de Reconstrucție

| Metrică | Valoare Măsurată |
| :--- | :--- |
| **Număr Cadre** | 59 fotografii (Motorola Edge 40 Neo) |
| **Camere Aliniate** | 59 / 59 (100%) |
| **Puncte 3D SfM** | 22,702 puncte |
| **Vârfuri Mesh** | 5,954 |
| **Triunghiuri** | 11,720 |
| **Atlas Textură** | 2048 × 1024 PBR |
| **Solid Watertight** | Da (`is_watertight = True`) |
| **Timp Reconstrucție** | ~14.2 secunde |

---

## 🚀 Rulare Rapidă

```bash
git clone https://github.com/lefterpatrickandrei-sketch/OmniScan3D.git
cd OmniScan3D
pip install pycolmap open3d trimesh opencv-python Pillow rembg scipy onnxruntime
python start_omniscan.py
```

---

## 📜 Licență
Distribuit sub licența **MIT**.
