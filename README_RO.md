# OmniScan 3D — Fotogrammetrie, Sinteză Geometrică & Reconstrucție PBR

[🇬🇧 English Version](README.md) | [🇷🇴 Versiunea în Română](README_RO.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA 13+](https://img.shields.io/badge/CUDA-13.3-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyCOLMAP 4.1+](https://img.shields.io/badge/PyCOLMAP-4.1.1-blue?style=flat)](https://github.com/colmap/pycolmap)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=flat&logo=three.js)](https://threejs.org/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Evaluator%20VLM-green)](https://developer.nvidia.com/nim)

**OmniScan 3D** este un motor complet de fotogrammetrie, viziune computațională și generare de asset-uri 3D PBR, conceput special pentru obiecte fizice dificile (suprafețe non-lambertiene, fără textură contrastantă sau geometrii cilindrice negre). Sistemul integrează calibrare bundle adjustment, detectare de plan RANSAC, sinteză de textură PBR pe macro-diagrame fără cusături și o buclă de auto-evaluare vizuală bazată pe **NVIDIA NIM Vision AI**.

---

## 🌐 Vizualizator 3D Interactiv în Browser

Modelul 3D și traiectoria celor 59 de camere pot fi explorate direct în browser:
* **Vizualizator Web Interactiv:** Deschide [`docs/index.html`](docs/index.html) sau accesează **[GitHub Pages Live](https://lefterpatrickandrei-sketch.github.io/OmniScan3D/)**.
* **Dashboard Web Local:** Rulează `python start_omniscan.py` și navighează la `http://localhost:8000`.

Interacțiuni disponibile: Rotație 360°, Turntable Auto-Rotate, Afișare/Ascundere 3D a Direcțiilor Camerelor (59 de piramide), Inspecție Fotografii Sursă & Date GPS, Suport AR (Realitate Augmentată pe mobil) și descărcare directă `.GLB`/`.OBJ`.

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

## 📐 Formulări Matematice & Algoritmi

### 1. Modelul de Proiecție al Camerei și Distorsiunea Radială (`SIMPLE_RADIAL`)

Pentru fiecare punct 3D din spațiul lumii $\mathbf{X}_w \in \mathbb{R}^3$, transformarea în coordonatele camerei $\mathbf{X}_c = [X_c, Y_c, Z_c]^T$ folosind matricea de rotație $\mathbf{R} \in SO(3)$ și vectorul de translație $\mathbf{t} \in \mathbb{R}^3$ este:

$$\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}$$

Coordonatele normalizate în planul imaginii $(x_n, y_n)$ și raza la pătrat $r^2 = x_n^2 + y_n^2$:

$$x_n = rac{X_c}{Z_c}, \quad y_n = rac{Y_c}{Z_c}$$

Coordonatele finale în pixeli $(u, v)$ cu distanța focală $f$, punctul principal $(c_x, c_y)$ și coeficientul de distorsiune radială $k_1$:

$$u = f \cdot x_n \left(1 + k_1 (x_n^2 + y_n^2)ight) + c_x$$

$$v = f \cdot y_n \left(1 + k_1 (x_n^2 + y_n^2)ight) + c_y$$

---

### 2. Convergența Axei Optice (Intersecția Razelor 3D prin Cele Mai Mici Pătrate)

Pentru a estima centrul de focalizare 3D $\mathbf{p}^*$ al celor $N$ raze de cameră calibrate cu centrele $\mathbf{C}_i$ și vectorii direcționali normalizați $\mathbf{v}_i$:

$$\mathbf{p}^* = rg\min_{\mathbf{p}} \sum_{i=1}^N \left\| (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T)(\mathbf{p} - \mathbf{C}_i) ight\|^2$$

Rezolvând sistemul liniar $\mathbf{A} \mathbf{p}^* = \mathbf{b}$:

$$\left( \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) ight) \mathbf{p}^* = \sum_{i=1}^N (\mathbf{I} - \mathbf{v}_i \mathbf{v}_i^T) \mathbf{C}_i$$

---

### 3. Estimarea Planului Mesei și a Vectorului Gravitațional prin RANSAC

Ecuația planului mesei $\mathbf{n}_{	ext{plan}} \cdot \mathbf{x} + d = 0$ este extrasă robust prin RANSAC din norul de puncte sparse $\mathcal{P} \subset \mathbb{R}^3$:

$$\mathbf{n}_{	ext{plan}} = rac{(\mathbf{p}_2 - \mathbf{p}_1) 	imes (\mathbf{p}_3 - \mathbf{p}_1)}{\|(\mathbf{p}_2 - \mathbf{p}_1) 	imes (\mathbf{p}_3 - \mathbf{p}_1)\|}$$

Axul longitudinal vertical al obiectului $\mathbf{u}_{	ext{up}}$ este determinat direct de vectorul normal opus mesei:

$$\mathbf{u}_{	ext{up}} = -\mathbf{n}_{	ext{plan}} = [0.1055, -0.5153, -0.8505]^T$$

---

### 4. Blending Unghiular Macro-Chart și Eliminarea Cusăturilor

Pentru a evita artefactele de tăiere verticală dintre camere, punctele de pe suprafață $\mathbf{p}(	heta, h)$ sunt mapate în macro-diagrame azimutale cu funcții continue de ponderare cosinus:

$$w_k(	heta) = \left[ \max\left(0, \cos\left(\gamma \cdot (	heta - 	heta_k)ight)ight) ight]^p$$

$$\mathbf{C}_{	ext{blended}}(	heta, h) = rac{\sum_{k=1}^K w_k(	heta) \cdot \mathbf{C}_k(\mathbf{p})}{\sum_{k=1}^K w_k(	heta)}$$

* În zona frontală a siglei $	heta \in [-35^\circ, +35^\circ]$, ponderea frontală este fixată $w_{	ext{front}} = 1.0$, garantând **claritate nativă maximă și 0% cusături pe emblema JBL**.

---

### 5. Generarea Hărților de Normale PBR din Derivate Sobel

Pentru a oferi relief tactil la iluminare virtuală fără a încărca geometria cu milioane de poligoane, vectorii normali $\mathbf{N} = [N_x, N_y, N_z]^T$ sunt generați din gradienții de luminanță ai imaginii $
abla I$:

$$N_x = -rac{\partial I}{\partial x} \cdot \sigma, \quad N_y = -rac{\partial I}{\partial y} \cdot \sigma, \quad N_z = 1.0$$

$$\mathbf{N}_{	ext{PBR}} = \left[ rac{1}{2}\left(rac{\mathbf{N}}{\|\mathbf{N}\|} + 1ight) ight] 	imes 255$$

---

## 🔬 Observații Tehnice, Incertitudini de Senzor & Note Fotogrammetrice de Interior

> [!NOTE]
> **Condiții Experimentale & Constrângeri Fizice:**  
> Setul de date a fost capturat folosind un smartphone comercial (**Motorola Edge 40 Neo**) în interiorul unei încăperi rezidențiale. Următoarele fenomene fizice și optice au fost identificate și gestionate pe parcursul reconstrucției:

### 1. Atenuarea Semnalului GPS & Efectul de Multipath în Clădire
* **Atenuare GNSS în Interior:** Semnalele de la sateliții GPS/GNSS sunt atenuate de planșeele și pereții din beton armat, suferind reflexii multiple (multipath).
* **Comparație de Precizie:** În timp ce **pozițiile relative ale camerelor** $\Delta \mathbf{C}_{ij}$ calculate prin triangulație epipolară au o precizie milimetrică ($\sigma_{\text{SfM}} pprox \pm 0.8\text{ mm}$), coordonatele geodezice absolute WGS84 (Lat $44.54852^\circ$, Lon $26.06934^\circ$, Alt $132.0\text{ m}$) au o incertitudine tipică de diluție de $\pm 5\text{--}15\text{ metri}$.

### 2. Disparitatea Punctelor Cheie SIFT (Obiect Negru vs. Fundal)
* **Distribuția Punctelor 3D Triangulate:**
  * **Puncte pe Fundal (Nervurile mesei din lemn, faianță, spătare scaun):** $20,976\text{ puncte } (92.4\%)$
  * **Puncte pe Corpul Boxei (Pânză textilă neagră mată):** $1,726\text{ puncte } (7.6\%)$
* **Impact Tehnic:** Pânza neagră mată absoarbe lumina și are gradienți locali aproape nuli $\nabla I \approx 0$. Acest lucru explică matematic de ce algoritmii clasici de tip MVS/Poisson au eșuat în generarea unei suprafețe coerente, confirmând necesitatea abordării geometrice hibride calibrate pe siluete.

### 3. Variația Distanței Focale („Lens Breathing”) & Rolling Shutter
* **Autofocus Dinamic:** Ajustarea automată a focalizării pe telefon produce mici variații ale distanței focale efective între prim-planuri și cadre generale ($f \in [1410, 1465]\text{ px}$). Modelul `SIMPLE_RADIAL` a rezolvat media optimă ($f = 1436.1\text{ px}$).
* **Obturator Electronic (Rolling Shutter):** Mișcarea mâinii în timpul capturii introduce mici deformări la nivel de linie de scanare, filtrate eficient prin algoritmul RANSAC.

### 4. Variația de Expunere HDR & Decuplarea Reflexiilor Speculare
* **Salturi de Luminozitate:** $19$ din cele $59$ de cadre au fost capturate automat în mod HDR (`_HDR.jpg`), creând diferențe de luminanță între unghiuri adiacente.
* **Soluția PBR:** Prin separarea modelului în textură albedo difuză curată și hartă de normale Sobel ($\mathbf{N}_{\text{PBR}}$), reflexiile parazite ale becurilor din cameră au fost complet eliminate din materialul 3D.

### 5. Ambiguitatea Scării Monoculare (Gauge Scale Factor)
* Fotogrammetria bazată pe o singură cameră este invariantă la o scalare arbitrară $\mathbf{X} \to \lambda \mathbf{X}$. Scara metrică reală ($H = 0.178\text{ m}$, $R = 0.034\text{ m}$) a fost ancorată calibrând distanța față de planul mesei determinat via RANSAC.

---

## 📊 Metrici de Performanță & Reconstrucție

| Metrică | Valoare Măsurată |
| :--- | :--- |
| **Număr Total Cadre** | 59 fotografii (Motorola Edge 40 Neo, 4096×2304) |
| **Camere Înregistrate** | 59 / 59 (Aliniere 100%) |
| **Puncte 3D Sparse (SfM)** | 22,702 puncte |
| **Vârfuri Geometrice (Mesh)** | 5,954 |
| **Triunghiuri (Poligoane)** | 11,720 |
| **Rezoluție Atlas Textură** | 2048 × 1024 (Albedo PBR + Hărți de Normale) |
| **Corp Solid (Watertight)** | Da (`is_watertight = True`) |
| **Formate de Export** | `.GLB` (Web 3D/AR), `.OBJ` (CAD), `.PLY` (Nor de puncte) |
| **Timp Reconstrucție GPU** | ~14.2 secunde |

---

## 🚀 Ghid de Instalare și Rulare Rapidă

```bash
# 1. Clonează repozitoriul
git clone https://github.com/lefterpatrickandrei-sketch/OmniScan3D.git
cd OmniScan3D

# 2. Instalează dependențele Python
pip install pycolmap open3d trimesh opencv-python Pillow rembg scipy onnxruntime

# 3. Pornește panoul de control local
python start_omniscan.py
```
Accesează **`http://localhost:8000`** în browser.

---

## 📜 Licență
Distribuit sub licența **MIT**. Consultați fișierul `LICENSE` pentru detalii suplimentare.
