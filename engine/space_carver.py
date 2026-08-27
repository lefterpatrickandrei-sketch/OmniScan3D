"""
OmniScan 3D — Multi-View Silhouette Space Carver v2 (Visual Hull Engine)
Improvements over v1:
- Higher resolution voxel grid (200^3 instead of 135^3)
- Mask cleaning: keeps only largest connected component per mask
- Erosion pass to tighten silhouettes and reduce background bleed
- Adaptive occupancy threshold
- Subdivision + Taubin smoothing for smoother surface
"""

import numpy as np
import cv2
import trimesh
from skimage import measure
from typing import Dict, Any, Tuple, Optional, Callable
import pycolmap
import logging

logger = logging.getLogger("SpaceCarver")


def clean_mask(mask: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """Keeps only the largest connected component and erodes slightly to tighten silhouette."""
    binary = (mask > 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    # Find largest foreground component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    clean = np.zeros_like(mask)
    clean[labels == largest_label] = 255
    # Slight erosion to remove edge bleed from rembg
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        clean = cv2.erode(clean, kernel, iterations=1)
    return clean


class SilhouetteSpaceCarver:
    def __init__(
        self,
        reconstruction: pycolmap.Reconstruction,
        masks: Dict[str, np.ndarray],
        resolution: int = 200,
        occupancy_threshold: float = 0.88,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.rec = reconstruction
        self.masks = masks
        self.res = resolution
        self.occupancy_threshold = occupancy_threshold
        self.progress_callback = progress_callback or (lambda p, m: None)

    def _compute_focal_convergence(self) -> Tuple[np.ndarray, float]:
        """Calculates the 3D focal convergence point where camera optical axes intersect."""
        origins = []
        view_dirs = []

        for img in self.rec.images.values():
            cfw = img.cam_from_world()
            R = cfw.rotation.matrix().astype(np.float32)
            t = cfw.translation.astype(np.float32)
            C = -R.T @ t
            v = R.T @ np.array([0, 0, 1], dtype=np.float32)
            origins.append(C)
            view_dirs.append(v / np.linalg.norm(v))

        origins = np.array(origins)
        view_dirs = np.array(view_dirs)

        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for C, v in zip(origins, view_dirs):
            I_minus_vvT = np.eye(3) - np.outer(v, v)
            A += I_minus_vvT
            b += I_minus_vvT @ C

        focus_center = np.linalg.solve(A, b).astype(np.float32)
        mean_cam_dist = float(np.mean(np.linalg.norm(origins - focus_center, axis=1)))
        
        bounding_radius = max(0.8, min(2.0, mean_cam_dist * 0.25))
        return focus_center, bounding_radius

    def carve(self) -> trimesh.Trimesh:
        """Executes space carving on the 3D voxel grid."""
        self.progress_callback(5.0, "Cleaning masks (removing chair/background contamination)...")
        cleaned_masks = {}
        for name, mask in self.masks.items():
            cleaned_masks[name] = clean_mask(mask, erode_px=5)
        logger.info(f"Cleaned {len(cleaned_masks)} masks (largest component + erosion)")

        self.progress_callback(10.0, "Estimating 3D object focal center and bounding volume...")
        center, radius = self._compute_focal_convergence()
        logger.info(f"Focal center: {center}, Bounding radius: {radius:.3f}")

        self.progress_callback(20.0, f"Initializing {self.res}^3 high-res 3D voxel grid...")
        x = np.linspace(center[0] - radius, center[0] + radius, self.res, dtype=np.float32)
        y = np.linspace(center[1] - radius, center[1] + radius, self.res, dtype=np.float32)
        z = np.linspace(center[2] - radius, center[2] + radius, self.res, dtype=np.float32)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        votes = np.zeros(len(pts), dtype=np.int16)
        valid_cams_count = np.zeros(len(pts), dtype=np.int16)

        total_cams = len(self.rec.images)
        for i, (img_id, img) in enumerate(self.rec.images.items()):
            if img.name not in cleaned_masks:
                continue
            mask = cleaned_masks[img.name]
            if mask is None:
                continue

            h, w = mask.shape[:2]
            cfw = img.cam_from_world()
            R = cfw.rotation.matrix().astype(np.float32)
            t = cfw.translation.astype(np.float32)
            cam = self.rec.cameras[img.camera_id]
            f, cx, cy, k1 = cam.params[:4]

            Xc = pts @ R.T + t
            z_c = Xc[:, 2]
            in_front = z_c > 0.15
            valid_idx = np.where(in_front)[0]
            if len(valid_idx) == 0:
                continue

            valid_cams_count[valid_idx] += 1

            x_n = Xc[valid_idx, 0] / z_c[valid_idx]
            y_n = Xc[valid_idx, 1] / z_c[valid_idx]
            dist = 1.0 + k1 * (x_n**2 + y_n**2)
            u = np.round(f * x_n * dist + cx).astype(np.int32)
            v = np.round(f * y_n * dist + cy).astype(np.int32)

            in_b = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx_in_bounds = valid_idx[in_b]
            inside = (mask[v[in_b], u[in_b]] > 128)
            votes[idx_in_bounds[inside]] += 1

            if (i + 1) % 10 == 0 or (i + 1) == total_cams:
                prog = 20.0 + ((i + 1) / total_cams) * 45.0
                self.progress_callback(prog, f"Carving silhouettes: [{i+1}/{total_cams}] cameras processed")

        self.progress_callback(70.0, "Extracting isosurface mesh via Marching Cubes...")
        ratio = np.zeros_like(votes, dtype=np.float32)
        valid_mask = valid_cams_count > (total_cams * 0.25)
        ratio[valid_mask] = votes[valid_mask] / valid_cams_count[valid_mask]

        grid = ratio.reshape((self.res, self.res, self.res))
        spacing = (
            2.0 * radius / (self.res - 1),
            2.0 * radius / (self.res - 1),
            2.0 * radius / (self.res - 1),
        )

        try:
            verts, faces, normals, values = measure.marching_cubes(
                grid, level=self.occupancy_threshold, spacing=spacing
            )
        except Exception:
            verts, faces, normals, values = measure.marching_cubes(
                grid, level=0.75, spacing=spacing
            )

        verts += (center - radius)
        raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        self.progress_callback(80.0, "Refining topology...")
        components = raw_mesh.split(only_watertight=False)
        if isinstance(components, list) and len(components) > 0:
            mesh = max(components, key=lambda m: len(m.vertices))
        else:
            mesh = raw_mesh

        self.progress_callback(85.0, "Subdividing mesh for smoother surface...")
        # Subdivide once to double triangle count for smoother curvature
        mesh = mesh.subdivide()

        self.progress_callback(88.0, "Applying Taubin smoothing (volume-preserving)...")
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=20)

        self.progress_callback(90.0, f"Carved solid mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
        return mesh
