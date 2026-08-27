"""
OmniScan 3D — Multi-View Texture Baker v2
Improvements over v1:
- Weighted multi-view blending (cos^2 weighting) instead of winner-takes-all
- Bilinear sampling instead of nearest-neighbor pixel lookup
- Higher resolution 4096x2048 texture atlas
- Bilateral filter for noise reduction while preserving logo edges
- Better inpainting with larger radius for untextured areas
- Adaptive radius sampling for non-uniform cylinder shape
"""

import numpy as np
import cv2
import trimesh
from PIL import Image
from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import pycolmap
import logging

logger = logging.getLogger("TextureBaker")


class MultiViewTextureBaker:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        reconstruction: pycolmap.Reconstruction,
        images_dict: Dict[str, np.ndarray],
        masks_dict: Dict[str, np.ndarray],
        texture_res: Tuple[int, int] = (4096, 2048),
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.mesh = mesh
        self.rec = reconstruction
        self.images = images_dict
        self.masks = masks_dict
        self.tex_w, self.tex_h = texture_res
        self.progress_callback = progress_callback or (lambda p, m: None)

    def _setup_cylindrical_uvs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Computes PCA coordinate frame and cylindrical UV unwrapping."""
        v = self.mesh.vertices.astype(np.float32)
        c = np.mean(v, axis=0)

        cov = np.cov(v.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        uz = eigenvecs[:, -1]
        if uz[2] < 0:
            uz = -uz
            
        ux = eigenvecs[:, 0]
        uy = np.cross(uz, ux)
        uy = uy / np.linalg.norm(uy)
        ux = np.cross(uy, uz)
        ux = ux / np.linalg.norm(ux)

        z_proj = (v - c) @ uz
        x_proj = (v - c) @ ux
        y_proj = (v - c) @ uy

        theta = np.arctan2(y_proj, x_proj)
        z_min, z_max = float(z_proj.min()), float(z_proj.max())

        u_coord = (theta + np.pi) / (2.0 * np.pi)
        v_coord = (z_proj - z_min) / max(1e-5, (z_max - z_min))
        uvs = np.stack([u_coord, v_coord], axis=1).astype(np.float32)

        return uvs, c, uz, ux, uy, z_min, z_max

    def _compute_adaptive_radius(self, c, uz, ux, uy, z_min, z_max, n_slices=64):
        """Computes radius as a function of z, capturing the tapered/rounded caps."""
        v = self.mesh.vertices.astype(np.float32)
        z_proj = (v - c) @ uz
        x_proj = (v - c) @ ux
        y_proj = (v - c) @ uy
        r_all = np.sqrt(x_proj**2 + y_proj**2)

        z_edges = np.linspace(z_min, z_max, n_slices + 1)
        radii = np.zeros(n_slices)
        z_centers = np.zeros(n_slices)
        for i in range(n_slices):
            in_slice = (z_proj >= z_edges[i]) & (z_proj < z_edges[i+1])
            if np.any(in_slice):
                radii[i] = np.percentile(r_all[in_slice], 90)
            else:
                radii[i] = np.mean(r_all)
            z_centers[i] = (z_edges[i] + z_edges[i+1]) / 2.0
        return z_centers, radii

    def _bilinear_sample(self, img, u_coords, v_coords):
        """Bilinear interpolation sampling from image at subpixel coordinates."""
        h, w = img.shape[:2]
        u0 = np.floor(u_coords).astype(np.int32)
        v0 = np.floor(v_coords).astype(np.int32)
        u1 = u0 + 1
        v1 = v0 + 1
        
        u0 = np.clip(u0, 0, w - 1)
        u1 = np.clip(u1, 0, w - 1)
        v0 = np.clip(v0, 0, h - 1)
        v1 = np.clip(v1, 0, h - 1)
        
        du = u_coords - u0.astype(np.float32)
        dv = v_coords - v0.astype(np.float32)
        
        c00 = img[v0, u0].astype(np.float32)
        c01 = img[v0, u1].astype(np.float32)
        c10 = img[v1, u0].astype(np.float32)
        c11 = img[v1, u1].astype(np.float32)
        
        du3 = du[:, None] if len(c00.shape) > 1 else du
        dv3 = dv[:, None] if len(c00.shape) > 1 else dv
        
        result = (c00 * (1 - du3) * (1 - dv3) + 
                  c01 * du3 * (1 - dv3) + 
                  c10 * (1 - du3) * dv3 + 
                  c11 * du3 * dv3)
        return result

    def bake(self, output_dir: Path) -> Dict[str, Path]:
        """Bakes textures with multi-view weighted blending and exports assets."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_callback(5.0, "Unwrapping cylindrical UV coordinates...")
        uvs, c, uz, ux, uy, z_min, z_max = self._setup_cylindrical_uvs()

        self.progress_callback(10.0, "Computing adaptive radius profile for tapered shape...")
        z_centers, radii = self._compute_adaptive_radius(c, uz, ux, uy, z_min, z_max)

        self.progress_callback(15.0, f"Generating {self.tex_w}x{self.tex_h} high-res texture grid...")
        u_lin = np.linspace(0, 1, self.tex_w, endpoint=False, dtype=np.float32)
        v_lin = np.linspace(0, 1, self.tex_h, endpoint=False, dtype=np.float32)
        U_grid, V_grid = np.meshgrid(u_lin, v_lin)

        theta_grid = U_grid * (2.0 * np.pi) - np.pi
        z_grid = V_grid * (z_max - z_min) + z_min

        # Interpolate radius for each z value
        r_grid = np.interp(z_grid.ravel(), z_centers, radii).reshape(z_grid.shape).astype(np.float32)

        # 3D sampling positions with adaptive radius
        pts_3d = (
            c[None, None, :]
            + z_grid[:, :, None] * uz[None, None, :]
            + (r_grid * np.cos(theta_grid))[:, :, None] * ux[None, None, :]
            + (r_grid * np.sin(theta_grid))[:, :, None] * uy[None, None, :]
        ).astype(np.float32)

        normals_3d = (
            np.cos(theta_grid)[:, :, None] * ux[None, None, :]
            + np.sin(theta_grid)[:, :, None] * uy[None, None, :]
        ).astype(np.float32)

        pts_flat = pts_3d.reshape(-1, 3)
        n_flat = normals_3d.reshape(-1, 3)

        # Weighted blending accumulators
        color_accum = np.zeros((len(pts_flat), 3), dtype=np.float64)
        weight_accum = np.zeros(len(pts_flat), dtype=np.float64)

        total_cams = len(self.rec.images)
        self.progress_callback(25.0, f"Ray-casting photographic pixels from {total_cams} views (weighted blending)...")

        for idx, (img_id, img) in enumerate(self.rec.images.items()):
            if img.name not in self.images or img.name not in self.masks:
                continue
            rgb = self.images[img.name]
            mask = self.masks[img.name]
            if rgb is None or mask is None:
                continue

            h, w = rgb.shape[:2]
            cfw = img.cam_from_world()
            R = cfw.rotation.matrix().astype(np.float32)
            t = cfw.translation.astype(np.float32)
            C = -R.T @ t
            cam = self.rec.cameras[img.camera_id]
            f, cx, cy, k1 = cam.params[:4]

            to_cam = C[None, :] - pts_flat
            dist_c = np.linalg.norm(to_cam, axis=1, keepdims=True)
            to_cam_dir = to_cam / np.maximum(dist_c, 1e-6)

            cos_a = np.sum(n_flat * to_cam_dir, axis=1)

            Xc = pts_flat @ R.T + t
            z_c = Xc[:, 2]
            valid = (z_c > 0.2) & (cos_a > 0.1)
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                continue

            x_n = Xc[valid_idx, 0] / z_c[valid_idx]
            y_n = Xc[valid_idx, 1] / z_c[valid_idx]
            d = 1.0 + k1 * (x_n**2 + y_n**2)
            u_p = f * x_n * d + cx
            v_p = f * y_n * d + cy

            in_b = (u_p >= 0) & (u_p < w - 1) & (v_p >= 0) & (v_p < h - 1)
            idx_in = valid_idx[in_b]
            u_in = u_p[in_b]
            v_in = v_p[in_b]

            # Check mask at integer positions
            u_int = np.round(u_in).astype(np.int32)
            v_int = np.round(v_in).astype(np.int32)
            in_m = mask[v_int, u_int] > 128
            idx_mask = idx_in[in_m]
            u_m = u_in[in_m]
            v_m = v_in[in_m]

            # Bilinear sample colors
            sampled = self._bilinear_sample(rgb, u_m, v_m)

            # Weight = cos^2(angle) for smooth blending
            w_arr = (cos_a[idx_mask] ** 2).astype(np.float64)

            color_accum[idx_mask] += sampled * w_arr[:, None]
            weight_accum[idx_mask] += w_arr

            if (idx + 1) % 10 == 0 or (idx + 1) == total_cams:
                prog = 25.0 + ((idx + 1) / total_cams) * 50.0
                self.progress_callback(prog, f"Blending views: [{idx+1}/{total_cams}] cameras integrated")

        self.progress_callback(80.0, "Normalizing blended colors...")
        has_color = weight_accum > 1e-6
        tex_colors = np.zeros((len(pts_flat), 3), dtype=np.uint8)
        tex_colors[has_color] = np.clip(
            color_accum[has_color] / weight_accum[has_color, None], 0, 255
        ).astype(np.uint8)

        self.progress_callback(85.0, "Post-processing texture: bilateral filter + inpainting...")
        texture_img = tex_colors.reshape((self.tex_h, self.tex_w, 3))

        # Inpaint untextured regions
        untextured = (weight_accum.reshape((self.tex_h, self.tex_w)) <= 1e-6).astype(np.uint8) * 255
        if np.any(untextured):
            texture_img = cv2.inpaint(texture_img, untextured, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

        # Bilateral filter: smooth noise but preserve edges (logo boundary)
        texture_img = cv2.bilateralFilter(texture_img, d=7, sigmaColor=35, sigmaSpace=7)

        tex_path = output_dir / "model_texture.png"
        pil_tex = Image.fromarray(texture_img)
        pil_tex.save(tex_path, format="PNG")

        self.progress_callback(90.0, "Creating textured 3D mesh with PBR materials...")
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=pil_tex,
            metallicFactor=0.05,
            roughnessFactor=0.85,
        )
        visual = trimesh.visual.TextureVisuals(
            uv=uvs,
            image=pil_tex,
            material=material
        )

        textured_mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices,
            faces=self.mesh.faces,
            visual=visual,
            process=False
        )

        glb_path = output_dir / "model.glb"
        obj_path = output_dir / "model.obj"
        ply_path = output_dir / "dense_cloud.ply"

        self.progress_callback(95.0, "Exporting .GLB, .OBJ, and .PLY assets...")
        textured_mesh.export(str(glb_path), file_type="glb")
        textured_mesh.export(str(obj_path), file_type="obj")

        pc = trimesh.PointCloud(vertices=self.mesh.vertices)
        pc.export(str(ply_path), file_type="ply")

        self.progress_callback(100.0, "Texture baking v2 complete!")
        return {
            "glb": glb_path,
            "obj": obj_path,
            "ply": ply_path,
            "texture": tex_path
        }
