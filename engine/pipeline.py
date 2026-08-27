"""
OmniScan 3D — Core Multi-View Reconstruction Pipeline
Combines:
1. EXIF Metadata & GPS Extraction
2. Orientation Normalization & AI Neural Object Segmentation
3. Robust 360° Camera Pose Calibration (SfM)
4. Multi-View Silhouette Space Carving (Visual Hull) for textureless/black objects
5. Multi-View Raycast Texture Baking with High-Resolution UV Atlas
6. Web-ready .GLB, .OBJ, .PLY, and Spatial GeoJSON Export
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import cv2
import numpy as np
import pycolmap
import open3d as o3d
import trimesh
from PIL import Image, ImageOps
import rembg

# Ensure base directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from engine.exif_tool import analyze_dataset
from engine.space_carver import SilhouetteSpaceCarver
from engine.texture_baker import MultiViewTextureBaker

logging.basicConfig(level=logging.INFO, format="[OmniScan %(levelname)s] %(message)s")
logger = logging.getLogger("OmniScanPipeline")


class PhotogrammetryPipeline:
    def __init__(self, project_dir: Path, progress_callback: Optional[Callable[[str, float, str], None]] = None):
        self.project_dir = Path(project_dir)
        self.images_dir = self.project_dir / "images"
        self.prepared_dir = self.project_dir / "prepared_images"
        self.masks_dir = self.project_dir / "masks"
        self.sparse_dir = self.project_dir / "sparse"
        self.output_dir = self.project_dir / "output"
        self.progress_callback = progress_callback or self._default_progress

        for d in [self.images_dir, self.prepared_dir, self.masks_dir, self.sparse_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _default_progress(self, stage: str, percent: float, message: str):
        logger.info(f"[{stage.upper()}] {percent:.1f}% — {message}")

    def update_progress(self, stage: str, percent: float, message: str):
        self.progress_callback(stage, percent, message)

    def run(self, quality: str = "high", use_ai_masking: bool = True) -> Dict[str, Any]:
        """Runs the complete 3D reconstruction pipeline."""
        start_time = time.time()
        results = {
            "project_id": self.project_dir.name,
            "status": "in_progress",
            "stages": {},
            "metrics": {},
            "outputs": {}
        }

        try:
            # ----------------------------------------------------
            # Stage 1: EXIF & GPS Analysis
            # ----------------------------------------------------
            self.update_progress("exif", 5.0, "Extracting camera parameters and GPS geotags...")
            exif_report = analyze_dataset(self.images_dir)
            results["exif_report"] = exif_report
            
            geo_manifest_path = self.output_dir / "geo_manifest.json"
            with open(geo_manifest_path, "w", encoding="utf-8") as f:
                json.dump(exif_report, f, indent=2)

            image_count = exif_report["total_images"]
            if image_count < 3:
                raise ValueError(f"Need at least 3 images for 3D reconstruction, found {image_count}.")

            self.update_progress("exif", 10.0, f"Analyzed {image_count} images. GPS Geotags: {exif_report['gps_tagged_count']}")

            # ----------------------------------------------------
            # Stage 2: Normalization & AI Object Segmentation
            # ----------------------------------------------------
            self.update_progress("segmentation", 15.0, "AI Neural Object Segmentation (isolating object from background)...")
            session = rembg.new_session("u2net")
            
            img_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            
            images_dict = {}
            masks_dict = {}

            for idx, img_p in enumerate(img_paths):
                p_current = 15.0 + (idx / len(img_paths)) * 25.0
                self.update_progress("segmentation", p_current, f"AI segmenting [{idx+1}/{len(img_paths)}]: {img_p.name}")
                
                target_img = self.prepared_dir / img_p.name
                mask_target = self.masks_dir / f"{img_p.name}.png"

                if not target_img.exists():
                    with Image.open(img_p) as im:
                        im_norm = ImageOps.exif_transpose(im)
                        w, h = im_norm.size
                        scale = 2048.0 / max(w, h)
                        if scale < 1.0:
                            im_norm = im_norm.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                        im_norm.save(target_img, quality=95)

                if not mask_target.exists() and use_ai_masking:
                    with Image.open(target_img) as im_norm:
                        mask = rembg.remove(im_norm, session=session, only_mask=True)
                        m_np = np.array(mask)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        m_dil = cv2.dilate(m_np, kernel, iterations=1)
                        Image.fromarray(m_dil).save(mask_target)

                # Load into memory cache for fast processing
                img_bgr = cv2.imread(str(target_img))
                if img_bgr is not None:
                    images_dict[img_p.name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                if mask_target.exists():
                    m_gray = cv2.imread(str(mask_target), cv2.IMREAD_GRAYSCALE)
                    masks_dict[img_p.name] = m_gray

            self.update_progress("segmentation", 42.0, "AI Object Masking complete.")

            # ----------------------------------------------------
            # Stage 3: Feature Extraction & Matching (SfM)
            # ----------------------------------------------------
            self.update_progress("features", 45.0, "Extracting full-scene features for camera calibration...")
            db_path = self.project_dir / "database.db"

            if not (self.sparse_dir / "0" / "cameras.bin").exists():
                ext_opt = pycolmap.FeatureExtractionOptions()
                ext_opt.max_image_size = 2048
                ext_opt.num_threads = 4

                pycolmap.extract_features(db_path, self.prepared_dir, extraction_options=ext_opt)
                self.update_progress("features", 55.0, "Feature extraction complete.")

                self.update_progress("matching", 58.0, "Matching viewpoints and estimating camera baselines...")
                pycolmap.match_exhaustive(db_path)
                self.update_progress("matching", 65.0, "Matching complete.")

                self.update_progress("sfm", 68.0, "Estimating 3D camera trajectory & ray triangulation...")
                maps = pycolmap.incremental_mapping(db_path, self.prepared_dir, self.sparse_dir)
                if not maps:
                    raise RuntimeError("SfM failed to reconstruct 3D trajectory. Check image overlap.")
                best_model_idx = max(maps.keys(), key=lambda k: len(maps[k].points3D))
                rec = maps[best_model_idx]
            else:
                self.update_progress("sfm", 65.0, "Loading calibrated camera trajectory from cache...")
                rec = pycolmap.Reconstruction(str(self.sparse_dir / "0"))

            num_registered = len(rec.images)
            self.update_progress("sfm", 70.0, f"Calibrated {num_registered}/{image_count} cameras.")

            # ----------------------------------------------------
            # Stage 4: Multi-View Silhouette Space Carving (Visual Hull)
            # ----------------------------------------------------
            self.update_progress("carving", 72.0, "Executing Multi-View Silhouette Space Carving for solid geometry...")
            carver = SilhouetteSpaceCarver(
                reconstruction=rec,
                masks=masks_dict,
                resolution=200,
                occupancy_threshold=0.88,
                progress_callback=lambda p, m: self.update_progress("carving", 72.0 + p * 0.15, m)
            )
            mesh = carver.carve()

            # ----------------------------------------------------
            # Stage 5: Multi-View Texture Baking (UV Atlas & Logos)
            # ----------------------------------------------------
            self.update_progress("texturing", 88.0, "Baking high-resolution UV texture atlas from 59 camera views...")
            baker = MultiViewTextureBaker(
                mesh=mesh,
                reconstruction=rec,
                images_dict=images_dict,
                masks_dict=masks_dict,
                texture_res=(4096, 2048),
                progress_callback=lambda p, m: self.update_progress("texturing", 88.0 + p * 0.10, m)
            )
            outputs = baker.bake(self.output_dir)

            # ----------------------------------------------------
            # Stage 6: Finalizing & Report
            # ----------------------------------------------------
            elapsed = round(time.time() - start_time, 2)
            results["status"] = "completed"
            results["elapsed_seconds"] = elapsed
            results["metrics"] = {
                "total_images": image_count,
                "registered_images": num_registered,
                "reconstructed_points": len(mesh.vertices),
                "mesh_vertices": len(mesh.vertices),
                "mesh_triangles": len(mesh.faces)
            }
            results["outputs"] = {
                "glb": str(outputs["glb"].name),
                "obj": str(outputs["obj"].name),
                "ply_dense": str(outputs["ply"].name),
                "texture": str(outputs["texture"].name),
                "manifest": str(geo_manifest_path.name)
            }

            summary_json_path = self.output_dir / "reconstruction_summary.json"
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            self.update_progress("completed", 100.0, f"3D Object Reconstruction completed in {elapsed}s!")
            return results

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}", exc_info=True)
            self.update_progress("failed", 100.0, f"Error: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results


if __name__ == "__main__":
    target_proj = BASE_DIR / "projects" / "Scan_Test2"
    pipeline = PhotogrammetryPipeline(target_proj)
    res = pipeline.run(quality="high", use_ai_masking=True)
    print("Pipeline result:", json.dumps(res, indent=2))
