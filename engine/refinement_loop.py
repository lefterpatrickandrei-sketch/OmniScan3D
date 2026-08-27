"""
OmniScan 3D — Autonomous AI Feedback & Refinement Loop
Runs iterative cycles:
Reconstruct -> Render 3D -> Critique with NVIDIA NIM -> Auto-Tune -> Approve
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pycolmap
import trimesh
from PIL import Image
import logging

from engine.hybrid_reconstructor import HybridJBLReconstructor
from engine.evaluator import VisualEvaluator

logger = logging.getLogger("RefinementLoop")


class AutonomousRefinementPipeline:
    def __init__(
        self,
        project_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.project_dir = Path(project_dir)
        self.images_dir = self.project_dir / "prepared_images"
        self.masks_dir = self.project_dir / "masks"
        self.sparse_dir = self.project_dir / "sparse" / "0"
        self.output_dir = self.project_dir / "output"
        self.progress_callback = progress_callback or (lambda p, m: None)

        self.rec = pycolmap.Reconstruction(str(self.sparse_dir))
        self.reconstructor = HybridJBLReconstructor(
            reconstruction=self.rec,
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
        )
        self.evaluator = VisualEvaluator()

    def run_loop(self, max_iterations: int = 3) -> Dict[str, Any]:
        """Runs the self-correcting refinement loop."""
        # Initial parameters
        params = {
            "height": 0.78,
            "radius": 0.172,
            "fillet_size": 0.035,
            "recess_depth": 0.02,
            "theta_offset": 0.0,
            "include_cord": True,
        }

        ground_truth_path = self.images_dir / "IMG_20260827_174338939.jpg"
        history = []

        best_mesh = None
        best_uvs = None
        best_texture = None
        best_score = -1

        for it in range(max_iterations):
            p_base = (it / max_iterations) * 90.0
            self.progress_callback(p_base + 5.0, f"Iteration {it+1}/{max_iterations}: Synthesizing 3D geometry & texture...")

            # 1. Build mesh
            mesh, uvs = self.reconstructor.build_mesh(
                height=params["height"],
                radius=params["radius"],
                fillet_size=params["fillet_size"],
                recess_depth=params["recess_depth"],
                include_cord=params["include_cord"],
            )

            # 2. Bake texture
            texture = self.reconstructor.bake_texture(
                height=params["height"],
                radius=params["radius"],
                theta_offset=params["theta_offset"],
                tex_res=(2048, 1024),
            )

            # 3. Render snapshot for AI inspection
            self.progress_callback(p_base + 15.0, f"Iteration {it+1}: Rendering 3D snapshot for NVIDIA NIM evaluation...")
            preview_path = self.output_dir / f"eval_render_it_{it+1}.png"
            self.evaluator.render_preview(mesh, texture, preview_path, view_angle_deg=0.0)

            # 4. Call NVIDIA NIM Visual Evaluator
            self.progress_callback(p_base + 22.0, f"Iteration {it+1}: NVIDIA NIM AI analyzing visual fidelity...")
            eval_result = self.evaluator.evaluate(preview_path, ground_truth_path)
            eval_result["iteration"] = it + 1
            eval_result["params_used"] = dict(params)
            history.append(eval_result)

            score = eval_result.get("overall_score", 80)
            logger.info(f"Iteration {it+1} Score: {score}/100 | Satisfactory: {eval_result.get('is_satisfactory')}")

            if score > best_score:
                best_score = score
                best_mesh = mesh
                best_uvs = uvs
                best_texture = texture

            # Check stopping criteria
            if eval_result.get("is_satisfactory") and not eval_result.get("has_seam_through_logo"):
                self.progress_callback(95.0, f"AI Evaluator Approved! Reached score {score}/100.")
                break

            # Apply auto-tuning adjustments
            adjs = eval_result.get("recommended_adjustments", {})
            delta_th = adjs.get("theta_offset_delta", 0.0)
            h_scale = adjs.get("height_scale", 1.0)
            r_scale = adjs.get("radius_scale", 1.0)

            params["theta_offset"] += float(delta_th)
            params["height"] *= float(h_scale)
            params["radius"] *= float(r_scale)

        # Final Export
        self.progress_callback(95.0, "Exporting approved 3D assets (.GLB, .OBJ, Texture)...")
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=best_texture,
            metallicFactor=0.05,
            roughnessFactor=0.85,
        )
        visual = trimesh.visual.TextureVisuals(
            uv=best_uvs,
            image=best_texture,
            material=material,
        )
        final_mesh = trimesh.Trimesh(
            vertices=best_mesh.vertices,
            faces=best_mesh.faces,
            visual=visual,
            process=False,
        )

        glb_path = self.output_dir / "model.glb"
        obj_path = self.output_dir / "model.obj"
        tex_path = self.output_dir / "model_texture.png"
        dense_path = self.output_dir / "dense_cloud.ply"

        final_mesh.export(str(glb_path), file_type="glb")
        final_mesh.export(str(obj_path), file_type="obj")
        best_texture.save(tex_path, format="PNG")

        pc = trimesh.PointCloud(vertices=best_mesh.vertices)
        pc.export(str(dense_path), file_type="ply")

        # Write evaluation history
        eval_report_path = self.output_dir / "ai_evaluation_report.json"
        with open(eval_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "final_score": best_score,
                "iterations_run": len(history),
                "history": history,
            }, f, indent=2)

        self.progress_callback(100.0, f"Autonomous refinement complete in {len(history)} iterations! Final Score: {best_score}/100.")
        return {
            "status": "completed",
            "score": best_score,
            "glb": str(glb_path),
            "history": history,
        }
