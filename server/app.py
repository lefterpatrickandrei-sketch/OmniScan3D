"""
OmniScan 3D — FastAPI Backend Server
Provides REST API, SSE live progress streaming, and static hosting for the Web 3D App.
"""

import os
import sys
import json
import shutil
import zipfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from engine.pipeline import PhotogrammetryPipeline
from engine.exif_tool import analyze_dataset

app = FastAPI(title="OmniScan 3D API", version="1.0.0")

# Enable CORS for local network and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS_DIR = BASE_DIR / "projects"
FRONTEND_DIR = BASE_DIR / "frontend"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# In-memory progress tracking
job_progress: Dict[str, Dict[str, Any]] = {}


def run_pipeline_task(project_id: str, quality: str = "medium"):
    project_dir = PROJECTS_DIR / project_id
    
    def on_progress(stage: str, percent: float, message: str):
        job_progress[project_id] = {
            "stage": stage,
            "percent": round(percent, 1),
            "message": message,
            "updated_at": datetime.now().isoformat()
        }

    pipeline = PhotogrammetryPipeline(project_dir, progress_callback=on_progress)
    result = pipeline.run(quality=quality)
    job_progress[project_id]["result"] = result


@app.get("/api/system")
def get_system_status():
    """Returns local GPU, CPU, and memory stats."""
    import platform
    import psutil

    info = {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.3)"
    }
    return info


@app.post("/api/upload")
async def upload_files(
    files: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    project_name: Optional[str] = Form(None)
):
    """Uploads photos or a ZIP archive to create a new reconstruction project."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = (project_name or "Scan").strip().replace(" ", "_")
    project_id = f"{clean_name}_{timestamp}"
    project_dir = PROJECTS_DIR / project_id
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    if zip_file and zip_file.filename:
        temp_zip = project_dir / "uploaded.zip"
        with open(temp_zip, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        with zipfile.ZipFile(temp_zip) as z:
            for item in z.infolist():
                if item.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not item.filename.startswith("__MACOSX"):
                    target = images_dir / Path(item.filename).name
                    with z.open(item) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    saved_count += 1
        
        if temp_zip.exists():
            temp_zip.unlink()

    if files:
        for file in files:
            if file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                target = images_dir / Path(file.filename).name
                with open(target, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                saved_count += 1

    if saved_count == 0:
        if project_dir.exists():
            shutil.rmtree(project_dir)
        raise HTTPException(status_code=400, detail="No valid images (.jpg, .jpeg, .png) were uploaded.")

    # Initialize progress status
    job_progress[project_id] = {
        "stage": "ready",
        "percent": 0.0,
        "message": f"Ready to process {saved_count} images.",
        "image_count": saved_count,
        "created_at": datetime.now().isoformat()
    }

    return {
        "project_id": project_id,
        "image_count": saved_count,
        "message": f"Successfully created project with {saved_count} images."
    }


@app.post("/api/process/{project_id}")
def start_reconstruction(project_id: str, quality: str = "medium", background_tasks: BackgroundTasks = None):
    """Triggers background 3D reconstruction."""
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found.")

    job_progress[project_id] = {
        "stage": "queued",
        "percent": 0.0,
        "message": "Reconstruction job queued...",
        "started_at": datetime.now().isoformat()
    }

    background_tasks.add_task(run_pipeline_task, project_id, quality)
    return {"project_id": project_id, "status": "started"}


@app.get("/api/progress/{project_id}")
def get_progress(project_id: str):
    """Returns real-time progress for a project."""
    if project_id in job_progress:
        return job_progress[project_id]
    
    # Check if project was already completed before server restart
    project_dir = PROJECTS_DIR / project_id
    summary_path = project_dir / "output" / "reconstruction_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "stage": "completed",
            "percent": 100.0,
            "message": "Reconstruction completed.",
            "result": data
        }

    return {"stage": "unknown", "percent": 0.0, "message": "No active job found."}


@app.get("/api/projects")
def list_projects():
    """Lists all scanned projects and their models."""
    projects = []
    for p in sorted(PROJECTS_DIR.iterdir(), key=os.path.getmtime, reverse=True):
        if p.is_dir():
            images_dir = p / "images"
            output_dir = p / "output"
            image_count = len(list(images_dir.glob("*.*"))) if images_dir.exists() else 0
            
            glb_exists = (output_dir / "model.glb").exists()
            summary_path = output_dir / "reconstruction_summary.json"
            geo_path = output_dir / "geo_manifest.json"

            summary = None
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    pass

            geo = None
            if geo_path.exists():
                try:
                    with open(geo_path, "r", encoding="utf-8") as f:
                        geo = json.load(f).get("geo_summary")
                except Exception:
                    pass

            projects.append({
                "project_id": p.name,
                "created_at": datetime.fromtimestamp(p.stat().st_ctime).isoformat(),
                "image_count": image_count,
                "has_model": glb_exists,
                "summary": summary,
                "geo": geo
            })
    return projects


@app.get("/api/models/{project_id}/{filename}")
def get_model_file(project_id: str, filename: str):
    """Serves model assets (.glb, .obj, .ply, .json)."""
    file_path = PROJECTS_DIR / project_id / "output" / filename
    if not file_path.exists():
        # Check images if requesting sample thumbnail
        img_path = PROJECTS_DIR / project_id / "images" / filename
        if img_path.exists():
            return FileResponse(img_path)
        raise HTTPException(status_code=404, detail="File not found.")
    
    media_types = {
        ".glb": "model/gltf-binary",
        ".obj": "text/plain",
        ".ply": "application/octet-stream",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg"
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type, filename=filename)


# Mount static frontend files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
