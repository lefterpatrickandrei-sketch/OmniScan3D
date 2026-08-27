"""
OmniScan 3D — EXIF & GPS Georeferencing Extractor
Extracts optical parameters, sensor models, and GPS metadata from image sets.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from PIL import Image, ExifTags


def _convert_to_degrees(value) -> float:
    """Helper to convert GPS rational tuples to decimal degrees."""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (TypeError, IndexError, ZeroDivisionError, ValueError):
        return 0.0


def extract_image_metadata(image_path: Path) -> Dict[str, Any]:
    """Extracts camera, optical, and GPS metadata from a single image."""
    meta = {
        "filename": image_path.name,
        "path": str(image_path),
        "width": 0,
        "height": 0,
        "make": "Unknown",
        "model": "Unknown",
        "datetime": None,
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "iso": None,
        "exposure_time": None,
        "f_number": None,
        "gps": None
    }

    try:
        with Image.open(image_path) as img:
            meta["width"], meta["height"] = img.size
            exif = img.getexif()
            if not exif:
                return meta

            exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            meta["make"] = str(exif_dict.get("Make", "Unknown")).strip()
            meta["model"] = str(exif_dict.get("Model", "Unknown")).strip()
            meta["datetime"] = str(exif_dict.get("DateTime", ""))
            
            # Focal length
            fl = exif_dict.get("FocalLength")
            if fl is not None:
                try:
                    meta["focal_length_mm"] = float(fl)
                except (ValueError, TypeError):
                    pass

            fl35 = exif_dict.get("FocalLengthIn35mmFilm")
            if fl35 is not None:
                try:
                    meta["focal_length_35mm"] = float(fl35)
                except (ValueError, TypeError):
                    pass

            # Exposure & ISO
            iso = exif_dict.get("ISOSpeedRatings")
            if iso is not None:
                meta["iso"] = int(iso) if isinstance(iso, (int, float)) else str(iso)
            
            fnum = exif_dict.get("FNumber")
            if fnum is not None:
                try:
                    meta["f_number"] = float(fnum)
                except (ValueError, TypeError):
                    pass

            exp = exif_dict.get("ExposureTime")
            if exp is not None:
                try:
                    meta["exposure_time"] = float(exp)
                except (ValueError, TypeError):
                    pass

            # GPS extraction
            gps_ifd = exif.get_ifd(0x8825)
            if gps_ifd:
                gps_data = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat_raw = gps_data.get("GPSLatitude")
                lat_ref = gps_data.get("GPSLatitudeRef", "N")
                lon_raw = gps_data.get("GPSLongitude")
                lon_ref = gps_data.get("GPSLongitudeRef", "E")
                alt_raw = gps_data.get("GPSAltitude")

                if lat_raw and lon_raw:
                    lat = _convert_to_degrees(lat_raw)
                    if lat_ref != "N":
                        lat = -lat
                    
                    lon = _convert_to_degrees(lon_raw)
                    if lon_ref != "E":
                        lon = -lon

                    alt = 0.0
                    if alt_raw is not None:
                        try:
                            alt = float(alt_raw)
                        except (ValueError, TypeError):
                            pass

                    meta["gps"] = {
                        "latitude": round(lat, 7),
                        "longitude": round(lon, 7),
                        "altitude": round(alt, 2),
                        "lat_ref": str(lat_ref),
                        "lon_ref": str(lon_ref),
                        "datestamp": str(gps_data.get("GPSDateStamp", "")),
                        "timestamp": str(gps_data.get("GPSTimeStamp", ""))
                    }

    except Exception as e:
        meta["error"] = str(e)

    return meta


def analyze_dataset(images_dir: Path) -> Dict[str, Any]:
    """Analyzes all images in a directory and generates comprehensive dataset report."""
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    records = []
    lats, lons, alts = [], [], []
    cameras = set()

    for p in image_files:
        info = extract_image_metadata(p)
        records.append(info)
        if (info["make"] != "Unknown" or info["model"] != "Unknown"):
            cameras.add(f"{info['make']} {info['model']}".strip())
        
        gps = info.get("gps")
        if gps and gps.get("latitude") and gps.get("longitude"):
            lats.append(gps["latitude"])
            lons.append(gps["longitude"])
            if gps.get("altitude"):
                alts.append(gps["altitude"])

    report = {
        "total_images": len(image_files),
        "valid_images": len(records),
        "cameras": list(cameras),
        "has_gps": len(lats) > 0,
        "gps_tagged_count": len(lats),
        "images": records
    }

    if lats and lons:
        report["geo_summary"] = {
            "center": {
                "latitude": round(sum(lats) / len(lats), 7),
                "longitude": round(sum(lons) / len(lons), 7),
                "altitude": round(sum(alts) / len(alts), 2) if alts else 0.0
            },
            "bounds": {
                "min_latitude": min(lats),
                "max_latitude": max(lats),
                "min_longitude": min(lons),
                "max_longitude": max(lons),
                "min_altitude": min(alts) if alts else 0.0,
                "max_altitude": max(alts) if alts else 0.0
            }
        }

    return report


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("c:/Users/lefpa/Downloads/OmniScan3D/projects/test_benchmark/images")
    if target.exists():
        res = analyze_dataset(target)
        print(f"Processed {res['total_images']} images. Cameras: {res['cameras']}. GPS tagged: {res['gps_tagged_count']}")
        if "geo_summary" in res:
            print("Geo center:", json.dumps(res["geo_summary"]["center"], indent=2))
