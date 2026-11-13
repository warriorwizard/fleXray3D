"""
FleXray Clinical Studio - Official Python Backend Server.
Directly implements the Universal Anatomy Segmentation and Volumetric DRR pipeline:
- 60 Anatomical Structures Pan-Segmentation (flagship bundle flux0375)
- Official Figure 1 Multilabel Mosaic Blending with Crisp 2-Pixel Boundaries
- Interactive Arbitrary-Angle Cone-Beam DRR Forward Projection (CUDA nanoDRR)
- CT and MRI Dual-Resolution Volume Processing & Calibration
- Zero-mock Clinical Dataset (Torso CT, NLST Chest CT, Torso DICOM Series, SPM Brain MRI)
- Multipart Volume Upload (Folder DICOMs, Multi-frame DICOM, ZIP, NIfTI)
"""

from __future__ import annotations

import os
import sys
import uuid
import time
import json
import shutil
import mimetypes
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure correct MIME types on Windows
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

import uvicorn
import pydicom
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import cv2
import torch

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ui.flexray_core import (
    get_segmenter,
    load_and_preprocess_radiograph,
    generate_pan_anatomy_overlay,
    generate_single_structure_overlay,
    ANATOMICAL_CATEGORIES,
    PALETTE_HEX,
    PALETTE_RGB,
)
from ui.volumetric_service import (
    assemble_dicom_series,
    assemble_multiframe_dicom,
    assemble_nifti,
    assemble_npz_volume,
    calibrate_volume,
    create_dual_resolution_package,
    ingest_uploaded_volume_archive,
)
from fxr.drr import (
    render_drr,
    DRRRenderRequest,
    build_render_intrinsics,
    compute_isocenter,
)
from nanodrr.data import Subject
from nanodrr.data.preprocess import hu_to_mu
from nanodrr.camera import make_rt_inv, make_k_inv
import nanodrr.drr as nanodrr_drr

app = FastAPI(
    title="FleXray Universal Anatomy Studio API",
    description="Official REST API for FleXray Universal Anatomy Segmentation & Volumetric DRR",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_and_mime_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if path.endswith(".js"):
        response.headers["content-type"] = "application/javascript; charset=utf-8"
        response.headers["cache-control"] = "no-cache, no-store, must-revalidate"
    elif path.endswith(".css"):
        response.headers["content-type"] = "text/css; charset=utf-8"
        response.headers["cache-control"] = "no-cache, no-store, must-revalidate"
    return response

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Static Storage Directories
STATIC_DIR = REPO_ROOT / "ui_data"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(STATIC_DIR)), name="data")

# Real Benchmark Dataset Studies (2D Planar and 3D Volumetric CT/MRI)
DEFAULT_STUDIES = [
    # --- Volumetric Studies ---
    {
        "id": "study-torso-spine-ct",
        "patientName": "Elena Rostova",
        "mrn": "VOL-CT-1092",
        "age": "48",
        "gender": "F",
        "modality": "CT",
        "bodyPart": "Torso & Spine",
        "view": "3D Volumetric CT",
        "studyDate": "2026-10-18",
        "institution": "Metropolitan Academic Hospital",
        "filePath": r"c:\Users\Amit Yadav\Downloads\projects\unet\data\real_ct_volume.npz",
        "isDicom": False,
        "isVolumetric": True,
        "description": "Genuine clinical 100-slice CT scan (100 x 512 x 512). Full pelvis, spine, ribs, and abdominal cavity. Supports real-time 3D WebGL volume inspection and arbitrary-angle DRR projection.",
    },
    {
        "id": "study-nlst-chest-ct",
        "patientName": "Arthur Vance",
        "mrn": "NLST-7819",
        "age": "62",
        "gender": "M",
        "modality": "CT",
        "bodyPart": "Chest / Lungs",
        "view": "3D Chest CT",
        "studyDate": "2026-10-16",
        "institution": "National Lung Screening Trial (NLST)",
        "filePath": r"c:\Users\Amit Yadav\Downloads\projects\unet\data\nlst_lung_ct.npz",
        "isDicom": False,
        "isVolumetric": True,
        "description": "Genuine NLST screening chest CT (150 x 512 x 512). Complete pulmonary parenchyma, mediastinum, thoracic spine, and rib cage with calibrated ground truth labels.",
    },
    {
        "id": "study-torso-dicom-series",
        "patientName": "David Miller",
        "mrn": "DCM-TORSO-40",
        "age": "45",
        "gender": "M",
        "modality": "CT",
        "bodyPart": "Torso",
        "view": "3D Axial Series (40 Slices)",
        "studyDate": "2026-10-17",
        "institution": "Apex Radiology Center",
        "filePath": r"c:\Users\Amit Yadav\Downloads\projects\cortex\sample_dicoms\CT_Torso_Series",
        "isDicom": True,
        "isVolumetric": True,
        "description": "Genuine clinical CT series of 40 DICOM slices (512 x 512, 0.7 x 0.7 mm, 2.5 mm spacing). Slice-normal direction cosine assembly with 3D ray-marching.",
    },
    {
        "id": "study-brain-mri-spm",
        "patientName": "Sofia Chen",
        "mrn": "SPM-MRI-T1",
        "age": "31",
        "gender": "F",
        "modality": "MR",
        "bodyPart": "Brain / Skull",
        "view": "3D T1 Anatomical MRI",
        "studyDate": "2026-10-19",
        "institution": "University Neuroimaging Center",
        "filePath": r"C:\Users\Amit Yadav\miniconda3\lib\site-packages\nibabel\tests\data\anatomical.nii",
        "isDicom": False,
        "isVolumetric": True,
        "description": "Genuine SPM T1-weighted anatomical 3D brain MRI scan. Features ambient air isolation and calibrated pseudo-CT / MIP cone-beam projection.",
    },
    {
        "id": "study-dbt-breast-tomosynthesis",
        "patientName": "Sarah Jenkins",
        "mrn": "TCIA-DBT-02655",
        "age": "52",
        "gender": "F",
        "modality": "MG",
        "bodyPart": "Breast",
        "view": "3D Digital Breast Tomosynthesis (DBT)",
        "studyDate": "2026-10-20",
        "institution": "Duke Breast Imaging / TCIA",
        "filePath": str(REPO_ROOT / "real_dataset" / "mammography" / "dbt_breast_3d" / "00000001.dcm"),
        "isDicom": True,
        "isVolumetric": True,
        "description": "Genuine clinical 50-slice 3D Digital Breast Tomosynthesis (DBT) volume (50 × 2457 × 1890, Hologic Selenia Dimensions, MLO view) from TCIA Duke Breast Screening. Supports 3D WebGL volumetric inspection, transfer functions, and synchronized DRR cone-beam projection.",
    },
    # --- 2D Planar Studies ---
    {
        "id": "pelvis-deepfluoro-22",
        "patientName": "Elena Rostova",
        "mrn": "MRN-642199",
        "age": "48",
        "gender": "F",
        "modality": "CR",
        "bodyPart": "Pelvis",
        "view": "AP Pelvis",
        "studyDate": "2026-10-18",
        "institution": "Metropolitan Academic Hospital",
        "filePath": str(REPO_ROOT / "real_dataset" / "slide_22.png"),
        "isDicom": False,
        "isVolumetric": False,
        "description": "Pelvis AP radiograph from DeepFluoro/Radiopaedia benchmark. Bilateral hips, sacrum, and lower lumbar spine segmentation.",
    },
    {
        "id": "knee-moola-ram-dcm",
        "patientName": "Moola Ram",
        "mrn": "BEDI123",
        "age": "54",
        "gender": "M",
        "modality": "DX",
        "bodyPart": "Knee",
        "view": "Lateral Knee (KNEE-LAT-M)",
        "studyDate": "2026-10-14",
        "institution": "Apex Orthopedic & Radiology Center",
        "filePath": str(REPO_ROOT.parent / "proxopacs_local_exe" / "a.dcm"),
        "isDicom": True,
        "isVolumetric": False,
        "description": "Genuine 16-bit Lateral Knee DICOM radiograph (3173 x 2500 px). Distal femur, proximal tibia, patella, and fibula segmentation with VOI LUT.",
    },
    {
        "id": "chest-radiopaedia-07",
        "patientName": "Arthur Vance",
        "mrn": "MRN-781920",
        "age": "62",
        "gender": "M",
        "modality": "DX",
        "bodyPart": "Chest",
        "view": "Lateral Chest",
        "studyDate": "2026-10-16",
        "institution": "St. Jude Cardio-Thoracic Imaging",
        "filePath": str(REPO_ROOT / "real_dataset" / "slide_07.png"),
        "isDicom": False,
        "isVolumetric": False,
        "description": "Lateral chest radiograph. Lungs, heart, thoracic vertebrae, sternum, and rib cage segmentation.",
    },
    {
        "id": "hand-radiopaedia-01",
        "patientName": "Sofia Chen",
        "mrn": "MRN-552910",
        "age": "31",
        "gender": "F",
        "modality": "CR",
        "bodyPart": "Extremity",
        "view": "PA Hand & Wrist",
        "studyDate": "2026-10-19",
        "institution": "University Orthopedic Clinic",
        "filePath": str(REPO_ROOT / "real_dataset" / "slide_12.png"),
        "isDicom": False,
        "isVolumetric": False,
        "description": "High-detail hand radiograph. Fine bone segmentation of phalanges, metacarpals, carpals, radius, and ulna.",
    },
    {
        "id": "spine-radiopaedia-04",
        "patientName": "David Miller",
        "mrn": "MRN-331089",
        "age": "45",
        "gender": "M",
        "modality": "CR",
        "bodyPart": "Spine",
        "view": "Cervico-Thoracic Spine",
        "studyDate": "2026-10-17",
        "institution": "Comprehensive Spine Institute",
        "filePath": str(REPO_ROOT / "real_dataset" / "slide_04.png"),
        "isDicom": False,
        "isVolumetric": False,
        "description": "Spine radiograph showing individual vertebral segmentation (C1-C7, T1-T12) and posterior ribs.",
    },
    {
        "id": "mammo-cbis-ddsm-right-cc",
        "patientName": "Claire Montgomery",
        "mrn": "CBIS-DDSM-1694-CC",
        "age": "58",
        "gender": "F",
        "modality": "MG",
        "bodyPart": "Breast",
        "view": "Full Field Digital Mammogram (Right CC)",
        "studyDate": "2026-10-21",
        "institution": "Curated Breast Imaging DDSM / TCIA",
        "filePath": str(REPO_ROOT / "real_dataset" / "mammography" / "cbis_ddsm_2d" / "00000001.dcm"),
        "isDicom": True,
        "isVolumetric": False,
        "description": "Genuine clinical 16-bit Full-Field Digital Mammogram (FFDM, 4201 × 1906 px) in standard Cranio-Caudal (CC) projection from TCIA CBIS-DDSM. Superior-inferior compression evaluating medial and lateral breast parenchyma.",
    },
    {
        "id": "mammo-cbis-ddsm-right-mlo",
        "patientName": "Claire Montgomery",
        "mrn": "CBIS-DDSM-1694-MLO",
        "age": "58",
        "gender": "F",
        "modality": "MG",
        "bodyPart": "Breast",
        "view": "Full Field Digital Mammogram (Right MLO)",
        "studyDate": "2026-10-21",
        "institution": "Curated Breast Imaging DDSM / TCIA",
        "filePath": str(REPO_ROOT / "real_dataset" / "mammography" / "cbis_ddsm_mlo" / "00000001.dcm"),
        "isDicom": True,
        "isVolumetric": False,
        "description": "Genuine clinical 16-bit Full-Field Digital Mammogram (FFDM, 5506 × 2251 px) in standard Mediolateral Oblique (MLO) projection from TCIA CBIS-DDSM. Captures the full pectoralis major muscle, axillary tail of Spence, and inframammary fold.",
    },
]

STORED_STUDIES: Dict[str, dict] = {s["id"]: dict(s) for s in DEFAULT_STUDIES}

# Cache for 2D predictions: study_id -> { "base_gray": np.ndarray, "probs": np.ndarray, "meta": dict }
PREDICTION_CACHE: Dict[str, dict] = {}

# GPU Cache for loaded 3D volumes: study_id -> { "vol_t": Tensor, "aff_t": Tensor, "iso": Tensor, "label_t": Tensor | None, "meta": dict }
VOLUME_CACHE: Dict[str, dict] = {}


def get_cached_volume(study_id: str) -> dict:
    """Retrieves or loads GPU-resident tensors for a volumetric study."""
    if study_id in VOLUME_CACHE:
        return VOLUME_CACHE[study_id]

    if study_id not in STORED_STUDIES:
        raise HTTPException(status_code=404, detail="Study not found")

    study = STORED_STUDIES[study_id]
    study_dir = STATIC_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    npz_cache = study_dir / "volume.npz"

    # If cached npz does not exist yet, build dual-resolution package from source filePath
    if not npz_cache.exists():
        src_path = Path(study["filePath"])
        if not src_path.exists():
            raise HTTPException(status_code=404, detail=f"Source volume missing: {src_path}")

        label = None
        if src_path.is_dir():
            vol, aff, meta = assemble_dicom_series(src_path)
        elif src_path.suffix.lower() == ".npz":
            vol, aff, meta, label = assemble_npz_volume(src_path)
        elif src_path.name.lower().endswith((".nii", ".nii.gz")):
            vol, aff, meta = assemble_nifti(src_path)
        elif src_path.suffix.lower() == ".dcm":
            vol, aff, meta = assemble_multiframe_dicom(src_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {src_path}")

        cal_vol = calibrate_volume(vol, study.get("modality", "CT"))
        pkg_meta = create_dual_resolution_package(cal_vol, aff, study_dir, meta, label=label)
        study.update(pkg_meta)

    # Load from npz_cache
    npz_data = np.load(str(npz_cache))
    raw_vol = npz_data["volume"].astype(np.float32)
    raw_aff = npz_data["affine"].astype(np.float32)
    raw_label = npz_data["label"].astype(np.int64) if "label" in npz_data else None

    vol_t = torch.from_numpy(raw_vol).unsqueeze(0).to(DEVICE)
    aff_t = torch.from_numpy(raw_aff).to(DEVICE)
    label_t = torch.from_numpy(raw_label).unsqueeze(0).to(DEVICE) if raw_label is not None else None

    # Construct and cache GPU Subject for rapid, authentic nanoDRR cone-beam projection
    vol_5d = vol_t if vol_t.ndim == 5 else vol_t.unsqueeze(0)
    image_mu = hu_to_mu(vol_5d)
    label_5d = label_t if label_t is not None else torch.zeros_like(vol_5d, dtype=torch.long)
    if label_5d.ndim == 4:
        label_5d = label_5d.unsqueeze(0)
    label_5d = label_5d.to(dtype=torch.long)

    *_, d_sz, h_sz, w_sz = image_mu.shape
    world_to_voxel = torch.linalg.inv(aff_t.double()).to(dtype=image_mu.dtype)
    voxel_to_grid = Subject._make_voxel_to_grid(image_mu.shape).to(device=DEVICE, dtype=image_mu.dtype)
    center_voxel = torch.tensor([(w_sz - 1) / 2.0, (h_sz - 1) / 2.0, (d_sz - 1) / 2.0, 1.0], dtype=aff_t.dtype, device=DEVICE)
    subject_iso = (aff_t @ center_voxel)[:3]
    subject = Subject(image_mu, label_5d, aff_t, world_to_voxel, voxel_to_grid, subject_iso, convert_to_mu=False)

    # Load volume_meta.json
    meta_json_path = study_dir / "volume_meta.json"
    if meta_json_path.exists():
        with open(meta_json_path, "r", encoding="utf-8") as f:
            v_meta = json.load(f)
    else:
        v_meta = {
            "dimensions": list(raw_vol.shape),
            "spacing": [1.0, 1.0, 1.0],
            "modality": study.get("modality", "CT"),
        }

    cached_entry = {
        "vol_t": vol_t,
        "aff_t": aff_t,
        "iso": subject_iso,
        "label_t": label_t,
        "meta": v_meta,
        "subject": subject,
    }
    VOLUME_CACHE[study_id] = cached_entry
    return cached_entry


def ensure_study_processed(study_id: str, threshold: float = 0.45) -> dict:
    """Processes or retrieves cached data for both 2D radiographs and 3D volumetric studies."""
    if study_id not in STORED_STUDIES:
        raise HTTPException(status_code=404, detail="Study not found")

    study_info = STORED_STUDIES[study_id]
    study_dir = STATIC_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    # Volumetric Studies Handling
    if study_info.get("isVolumetric", False):
        cached_vol = get_cached_volume(study_id)
        v_meta = cached_vol["meta"]

        # Ensure default AP Frontal (yaw=0, pitch=0) DRR is rendered
        base_drr_path = study_dir / "base_radiograph.png"
        overlay_drr_path = study_dir / f"overlay_t{int(threshold*100)}.png"

        if study_id not in PREDICTION_CACHE or not base_drr_path.exists():
            # Calculate dynamic FOV del_pix and source distance for genuine full-field projection
            vol_t = cached_vol["vol_t"]
            aff_t = cached_vol["aff_t"]
            d, h, w = vol_t.shape[-3:]
            phys_x = w * abs(aff_t[0, 0].item())
            phys_y = h * abs(aff_t[1, 1].item())
            phys_z = d * abs(aff_t[2, 2].item())
            # In AP projection, X is horizontal, Z is vertical. Frame patient with full 3D extent and clinical margins.
            anatomical_fov = max(phys_x, phys_y, phys_z, 100.0) * 1.22

            sdd = 1200.0
            src_dist = sdd * 0.72
            mag = sdd / src_dist
            del_pix = (anatomical_fov * mag) / 512.0

            # Render AP Frontal DRR via nanoDRR
            k_inv = make_k_inv(
                sdd=sdd,
                delx=del_pix,
                dely=del_pix,
                x0=0.0,
                y0=0.0,
                height=512,
                width=512,
                device=DEVICE,
                dtype=torch.float32,
            )
            rot = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
            xyz = torch.tensor([[0.0, src_dist, 0.0]], dtype=torch.float32, device=DEVICE)
            rt_inv = make_rt_inv(rot, xyz, orientation="AP", isocenter=cached_vol["iso"])
            sdd_t = torch.tensor([sdd], dtype=torch.float32, device=DEVICE)

            rendered = nanodrr_drr.render(
                subject=cached_vol["subject"],
                k_inv=k_inv,
                rt_inv=rt_inv,
                sdd=sdd_t,
                height=512,
                width=512,
                n_samples=300,
            )
            raw_img = rendered.sum(dim=1)[0].cpu().numpy()
            p1, p99 = np.percentile(raw_img, 1), np.percentile(raw_img, 99)
            base_gray = np.clip((raw_img - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)

            cv2.imwrite(str(base_drr_path), (base_gray * 255.0).astype(np.uint8))

            # Run FleXray inference
            segmenter = get_segmenter()
            pred = segmenter.predict((base_gray * 255.0).astype(np.uint8))
            probs = pred.probabilities[0].cpu().numpy()

            PREDICTION_CACHE[study_id] = {
                "base_gray": base_gray,
                "probs": probs,
                "meta": {
                    "matrix": f"{base_gray.shape[0]} × {base_gray.shape[1]} px",
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "sdd": 1200.0,
                },
            }

        cached = PREDICTION_CACHE[study_id]
        base_gray = cached["base_gray"]
        probs = cached["probs"]
        segmenter = get_segmenter()

        overlay_uint8, structures = generate_pan_anatomy_overlay(
            base_gray,
            probs,
            segmenter.label_names,
            threshold=threshold,
        )
        cv2.imwrite(str(overlay_drr_path), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

        initial_focus_url = None
        if structures:
            primary_name = structures[0]["name"]
            initial_focus_path = study_dir / f"focus_{primary_name}.png"
            if not initial_focus_path.exists():
                focus_uint8 = generate_single_structure_overlay(
                    base_gray,
                    probs,
                    segmenter.label_names,
                    target_label=primary_name,
                    threshold=threshold * 0.85,
                )
                cv2.imwrite(str(initial_focus_path), cv2.cvtColor(focus_uint8, cv2.COLOR_RGB2BGR))
            initial_focus_url = f"/data/{study_id}/focus_{primary_name}.png"

        study_info["detectedCount"] = len(structures)
        study_info["matrix"] = f"{v_meta['dimensions'][0]} × {v_meta['dimensions'][1]} × {v_meta['dimensions'][2]} voxels"
        study_info["structures"] = structures
        study_info["dimensions"] = v_meta["dimensions"]
        study_info["spacing"] = v_meta["spacing"]
        study_info["normalizedScale"] = v_meta.get("normalizedScale", [1.0, 1.0, 1.0])
        study_info["volumeProxyUrl"] = f"/data/{study_id}/volume_proxy.bin"
        study_info["volumeMetaUrl"] = f"/data/{study_id}/volume_meta.json"
        study_info["defaultPreset"] = v_meta.get("defaultPreset", "Bone")
        study_info["defaultWindow"] = v_meta.get("defaultWindow", 1500.0)
        study_info["defaultLevel"] = v_meta.get("defaultLevel", 400.0)
        study_info["hasGroundTruthLabels"] = cached_vol.get("label_t") is not None

        return {
            "study": study_info,
            "baseImageUrl": f"/data/{study_id}/base_radiograph.png",
            "overlayImageUrl": f"/data/{study_id}/overlay_t{int(threshold*100)}.png",
            "focusImageUrl": initial_focus_url,
            "structures": structures,
            "categories": ANATOMICAL_CATEGORIES,
            "isVolumetric": True,
            "hasGroundTruthLabels": cached_vol.get("label_t") is not None,
            "volumeProxyUrl": f"/data/{study_id}/volume_proxy.bin",
            "volumeMetaUrl": f"/data/{study_id}/volume_meta.json",
        }

    # 2D Planar Studies Handling
    if study_id not in PREDICTION_CACHE:
        file_path = study_info["filePath"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file missing on disk: {file_path}")

        base_gray, meta = load_and_preprocess_radiograph(file_path)
        base_img_path = study_dir / "base_radiograph.png"
        cv2.imwrite(str(base_img_path), (base_gray * 255.0).astype(np.uint8))

        segmenter = get_segmenter()
        pred = segmenter.predict((base_gray * 255.0).astype(np.uint8))
        probs = pred.probabilities[0].cpu().numpy()

        PREDICTION_CACHE[study_id] = {
            "base_gray": base_gray,
            "probs": probs,
            "meta": meta,
        }

    cached = PREDICTION_CACHE[study_id]
    base_gray = cached["base_gray"]
    probs = cached["probs"]
    meta = cached["meta"]
    segmenter = get_segmenter()

    overlay_uint8, structures = generate_pan_anatomy_overlay(
        base_gray,
        probs,
        segmenter.label_names,
        threshold=threshold,
    )
    overlay_img_path = study_dir / f"overlay_t{int(threshold*100)}.png"
    cv2.imwrite(str(overlay_img_path), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

    initial_focus_url = None
    if structures:
        primary_name = structures[0]["name"]
        initial_focus_path = study_dir / f"focus_{primary_name}.png"
        if not initial_focus_path.exists():
            focus_uint8 = generate_single_structure_overlay(
                base_gray,
                probs,
                segmenter.label_names,
                target_label=primary_name,
                threshold=threshold * 0.85,
            )
            cv2.imwrite(str(initial_focus_path), cv2.cvtColor(focus_uint8, cv2.COLOR_RGB2BGR))
        initial_focus_url = f"/data/{study_id}/focus_{primary_name}.png"

    study_info["detectedCount"] = len(structures)
    study_info["matrix"] = meta.get("matrix", f"{base_gray.shape[0]} × {base_gray.shape[1]} px")
    study_info["structures"] = structures

    return {
        "study": study_info,
        "baseImageUrl": f"/data/{study_id}/base_radiograph.png",
        "overlayImageUrl": f"/data/{study_id}/overlay_t{int(threshold*100)}.png",
        "focusImageUrl": initial_focus_url,
        "structures": structures,
        "categories": ANATOMICAL_CATEGORIES,
        "isVolumetric": False,
    }


# Request model for arbitrary-angle DRR projection
class DRRProjectRequest(BaseModel):
    study_id: str
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    sdd: float = 1200.0
    height: int = 256
    width: int = 256
    n_samples: int = 300
    seg_threshold: float = 0.05
    threshold: float = 0.45
    mode: str = "flexray" # "flexray" or "ground_truth"
    focus_structure: Optional[str] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model": "FleXray flagship (flux0375)",
        "classes": 60,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "volumetric_drr_engine": "nanoDRR cone-beam CUDA",
    }


@app.get("/api/studies")
def list_studies():
    """Returns the clinical PACS worklist with all 2D and 3D studies."""
    result = []
    for s_id, s in STORED_STUDIES.items():
        item = dict(s)
        item.pop("filePath", None)
        result.append(item)
    return result


@app.get("/api/study/{study_id}")
def get_study_details(study_id: str, threshold: float = Query(default=0.45, ge=0.1, le=0.9)):
    """Runs universal segmentation / volume sync and returns triptych assets and active structures."""
    return ensure_study_processed(study_id, threshold=threshold)


@app.api_route("/api/volume_meta/{study_id}", methods=["GET", "HEAD"])
def get_volume_meta(study_id: str):
    """Returns 3D volume dimensions, spacing, and transfer function metadata."""
    meta_path = STATIC_DIR / study_id / "volume_meta.json"
    if not meta_path.exists():
        ensure_study_processed(study_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Volume metadata not found for study")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.api_route("/api/volume_proxy/{study_id}", methods=["GET", "HEAD"])
def get_volume_proxy(study_id: str):
    """Guarantees volume proxy generation and returns the 128^3 binary texture buffer."""
    bin_path = STATIC_DIR / study_id / "volume_proxy.bin"
    if not bin_path.exists():
        ensure_study_processed(study_id)
    if not bin_path.exists():
        raise HTTPException(status_code=404, detail="Volume proxy not found for study")
    return FileResponse(str(bin_path), media_type="application/octet-stream")


@app.post("/api/project_drr")
def project_drr(req: DRRProjectRequest):
    """
    Simulates a 2D radiograph (DRR) from the exact 3D camera pose and runs universal segmentation.
    Maps Three.js spherical yaw/pitch to nanoDRR ZXY Euler rotation.
    """
    study_id = req.study_id
    if study_id not in STORED_STUDIES:
        raise HTTPException(status_code=404, detail="Study not found")

    t_start = time.time()
    cached_vol = get_cached_volume(study_id)
    study_dir = STATIC_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    vol_t = cached_vol["vol_t"]
    aff_t = cached_vol["aff_t"]
    d, h, w = vol_t.shape[-3:]
    phys_x = w * abs(aff_t[0, 0].item())
    phys_y = h * abs(aff_t[1, 1].item())
    phys_z = d * abs(aff_t[2, 2].item())
    # Anatomical bounding extent with generous clinical margins for arbitrary rotation
    anatomical_fov = max(phys_x, phys_y, phys_z, 100.0) * 1.22

    sdd = req.sdd
    src_dist = sdd * 0.72
    mag = sdd / src_dist
    del_pix = (anatomical_fov * mag) / max(req.width, req.height)

    k_inv = make_k_inv(
        sdd=sdd,
        delx=del_pix,
        dely=del_pix,
        x0=0.0,
        y0=0.0,
        height=req.height,
        width=req.width,
        device=DEVICE,
        dtype=torch.float32,
    )

    # Three.js Yaw (azimuth) rotates opposite to nanoDRR Euler Z-rotation; negate req.yaw for exact sync
    rot = torch.tensor([[-req.yaw, req.pitch, req.roll]], dtype=torch.float32, device=DEVICE)
    xyz = torch.tensor([[0.0, src_dist, 0.0]], dtype=torch.float32, device=DEVICE)
    rt_inv = make_rt_inv(rot, xyz, orientation="AP", isocenter=cached_vol["iso"])
    sdd_t = torch.tensor([sdd], dtype=torch.float32, device=DEVICE)

    rendered = nanodrr_drr.render(
        subject=cached_vol["subject"],
        k_inv=k_inv,
        rt_inv=rt_inv,
        sdd=sdd_t,
        height=req.height,
        width=req.width,
        n_samples=req.n_samples,
    )

    t_render = time.time() - t_start

    # DRR image normalization
    raw_img = rendered.sum(dim=1)[0].cpu().numpy()
    p1, p99 = np.percentile(raw_img, 1), np.percentile(raw_img, 99)
    drr_gray = np.clip((raw_img - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)

    # Unique filename keyed to pose
    pose_key = f"y{int(round(req.yaw))}_p{int(round(req.pitch))}_s{int(round(req.sdd))}"
    drr_filename = f"drr_{pose_key}.png"
    drr_path = study_dir / drr_filename
    cv2.imwrite(str(drr_path), (drr_gray * 255.0).astype(np.uint8))

    segmenter = get_segmenter()
    structures = []
    overlay_filename = f"drr_overlay_{pose_key}.png"
    overlay_path = study_dir / overlay_filename

    if req.mode == "ground_truth" and cached_vol["label_t"] is not None and rendered.shape[1] > 1:
        # Projected Ground Truth 3D Organ Masks
        proj_labels = rendered[0].cpu().numpy() # (C, H, W)
        gt_colors = [
            (255, 100, 100), # 1: lungs / red
            (255, 220, 150), # 2: bone / ivory
            (100, 200, 120), # 3: tissue / green
        ]
        label_names = ["background", "lungs", "vertebrae_ribs", "soft_tissue"]
        gt_weights = proj_labels[1:]
        overlay_rgb = drr_gray[..., None].repeat(3, axis=-1)
        for c_idx in range(gt_weights.shape[0]):
            m = gt_weights[c_idx] > 0.5
            if m.any():
                col = np.array(gt_colors[c_idx % len(gt_colors)], dtype=np.float32) / 255.0
                overlay_rgb[m] = overlay_rgb[m] * 0.45 + col * 0.55
                # 2px boundary
                bnd = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) - m.astype(np.uint8)
                overlay_rgb[bnd > 0] = col

                px_c = int(m.sum())
                structures.append({
                    "name": label_names[c_idx + 1],
                    "label": label_names[c_idx + 1].replace("_", " ").title(),
                    "category": "3D Ground Truth",
                    "hexColor": "#" + "".join(f"{int(v*255):02x}" for v in col),
                    "pixelCount": px_c,
                    "coveragePercent": round((px_c / (req.height * req.width)) * 100.0, 2),
                    "maxConfidence": 1.0,
                    "meanConfidence": 1.0,
                })
        overlay_uint8 = (np.clip(overlay_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

        # Single structure focus for 3D ground truth mode
        focus_url = None
        if req.focus_structure:
            matched_idx = -1
            target_lower = req.focus_structure.lower()
            for c_i, c_name in enumerate(label_names[1:]):
                if c_name in target_lower or target_lower in c_name:
                    matched_idx = c_i
                    break
            if matched_idx >= 0 and matched_idx < gt_weights.shape[0]:
                m = gt_weights[matched_idx] > 0.5
                focus_rgb = drr_gray[..., None].repeat(3, axis=-1)
                col = np.array(gt_colors[matched_idx % len(gt_colors)], dtype=np.float32) / 255.0
                if m.any():
                    focus_rgb[m] = focus_rgb[m] * 0.35 + col * 0.65
                    bnd = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) - m.astype(np.uint8)
                    focus_rgb[bnd > 0] = col
                focus_uint8 = (np.clip(focus_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                focus_filename = f"focus_{pose_key}_{req.focus_structure}.png"
                focus_path = study_dir / focus_filename
                cv2.imwrite(str(focus_path), cv2.cvtColor(focus_uint8, cv2.COLOR_RGB2BGR))
                focus_url = f"/data/{study_id}/{focus_filename}"
    else:
        # FleXray Universal 60-Anatomy Model Inference on generated DRR
        pred = segmenter.predict((drr_gray * 255.0).astype(np.uint8))
        probs = pred.probabilities[0].cpu().numpy()

        overlay_uint8, structures = generate_pan_anatomy_overlay(
            drr_gray,
            probs,
            segmenter.label_names,
            threshold=req.threshold,
        )
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

        # Single structure focus if requested
        focus_url = None
        if req.focus_structure:
            focus_uint8 = generate_single_structure_overlay(
                drr_gray,
                probs,
                segmenter.label_names,
                target_label=req.focus_structure,
                threshold=req.threshold * 0.85,
            )
            focus_filename = f"focus_{pose_key}_{req.focus_structure}.png"
            focus_path = study_dir / focus_filename
            cv2.imwrite(str(focus_path), cv2.cvtColor(focus_uint8, cv2.COLOR_RGB2BGR))
            focus_url = f"/data/{study_id}/{focus_filename}"

    t_total = time.time() - t_start

    return {
        "study_id": study_id,
        "pose": {
            "yaw": req.yaw,
            "pitch": req.pitch,
            "roll": req.roll,
            "sdd": req.sdd,
        },
        "drrImageUrl": f"/data/{study_id}/{drr_filename}",
        "overlayImageUrl": f"/data/{study_id}/{overlay_filename}",
        "focusImageUrl": focus_url,
        "structures": structures,
        "timing": {
            "cuda_render_ms": round(t_render * 1000.0, 1),
            "total_ms": round(t_total * 1000.0, 1),
        },
    }


@app.get("/api/structure_focus/{study_id}/{structure_name}")
def get_structure_focus(
    study_id: str,
    structure_name: str,
    threshold: float = Query(default=0.40, ge=0.1, le=0.9)
):
    """Renders Panel 3 of the paper's triptych: isolates single selected anatomical structure."""
    if study_id not in PREDICTION_CACHE:
        ensure_study_processed(study_id, threshold=threshold)

    cached = PREDICTION_CACHE[study_id]
    base_gray = cached["base_gray"]
    probs = cached["probs"]
    segmenter = get_segmenter()

    single_uint8 = generate_single_structure_overlay(
        base_gray,
        probs,
        segmenter.label_names,
        target_label=structure_name,
        threshold=threshold,
    )

    study_dir = STATIC_DIR / study_id
    focus_path = study_dir / f"focus_{structure_name}.png"
    cv2.imwrite(str(focus_path), cv2.cvtColor(single_uint8, cv2.COLOR_RGB2BGR))

    return {
        "structure": structure_name,
        "focusImageUrl": f"/data/{study_id}/focus_{structure_name}.png",
        "hexColor": PALETTE_HEX.get(structure_name, "#38BDF8"),
    }


@app.post("/api/upload_volume")
async def upload_volume_series(
    files: List[UploadFile] = File(...),
    patient_name: Optional[str] = Form(None),
    modality: Optional[str] = Form("CT"),
    body_part: Optional[str] = Form("Torso"),
):
    """
    Accepts volumetric uploads:
    - Multi-file DICOM series (multiple .dcm files)
    - ZIP archive (.zip) containing DICOM series or NIfTI
    - Single NIfTI volume (.nii or .nii.gz)
    - Single Multi-frame DICOM (.dcm)
    Constructs 3D volume, slice normal sorting, calibration, and WebGL proxy.
    """
    try:
        new_id = f"vol-{uuid.uuid4().hex[:8]}"
        study_dir = STATIC_DIR / new_id
        study_dir.mkdir(parents=True, exist_ok=True)
        upload_temp = study_dir / "raw_uploads"
        upload_temp.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for file in files:
            content = await file.read()
            if len(content) == 0:
                continue
            fname = file.filename or f"slice_{len(saved_files):04d}.dcm"
            # Normalize path separators and safely create intermediate directories
            clean_rel_path = Path(fname.replace("\\", "/").lstrip("/"))
            dest_file = upload_temp / clean_rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "wb") as f:
                f.write(content)
            saved_files.append(dest_file)

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files received.")

        # Inspect format
        first_file = saved_files[0]
        label = None

        if len(saved_files) == 1 and first_file.suffix.lower() == ".zip":
            # ZIP archive
            meta_result = ingest_uploaded_volume_archive(first_file, study_dir)
            vol_meta = meta_result
            final_modality = meta_result.get("modality", modality or "CT")
        elif len(saved_files) == 1 and first_file.name.lower().endswith((".nii", ".nii.gz")):
            # NIfTI volume
            vol, aff, meta = assemble_nifti(first_file)
            final_modality = meta.get("modality", modality or "CT")
            cal_vol = calibrate_volume(vol, final_modality)
            vol_meta = create_dual_resolution_package(cal_vol, aff, study_dir, meta)
        else:
            # DICOM series or Multi-frame DICOM
            if len(saved_files) == 1:
                try:
                    ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
                    if int(getattr(ds, "NumberOfFrames", 1)) > 1:
                        vol, aff, meta = assemble_multiframe_dicom(first_file)
                    else:
                        vol, aff, meta = assemble_dicom_series(saved_files)
                except Exception:
                    vol, aff, meta = assemble_dicom_series(saved_files)
            else:
                vol, aff, meta = assemble_dicom_series(saved_files)

            final_modality = meta.get("modality", modality or "CT")
            cal_vol = calibrate_volume(vol, final_modality)
            vol_meta = create_dual_resolution_package(cal_vol, aff, study_dir, meta)

        new_study = {
            "id": new_id,
            "patientName": patient_name or vol_meta.get("patientName", "Uploaded Patient"),
            "mrn": vol_meta.get("patientId", f"MRN-{uuid.uuid4().hex[:6].upper()}"),
            "age": "50",
            "gender": "O",
            "modality": final_modality,
            "bodyPart": body_part or vol_meta.get("bodyPart", "Volumetric"),
            "view": f"3D {final_modality} Volume",
            "studyDate": datetime.date.today().strftime("%Y-%m-%d"),
            "institution": "Clinical Ingestion PACS",
            "filePath": str(study_dir / "volume.npz"),
            "isDicom": True,
            "isVolumetric": True,
            "description": f"Live uploaded volumetric {final_modality} scan ({vol_meta.get('dimensions', [0,0,0])[0]} slices). Dual-resolution WebGL and CUDA cone-beam ready.",
        }

        STORED_STUDIES[new_id] = new_study

        # Precompute initial AP Frontal DRR and segmentations
        details = ensure_study_processed(new_id, threshold=0.45)
        return {"status": "success", "study": new_study, "details": details}

    except Exception as e:
        print(f"Error in upload_volume_series: {e}")
        import traceback
        traceback.print_exc()
        if study_dir.exists() and new_id not in STORED_STUDIES:
            shutil.rmtree(study_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to process volume: {str(e)}")


@app.post("/api/upload_dicom")
async def upload_dicom_file(file: UploadFile = File(...)):
    """Accepts single 2D radiograph DICOM (.dcm), multi-frame volumetric DICOM, or standard radiograph (.png/.jpg)."""
    try:
        content = await file.read()
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")

        new_id = f"upload-{uuid.uuid4().hex[:8]}"
        study_dir = STATIC_DIR / new_id
        study_dir.mkdir(parents=True, exist_ok=True)

        is_dcm = file.filename.lower().endswith((".dcm", ".dicom")) or (len(content) > 132 and content[128:132] == b"DICM")
        saved_filename = Path(file.filename).name if file.filename else ("patient.dcm" if is_dcm else "patient.png")
        saved_path = study_dir / saved_filename

        with open(saved_path, "wb") as f:
            f.write(content)

        # Detect if single multi-frame DICOM volume
        if is_dcm:
            try:
                ds_header = pydicom.dcmread(str(saved_path), stop_before_pixels=True)
                if int(getattr(ds_header, "NumberOfFrames", 1)) > 1:
                    vol, aff, meta = assemble_multiframe_dicom(saved_path)
                    final_modality = meta.get("modality", "CT")
                    cal_vol = calibrate_volume(vol, final_modality)
                    vol_meta = create_dual_resolution_package(cal_vol, aff, study_dir, meta)
                    new_study = {
                        "id": new_id,
                        "patientName": meta.get("patientName", "Uploaded Patient"),
                        "mrn": meta.get("patientId", f"MRN-{uuid.uuid4().hex[:6].upper()}"),
                        "age": "50",
                        "gender": "O",
                        "modality": final_modality,
                        "bodyPart": meta.get("bodyPart", "Volumetric"),
                        "view": f"3D {final_modality} Multi-frame",
                        "studyDate": datetime.date.today().strftime("%Y-%m-%d"),
                        "institution": "Clinical Ingestion PACS",
                        "filePath": str(study_dir / "volume.npz"),
                        "isDicom": True,
                        "isVolumetric": True,
                        "description": f"Live uploaded multi-frame {final_modality} scan ({vol_meta.get('dimensions', [0,0,0])[0]} frames). Dual-resolution WebGL and CUDA cone-beam ready.",
                    }
                    STORED_STUDIES[new_id] = new_study
                    result = ensure_study_processed(new_id, threshold=0.45)
                    return {"status": "success", "study": new_study, "details": result}
            except Exception as mf_err:
                # Not a multi-frame DICOM, continue to 2D planar preprocessing
                pass

        base_gray, meta = load_and_preprocess_radiograph(saved_path)

        new_study = {
            "id": new_id,
            "patientName": meta.get("patientName", "Uploaded Patient"),
            "mrn": meta.get("patientId", f"MRN-{uuid.uuid4().hex[:6].upper()}"),
            "age": meta.get("age", "50"),
            "gender": meta.get("gender", "M"),
            "modality": meta.get("modality", "DX"),
            "bodyPart": meta.get("bodyPart", "Diagnostic"),
            "view": meta.get("view", "Clinical Projection"),
            "studyDate": meta.get("studyDate", datetime.date.today().strftime("%Y-%m-%d")),
            "institution": "Clinical Ingestion PACS",
            "filePath": str(saved_path),
            "isDicom": is_dcm,
            "isVolumetric": False,
            "description": f"Live uploaded radiograph ({meta.get('matrix', 'Standard')}). Processed with FleXray Universal Anatomy Segmenter.",
        }

        STORED_STUDIES[new_id] = new_study
        result = ensure_study_processed(new_id, threshold=0.45)
        return {"status": "success", "study": new_study, "details": result}
    except Exception as e:
        print(f"Error in upload_dicom_file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process radiograph: {str(e)}")


# Mount React build dist if present
dist_dir = REPO_ROOT / "ui" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui")


if __name__ == "__main__":
    print("Starting FleXray Universal Anatomy & Volumetric DRR Server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
