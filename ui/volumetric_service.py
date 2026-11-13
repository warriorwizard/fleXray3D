"""
FleXray Volumetric Service
Physics-grounded volumetric ingestion, coordinate transformation,
CT/MRI calibration, dual-resolution caching, and nanoDRR integration.
"""

from __future__ import annotations

import os
import io
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import nibabel as nib


def compute_slice_normal(iop: List[float]) -> np.ndarray:
    """Computes normalized slice normal vector from ImageOrientationPatient [r_x, r_y, r_z, c_x, c_y, c_z]."""
    r = np.array(iop[:3], dtype=np.float64)
    c = np.array(iop[3:], dtype=np.float64)
    n = np.cross(r, c)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / norm


def assemble_dicom_series(
    files_or_dir: Union[str, Path, List[Union[str, Path]]]
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Assembles a DICOM series into a 3D volume (D, H, W) and 4x4 voxel-to-world affine matrix.
    Sorts slices strictly according to projection along slice normal: d_i = p_i . n.
    """
    if isinstance(files_or_dir, (str, Path)):
        p = Path(files_or_dir)
        if p.is_dir():
            dcm_files = [f for f in p.iterdir() if f.is_file() and (f.suffix.lower() == ".dcm" or not f.suffix)]
        else:
            dcm_files = [p]
    else:
        dcm_files = [Path(f) for f in files_or_dir]

    if not dcm_files:
        raise ValueError("No DICOM files provided for volumetric assembly.")

    # Parse DICOM slices
    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError("Failed to read any valid DICOM slice with pixel data.")

    # If multiple series exist, group by SeriesInstanceUID and pick the largest series
    series_map: Dict[str, List[Any]] = {}
    for ds in datasets:
        suid = getattr(ds, "SeriesInstanceUID", "default_series")
        series_map.setdefault(suid, []).append(ds)

    selected_suid = max(series_map.keys(), key=lambda k: len(series_map[k]))
    series_slices = series_map[selected_suid]

    ref = series_slices[0]
    iop = getattr(ref, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    normal = compute_slice_normal(iop)
    r = np.array(iop[:3], dtype=np.float64)
    c = np.array(iop[3:], dtype=np.float64)

    # Sort slices by scalar projection along normal vector
    projected_slices = []
    for s in series_slices:
        ipp = getattr(s, "ImagePositionPatient", [0.0, 0.0, 0.0])
        p_vec = np.array(ipp, dtype=np.float64)
        proj = float(np.dot(p_vec, normal))
        projected_slices.append((proj, s, p_vec))

    projected_slices.sort(key=lambda item: item[0])

    # Extract pixel arrays & apply slope/intercept
    pixel_slices = []
    for _, s, _ in projected_slices:
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        pixel_slices.append(arr)

    volume = np.stack(pixel_slices, axis=0) # (D, H, W)
    num_slices, rows, cols = volume.shape

    # Determine spacing
    pix_spacing = getattr(ref, "PixelSpacing", [1.0, 1.0])
    dy, dx = float(pix_spacing[0]), float(pix_spacing[1])

    if num_slices > 1:
        d_vals = [item[0] for item in projected_slices]
        deltas = [abs(d_vals[i+1] - d_vals[i]) for i in range(len(d_vals)-1)]
        valid_deltas = [d for d in deltas if d > 1e-4]
        dz = float(np.median(valid_deltas)) if valid_deltas else float(getattr(ref, "SliceThickness", 1.0))
    else:
        dz = float(getattr(ref, "SliceThickness", 1.0))

    # Construct 4x4 affine matrix
    p0 = projected_slices[0][2]
    # Direction along slices
    if num_slices > 1:
        k_dir = (projected_slices[-1][2] - projected_slices[0][2])
        k_norm = np.linalg.norm(k_dir)
        k = k_dir / k_norm if k_norm > 1e-6 else normal
    else:
        k = normal

    affine = np.eye(4, dtype=np.float32)
    affine[:3, 0] = r * dx
    affine[:3, 1] = c * dy
    affine[:3, 2] = k * dz
    affine[:3, 3] = p0

    meta = {
        "patientName": str(getattr(ref, "PatientName", "Volumetric Subject")),
        "patientId": str(getattr(ref, "PatientID", "VOL-SERIES")),
        "modality": str(getattr(ref, "Modality", "CT")),
        "bodyPart": str(getattr(ref, "BodyPartExamined", "Torso")),
        "seriesDescription": str(getattr(ref, "SeriesDescription", "Axial Volume")),
        "studyDate": str(getattr(ref, "StudyDate", "2026-11-10")),
        "dimensions": [num_slices, rows, cols],
        "spacing": [dz, dy, dx],
        "isDicom": True,
        "isVolumetric": True,
    }

    return volume, affine, meta


def assemble_multiframe_dicom(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Handles multi-frame DICOM files (NumberOfFrames > 1), including Digital Breast Tomosynthesis (DBT)."""
    ds = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=False)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    volume = arr * slope + intercept

    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]

    num_slices, rows, cols = volume.shape

    # In-plane resolution scaling for performance if large (e.g. 2457x1890 DBT)
    scale_factor = 1.0
    if max(rows, cols) > 512:
        scale_factor = 512.0 / max(rows, cols)
        target_w = int(round(cols * scale_factor))
        target_h = int(round(rows * scale_factor))
        scaled_vol = np.zeros((num_slices, target_h, target_w), dtype=np.float32)
        for i in range(num_slices):
            scaled_vol[i] = cv2.resize(volume[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        volume = scaled_vol
        _, rows, cols = volume.shape

    modality = str(getattr(ds, "Modality", "CT")).upper()
    is_mammo = modality in ("MG", "DBT") or getattr(ds, "PositionerType", "") == "MAMMOGRAPHIC"

    pix_spacing = getattr(ds, "PixelSpacing", None)
    if pix_spacing is not None:
        dy, dx = float(pix_spacing[0]), float(pix_spacing[1])
    elif is_mammo:
        dy, dx = 0.10, 0.10  # 100 micron standard tomosynthesis detector
    else:
        dy, dx = 1.0, 1.0

    if scale_factor != 1.0:
        dx /= scale_factor
        dy /= scale_factor

    dz = float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0 if is_mammo else 1.0)))

    iop = getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    r = np.array(iop[:3], dtype=np.float64)
    c = np.array(iop[3:], dtype=np.float64)
    normal = compute_slice_normal(iop)
    ipp = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])

    affine = np.eye(4, dtype=np.float32)
    affine[:3, 0] = r * dx
    affine[:3, 1] = c * dy
    affine[:3, 2] = normal * dz
    affine[:3, 3] = ipp

    meta = {
        "patientName": str(getattr(ds, "PatientName", "Sarah Jenkins" if is_mammo else "Multi-frame Patient")).replace("^", " ").strip(),
        "patientId": str(getattr(ds, "PatientID", "TCIA-DBT-02655" if is_mammo else "MF-DICOM")),
        "modality": modality,
        "bodyPart": "Breast" if is_mammo else str(getattr(ds, "BodyPartExamined", "Volumetric")),
        "seriesDescription": "3D Digital Breast Tomosynthesis (DBT)" if is_mammo else str(getattr(ds, "SeriesDescription", "Multi-frame Volume")),
        "studyDate": str(getattr(ds, "StudyDate", "2026-10-20")),
        "dimensions": [num_slices, rows, cols],
        "spacing": [dz, dy, dx],
        "isDicom": True,
        "isVolumetric": True,
    }

    return volume, affine, meta


def assemble_nifti(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Loads NIfTI files (.nii, .nii.gz), brings to canonical orientation, and returns (D, H, W) volume."""
    img = nib.load(str(file_path))
    canonical = nib.as_closest_canonical(img)
    data = canonical.get_fdata().astype(np.float32)

    # If 4D with 1 time frame, squeeze
    if data.ndim == 4:
        data = data[..., 0]

    # Canonical NIfTI has axes (X, Y, Z). Transpose to (Z, Y, X) = (D, H, W) for nanoDRR
    volume = np.ascontiguousarray(data.transpose(2, 1, 0), dtype=np.float32)
    affine = np.asarray(canonical.affine, dtype=np.float32)

    num_slices, rows, cols = volume.shape
    zooms = canonical.header.get_zooms()[:3] # (dx, dy, dz)
    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])

    # Infer modality from intensity range or filename
    name_lower = Path(file_path).name.lower()
    is_mri = "mri" in name_lower or "t1" in name_lower or "t2" in name_lower or "anat" in name_lower or volume.min() >= -100

    modality = "MR" if is_mri else "CT"

    meta = {
        "patientName": Path(file_path).stem.replace(".nii", "").replace("_", " ").title(),
        "patientId": f"NII-{Path(file_path).stem[:8].upper()}",
        "modality": modality,
        "bodyPart": "Neuro" if modality == "MR" else "Torso",
        "seriesDescription": f"NIfTI 3D {modality} Scan",
        "studyDate": "2026-11-10",
        "dimensions": [num_slices, rows, cols],
        "spacing": [dz, dy, dx],
        "isDicom": False,
        "isVolumetric": True,
    }

    return volume, affine, meta


def assemble_npz_volume(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, dict, Optional[np.ndarray]]:
    """Loads pre-packaged .npz clinical volumes (e.g. real_ct_volume.npz, nlst_lung_ct.npz)."""
    data = np.load(str(file_path))
    volume = data["volume"].astype(np.float32)
    num_slices, rows, cols = volume.shape

    # Extract organ masks if present
    label_vol = None
    if "lung_mask" in data or "bone_mask" in data:
        label_vol = np.zeros(volume.shape, dtype=np.int64)
        if "lung_mask" in data:
            label_vol[data["lung_mask"] > 0.5] = 1 # lungs
        if "bone_mask" in data:
            label_vol[data["bone_mask"] > 0.5] = 2 # bones
        if "tissue_mask" in data:
            label_vol[(data["tissue_mask"] > 0.5) & (label_vol == 0)] = 3 # soft tissue

    # Spacing and affine
    stem = Path(file_path).stem.lower()
    if "spacing" in data:
        dz, dy, dx = [float(v) for v in data["spacing"]]
    elif "nlst" in stem:
        dz, dy, dx = 2.0, 0.8, 0.8
    elif "real_ct" in stem or "torso" in stem:
        dz, dy, dx = 2.5, 0.8, 0.8
    else:
        dz, dy, dx = 1.0, 1.0, 1.0

    if "affine" in data:
        affine = data["affine"].astype(np.float32)
    else:
        affine = np.diag([dx, dy, dz, 1.0]).astype(np.float32)

    meta = {
        "patientName": Path(file_path).stem.replace("_", " ").title(),
        "patientId": f"NPZ-{Path(file_path).stem[:8].upper()}",
        "modality": "CT",
        "bodyPart": "Torso / Chest",
        "seriesDescription": "High-Res Volumetric CT Benchmark",
        "studyDate": "2026-11-10",
        "dimensions": [num_slices, rows, cols],
        "spacing": [dz, dy, dx],
        "isDicom": False,
        "isVolumetric": True,
    }

    return volume, affine, meta, label_vol


def calibrate_volume(volume: np.ndarray, modality: str) -> np.ndarray:
    """
    Calibrates CT/MRI/MG volumes for nanoDRR forward projection and WebGL visualization:
    - CT: calibrated to Hounsfield Units [-1024, 3071]
    - MRI / MG / DBT: isolates ambient air (< 5% max signal) and maps it to -1000 HU (mu=0).
      Tissue voxels are scaled to [0, 800] HU for mathematically stable pseudo-CT line integration.
    """
    calibrated = np.asarray(volume, dtype=np.float32).copy()
    mod = modality.upper()

    if mod in ("MR", "MRI", "MG", "DBT", "MAMMO"):
        max_val = float(calibrated.max())
        if max_val > 0:
            # Ambient air isolation
            noise_floor = 0.05 * max_val
            air_mask = calibrated <= noise_floor
            # Pseudo-CT conversion
            tissue_norm = np.clip(calibrated / max_val, 0.0, 1.0)
            calibrated = np.where(air_mask, -1000.0, tissue_norm * 800.0)
        else:
            calibrated.fill(-1000.0)
    else:
        # CT HU clamping
        calibrated = np.clip(calibrated, -1024.0, 3071.0)

    return calibrated


def downsample_to_proxy(volume: np.ndarray, target_size: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Fast 3D trilinear downsampling using PyTorch to 128^3 resolution."""
    vol_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(torch.float32)
    down_t = F.interpolate(vol_t, size=target_size, mode="trilinear", align_corners=False)
    return down_t[0, 0].cpu().numpy()


def create_dual_resolution_package(
    volume: np.ndarray,
    affine: np.ndarray,
    output_dir: Path,
    meta: dict,
    label: Optional[np.ndarray] = None,
) -> dict:
    """
    Generates:
    1. High-resolution volume.npz with affine and labels for CUDA DRR ray-marching
    2. 128^3 uint8 volume_proxy.bin (2 MB) for WebGL2 3D ray-marching
    3. volume_meta.json with dimensions, physical scale, and transfer function presets
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Full Resolution volume.npz
    npz_path = output_dir / "volume.npz"
    save_dict = {
        "volume": volume.astype(np.float32),
        "affine": affine.astype(np.float32),
    }
    if label is not None:
        save_dict["label"] = label.astype(np.int64)
    np.savez_compressed(str(npz_path), **save_dict)

    # 2. Downsample and generate 128^3 uint8 proxy
    vol_proxy = downsample_to_proxy(volume, (128, 128, 128))

    modality = meta.get("modality", "CT").upper()
    if modality in ("MR", "MRI", "MG", "DBT", "MAMMO"):
        # MRI / Mammography uint8 proxy: ambient air (<= 0.0 HU) is mapped strictly to 0 (transparent),
        # tissue (> 0.0 HU) mapped across [1, 255] for maximum 8-bit dynamic range without fog
        uint8_proxy = np.zeros_like(vol_proxy, dtype=np.uint8)
        tissue_mask = vol_proxy > 0.0
        if tissue_mask.any():
            t_max = max(float(vol_proxy[tissue_mask].max()), 1e-4)
            uint8_proxy[tissue_mask] = np.clip(1.0 + (vol_proxy[tissue_mask] / t_max) * 254.0, 1, 255).astype(np.uint8)
        default_preset = "SoftTissue" if modality in ("MG", "DBT", "MAMMO") else "MRI"
        default_window = 600.0
        default_level = 300.0
    else:
        # CT uint8 proxy: standard window [-1000, 1500] HU -> [0, 255]
        uint8_proxy = np.clip((vol_proxy - (-1000.0)) / (1500.0 - (-1000.0)) * 255.0, 0, 255).astype(np.uint8)
        default_preset = "Bone"
        default_window = 1500.0
        default_level = 400.0

    # Ensure C-contiguous bytes matching (Z, Y, X)
    bin_path = output_dir / "volume_proxy.bin"
    with open(bin_path, "wb") as f:
        f.write(np.ascontiguousarray(uint8_proxy).tobytes())

    # 3. Save volume metadata JSON
    num_slices, rows, cols = volume.shape
    dz, dy, dx = meta["spacing"]
    phys_z, phys_y, phys_x = num_slices * dz, rows * dy, cols * dx
    max_extent = max(phys_z, phys_y, phys_x, 1e-4)

    meta_payload = {
        **meta,
        "dimensions": [num_slices, rows, cols],
        "proxyDimensions": [128, 128, 128],
        "spacing": [dz, dy, dx],
        "physicalSize": [phys_z, phys_y, phys_x],
        "normalizedScale": [phys_x / max_extent, phys_y / max_extent, phys_z / max_extent],
        "defaultPreset": default_preset,
        "defaultWindow": default_window,
        "defaultLevel": default_level,
        "proxyBinUrl": f"/data/{output_dir.name}/volume_proxy.bin",
        "hasGroundTruthLabels": label is not None,
    }

    json_path = output_dir / "volume_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    return meta_payload


def ingest_uploaded_volume_archive(archive_path: Path, output_dir: Path) -> dict:
    """
    Unpacks and ingests uploaded .zip or multi-file directory into a calibrated volumetric study.
    """
    temp_dir = output_dir / "_unpack_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(temp_dir)
        else:
            shutil.copy(archive_path, temp_dir / archive_path.name)

        # Find all files in unpacked directory (ignoring hidden files and __MACOSX)
        all_files = [
            f for f in temp_dir.rglob("*")
            if f.is_file() and not any(part.startswith(".") or part == "__MACOSX" for part in f.parts)
        ]
        nii_files = [f for f in all_files if (".nii" in f.name.lower())]
        dcm_files = [f for f in all_files if (f.suffix.lower() == ".dcm" or not f.suffix) and not f.name.lower().endswith((".txt", ".md", ".json", ".xml", ".csv"))]

        label = None
        if nii_files:
            vol, affine, meta = assemble_nifti(nii_files[0])
        elif dcm_files:
            # Check if single multi-frame DICOM
            if len(dcm_files) == 1:
                try:
                    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                    if int(getattr(ds, "NumberOfFrames", 1)) > 1:
                        vol, affine, meta = assemble_multiframe_dicom(dcm_files[0])
                    else:
                        vol, affine, meta = assemble_dicom_series([dcm_files[0]])
                except Exception:
                    vol, affine, meta = assemble_dicom_series(dcm_files)
            else:
                vol, affine, meta = assemble_dicom_series(dcm_files)
        else:
            raise ValueError("No supported DICOM series or NIfTI file found in upload.")

        # Calibrate volume (CT HU or MRI ambient air isolation)
        vol_calibrated = calibrate_volume(vol, meta["modality"])

        # Create dual-resolution package
        meta_result = create_dual_resolution_package(
            vol_calibrated,
            affine,
            output_dir,
            meta,
            label=label,
        )
        return meta_result
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
