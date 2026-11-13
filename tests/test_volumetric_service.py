"""
Tests for FleXray Volumetric Service (CT/MRI Ingestion, Direction Cosines, Calibration, and Caching).
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from ui.volumetric_service import (
    compute_slice_normal,
    assemble_dicom_series,
    assemble_nifti,
    assemble_npz_volume,
    calibrate_volume,
    create_dual_resolution_package,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_compute_slice_normal_orthogonal():
    # Standard Axial IOP: Row = (1, 0, 0), Col = (0, 1, 0) -> Normal = (0, 0, 1)
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    n = compute_slice_normal(iop)
    assert np.allclose(n, [0.0, 0.0, 1.0])

    # Coronal IOP: Row = (1, 0, 0), Col = (0, 0, -1) -> Normal = (0, 1, 0)
    iop_cor = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    n_cor = compute_slice_normal(iop_cor)
    assert np.allclose(n_cor, [0.0, 1.0, 0.0])


def test_dicom_series_assembly_monotonic():
    dcm_dir = Path(r"c:\Users\Amit Yadav\Downloads\projects\cortex\sample_dicoms\CT_Torso_Series")
    if not dcm_dir.exists():
        pytest.skip("CT_Torso_Series not found on disk")

    vol, affine, meta = assemble_dicom_series(dcm_dir)

    assert vol.ndim == 3
    assert vol.shape == (40, 512, 512)
    assert affine.shape == (4, 4)
    assert meta["modality"] == "CT"
    assert meta["spacing"][0] == 2.5 # dz
    assert meta["spacing"][1] == 0.7 # dy
    assert meta["spacing"][2] == 0.7 # dx


def test_nifti_assembly_and_mri_calibration(tmp_path: Path):
    nii_path = Path(r"C:\Users\Amit Yadav\miniconda3\lib\site-packages\nibabel\tests\data\anatomical.nii")
    if not nii_path.exists():
        pytest.skip("anatomical.nii not found in nibabel testing data")

    vol, affine, meta = assemble_nifti(nii_path)
    assert vol.ndim == 3
    assert affine.shape == (4, 4)
    assert meta["modality"] == "MR"

    # Calibration: ambient air must map to -1000 HU
    calibrated = calibrate_volume(vol, "MR")
    assert calibrated.min() == -1000.0
    # Tissue must be non-negative
    tissue_voxels = calibrated[calibrated > -900.0]
    assert len(tissue_voxels) > 0
    assert tissue_voxels.min() >= 0.0
    assert tissue_voxels.max() <= 800.0

    # Package export
    pkg = create_dual_resolution_package(calibrated, affine, tmp_path / "mri_pkg", meta)
    assert (tmp_path / "mri_pkg" / "volume.npz").exists()
    bin_file = tmp_path / "mri_pkg" / "volume_proxy.bin"
    assert bin_file.exists()
    assert bin_file.stat().st_size == 128 * 128 * 128 # exactly 2,097,152 bytes


def test_npz_volume_assembly():
    npz_path = Path(r"c:\Users\Amit Yadav\Downloads\projects\unet\data\real_ct_volume.npz")
    if not npz_path.exists():
        pytest.skip("real_ct_volume.npz not found")

    vol, affine, meta, label = assemble_npz_volume(npz_path)
    assert vol.shape == (100, 512, 512)
    assert affine.shape == (4, 4)
    assert label is not None
    assert label.shape == (100, 512, 512)


def test_mri_proxy_ambient_air_isolation(tmp_path: Path):
    # Simulated MRI volume: 0 is ambient air, 200..1000 is brain tissue
    vol = np.zeros((32, 64, 64), dtype=np.float32)
    vol[:, 16:48, 16:48] = 500.0 # brain tissue

    calibrated = calibrate_volume(vol, "MR")
    # Air must be -1000, tissue > 0
    assert calibrated[0, 0, 0] == -1000.0
    assert calibrated[16, 32, 32] > 0.0

    meta = {"modality": "MR", "spacing": [1.0, 1.0, 1.0]}
    pkg = create_dual_resolution_package(calibrated, np.eye(4), tmp_path / "mri_test", meta)

    bin_data = np.fromfile(tmp_path / "mri_test" / "volume_proxy.bin", dtype=np.uint8)
    # Air voxels must be strictly 0 for complete transparency
    assert bin_data[0] == 0
    # Tissue voxels must have strong signal
    assert bin_data.max() > 200
