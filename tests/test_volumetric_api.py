"""
REST API tests for FleXray Volumetric DRR & Segmentation Endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ui.server import app

client = TestClient(app)


def test_health_check_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "volumetric_drr_engine" in data


def test_studies_includes_volumetric_datasets():
    res = client.get("/api/studies")
    assert res.status_code == 200
    studies = res.json()
    ids = [s["id"] for s in studies]

    # Verify all 4 genuine clinical volumetric studies are registered
    assert "study-torso-spine-ct" in ids
    assert "study-nlst-chest-ct" in ids
    assert "study-torso-dicom-series" in ids
    assert "study-brain-mri-spm" in ids

    # Check volumetric flag
    vol_studies = [s for s in studies if s.get("isVolumetric")]
    assert len(vol_studies) >= 4


def test_project_drr_endpoint():
    payload = {
        "study_id": "study-torso-spine-ct",
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "sdd": 1200.0,
        "height": 128,
        "width": 128,
        "n_samples": 150,
        "threshold": 0.45,
    }
    res = client.post("/api/project_drr", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["study_id"] == "study-torso-spine-ct"
    assert "drrImageUrl" in data
    assert "overlayImageUrl" in data
    assert "structures" in data
    assert "timing" in data
    assert data["timing"]["cuda_render_ms"] > 0


def test_project_drr_ground_truth_mode_with_focus():
    payload = {
        "study_id": "study-torso-spine-ct",
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "sdd": 1200.0,
        "height": 128,
        "width": 128,
        "n_samples": 100,
        "mode": "ground_truth",
        "focus_structure": "vertebrae_ribs",
    }
    res = client.post("/api/project_drr", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["study_id"] == "study-torso-spine-ct"
    assert data["focusImageUrl"] is not None
    assert "focus_" in data["focusImageUrl"]


def test_upload_volume_zip_archive():
    import io, zipfile
    from pathlib import Path

    nii_path = Path(r"C:\Users\Amit Yadav\miniconda3\lib\site-packages\nibabel\tests\data\anatomical.nii")
    if not nii_path.exists():
        pytest.skip("anatomical.nii missing")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.write(nii_path, arcname="brain/anatomical.nii")
        zf.writestr("brain/README.txt", "Brain MRI clinical scan")
    zip_buf.seek(0)

    files = [("files", ("test_brain.zip", zip_buf.getvalue(), "application/zip"))]
    res = client.post("/api/upload_volume", files=files, data={"patient_name": "Test MRI Subject", "modality": "MR"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["study"]["isVolumetric"] is True
    assert "volumeProxyUrl" in data["study"]
    assert "details" in data


def test_upload_volume_relative_folder_path():
    # Verify subdirectories in filename do not crash with FileNotFoundError
    files = [
        ("files", ("CT_Folder/slice_0001.dcm", b"NOT_A_DICOM", "application/dicom"))
    ]
    res = client.post("/api/upload_volume", files=files)
    # Expected clean 400 rejection (not 500 or FileNotFoundError crash)
    assert res.status_code == 400
    assert "Failed to process volume" in res.json()["detail"]


def test_drr_yaw_direction_synchrony():
    import cv2
    import numpy as np
    from pathlib import Path

    # Verify that positive yaw (+72) projects lateral chest with spine on the right (col > 256)
    # and negative yaw (-72) projects lateral chest with spine on the left (col < 256),
    # maintaining exact 1:1 orientation sync with Three.js camera orbit.
    pos_res = client.post(
        "/api/project_drr",
        json={
            "study_id": "study-nlst-chest-ct",
            "yaw": 72.0,
            "pitch": 0.0,
            "roll": 0.0,
            "sdd": 1200.0,
            "height": 512,
            "width": 512,
            "focus_structure": "lungs",
        },
    )
    assert pos_res.status_code == 200
    pos_data = pos_res.json()
    assert pos_data["focusImageUrl"] is not None

    drr_pos_path = Path("ui_data") / "study-nlst-chest-ct" / f"drr_{Path(pos_data['drrImageUrl']).name}"
    if not drr_pos_path.exists():
        drr_pos_path = Path(__file__).resolve().parent.parent / "ui_data" / "study-nlst-chest-ct" / f"drr_{Path(pos_data['drrImageUrl']).name}"
    # Read generated image from disk
    drr_pos_disk = Path(__file__).resolve().parent.parent / "ui_data" / "study-nlst-chest-ct" / Path(pos_data['drrImageUrl']).name
    img_pos = cv2.imread(str(drr_pos_disk), cv2.IMREAD_GRAYSCALE)
    col_peak_pos = int(np.argmax(img_pos.mean(axis=0)))
    assert col_peak_pos > 256, f"For yaw=+72, spine peak should be on the right half, got col={col_peak_pos}"

    neg_res = client.post(
        "/api/project_drr",
        json={
            "study_id": "study-nlst-chest-ct",
            "yaw": -72.0,
            "pitch": 0.0,
            "roll": 0.0,
            "sdd": 1200.0,
            "height": 512,
            "width": 512,
            "focus_structure": "lungs",
        },
    )
    assert neg_res.status_code == 200
    neg_data = neg_res.json()
    assert neg_data["focusImageUrl"] is not None

    drr_neg_disk = Path(__file__).resolve().parent.parent / "ui_data" / "study-nlst-chest-ct" / Path(neg_data['drrImageUrl']).name
    img_neg = cv2.imread(str(drr_neg_disk), cv2.IMREAD_GRAYSCALE)
    col_peak_neg = int(np.argmax(img_neg.mean(axis=0)))
    assert col_peak_neg < 256, f"For yaw=-72, spine peak should be on the left half, got col={col_peak_neg}"
