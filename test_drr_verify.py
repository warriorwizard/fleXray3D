import sys
sys.path.insert(0, r"c:\Users\Amit Yadav\Downloads\projects\fleXray")
from pathlib import Path
import numpy as np
import torch, cv2
from fxr.drr import render_drr, DRRRenderRequest, build_render_intrinsics, compute_isocenter
from ui.flexray_core import get_segmenter, generate_pan_anatomy_overlay

for study_id in ["study-nlst-chest-ct", "study-torso-spine-ct"]:
    npz_path = Path(rf"c:\Users\Amit Yadav\Downloads\projects\fleXray\ui_data\{study_id}\volume.npz")
    if not npz_path.exists():
        print(f"Skipping {study_id}: file not found")
        continue

    data = np.load(npz_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    vol = torch.from_numpy(data["volume"].astype(np.float32)).unsqueeze(0).to(DEVICE)
    aff = torch.from_numpy(data["affine"].astype(np.float32)).to(DEVICE)
    iso = compute_isocenter(vol, None, aff, "volume_center").to(DEVICE)

    d, h, w = vol.shape[-3:]
    phys_x = w * abs(aff[0,0].item())
    phys_y = h * abs(aff[1,1].item())
    phys_z = d * abs(aff[2,2].item())
    max_diam = max(phys_x, phys_y, phys_z)
    sdd = 1200.0
    src_dist = sdd * 0.72
    mag = sdd / src_dist
    del_pix = (max_diam * mag * 1.12) / 256.0

    intrinsics = build_render_intrinsics(
        sdd=sdd,
        delx=del_pix,
        dely=del_pix,
        x0=0.0,
        y0=0.0,
        height=256,
        width=256,
        device=DEVICE,
    )

    req = DRRRenderRequest(
        rot=torch.tensor([0.0, 0.0, 0.0], device=DEVICE),
        xyz=torch.tensor([0.0, src_dist, 0.0], device=DEVICE),
        intrinsics=intrinsics,
        isocenter=iso,
        n_samples=300,
        seg_threshold=0.05,
    )

    res = render_drr(volume=vol, label=None, affine=aff, request=req)
    img = res.images[0, 0].cpu().numpy()
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    base_gray = np.clip((img - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
    out_img_path = f"verify_{study_id}.png"
    cv2.imwrite(out_img_path, (base_gray * 255.0).astype(np.uint8))

    segmenter = get_segmenter()
    pred = segmenter.predict((base_gray * 255.0).astype(np.uint8))
    probs = pred.probabilities[0].cpu().numpy()
    overlay_uint8, structures = generate_pan_anatomy_overlay(base_gray, probs, segmenter.label_names, threshold=0.45)
    out_overlay_path = f"verify_overlay_{study_id}.png"
    cv2.imwrite(out_overlay_path, cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

    print(f"[{study_id}] Detected {len(structures)} structures.")
    for s in structures[:6]:
        print(f"  {s['name']}: {s['maxConfidence']*100:.1f}% conf, {s['coveragePercent']}% cov")

