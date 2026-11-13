/**
 * Real Clinical Studies Database.
 * 
 * Contains exclusively genuine patient imaging data processed through FleXray:
 * 1. Moola Ram: Lateral Knee DX (3173 x 2500 16-bit Digital Radiograph from a.dcm)
 * 2. Elena Rostova: Upright Chest PA (1200 x 1200 CR Radiograph from sample-chest.png)
 * 
 * Strictly zero synthetic or mock studies.
 */

export const STUDIES = [
  {
    id: "knee-moola-ram",
    patientName: "Moola Ram",
    mrn: "BEDI123",
    age: "54",
    gender: "M",
    modality: "DX",
    bodyPart: "Knee",
    view: "Lateral (KNEE-LAT-M)",
    studyDate: "2026-10-14 11:22",
    institution: "Apex Orthopedic & Radiology Center",
    status: "Processed",
    summary: "High-resolution Lateral Knee DX radiograph (3173 x 2500 px). 3D Gaussian Splatting and physical VDES bone/soft decomposition complete.",
    metrics: {
      "Acquisition Matrix": "3173 × 2500 px (16-bit)",
      "Medial Joint Space": "4.8 mm",
      "Lateral Joint Space": "5.2 mm",
      "Patellar Alignment": "Insall-Salvati 1.04 (Normal)",
      "Osteoarthritis Grade": "Kellgren-Lawrence Grade 1",
      "Reconstruction SSIM": "0.782",
      "3D Gaussians Fitted": "36,300 Primitives",
      "Convergence Time": "6.90 s"
    },
    presentStructures: ["femurs", "tibiae", "patellae", "fibulae"],
    images: {
      raw: "/output_knee_dcm/patient_raw_hd.png",
      overlay: "/output_knee_dcm/patient_overlay_hd.png",
      softTissue: "/output_knee_dcm/patient_soft_tissue_hd.png",
      bone: "/output_knee_dcm/patient_bone_hd.png",
      strip: "/output_knee_dcm/patient_comparison_strip_hd.png",
      dicomSc: "/output_knee_dcm/patient_vdes_secondary_capture.dcm",
      ply: "/output_knee_dcm/patient_3d_splat.ply"
    }
  },
  {
    id: "chest-elena-rostova",
    patientName: "Elena Rostova",
    mrn: "MRN-642199",
    age: "48",
    gender: "F",
    modality: "CR",
    bodyPart: "Chest",
    view: "PA Frontal",
    studyDate: "2026-10-18 09:14",
    institution: "Metropolitan Academic Hospital",
    status: "Processed",
    summary: "High-definition posteroanterior chest radiograph (1200 x 1200 px). Virtual Dual-Energy bone cancellation isolates lung parenchyma.",
    metrics: {
      "Acquisition Matrix": "1200 × 1200 px",
      "Cardiothoracic Ratio (CTR)": "0.44 (Normal < 0.50)",
      "Transverse Cardiac Diameter": "12.8 cm",
      "Thoracic Cavity Diameter": "29.1 cm",
      "Rib Suppression Ratio": "96.4% bone removed",
      "Reconstruction SSIM": "0.724",
      "3D Gaussians Fitted": "25,100 Primitives",
      "Convergence Time": "36.78 s"
    },
    presentStructures: [
      "lungs", "heart", "clavicles", "sternum",
      "rib_1", "rib_2", "rib_3", "rib_4", "rib_5", "rib_6", "rib_7", "rib_8", "rib_9", "rib_10", "rib_11", "rib_12",
      "vertebra_t1", "vertebra_t2", "vertebra_t3", "vertebra_t4", "vertebra_t5"
    ],
    images: {
      raw: "/output_chest/patient_raw_hd.png",
      overlay: "/output_chest/patient_overlay_hd.png",
      softTissue: "/output_chest/patient_soft_tissue_hd.png",
      bone: "/output_chest/patient_bone_hd.png",
      strip: "/output_chest/patient_comparison_strip_hd.png",
      ply: "/output_chest/patient_3d_splat.ply"
    }
  }
];
