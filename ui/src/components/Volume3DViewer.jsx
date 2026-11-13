import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  RotateCw,
  Box,
  Layers,
  Sliders,
  Sparkles,
  RefreshCw,
  Camera,
  Compass,
  Eye,
  SlidersHorizontal
} from 'lucide-react';

// Preset configurations for Window / Level and Transfer Functions
const PRESETS = {
  Bone: {
    label: 'Bone (CT)',
    id: 0,
    window: 1500.0,
    level: 400.0,
    opacity: 1.2,
    colorScheme: 'bone',
  },
  SoftTissue: {
    label: 'Soft Tissue',
    id: 1,
    window: 400.0,
    level: 50.0,
    opacity: 0.9,
    colorScheme: 'soft_tissue',
  },
  Lung: {
    label: 'Lung Parenchyma',
    id: 2,
    window: 1500.0,
    level: -600.0,
    opacity: 0.8,
    colorScheme: 'lung',
  },
  MRI: {
    label: 'MRI Contrast',
    id: 3,
    window: 800.0,
    level: 400.0,
    opacity: 1.0,
    colorScheme: 'mri',
  },
  Mammography: {
    label: 'Mammography',
    id: 4,
    window: 600.0,
    level: 300.0,
    opacity: 1.1,
    colorScheme: 'mammo',
  },
};

const CLINICAL_ANGLES = [
  { name: 'AP Frontal', yaw: 0, pitch: 0 },
  { name: 'PA 180°', yaw: 180, pitch: 0 },
  { name: 'Lateral 90°', yaw: 90, pitch: 0 },
  { name: 'Lateral -90°', yaw: -90, pitch: 0 },
  { name: 'LAO Oblique 45°', yaw: 45, pitch: 0 },
  { name: 'RAO Oblique -45°', yaw: -45, pitch: 0 },
  { name: 'Cranial Tilt 30°', yaw: 0, pitch: 30 },
  { name: 'Caudal Tilt -30°', yaw: 0, pitch: -30 },
];

// Vertex shader for WebGL2 Volume Ray Marcher
const vertexShader = `
varying vec3 vOrigin;
varying vec3 vCamOrigin;
varying vec3 vDirection;
varying vec3 vLocalLightDir;

void main() {
    vOrigin = position + 0.5;
    vec4 camLocal = inverse(modelMatrix) * vec4(cameraPosition, 1.0);
    vCamOrigin = camLocal.xyz + 0.5;
    vDirection = vOrigin - vCamOrigin;
    vLocalLightDir = normalize(mat3(inverse(modelMatrix)) * vec3(0.4, 0.8, 0.6));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// Fragment shader: high-performance ray marcher with transfer functions and gradient lighting
const fragmentShader = `
precision highp float;
precision highp sampler3D;

varying vec3 vOrigin;
varying vec3 vCamOrigin;
varying vec3 vDirection;
varying vec3 vLocalLightDir;

uniform sampler3D u_data;
uniform float u_window;
uniform float u_level;
uniform int u_preset;
uniform float u_opacity;
uniform vec3 u_scale;
uniform int u_modality; // 0: CT, 1: MR

vec2 hitBox(vec3 orig, vec3 dir) {
    vec3 box_min = vec3(0.0);
    vec3 box_max = vec3(1.0);
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin_tmp = (box_min - orig) * inv_dir;
    vec3 tmax_tmp = (box_max - orig) * inv_dir;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}

// Finite-difference central gradient for realistic 3D clinical shading
vec3 computeNormal(vec3 p) {
    float eps = 1.0 / 128.0;
    float dx = texture(u_data, p + vec3(eps, 0.0, 0.0)).r - texture(u_data, p - vec3(eps, 0.0, 0.0)).r;
    float dy = texture(u_data, p + vec3(0.0, eps, 0.0)).r - texture(u_data, p - vec3(0.0, eps, 0.0)).r;
    float dz = texture(u_data, p + vec3(0.0, 0.0, eps)).r - texture(u_data, p - vec3(0.0, 0.0, eps)).r;
    return normalize(vec3(-dx, -dy, -dz) + 1e-5);
}

void main() {
    vec3 rayDir = normalize(vDirection);
    vec3 rayOrig = vCamOrigin;

    vec2 bounds = hitBox(rayOrig, rayDir);
    if (bounds.x > bounds.y || bounds.y < 0.0) discard;

    bounds.x = max(bounds.x, 0.0);
    vec3 pStart = rayOrig + rayDir * bounds.x;
    vec3 pEnd = rayOrig + rayDir * bounds.y;

    float pathLen = length(pEnd - pStart);
    if (pathLen < 0.001) discard;

    int numSteps = 220;
    float stepSize = pathLen / float(numSteps);
    vec3 stepVec = rayDir * stepSize;
    vec3 currentPos = pStart;

    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    // Window / Level in [0, 1] texture coordinates
    float wMin, wMax;
    if (u_modality == 1) { // MR
        // In MR proxy: 0 is air, tissue is in (0, 1]
        wMin = clamp((u_level - u_window * 0.5) / 800.0, 0.0, 1.0);
        wMax = clamp((u_level + u_window * 0.5) / 800.0, 0.02, 1.0);
    } else { // CT: texture 0..1 maps to -1000..1500 HU (range = 2500)
        float huMin = u_level - u_window * 0.5;
        float huMax = u_level + u_window * 0.5;
        wMin = clamp((huMin - (-1000.0)) / 2500.0, 0.0, 1.0);
        wMax = clamp((huMax - (-1000.0)) / 2500.0, 0.0, 1.0);
    }
    float wRange = max(wMax - wMin, 0.001);

    for (int i = 0; i < 220; i++) {
        if (accumAlpha >= 0.96) break;

        float val = texture(u_data, currentPos).r;
        float normVal = clamp((val - wMin) / wRange, 0.0, 1.0);

        if (normVal > 0.02) {
            vec3 col = vec3(normVal);
            float alpha = 0.0;

            if (u_preset == 0) { // Bone
                float boneWeight = smoothstep(0.25, 0.75, normVal);
                col = mix(vec3(0.85, 0.45, 0.35), vec3(0.98, 0.93, 0.82), boneWeight);
                alpha = pow(normVal, 2.8) * 0.45 * u_opacity;
            } else if (u_preset == 1) { // Soft Tissue
                col = mix(vec3(0.70, 0.30, 0.30), vec3(0.95, 0.85, 0.75), normVal);
                alpha = pow(normVal, 1.8) * 0.25 * u_opacity;
            } else if (u_preset == 2) { // Lung
                float lungVal = 1.0 - normVal;
                col = mix(vec3(0.20, 0.65, 0.95), vec3(0.95, 0.95, 0.95), normVal);
                alpha = smoothstep(0.1, 0.6, lungVal) * 0.20 * u_opacity;
            } else if (u_preset == 4) { // Mammography / Breast
                col = mix(vec3(0.92, 0.62, 0.58), vec3(1.0, 0.98, 0.92), smoothstep(0.15, 0.85, normVal));
                alpha = pow(normVal, 1.6) * 0.40 * u_opacity;
            } else { // MRI
                col = vec3(normVal, normVal * 0.98, normVal * 1.05);
                alpha = pow(normVal, 2.0) * 0.35 * u_opacity;
            }

            // Normal and diffuse lighting
            if (alpha > 0.01) {
                vec3 norm = computeNormal(currentPos);
                float diff = clamp(dot(norm, vLocalLightDir), 0.3, 1.0);
                col *= (diff * 0.8 + 0.2);

                accumColor += (1.0 - accumAlpha) * col * alpha;
                accumAlpha += (1.0 - accumAlpha) * alpha;
            }
        }

        currentPos += stepVec;
    }

    if (accumAlpha < 0.01) discard;

    gl_FragColor = vec4(accumColor, accumAlpha);
}
`;

export default function Volume3DViewer({
  study,
  onCameraChange,
  activePose,
  syncLoading = false,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const volumeMeshRef = useRef(null);
  const frustumGroupRef = useRef(null);
  const debounceTimerRef = useRef(null);
  const onCameraChangeRef = useRef(onCameraChange);

  useEffect(() => {
    onCameraChangeRef.current = onCameraChange;
  }, [onCameraChange]);

  // Transfer function & rendering state
  const isMRI = study.modality === 'MR' || study.modality === 'MRI';
  const isMG = study.modality === 'MG';
  const isMRorMG = isMRI || isMG;
  const initialPresetKey = isMG ? 'Mammography' : (isMRI ? 'MRI' : 'Bone');
  const [activePreset, setActivePreset] = useState(initialPresetKey);
  const [windowVal, setWindowVal] = useState(isMRorMG ? 600 : 1500);
  const [levelVal, setLevelVal] = useState(isMRorMG ? 300 : 400);
  const [opacityVal, setOpacityVal] = useState(1.0);
  const [showFrustum, setShowFrustum] = useState(false);

  // Re-synchronize presets and window/level when switching studies
  useEffect(() => {
    const isMR = study.modality === 'MR' || study.modality === 'MRI';
    const isMammo = study.modality === 'MG';
    const defaultPresetKey = isMammo ? 'Mammography' : (isMR ? 'MRI' : (study.defaultPreset || 'Bone'));
    const p = PRESETS[defaultPresetKey] || (isMR ? PRESETS.MRI : PRESETS.Bone);
    setActivePreset(defaultPresetKey);
    setWindowVal(study.defaultWindow || p.window);
    setLevelVal(study.defaultLevel || p.level);
    setOpacityVal(p.opacity);
  }, [study.id, study.modality]);

  // Live HUD Pose State
  const [hudPose, setHudPose] = useState({
    yaw: 0.0,
    pitch: 0.0,
    roll: 0.0,
    sdd: 1200.0,
    viewName: 'AP Frontal',
  });

  const [isLoadingVolume, setIsLoadingVolume] = useState(true);
  const [loadError, setLoadError] = useState(null);

  // Helper: map spherical coordinates to nanoDRR ZXY Euler
  const extractPoseFromControls = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return null;
    const cam = cameraRef.current;
    const target = controlsRef.current.target;

    const offset = new THREE.Vector3().subVectors(cam.position, target);
    const radius = offset.length();

    // Azimuth (Yaw) around Y axis in degrees
    const azimuth = THREE.MathUtils.radToDeg(Math.atan2(offset.x, offset.z));
    // Polar (Pitch) relative to horizontal plane
    const polar = THREE.MathUtils.radToDeg(Math.acos(Math.max(-1, Math.min(1, offset.y / radius))));
    const pitch = 90.0 - polar;

    const yawRound = Math.round(azimuth * 10) / 10;
    const pitchRound = Math.round(pitch * 10) / 10;

    // Detect view name preset
    let viewName = 'Arbitrary Orbit';
    if (Math.abs(yawRound) < 3 && Math.abs(pitchRound) < 3) viewName = 'AP Frontal (0°, 0°)';
    else if (Math.abs(Math.abs(yawRound) - 180) < 3 && Math.abs(pitchRound) < 3) viewName = 'PA Frontal (180°, 0°)';
    else if (Math.abs(yawRound - 90) < 3 && Math.abs(pitchRound) < 3) viewName = 'Lateral Right (90°, 0°)';
    else if (Math.abs(yawRound - (-90)) < 3 && Math.abs(pitchRound) < 3) viewName = 'Lateral Left (-90°, 0°)';
    else if (Math.abs(yawRound - 45) < 3 && Math.abs(pitchRound) < 3) viewName = 'LAO Oblique (45°, 0°)';
    else if (Math.abs(yawRound - (-45)) < 3 && Math.abs(pitchRound) < 3) viewName = 'RAO Oblique (-45°, 0°)';
    else if (Math.abs(yawRound) < 3 && Math.abs(pitchRound - 30) < 3) viewName = 'Cranial Tilt (0°, 30°)';
    else if (Math.abs(yawRound) < 3 && Math.abs(pitchRound - (-30)) < 3) viewName = 'Caudal Tilt (0°, -30°)';

    return {
      yaw: yawRound,
      pitch: pitchRound,
      roll: 0.0,
      sdd: 1200.0,
      viewName,
    };
  }, []);

  // Update virtual cone-beam frustum in scene
  const updateFrustum = useCallback((yaw, pitch, sdd = 1200) => {
    if (!frustumGroupRef.current) return;
    const group = frustumGroupRef.current;

    // Virtual source position at distance sdd along projection angle
    const yawRad = THREE.MathUtils.degToRad(yaw);
    const pitchRad = THREE.MathUtils.degToRad(pitch);

    const dist = 1.4; // normalized visual distance
    const srcX = dist * Math.sin(yawRad) * Math.cos(pitchRad);
    const srcY = dist * Math.sin(pitchRad);
    const srcZ = dist * Math.cos(yawRad) * Math.cos(pitchRad);

    const sourcePos = new THREE.Vector3(srcX, srcY, srcZ);
    const detPos = new THREE.Vector3(-srcX * 0.7, -srcY * 0.7, -srcZ * 0.7);

    // Update apex sphere
    const apex = group.getObjectByName('focalApex');
    if (apex) apex.position.copy(sourcePos);

    // Update detector plane
    const detMesh = group.getObjectByName('detPlane');
    if (detMesh) {
      detMesh.position.copy(detPos);
      detMesh.lookAt(sourcePos);
    }

    // Update ray lines
    const rays = group.getObjectByName('coneRays');
    if (rays && detMesh) {
      const geom = rays.geometry;
      const posAttr = geom.attributes.position;
      const halfW = 0.55;
      const halfH = 0.55;
      const corners = [
        new THREE.Vector3(-halfW, -halfH, 0).applyMatrix4(detMesh.matrixWorld),
        new THREE.Vector3(halfW, -halfH, 0).applyMatrix4(detMesh.matrixWorld),
        new THREE.Vector3(halfW, halfH, 0).applyMatrix4(detMesh.matrixWorld),
        new THREE.Vector3(-halfW, halfH, 0).applyMatrix4(detMesh.matrixWorld),
      ];
      let pIdx = 0;
      for (let i = 0; i < 4; i++) {
        posAttr.setXYZ(pIdx++, sourcePos.x, sourcePos.y, sourcePos.z);
        posAttr.setXYZ(pIdx++, corners[i].x, corners[i].y, corners[i].z);
      }
      posAttr.needsUpdate = true;
    }
  }, []);

  // Preset click handler: animate orbit to target clinical angle
  const handlePresetClick = (targetYaw, targetPitch) => {
    if (!cameraRef.current || !controlsRef.current) return;
    const radius = cameraRef.current.position.length();
    const yawRad = THREE.MathUtils.degToRad(targetYaw);
    const polarRad = THREE.MathUtils.degToRad(90.0 - targetPitch);

    const x = radius * Math.sin(polarRad) * Math.sin(yawRad);
    const y = radius * Math.cos(polarRad);
    const z = radius * Math.sin(polarRad) * Math.cos(yawRad);

    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(0, 0, 0);
    controlsRef.current.update();

    const pose = extractPoseFromControls();
    if (pose) {
      setHudPose(pose);
      updateFrustum(pose.yaw, pose.pitch, pose.sdd);
      if (onCameraChangeRef.current) onCameraChangeRef.current(pose);
    }
  };

  // Initialize Three.js scene and load 3D Data3DTexture proxy
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x06080e);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 2.75);
    cameraRef.current = camera;

    // Renderer (WebGL2)
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.85;
    controls.zoomSpeed = 1.0;
    controls.minDistance = 0.8;
    controls.maxDistance = 6.0;
    controlsRef.current = controls;

    // Bounding Box Guide
    const boxGeom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
    const boxEdges = new THREE.EdgesGeometry(boxGeom);
    const boxMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.18,
    });
    const boxWire = new THREE.LineSegments(boxEdges, boxMat);
    // Bounding wireframe omitted for pure clinical PACS aesthetic

    // Virtual Cone-Beam Frustum Visualization (rays and detector plane without occluding center blob)
    const frustumGroup = new THREE.Group();
    frustumGroupRef.current = frustumGroup;

    // Detector Plane
    const detGeom = new THREE.PlaneGeometry(1.1, 1.1);
    const detMat = new THREE.MeshBasicMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
    });
    const detPlane = new THREE.Mesh(detGeom, detMat);
    detPlane.name = 'detPlane';
    frustumGroup.add(detPlane);

    // Cone beam ray edges (8 vertices for 4 rays)
    const rayPos = new Float32Array(8 * 3);
    const rayGeom = new THREE.BufferGeometry();
    rayGeom.setAttribute('position', new THREE.BufferAttribute(rayPos, 3));
    const rayMat = new THREE.LineBasicMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.20,
    });
    const coneRays = new THREE.LineSegments(rayGeom, rayMat);
    coneRays.name = 'coneRays';
    frustumGroup.add(coneRays);

    scene.add(frustumGroup);

    // Subtle professional PACS isocenter crosshair marker (fixed at volume center 0,0,0)
    const isocenterGroup = new THREE.Group();
    isocenterGroup.name = 'isocenterCrosshair';

    const crossArm = 0.08;
    const crosshairGeom = new THREE.BufferGeometry();
    const crossPositions = new Float32Array([
      -crossArm, 0, 0,  crossArm, 0, 0,
      0, -crossArm, 0,  0, crossArm, 0,
      0, 0, -crossArm,  0, 0, crossArm,
    ]);
    crosshairGeom.setAttribute('position', new THREE.BufferAttribute(crossPositions, 3));
    const crosshairMat = new THREE.LineBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.25,
    });
    const crosshairLines = new THREE.LineSegments(crosshairGeom, crosshairMat);
    isocenterGroup.add(crosshairLines);

    scene.add(isocenterGroup);

    // Fetch and bind 128^3 uint8 binary proxy volume
    setIsLoadingVolume(true);
    setLoadError(null);

    const proxyBinUrl = `/api/volume_proxy/${study.id}`;

    fetch(proxyBinUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${proxyBinUrl}`);
        return res.arrayBuffer();
      })
      .then((buffer) => {
        const uint8Data = new Uint8Array(buffer);
        const texture = new THREE.Data3DTexture(uint8Data, 128, 128, 128);
        texture.format = THREE.RedFormat;
        texture.type = THREE.UnsignedByteType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.unpackAlignment = 1;
        texture.needsUpdate = true;

        const scaleArr = study.normalizedScale || [1.0, 1.0, 1.0];
        const volMat = new THREE.ShaderMaterial({
          vertexShader,
          fragmentShader,
          side: THREE.BackSide,
          transparent: true,
          premultipliedAlpha: true,
          uniforms: {
            u_data: { value: texture },
            u_window: { value: windowVal },
            u_level: { value: levelVal },
            u_preset: { value: PRESETS[activePreset].id },
            u_opacity: { value: opacityVal },
            u_scale: { value: new THREE.Vector3(scaleArr[0], scaleArr[1], scaleArr[2]) },
            u_modality: { value: isMRorMG ? 1 : 0 },
          },
        });

        const volumeGeom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
        const volumeMesh = new THREE.Mesh(volumeGeom, volMat);
        volumeMesh.scale.set(scaleArr[0], scaleArr[1], scaleArr[2]);
        volumeMesh.rotation.x = -Math.PI / 2;
        volumeMeshRef.current = volumeMesh;
        scene.add(volumeMesh);

        boxWire.scale.set(scaleArr[0], scaleArr[1], scaleArr[2]);
        boxWire.rotation.x = -Math.PI / 2;

        setIsLoadingVolume(false);
        updateFrustum(0, 0, 1200);
      })
      .catch((err) => {
        console.error('Failed to load 3D volume proxy:', err);
        setLoadError(err.message);
        setIsLoadingVolume(false);
      });

    // Orbit controls change listener (Debounced 200 ms sync)
    const handleControlsChange = () => {
      const pose = extractPoseFromControls();
      if (!pose) return;
      setHudPose(pose);
      updateFrustum(pose.yaw, pose.pitch, pose.sdd);

      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = setTimeout(() => {
        if (onCameraChangeRef.current) onCameraChangeRef.current(pose);
      }, 200);
    };

    controls.addEventListener('change', handleControlsChange);

    // Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler with ResizeObserver
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });
    resizeObserver.observe(container);
    window.addEventListener('resize', handleResize);

    return () => {
      if (animId) cancelAnimationFrame(animId);
      controls.removeEventListener('change', handleControlsChange);
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
      renderer.dispose();
      if (container) container.innerHTML = '';
    };
  }, [study.id]);

  // Sync Uniforms when Transfer Function or Window/Level sliders change
  useEffect(() => {
    if (!volumeMeshRef.current) return;
    const mat = volumeMeshRef.current.material;
    mat.uniforms.u_window.value = windowVal;
    mat.uniforms.u_level.value = levelVal;
    mat.uniforms.u_preset.value = PRESETS[activePreset].id;
    mat.uniforms.u_opacity.value = opacityVal;
  }, [windowVal, levelVal, activePreset, opacityVal]);

  // Sync Frustum visibility
  useEffect(() => {
    if (frustumGroupRef.current) {
      frustumGroupRef.current.visible = showFrustum;
    }
  }, [showFrustum]);

  return (
    <div className="relative w-full h-full flex flex-col bg-[#06080e] overflow-hidden select-none min-h-0 min-w-0">
      {/* 3D WebGL Canvas */}
      <div ref={mountRef} className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing min-h-0 min-w-0" />

      {/* Top Floating PACS Controls */}
      <div className="absolute top-2 inset-x-2 flex items-start justify-between pointer-events-none z-10 gap-2">
        {/* Preset Angles Pill */}
        <div className="pointer-events-auto bg-slate-950/85 backdrop-blur-md border border-slate-800/80 rounded-xl px-2 py-1 shadow-xl flex items-center gap-1 text-xs max-w-[calc(100%-60px)] overflow-x-auto no-scrollbar shrink-0">
          <span className="text-[10px] font-mono text-cyan-400 font-bold px-1.5 py-0.5 rounded bg-cyan-950/40 border border-cyan-800/40 whitespace-nowrap shrink-0">
            {hudPose.viewName}
          </span>
          <span className="text-[10px] font-mono text-slate-300 px-1 whitespace-nowrap shrink-0">
            {hudPose.yaw}° / {hudPose.pitch}°
          </span>
          <div className="h-3.5 w-px bg-slate-800 mx-0.5 shrink-0" />
          <div className="flex items-center gap-1 shrink">
            {CLINICAL_ANGLES.map((ang) => {
              const isActive =
                Math.abs(hudPose.yaw - ang.yaw) < 3 && Math.abs(hudPose.pitch - ang.pitch) < 3;
              return (
                <button
                  key={ang.name}
                  onClick={() => handlePresetClick(ang.yaw, ang.pitch)}
                  className={`px-1.5 py-0.5 rounded-md border text-[10px] font-medium transition-all whitespace-nowrap shrink-0 ${
                    isActive
                      ? 'bg-cyan-500 text-black border-cyan-400 font-bold shadow-md shadow-cyan-500/20'
                      : 'bg-slate-900/90 hover:bg-slate-800 border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300'
                  }`}
                >
                  {ang.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Status / Frustum Pill */}
        <div className="pointer-events-auto flex items-center gap-1.5 shrink-0">
          {syncLoading && (
            <div className="flex items-center gap-1.5 text-[10px] text-cyan-400 bg-cyan-950/80 border border-cyan-500/40 px-2 py-0.5 rounded-xl shadow-lg backdrop-blur-md whitespace-nowrap">
              <RefreshCw className="w-3 h-3 animate-spin" />
              <span className="hidden sm:inline">Sync...</span>
            </div>
          )}
          <button
            onClick={() => setShowFrustum(!showFrustum)}
            title="Toggle virtual X-ray cone beam"
            className={`p-1.5 rounded-xl border text-[11px] font-semibold transition-colors shadow-lg backdrop-blur-md ${
              showFrustum
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                : 'bg-slate-950/80 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Bottom Floating Control Bar: Window/Level & Transfer Function */}
      <div className="absolute bottom-2 inset-x-2 flex items-center justify-between gap-2 pointer-events-none z-10 flex-wrap sm:flex-nowrap">
        {/* Transfer Function Selector */}
        <div className="pointer-events-auto bg-slate-950/85 backdrop-blur-md border border-slate-800/80 rounded-xl p-1 flex items-center gap-1 shadow-xl shrink-0">
          {Object.entries(PRESETS).map(([key, p]) => (
            <button
              key={key}
              onClick={() => {
                setActivePreset(key);
                setWindowVal(p.window);
                setLevelVal(p.level);
                setOpacityVal(p.opacity);
              }}
              className={`px-2 py-0.5 rounded-lg text-[10px] font-semibold transition-all whitespace-nowrap ${
                activePreset === key
                  ? 'bg-cyan-500 text-black shadow-md font-bold'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Compact Window / Level Sliders */}
        <div className="pointer-events-auto bg-slate-950/85 backdrop-blur-md border border-slate-800/80 rounded-xl px-2.5 py-1 shadow-xl flex items-center gap-2 text-xs text-slate-300 font-mono shrink-0">
          <div className="flex items-center gap-1">
            <span className="text-slate-400 text-[10px] font-bold">W</span>
            <input
              type="range"
              min="100"
              max="2500"
              step="50"
              value={windowVal}
              onChange={(e) => setWindowVal(parseFloat(e.target.value))}
              className="w-14 accent-cyan-400 cursor-pointer h-1 bg-slate-800 rounded-lg"
            />
            <span className="text-cyan-300 text-[10px] font-bold w-8 text-right">{Math.round(windowVal)}</span>
          </div>

          <div className="h-3 w-px bg-slate-800" />

          <div className="flex items-center gap-1">
            <span className="text-slate-400 text-[10px] font-bold">L</span>
            <input
              type="range"
              min="-800"
              max="1200"
              step="25"
              value={levelVal}
              onChange={(e) => setLevelVal(parseFloat(e.target.value))}
              className="w-14 accent-cyan-400 cursor-pointer h-1 bg-slate-800 rounded-lg"
            />
            <span className="text-cyan-300 text-[10px] font-bold w-8 text-right">{Math.round(levelVal)}</span>
          </div>
        </div>
      </div>

      {/* Loading Overlay */}
      {isLoadingVolume && (
        <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30">
          <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
          <div className="text-xs font-mono text-slate-300">
            Streaming 3D Volume Proxy (128³ uint8)...
          </div>
        </div>
      )}

      {/* Error Overlay */}
      {loadError && (
        <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center gap-2 z-30 text-rose-400 p-6 text-center">
          <div className="font-bold text-sm">Failed to Load 3D Volume</div>
          <div className="text-xs font-mono text-slate-400">{loadError}</div>
        </div>
      )}
    </div>
  );
}
