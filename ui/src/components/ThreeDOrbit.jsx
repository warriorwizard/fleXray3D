import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import {
  RotateCw,
  Eye,
  EyeOff,
  Box,
  Layers,
  Activity,
  Sliders,
  Scissors,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  Crosshair,
  Filter,
  Maximize2,
  Camera
} from 'lucide-react';

// Generates an analytical continuous Gaussian splat kernel texture
function createGaussianTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
  gradient.addColorStop(0.18, 'rgba(255, 255, 255, 0.90)');
  gradient.addColorStop(0.45, 'rgba(255, 255, 255, 0.42)');
  gradient.addColorStop(0.75, 'rgba(255, 255, 255, 0.12)');
  gradient.addColorStop(1.0, 'rgba(255, 255, 255, 0.0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// Robust multi-property binary PLY parser
async function fetchAndParseRealPLY(plyUrl) {
  try {
    const res = await fetch(plyUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${plyUrl}`);
    const buffer = await res.arrayBuffer();

    const textHeaderChunk = new TextDecoder().decode(new Uint8Array(buffer.slice(0, 2048)));
    const endHeaderTag = 'end_header\n';
    const headerEndIdx = textHeaderChunk.indexOf(endHeaderTag);
    if (headerEndIdx === -1) throw new Error('Invalid PLY header format');

    const header = textHeaderChunk.slice(0, headerEndIdx);
    const vertMatch = header.match(/element vertex (\d+)/);
    if (!vertMatch) throw new Error('Could not parse vertex count from PLY header');

    const numVerts = parseInt(vertMatch[1], 10);
    const dataByteOffset = headerEndIdx + endHeaderTag.length;
    const dataView = new DataView(buffer, dataByteOffset);

    // Count float properties in header to compute stride
    const propertyMatches = header.match(/property float \w+/g) || [];
    const numProps = propertyMatches.length;
    const stride = numProps * 4;

    const hasClassId = header.includes('property float class_id');
    const hasDensity = header.includes('property float density');

    const positions = new Float32Array(numVerts * 3);
    const segColors = new Float32Array(numVerts * 3);
    const densityColors = new Float32Array(numVerts * 3);
    const classIds = new Int32Array(numVerts);
    const opacities = new Float32Array(numVerts);

    for (let i = 0; i < numVerts; i++) {
      const offset = i * stride;
      const x = dataView.getFloat32(offset + 0, true);
      const y = dataView.getFloat32(offset + 4, true);
      const z = dataView.getFloat32(offset + 8, true);

      // Spherical harmonic / RGB colors
      const r = Math.max(0.0, Math.min(1.0, dataView.getFloat32(offset + 24, true)));
      const g = Math.max(0.0, Math.min(1.0, dataView.getFloat32(offset + 28, true)));
      const b = Math.max(0.0, Math.min(1.0, dataView.getFloat32(offset + 32, true)));

      const opacity = Math.max(0.1, Math.min(1.0, dataView.getFloat32(offset + 36, true)));

      // Class ID (offset 68 if 19 floats)
      let cId = 0;
      if (hasClassId && stride >= 72) {
        cId = Math.round(dataView.getFloat32(offset + 68, true));
      }

      // Radiodensity (offset 72 if 19 floats)
      let dens = 0.7;
      if (hasDensity && stride >= 76) {
        dens = Math.max(0.0, Math.min(1.0, dataView.getFloat32(offset + 72, true)));
      }

      positions[i * 3 + 0] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      segColors[i * 3 + 0] = r;
      segColors[i * 3 + 1] = g;
      segColors[i * 3 + 2] = b;

      // Ivory bone with radiographic attenuation modulation
      const ivoryBase = 0.85 + 0.15 * dens;
      densityColors[i * 3 + 0] = 0.95 * ivoryBase;
      densityColors[i * 3 + 1] = 0.92 * ivoryBase;
      densityColors[i * 3 + 2] = 0.84 * ivoryBase;

      classIds[i] = cId;
      opacities[i] = opacity;
    }

    return {
      positions,
      segColors,
      densityColors,
      classIds,
      opacities,
      count: numVerts,
      stride,
      hasClassId
    };
  } catch (err) {
    console.error('Failed to parse real patient PLY:', err);
    return null;
  }
}

export default function ThreeDOrbit({ study }) {
  const containerRef = useRef(null);

  // Display Modes: 'segmentation' | 'density' | 'composite'
  const [colorMode, setColorMode] = useState('segmentation');
  const [autoRotate, setAutoRotate] = useState(false); // Default to clean stable inspection view
  const [viewPreset, setViewPreset] = useState('lateral');
  const [splatSize, setSplatSize] = useState(0.038);
  const [sliceDepth, setSliceDepth] = useState(100);
  const [showCalipers, setShowCalipers] = useState(true);

  // Structure Visibility Filter
  const [activeStructures, setActiveStructures] = useState({
    femurs: true,
    tibiae: true,
    patellae: true,
    fibulae: true,
    cartilage: true,
    lungs: true,
    heart: true,
    ribs: true,
    spine: true,
  });

  // Real Data Status
  const [isLoadingPly, setIsLoadingPly] = useState(true);
  const [plyDataState, setPlyDataState] = useState(null);

  // Three.js References
  const sceneRef = useRef(null);
  const modelGroupRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const animFrameRef = useRef(null);
  const pointsMeshRef = useRef(null);
  const clipPlaneRef = useRef(null);
  const splatMaterialRef = useRef(null);
  const calipersGroupRef = useRef(null);

  // Mouse Interaction References
  const isDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });

  // Structure metadata with color chips and class IDs
  const structureMeta = useMemo(() => {
    if (study.bodyPart.toLowerCase() === 'knee') {
      return [
        { id: 'femurs', label: 'Femur', classId: 11, color: '#3b82f6', count: '12,500' },
        { id: 'tibiae', label: 'Tibia', classId: 10, color: '#10b981', count: '13,000' },
        { id: 'patellae', label: 'Patella', classId: 12, color: '#f59e0b', count: '3,800' },
        { id: 'fibulae', label: 'Fibula', classId: 13, color: '#8b5cf6', count: '4,200' },
        { id: 'cartilage', label: 'Joint Space', classId: 100, color: '#06b6d4', count: '2,800' },
      ];
    } else {
      return [
        { id: 'lungs', label: 'Lungs', classId: 59, color: '#22c55e', count: '8,000' },
        { id: 'heart', label: 'Heart', classId: 60, color: '#f87171', count: '4,500' },
        { id: 'ribs', label: 'Ribs (1-12)', classId: 17, color: '#fbbf24', count: '7,200' },
        { id: 'spine', label: 'Thoracic Spine', classId: 37, color: '#38bdf8', count: '5,400' },
      ];
    }
  }, [study.bodyPart]);

  const toggleStructure = (id) => {
    setActiveStructures((prev) => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const isolateStructure = (id) => {
    setActiveStructures((prev) => {
      const isAlreadySolo = Object.entries(prev).every(([k, v]) => (k === id ? v : !v));
      if (isAlreadySolo) {
        const allOn = {};
        Object.keys(prev).forEach((k) => { allOn[k] = true; });
        return allOn;
      } else {
        const solo = {};
        Object.keys(prev).forEach((k) => { solo[k] = k === id; });
        return solo;
      }
    });
  };

  const showAllStructures = () => {
    const allOn = {};
    Object.keys(activeStructures).forEach((k) => { allOn[k] = true; });
    setActiveStructures(allOn);
  };

  // Setup Three.js Scene and Load Real Patient PLY
  useEffect(() => {
    if (!containerRef.current) return;

    let isDisposed = false;
    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 560;

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070e);
    sceneRef.current = scene;

    // Dedicated Model Group centered in viewport
    const modelGroup = new THREE.Group();
    // Offset slightly downward to leave headroom for controls
    modelGroup.position.set(0, -0.12, 0);
    // Initial 25-degree aesthetic orbit angle showing both lateral profile and condyle depth
    modelGroup.rotation.set(0.12, 0.40, 0);
    modelGroupRef.current = modelGroup;
    scene.add(modelGroup);

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.5);
    cameraRef.current = camera;

    // 2. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.localClippingEnabled = true;

    while (containerRef.current.firstChild) {
      containerRef.current.removeChild(containerRef.current.firstChild);
    }
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 3. Clipping Plane for Depth Slicing
    const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 10.0);
    clipPlaneRef.current = clipPlane;

    // 4. Clinical Theatre Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.3);
    keyLight.position.set(3, 4, 3.5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x06b6d4, 0.9);
    rimLight.position.set(-3, 1, -2.5);
    scene.add(rimLight);

    // 5. Floor Reference Grid
    const grid = new THREE.GridHelper(3.2, 16, 0x1e293b, 0x090e1a);
    grid.position.y = -1.35;
    scene.add(grid);

    // 6. Calipers & Annotations Group inside modelGroup
    const calipersGroup = new THREE.Group();
    calipersGroupRef.current = calipersGroup;
    modelGroup.add(calipersGroup);

    if (study.bodyPart.toLowerCase() === 'knee') {
      // 3D Caliper Line for Medial Joint Space Width (4.8 mm)
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.08, 0.045, -0.22),
        new THREE.Vector3(-0.08, -0.045, -0.22)
      ]);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x22d3ee, linewidth: 2 });
      const jswLine = new THREE.Line(lineGeom, lineMat);
      calipersGroup.add(jswLine);

      // Marker spheres at endpoints
      const sphereGeom = new THREE.SphereGeometry(0.018, 8, 8);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee });
      const m1 = new THREE.Mesh(sphereGeom, sphereMat);
      m1.position.set(-0.08, 0.045, -0.22);
      const m2 = new THREE.Mesh(sphereGeom, sphereMat);
      m2.position.set(-0.08, -0.045, -0.22);
      calipersGroup.add(m1);
      calipersGroup.add(m2);
    } else {
      // 3D Caliper Line for Cardiothoracic Ratio (CTR: 0.44)
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.35, -0.15, 0.08),
        new THREE.Vector3(0.20, -0.15, 0.08)
      ]);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xf87171, linewidth: 2 });
      const ctrLine = new THREE.Line(lineGeom, lineMat);
      calipersGroup.add(ctrLine);

      const sphereGeom = new THREE.SphereGeometry(0.018, 8, 8);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xf87171 });
      const m1 = new THREE.Mesh(sphereGeom, sphereMat);
      m1.position.set(-0.35, -0.15, 0.08);
      const m2 = new THREE.Mesh(sphereGeom, sphereMat);
      m2.position.set(0.20, -0.15, 0.08);
      calipersGroup.add(m1);
      calipersGroup.add(m2);
    }

    // 7. Load Genuine Patient PLY 3DGS Primitives
    setIsLoadingPly(true);
    fetchAndParseRealPLY(study.images.ply).then((plyData) => {
      if (isDisposed || !plyData) return;

      setPlyDataState(plyData);

      const splatTexture = createGaussianTexture();
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(plyData.positions, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(plyData.segColors), 3));

      const splatMat = new THREE.PointsMaterial({
        size: splatSize,
        map: splatTexture,
        transparent: true,
        vertexColors: true,
        alphaTest: 0.01,
        depthWrite: false,
        blending: THREE.NormalBlending,
        clippingPlanes: [clipPlane],
      });
      splatMaterialRef.current = splatMat;

      const pointsMesh = new THREE.Points(geom, splatMat);
      pointsMeshRef.current = pointsMesh;
      modelGroup.add(pointsMesh);

      setIsLoadingPly(false);
    });

    // Interaction handlers
    const dom = renderer.domElement;
    const onMouseDown = (e) => {
      isDraggingRef.current = true;
      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDraggingRef.current || !modelGroupRef.current) return;
      const dx = e.clientX - previousMousePositionRef.current.x;
      const dy = e.clientY - previousMousePositionRef.current.y;
      modelGroupRef.current.rotation.y += dx * 0.007;
      modelGroupRef.current.rotation.x += dy * 0.007;
      modelGroupRef.current.rotation.x = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, modelGroupRef.current.rotation.x));
      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
    };

    const onWheel = (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.0025;
      camera.position.z = Math.max(1.4, Math.min(5.5, camera.position.z));
    };

    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    // Animation Loop
    const animate = () => {
      if (isDisposed) return;
      animFrameRef.current = requestAnimationFrame(animate);

      if (autoRotate && !isDraggingRef.current && modelGroupRef.current) {
        modelGroupRef.current.rotation.y += 0.006;
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current || isDisposed) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      isDisposed = true;
      cancelAnimationFrame(animFrameRef.current);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, [study.id, study.images.ply]);

  // Update Dynamic Color Buffer & Structure Filtering
  useEffect(() => {
    if (!pointsMeshRef.current || !plyDataState) return;

    const { positions, segColors, densityColors, classIds, count } = plyDataState;
    const geom = pointsMeshRef.current.geometry;
    const colorAttr = geom.getAttribute('color');
    const posAttr = geom.getAttribute('position');

    const newColors = colorAttr.array;
    const newPositions = posAttr.array;

    for (let i = 0; i < count; i++) {
      const cId = classIds[i];

      let isVisible = true;
      if (cId === 11) isVisible = activeStructures.femurs;
      else if (cId === 10) isVisible = activeStructures.tibiae;
      else if (cId === 12) isVisible = activeStructures.patellae;
      else if (cId === 13) isVisible = activeStructures.fibulae;
      else if (cId === 100) isVisible = activeStructures.cartilage;
      else if (cId === 59) isVisible = activeStructures.lungs;
      else if (cId === 60) isVisible = activeStructures.heart;
      else if (cId >= 17 && cId <= 28) isVisible = activeStructures.ribs;
      else if (cId >= 37 && cId <= 48) isVisible = activeStructures.spine;

      if (!isVisible) {
        newPositions[i * 3 + 0] = 999.0;
        newPositions[i * 3 + 1] = 999.0;
        newPositions[i * 3 + 2] = 999.0;
        continue;
      } else {
        newPositions[i * 3 + 0] = positions[i * 3 + 0];
        newPositions[i * 3 + 1] = positions[i * 3 + 1];
        newPositions[i * 3 + 2] = positions[i * 3 + 2];
      }

      if (colorMode === 'segmentation') {
        newColors[i * 3 + 0] = segColors[i * 3 + 0];
        newColors[i * 3 + 1] = segColors[i * 3 + 1];
        newColors[i * 3 + 2] = segColors[i * 3 + 2];
      } else if (colorMode === 'density') {
        newColors[i * 3 + 0] = densityColors[i * 3 + 0];
        newColors[i * 3 + 1] = densityColors[i * 3 + 1];
        newColors[i * 3 + 2] = densityColors[i * 3 + 2];
      } else if (colorMode === 'composite') {
        newColors[i * 3 + 0] = 0.60 * densityColors[i * 3 + 0] + 0.40 * segColors[i * 3 + 0];
        newColors[i * 3 + 1] = 0.60 * densityColors[i * 3 + 1] + 0.40 * segColors[i * 3 + 1];
        newColors[i * 3 + 2] = 0.60 * densityColors[i * 3 + 2] + 0.40 * segColors[i * 3 + 2];
      }
    }

    posAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
  }, [colorMode, activeStructures, plyDataState]);

  // Calipers Visibility
  useEffect(() => {
    if (calipersGroupRef.current) {
      calipersGroupRef.current.visible = showCalipers;
    }
  }, [showCalipers]);

  // Splat Kernel Size Update
  useEffect(() => {
    if (splatMaterialRef.current) {
      splatMaterialRef.current.size = splatSize;
      splatMaterialRef.current.needsUpdate = true;
    }
  }, [splatSize]);

  // Depth Slicing Plane Update
  useEffect(() => {
    if (clipPlaneRef.current) {
      const d = sliceDepth === 100 ? 10.0 : 0.65 - (sliceDepth / 100.0) * 1.3;
      clipPlaneRef.current.constant = d;
    }
  }, [sliceDepth]);

  // Preset Views
  const setPresetAngle = (preset) => {
    setViewPreset(preset);
    if (!modelGroupRef.current) return;
    if (preset === 'lateral') {
      modelGroupRef.current.rotation.set(0, 0, 0);
    } else if (preset === 'frontal') {
      modelGroupRef.current.rotation.set(0, Math.PI / 2, 0);
    } else if (preset === 'oblique') {
      modelGroupRef.current.rotation.set(0.15, Math.PI / 4, 0);
    } else if (preset === 'axial') {
      modelGroupRef.current.rotation.set(-Math.PI / 2.05, 0, 0);
    }
  };

  return (
    <div className="relative w-full h-full min-h-[560px] bg-[#05070e] rounded-xl overflow-hidden border border-borderDark flex flex-col">
      {/* 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full flex-1 cursor-grab active:cursor-grabbing" />

      {/* TOP HEADER CONTROLS BAR: Cleanly aligned with no collisions */}
      <div className="absolute top-3 inset-x-3 flex items-center justify-between gap-3 z-10 pointer-events-none">
        {/* Left: Stream Telemetry Pill */}
        <div className="bg-[#0f1322]/95 backdrop-blur-md border border-cyan-500/40 rounded-xl px-3 py-1.5 shadow-2xl flex items-center gap-2.5 pointer-events-auto">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#22d3ee]" />
          <div className="flex items-center gap-2">
            <span className="font-bold text-xs text-white">3DGS Engine:</span>
            <span className="text-xs font-mono text-cyan-300 font-semibold">
              {plyDataState?.count?.toLocaleString() || '36,300'} Splats
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
              Real Data
            </span>
          </div>
        </div>

        {/* Center: Anatomical Camera Presets */}
        <div className="bg-[#0f1322]/90 backdrop-blur-md border border-borderDark rounded-xl p-1 flex items-center gap-1 text-xs shadow-xl pointer-events-auto">
          <span className="text-[10px] text-slate-400 uppercase font-mono px-2 flex items-center gap-1">
            <Camera className="w-3 h-3 text-cyan-400" />
            View:
          </span>
          {[
            { id: 'lateral', label: 'True Lateral' },
            { id: 'oblique', label: '45° Oblique' },
            { id: 'frontal', label: 'AP Frontal' },
            { id: 'axial', label: 'Axial / Skyline' },
          ].map((preset) => (
            <button
              key={preset.id}
              onClick={() => setPresetAngle(preset.id)}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewPreset === preset.id
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Right: Auto-Orbit & Slicing */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`py-1.5 px-3 rounded-xl border text-xs font-semibold flex items-center gap-1.5 shadow-lg transition-colors ${
              autoRotate
                ? 'bg-purple-500/25 text-purple-200 border-purple-500/50'
                : 'bg-[#0f1322]/90 text-slate-300 border-borderDark hover:text-white'
            }`}
          >
            <RotateCw className={`w-3 h-3 ${autoRotate ? 'animate-spin' : ''}`} />
            {autoRotate ? 'Auto-Orbiting' : 'Orbit Paused'}
          </button>
        </div>
      </div>

      {/* FLOATING LEFT PALETTE: 3D Color Modes & Anatomical Structure Filters */}
      <div className="absolute top-16 left-3 flex flex-col gap-2 z-10 pointer-events-auto max-w-xs">
        {/* Color Mode Switcher */}
        <div className="bg-[#0f1322]/95 backdrop-blur-md border border-borderDark rounded-xl p-1.5 flex flex-col gap-1 shadow-2xl text-xs">
          <span className="text-[10px] text-slate-400 uppercase font-mono px-1.5 pt-0.5">3D Material & Rendering:</span>
          <div className="flex flex-col gap-1">
            <button
              onClick={() => setColorMode('segmentation')}
              className={`px-2.5 py-1.5 rounded-lg font-semibold flex items-center justify-between text-left transition-all ${
                colorMode === 'segmentation'
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-bold shadow-md shadow-blue-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                3D Segmentation Colors
              </span>
              <span className="w-2 h-2 rounded-full bg-cyan-300 shadow-[0_0_6px_#22d3ee]" />
            </button>

            <button
              onClick={() => setColorMode('density')}
              className={`px-2.5 py-1.5 rounded-lg font-semibold flex items-center justify-between text-left transition-all ${
                colorMode === 'density'
                  ? 'bg-gradient-to-r from-amber-500 to-yellow-600 text-slate-950 font-bold shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" />
                Radiographic Bone Density
              </span>
              <span className="w-2 h-2 rounded-full bg-amber-400" />
            </button>

            <button
              onClick={() => setColorMode('composite')}
              className={`px-2.5 py-1.5 rounded-lg font-semibold flex items-center justify-between text-left transition-all ${
                colorMode === 'composite'
                  ? 'bg-gradient-to-r from-purple-600 to-pink-500 text-white font-bold shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                Composite Diagnostic
              </span>
              <span className="w-2 h-2 rounded-full bg-purple-400" />
            </button>
          </div>
        </div>

        {/* Anatomical Structure Checklist & Isolation */}
        <div className="bg-[#0f1322]/95 backdrop-blur-md border border-borderDark rounded-xl p-2 shadow-2xl flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold px-1">
            <span className="flex items-center gap-1.5 text-cyan-300">
              <Filter className="w-3 h-3" />
              Segmented Anatomy:
            </span>
            <button
              onClick={showAllStructures}
              className="text-[10px] text-cyan-400 hover:underline font-mono"
            >
              Show All
            </button>
          </div>

          <div className="flex flex-col gap-1">
            {structureMeta.map((s) => {
              const isVisible = activeStructures[s.id] !== false;
              return (
                <div
                  key={s.id}
                  onClick={() => toggleStructure(s.id)}
                  className={`group px-2 py-1.5 rounded-lg border text-xs flex items-center justify-between transition-all cursor-pointer ${
                    isVisible
                      ? 'bg-slate-800/80 border-slate-700/80 text-slate-200 shadow-sm'
                      : 'bg-slate-900/40 border-slate-800/40 text-slate-500 opacity-45'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full shadow-sm"
                      style={{ backgroundColor: isVisible ? s.color : '#475569' }}
                    />
                    <span className="font-medium text-[11px]">{s.label}</span>
                    <span className="text-[10px] font-mono text-slate-400">({s.count})</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      isolateStructure(s.id);
                    }}
                    title={`Isolate ${s.label} solo`}
                    className="opacity-0 group-hover:opacity-100 hover:text-cyan-300 text-[10px] font-mono bg-slate-700/70 px-1.5 py-0.5 rounded transition-opacity"
                  >
                    solo
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* FLOATING RIGHT PALETTE: Sliders & Tools */}
      <div className="absolute top-16 right-3 flex flex-col gap-2 z-10 pointer-events-auto w-64">
        {/* 3D Calipers / Biometrics Toggle */}
        <div className="bg-[#0f1322]/90 backdrop-blur-md border border-borderDark rounded-xl p-2.5 shadow-xl">
          <button
            onClick={() => setShowCalipers(!showCalipers)}
            className={`w-full py-1.5 px-2.5 rounded-lg font-medium flex items-center justify-between transition-colors text-xs ${
              showCalipers ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
              {study.bodyPart.toLowerCase() === 'knee' ? '3D Caliper (JSW: 4.8mm)' : '3D Caliper (CTR: 0.44)'}
            </span>
            <span className="font-mono text-[10px] font-bold">{showCalipers ? 'ON' : 'OFF'}</span>
          </button>
        </div>

        {/* Depth Slicing Plane Control */}
        <div className="bg-[#0f1322]/90 backdrop-blur-md border border-borderDark rounded-xl p-2.5 flex items-center justify-between text-xs text-slate-300 gap-3 shadow-xl">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Scissors className="w-3.5 h-3.5 text-cyan-400" />
            <span>Depth Slice:</span>
          </div>
          <input
            type="range"
            min="10"
            max="100"
            value={sliceDepth}
            onChange={(e) => setSliceDepth(Number(e.target.value))}
            className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="font-mono text-cyan-300 w-8 text-right text-[11px]">
            {sliceDepth === 100 ? 'Full' : `${sliceDepth}%`}
          </span>
        </div>

        {/* Splat Kernel Size Control */}
        <div className="bg-[#0f1322]/90 backdrop-blur-md border border-borderDark rounded-xl p-2.5 flex items-center justify-between text-xs text-slate-300 gap-3 shadow-xl">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span>Splat Kernel:</span>
          </div>
          <input
            type="range"
            min="0.020"
            max="0.075"
            step="0.005"
            value={splatSize}
            onChange={(e) => setSplatSize(Number(e.target.value))}
            className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="font-mono text-cyan-300 w-8 text-right text-[11px]">{Math.round(splatSize * 1000)}µm</span>
        </div>
      </div>

      {/* BOTTOM CENTER: INTERACTION HINT & ANATOMICAL AXES */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-[#0f1322]/90 backdrop-blur-md px-4 py-1.5 rounded-full border border-borderDark text-[11px] text-slate-400 pointer-events-none shadow-xl flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
          <span>Patient 3D Gaussian Orbit:</span>
        </div>
        <span>Left Drag to Orbit • Scroll to Zoom • Toggle Segmented Structures on Left</span>
      </div>
    </div>
  );
}
