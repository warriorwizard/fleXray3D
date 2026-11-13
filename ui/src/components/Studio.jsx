import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ArrowLeft,
  Activity,
  Layers,
  Crosshair,
  Sliders,
  Sun,
  Eye,
  EyeOff,
  CheckCircle2,
  Sparkles,
  Search,
  Filter,
  Maximize2,
  RefreshCw,
  Info,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  Box,
  LayoutGrid,
  Columns,
  Square,
  AlertTriangle
} from 'lucide-react';
import Volume3DViewer from './Volume3DViewer';

export default function Studio({ study, onBackToWorklist }) {
  const [threshold, setThreshold] = useState(0.45);
  const [overlayOpacity, setOverlayOpacity] = useState(60);
  const [isInverted, setIsInverted] = useState(false);
  const [showBoundaries, setShowBoundaries] = useState(true);

  // Volumetric vs 2D Planar Layout State
  const isVolumetric = !!study.isVolumetric;
  const [viewMode, setViewMode] = useState(isVolumetric ? 'split' : 'triptych'); // 'split' | '3d_only' | 'drr_only' | 'triptych'
  const [projectionMode, setProjectionMode] = useState('flexray'); // 'flexray' | 'ground_truth'
  const [drrLayoutMode, setDrrLayoutMode] = useState('triptych'); // 'triptych' | 'duo' | 'focus'
  const [focusTab, setFocusTab] = useState('mosaic'); // 'raw' | 'mosaic' | 'single'
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Active Data State
  const [isLoading, setIsLoading] = useState(true);
  const [studyData, setStudyData] = useState(null);
  const [selectedStructure, setSelectedStructure] = useState(null);
  const [focusImageUrl, setFocusImageUrl] = useState(null);
  const [isLoadingFocus, setIsLoadingFocus] = useState(false);
  const [searchAnatomy, setSearchAnatomy] = useState('');
  const [collapsedCategories, setCollapsedCategories] = useState({});
  const [showActiveOnly, setShowActiveOnly] = useState(false);

  const selectedStructureRef = useRef(selectedStructure);
  useEffect(() => {
    selectedStructureRef.current = selectedStructure;
  }, [selectedStructure]);

  const projectionModeRef = useRef(projectionMode);
  useEffect(() => {
    projectionModeRef.current = projectionMode;
  }, [projectionMode]);

  // 3D Live DRR Sync State
  const [currentPose, setCurrentPose] = useState({
    yaw: 0.0,
    pitch: 0.0,
    roll: 0.0,
    sdd: 1200.0,
    viewName: 'AP Frontal',
  });
  const [isProjecting, setIsProjecting] = useState(false);
  const [drrData, setDrrData] = useState(null);
  const activeReqIdRef = useRef(0);

  // Reset when switching studies
  useEffect(() => {
    setSelectedStructure(null);
    selectedStructureRef.current = null;
    setFocusImageUrl(null);
    setDrrData(null);
    setStudyData(null);
    setIsLoading(true);
    setCurrentPose({ yaw: 0.0, pitch: 0.0, roll: 0.0, sdd: 1200.0, viewName: 'AP Frontal' });
  }, [study.id]);

  // Initial load or threshold change for 2D and initial volume metadata
  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    fetch(`/api/study/${study.id}?threshold=${threshold}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        setStudyData(data);
        setIsLoading(false);

        // Auto-select primary detected structure
        if (data.structures && data.structures.length > 0) {
          const currentTarget = selectedStructureRef.current;
          if (!currentTarget) {
            const first = data.structures[0].name;
            setSelectedStructure(first);
            selectedStructureRef.current = first;
            if (data.focusImageUrl) {
              setFocusImageUrl(data.focusImageUrl);
            } else if (!isVolumetric) {
              fetch(`/api/structure_focus/${study.id}/${first}?threshold=${threshold}`)
                .then((r) => r.json())
                .then((fData) => {
                  if (fData.focusImageUrl) setFocusImageUrl(fData.focusImageUrl);
                })
                .catch(console.error);
            }
          } else {
            if (data.focusImageUrl && currentTarget === data.structures[0].name) {
              setFocusImageUrl(data.focusImageUrl);
            } else if (!isVolumetric) {
              fetch(`/api/structure_focus/${study.id}/${currentTarget}?threshold=${threshold}`)
                .then((r) => r.json())
                .then((fData) => {
                  if (fData.focusImageUrl) setFocusImageUrl(fData.focusImageUrl);
                })
                .catch(console.error);
            }
          }
        }
      })
      .catch((err) => {
        console.error('Failed to load study details:', err);
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [study.id, threshold]);

  // Arbitrary-angle DRR projection function (called when camera stops orbiting or mode changes)
  const triggerDrrProjection = useCallback((pose, targetStructure = null, mode = null) => {
    if (!isVolumetric) return;

    const actualStructure = targetStructure || selectedStructureRef.current || selectedStructure || studyData?.structures?.[0]?.name;
    const actualMode = mode || projectionModeRef.current || projectionMode;

    const reqId = ++activeReqIdRef.current;
    setIsProjecting(true);

    const payload = {
      study_id: study.id,
      yaw: pose.yaw,
      pitch: pose.pitch,
      roll: pose.roll || 0.0,
      sdd: pose.sdd || 1200.0,
      height: 512,
      width: 512,
      n_samples: 300,
      seg_threshold: 0.05,
      threshold,
      mode: actualMode,
      focus_structure: actualStructure,
    };

    fetch('/api/project_drr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (reqId !== activeReqIdRef.current) return;
        setDrrData(data);
        setIsProjecting(false);
      })
      .catch((err) => {
        console.error('DRR projection failed:', err);
        if (reqId === activeReqIdRef.current) setIsProjecting(false);
      });
  }, [isVolumetric, study.id, selectedStructure, studyData, projectionMode, threshold]);

  // Handle 3D camera change from Volume3DViewer
  const handleCameraChange = useCallback((pose) => {
    setCurrentPose(pose);
    const target = selectedStructureRef.current || selectedStructure || studyData?.structures?.[0]?.name;
    triggerDrrProjection(pose, target);
  }, [triggerDrrProjection, selectedStructure, studyData]);

  // Handle structure selection
  const handleSelectStructure = (structureName) => {
    setSelectedStructure(structureName);
    selectedStructureRef.current = structureName;
    if (isVolumetric) {
      triggerDrrProjection(currentPose, structureName);
    } else {
      setIsLoadingFocus(true);
      fetch(`/api/structure_focus/${study.id}/${structureName}?threshold=${threshold}`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data) => {
          setFocusImageUrl(data.focusImageUrl);
          setIsLoadingFocus(false);
        })
        .catch((err) => {
          console.error('Failed to fetch structure focus:', err);
          setIsLoadingFocus(false);
        });
    }
  };

  // Toggle category in taxonomy browser
  const toggleCategory = (cat) => {
    setCollapsedCategories((prev) => ({
      ...prev,
      [cat]: !prev[cat],
    }));
  };

  // Active structures: either from dynamic DRR projection or static 2D
  const activeStructures = isVolumetric && drrData?.structures ? drrData.structures : (studyData?.structures || []);

  const detectedMap = React.useMemo(() => {
    const map = {};
    activeStructures.forEach((s) => {
      map[s.name] = s;
    });
    return map;
  }, [activeStructures]);

  const currentFocusMeta = selectedStructure ? detectedMap[selectedStructure] : null;

  // Active Image Assets
  const activeBaseImage = isVolumetric && drrData?.drrImageUrl ? drrData.drrImageUrl : studyData?.baseImageUrl;
  const activeOverlayImage = isVolumetric && drrData?.overlayImageUrl ? drrData.overlayImageUrl : studyData?.overlayImageUrl;
  const activeFocusImage = isVolumetric && drrData?.focusImageUrl ? drrData.focusImageUrl : (focusImageUrl || activeOverlayImage);

  const renderPanel1 = () => (
    <div className="bg-[#0e111a] border border-slate-800/80 rounded-xl overflow-hidden flex flex-col shadow-xl min-w-0 min-h-0 flex-1">
      {drrLayoutMode !== 'focus' && (
        <div className="px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between gap-2 min-w-0">
          <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5 truncate">
            <span className="w-2 h-2 rounded-full bg-slate-400 flex-shrink-0" />
            <span className="truncate">{isVolumetric ? '1. Physics DRR' : '1. Input Radiograph'}</span>
          </span>
          <span className="text-[10px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 flex-shrink-0 whitespace-nowrap">
            {isVolumetric ? 'Cone-Beam' : 'VOI LUT'}
          </span>
        </div>
      )}
      <div className="flex-1 p-1 flex items-center justify-center bg-[#05070c] relative overflow-hidden min-h-0 min-w-0">
        {isLoading || isProjecting ? (
          <div className="flex flex-col items-center gap-2 text-slate-500 text-xs">
            <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
            <span>{isVolumetric ? 'Ray-marching DRR on CUDA...' : 'Loading radiograph...'}</span>
          </div>
        ) : (
          <img
            src={activeBaseImage}
            alt="Base Radiograph"
            className={`w-full h-full object-contain rounded select-none pointer-events-none transition-all ${
              isInverted ? 'filter invert contrast-125' : 'filter contrast-110'
            }`}
          />
        )}
        <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur-md text-[10px] font-mono text-slate-400 border border-slate-800">
          {isVolumetric ? `${currentPose.yaw}° / ${currentPose.pitch}°` : study.view}
        </div>
      </div>
    </div>
  );

  const renderPanel2 = () => (
    <div className="bg-[#0e111a] border border-slate-800/80 rounded-xl overflow-hidden flex flex-col shadow-xl min-w-0 min-h-0 flex-1">
      {drrLayoutMode !== 'focus' && (
        <div className="px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between gap-2 min-w-0">
          <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5 truncate">
            <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#22d3ee] flex-shrink-0" />
            <span className="truncate">2. All Labels (Mosaic)</span>
          </span>
          <span className="text-[10px] font-mono text-cyan-300 px-1.5 py-0.5 rounded bg-cyan-950/40 border border-cyan-800/40 flex-shrink-0 whitespace-nowrap">
            {activeStructures.length} active
          </span>
        </div>
      )}
      <div className="flex-1 p-1 flex items-center justify-center bg-[#05070c] relative overflow-hidden min-h-0 min-w-0">
        {isLoading || isProjecting ? (
          <div className="flex flex-col items-center gap-2 text-slate-500 text-xs">
            <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
            <span>Segmenting structures...</span>
          </div>
        ) : (
          <img
            src={activeOverlayImage}
            alt="Multilabel Mosaic"
            style={{ opacity: overlayOpacity / 100 }}
            className={`w-full h-full object-contain rounded select-none pointer-events-none transition-all ${
              isInverted ? 'filter invert' : ''
            }`}
          />
        )}
        <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur-md text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
          {projectionMode === 'ground_truth' ? '3D GT' : `${activeStructures.length} Structures Active`}
        </div>
      </div>
    </div>
  );

  const renderPanel3 = () => (
    <div className="bg-[#0e111a] border border-slate-800/80 rounded-xl overflow-hidden flex flex-col shadow-xl min-w-0 min-h-0 flex-1">
      {drrLayoutMode !== 'focus' && (
        <div className="px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between gap-2 min-w-0">
          <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5 truncate">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: currentFocusMeta?.hexColor || '#38bdf8' }}
            />
            <span className="truncate">3. {currentFocusMeta?.label || selectedStructure || 'Structure Focus'}</span>
          </span>
          {currentFocusMeta && (
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-1.5 py-0.5 rounded flex-shrink-0 whitespace-nowrap">
              {Math.round(currentFocusMeta.maxConfidence * 100)}%
            </span>
          )}
        </div>
      )}
      <div className="flex-1 p-1 flex items-center justify-center bg-[#05070c] relative overflow-hidden min-h-0 min-w-0">
        {isLoadingFocus || isProjecting ? (
          <div className="flex flex-col items-center gap-2 text-slate-500 text-xs">
            <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
            <span>Isolating {selectedStructure}...</span>
          </div>
        ) : (
          <img
            src={activeFocusImage}
            alt="Single Structure Focus"
            className={`w-full h-full object-contain rounded select-none pointer-events-none transition-all ${
              isInverted ? 'filter invert' : ''
            }`}
          />
        )}

        {/* Structure Statistics Box */}
        {currentFocusMeta && (
          <div className="absolute bottom-2 inset-x-2 bg-slate-950/85 backdrop-blur-md border border-slate-800/90 rounded-lg p-2 flex items-center justify-between text-[11px] font-mono shadow-2xl">
            <div className="flex items-center gap-2 truncate">
              <span className="text-slate-400">Coverage:</span>
              <strong className="text-cyan-300">{currentFocusMeta.coveragePercent}%</strong>
              <span className="text-slate-500 hidden sm:inline">({currentFocusMeta.pixelCount?.toLocaleString()} px)</span>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-slate-400">Mean Prob:</span>
              <strong className="text-emerald-400">
                {Math.round((currentFocusMeta.meanConfidence || currentFocusMeta.maxConfidence) * 100)}%
              </strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex-1 flex flex-col w-full h-full min-h-0 min-w-0 bg-[#08090e] overflow-hidden">
      {/* Top Patient Header */}
      <div className="bg-[#0e111a] border-b border-slate-800/80 px-3 py-1 flex items-center justify-between gap-2 shrink-0 min-w-0 overflow-x-auto no-scrollbar">
        <div className="flex items-center gap-2 shrink-0 min-w-0">
          <button
            onClick={onBackToWorklist}
            className="px-2 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 transition-colors flex items-center gap-1.5 text-xs font-semibold shadow-sm shrink-0"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Worklist</span>
          </button>

          <div className="h-4 w-px bg-slate-800 shrink-0" />

          <div className="flex items-center gap-1.5 min-w-0">
            <h2 className="text-xs sm:text-sm font-bold text-slate-100 truncate max-w-[120px] sm:max-w-[200px] md:max-w-xs">{study.patientName}</h2>
            <span className="text-[10px] sm:text-[11px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 shrink-0">
              {study.mrn}
            </span>
            <span className="text-[10px] sm:text-[11px] px-1.5 py-0.5 rounded-md font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shrink-0 whitespace-nowrap">
              {study.modality} • {study.bodyPart}
            </span>
            {study.modality === 'MG' && (
              <span className="text-[10px] sm:text-[11px] px-1.5 py-0.5 rounded-md font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30 flex items-center gap-1 font-mono shrink-0" title="FleXray's 60-structure taxonomy was trained exclusively on whole-body CT/X-ray (chest, spine, limbs). Mammography activations represent out-of-domain evaluation.">
                <AlertTriangle className="w-3 h-3 text-amber-400" />
                <span className="hidden lg:inline">Out-of-Domain Modality</span>
              </span>
            )}
            {isVolumetric && (
              <span className="text-[10px] sm:text-[11px] px-1.5 py-0.5 rounded-md font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30 flex items-center gap-1 font-mono shrink-0">
                <Box className="w-3 h-3" />
                <span className="hidden md:inline">3D PACS</span>
              </span>
            )}
          </div>
        </div>

        {/* Global Controls & Layout Toggles */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Volumetric Viewport Mode Toggle */}
          {isVolumetric && (
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs shrink-0">
              <button
                onClick={() => setViewMode('split')}
                className={`px-2 py-1 rounded-md font-semibold flex items-center gap-1 text-[11px] transition-all ${
                  viewMode === 'split'
                    ? 'bg-cyan-500 text-black shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Columns className="w-3 h-3" />
                <span>Dual</span>
              </button>
              <button
                onClick={() => setViewMode('3d_only')}
                className={`px-2 py-1 rounded-md font-semibold flex items-center gap-1 text-[11px] transition-all ${
                  viewMode === '3d_only'
                    ? 'bg-cyan-500 text-black shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Box className="w-3 h-3" />
                <span>3D Only</span>
              </button>
              <button
                onClick={() => setViewMode('drr_only')}
                className={`px-2 py-1 rounded-md font-semibold flex items-center gap-1 text-[11px] transition-all ${
                  viewMode === 'drr_only'
                    ? 'bg-cyan-500 text-black shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <LayoutGrid className="w-3 h-3" />
                <span>2D DRR</span>
              </button>
            </div>
          )}

          {/* Probability Threshold Slider */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 shrink-0">
            <span className="text-slate-400 text-[10px] font-mono">τ:</span>
            <input
              type="range"
              min="0.20"
              max="0.80"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-12 sm:w-14 accent-cyan-400 cursor-pointer h-1 bg-slate-800 rounded-lg"
            />
            <span className="font-mono text-cyan-300 font-bold text-[10px] w-7 text-right">
              {Math.round(threshold * 100)}%
            </span>
          </div>

          {/* Overlay Opacity Slider */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 shrink-0">
            <span className="text-slate-400 text-[10px] font-mono">Opacity:</span>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseInt(e.target.value, 10))}
              className="w-12 sm:w-14 accent-cyan-400 cursor-pointer h-1 bg-slate-800 rounded-lg"
            />
            <span className="font-mono text-cyan-300 font-bold text-[10px] w-7 text-right">
              {overlayOpacity}%
            </span>
          </div>

          {/* Invert LUT Toggle */}
          <button
            onClick={() => setIsInverted(!isInverted)}
            className={`px-2 py-1 rounded-lg border text-[11px] font-semibold flex items-center gap-1 transition-colors shrink-0 ${
              isInverted
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <Sun className="w-3 h-3" />
            <span>Invert</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden min-h-0 min-w-0 w-full">
        {/* Volumetric Dual-Viewport Layout */}
        {isVolumetric && (viewMode === 'split' || viewMode === '3d_only') && (
          <div
            className={`border-r border-slate-800/80 relative flex flex-col min-h-0 min-w-0 ${
              viewMode === '3d_only' ? 'flex-1' : 'flex-1'
            }`}
          >
            <div className="flex-1 relative overflow-hidden bg-black min-h-0 min-w-0">
              <Volume3DViewer
                study={study}
                onCameraChange={handleCameraChange}
                activePose={currentPose}
                syncLoading={isProjecting}
              />
            </div>
          </div>
        )}

        {/* 2D Radiograph & Projection Panels */}
        {(!isVolumetric || viewMode === 'split' || viewMode === 'drr_only') && (
          <div className="flex-1 flex flex-col p-1.5 overflow-hidden gap-1.5 min-h-0 min-w-0">
            {/* Sleek Single-Line PACS Toolbar */}
            <div className="flex items-center justify-between gap-1.5 px-1 text-xs shrink-0 min-w-0">
              <div className="flex items-center gap-1.5 min-w-0 shrink-0">
                <span className="font-bold text-cyan-400 text-xs flex items-center gap-1 truncate">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 shrink-0" />
                  <span className="truncate">{isVolumetric ? `Synchronized DRR (${drrLayoutMode === 'triptych' ? '3-View' : drrLayoutMode === 'duo' ? 'Dual' : 'Focus'})` : `Clinical Radiograph (${study.view || study.modality})`}</span>
                </span>
                {drrData?.timing && (
                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono text-[10px] shrink-0">
                    {drrData.timing.cuda_render_ms}ms CUDA
                  </span>
                )}
                {isVolumetric && (
                  <span className="text-[10px] font-mono text-slate-400 hidden xl:inline shrink-0">
                    {currentPose.yaw}° / {currentPose.pitch}°
                  </span>
                )}
              </div>

              {/* View Layout Controls & Mode Switches */}
              <div className="flex items-center gap-1 shrink-0">
                {/* 2D DRR Layout Switcher */}
                <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[10px]">
                  <button
                    onClick={() => setDrrLayoutMode('focus')}
                    title="Single Large Focus View"
                    className={`px-1.5 py-0.5 rounded font-semibold flex items-center gap-1 transition-all ${
                      drrLayoutMode === 'focus'
                        ? 'bg-cyan-500 text-black shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    <Square className="w-2.5 h-2.5" />
                    <span>Focus</span>
                  </button>
                  <button
                    onClick={() => setDrrLayoutMode('duo')}
                    title="2-Panel Side-by-Side View"
                    className={`px-1.5 py-0.5 rounded font-semibold flex items-center gap-1 transition-all ${
                      drrLayoutMode === 'duo'
                        ? 'bg-cyan-500 text-black shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    <LayoutGrid className="w-2.5 h-2.5" />
                    <span>Duo</span>
                  </button>
                  <button
                    onClick={() => setDrrLayoutMode('triptych')}
                    title="3-Panel Triptych View"
                    className={`px-1.5 py-0.5 rounded font-semibold flex items-center gap-1 transition-all ${
                      drrLayoutMode === 'triptych'
                        ? 'bg-cyan-500 text-black shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    <Columns className="w-2.5 h-2.5" />
                    <span>Triptych</span>
                  </button>
                </div>

                {/* Sub-tabs when in Focus Mode */}
                {drrLayoutMode === 'focus' && (
                  <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[10px]">
                    <button
                      onClick={() => setFocusTab('mosaic')}
                      className={`px-1.5 py-0.5 rounded font-medium transition-all ${
                        focusTab === 'mosaic' ? 'bg-cyan-500 text-black font-semibold' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      All Labels
                    </button>
                    <button
                      onClick={() => setFocusTab('raw')}
                      className={`px-1.5 py-0.5 rounded font-medium transition-all ${
                        focusTab === 'raw' ? 'bg-cyan-500 text-black font-semibold' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {isVolumetric ? 'Raw DRR' : 'Raw'}
                    </button>
                    <button
                      onClick={() => setFocusTab('single')}
                      className={`px-1.5 py-0.5 rounded font-medium transition-all ${
                        focusTab === 'single' ? 'bg-cyan-500 text-black font-semibold' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Structure
                    </button>
                  </div>
                )}

                {/* Mode Switch for Volumetric Studies */}
                {isVolumetric && (
                  <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[10px]">
                    <button
                      onClick={() => {
                        setProjectionMode('flexray');
                        triggerDrrProjection(currentPose, selectedStructure, 'flexray');
                      }}
                      className={`px-1.5 py-0.5 rounded font-semibold transition-all ${
                        projectionMode === 'flexray'
                          ? 'bg-cyan-500 text-black shadow'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      FleXray
                    </button>
                    <button
                      onClick={() => {
                        setProjectionMode('ground_truth');
                        triggerDrrProjection(currentPose, selectedStructure, 'ground_truth');
                      }}
                      className={`px-1.5 py-0.5 rounded font-semibold transition-all ${
                        projectionMode === 'ground_truth'
                          ? 'bg-amber-500 text-black shadow'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      3D GT
                    </button>
                  </div>
                )}

                <div className="text-[10px] font-mono text-cyan-400 shrink-0 px-1">
                  {activeStructures.length} act
                </div>
              </div>
            </div>

            {/* Panels Display */}
            {drrLayoutMode === 'triptych' && (
              isVolumetric && viewMode === 'split' ? (
                /* In Split Dual-View mode: 2-Tier Stacked Layout (Tier 1: Top DRR, Tier 2: Mosaic + Focus side-by-side) */
                <div className="flex-1 flex flex-col gap-2 min-h-0 overflow-hidden">
                  <div className="flex-1 min-h-0 flex">
                    {renderPanel1()}
                  </div>
                  <div className="flex-1 min-h-0 grid grid-cols-2 gap-2">
                    {renderPanel2()}
                    {renderPanel3()}
                  </div>
                </div>
              ) : (
                /* In Full-width or 2D Radiograph mode: 3-column Triptych */
                <div className="flex-1 grid grid-cols-3 gap-2.5 min-h-0">
                  {renderPanel1()}
                  {renderPanel2()}
                  {renderPanel3()}
                </div>
              )
            )}
            {drrLayoutMode === 'duo' && (
              <div className="flex-1 grid grid-cols-2 gap-2.5 min-h-0">
                {renderPanel1()}
                {renderPanel2()}
              </div>
            )}
            {drrLayoutMode === 'focus' && (
              <div className="flex-1 flex flex-col min-h-0">
                {focusTab === 'raw' && renderPanel1()}
                {focusTab === 'mosaic' && renderPanel2()}
                {focusTab === 'single' && renderPanel3()}
              </div>
            )}
          </div>
        )}

        {/* Right Sidebar: 60-Anatomical Taxonomy Browser */}
        {viewMode !== '3d_only' && (
          isSidebarOpen ? (
            <div className="w-72 shrink-0 bg-[#0e111a] border-l border-slate-800/80 flex flex-col overflow-hidden min-h-0 transition-all duration-200">
            {/* Header */}
            <div className="p-3 border-b border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  60-Anatomy Taxonomy
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setShowActiveOnly(!showActiveOnly)}
                    title={showActiveOnly ? "Show all 60 anatomical structures" : "Filter to active detected structures only"}
                    className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
                      showActiveOnly
                        ? 'bg-cyan-500 text-black font-bold border-cyan-400 shadow'
                        : 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30 hover:bg-cyan-500/20'
                    }`}
                  >
                    {activeStructures.length} Active
                  </button>
                  <button
                    onClick={() => setIsSidebarOpen(false)}
                    title="Collapse Taxonomy Sidebar"
                    className="p-1 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Quick Filter Search */}
              <div className="relative mt-2">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Find structure (e.g. femur, rib)..."
                  value={searchAnatomy}
                  onChange={(e) => setSearchAnatomy(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-[#090b12] border border-slate-800 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Quick Filter & Collapse Controls */}
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/60 text-[10px]">
                <div className="flex items-center gap-1 bg-[#090b12] p-0.5 rounded-lg border border-slate-800">
                  <button
                    onClick={() => setShowActiveOnly(false)}
                    className={`px-2 py-0.5 rounded font-medium transition-colors ${
                      !showActiveOnly ? 'bg-cyan-500 text-black font-bold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    All (60)
                  </button>
                  <button
                    onClick={() => setShowActiveOnly(true)}
                    className={`px-2 py-0.5 rounded font-medium transition-colors ${
                      showActiveOnly ? 'bg-cyan-500 text-black font-bold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Active ({activeStructures.length})
                  </button>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <button
                    onClick={() => {
                      const allCollapsed = {};
                      Object.keys(studyData?.categories || {}).forEach((k) => { allCollapsed[k] = false; });
                      setCollapsedCategories(allCollapsed);
                    }}
                    title="Expand all categories"
                    className="hover:text-cyan-400 transition-colors"
                  >
                    Expand
                  </button>
                  <span className="text-slate-600">|</span>
                  <button
                    onClick={() => {
                      const allCollapsed = {};
                      Object.keys(studyData?.categories || {}).forEach((k) => { allCollapsed[k] = true; });
                      setCollapsedCategories(allCollapsed);
                    }}
                    title="Collapse all categories"
                    className="hover:text-cyan-400 transition-colors"
                  >
                    Collapse
                  </button>
                </div>
              </div>
            </div>

            {/* Categorized Structures List */}
            <div className="flex-1 overflow-y-auto p-2.5 flex flex-col gap-2.5">
              {studyData?.categories &&
                Object.entries(studyData.categories).map(([category, items]) => {
                  const isCollapsed = collapsedCategories[category];
                  let matchingItems = items.filter((name) =>
                    name.toLowerCase().includes(searchAnatomy.toLowerCase())
                  );
                  if (showActiveOnly) {
                    matchingItems = matchingItems.filter((name) => detectedMap[name]);
                  }
                  if (matchingItems.length === 0) return null;

                  const detectedCount = matchingItems.filter((name) => detectedMap[name]).length;

                  return (
                    <div key={category} className="shrink-0 border border-slate-800/80 rounded-xl overflow-hidden bg-slate-900/40">
                      <button
                        onClick={() => toggleCategory(category)}
                        className="w-full px-3 py-2 bg-slate-900/90 hover:bg-slate-800/80 flex items-center justify-between text-xs font-semibold text-slate-300 transition-colors"
                      >
                        <span className="flex items-center gap-1.5 truncate">
                          {isCollapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                          {category}
                        </span>
                        <span
                          className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                            detectedCount > 0
                              ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                              : 'text-slate-500'
                          }`}
                        >
                          {detectedCount} active
                        </span>
                      </button>

                      {!isCollapsed && (
                        <div className="p-1.5 flex flex-col gap-1">
                          {matchingItems.map((name) => {
                            const meta = detectedMap[name];
                            const isDetected = !!meta;
                            const isSelected = selectedStructure === name;
                            const hexColor = meta?.hexColor || '#475569';

                            return (
                              <button
                                key={name}
                                onClick={() => handleSelectStructure(name)}
                                className={`w-full shrink-0 px-2.5 py-1.5 rounded-lg text-xs flex items-center justify-between transition-all text-left ${
                                  isSelected
                                    ? 'bg-cyan-500/15 border border-cyan-500/50 text-white shadow'
                                    : isDetected
                                    ? 'hover:bg-slate-800/60 text-slate-200 border border-transparent'
                                    : 'opacity-40 hover:opacity-75 text-slate-500 border border-transparent'
                                }`}
                              >
                                <div className="flex items-center gap-2 truncate">
                                  <span
                                    className="w-2.5 h-2.5 rounded-full shadow-sm flex-shrink-0"
                                    style={{ backgroundColor: hexColor }}
                                  />
                                  <span className="capitalize truncate font-medium">
                                    {name.replace('_', ' ')}
                                  </span>
                                </div>

                                {isDetected ? (
                                  <span className="text-[10px] font-mono text-cyan-300 ml-1">
                                    {meta.coveragePercent}%
                                  </span>
                                ) : (
                                  <span className="text-[10px] font-mono text-slate-600">
                                    —
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        ) : (
          <div className="w-10 shrink-0 bg-[#0e111a] border-l border-slate-800/80 flex flex-col items-center py-3 gap-4 overflow-hidden select-none min-h-0 transition-all duration-200">
            <button
              onClick={() => setIsSidebarOpen(true)}
              title="Expand 60-Anatomy Taxonomy Sidebar"
              className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-cyan-400 border border-slate-800 hover:border-cyan-500/40 transition-colors shadow"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div
              onClick={() => setIsSidebarOpen(true)}
              className="flex-1 flex flex-col items-center justify-center cursor-pointer gap-2"
            >
              <Layers className="w-4 h-4 text-slate-500 hover:text-cyan-400" />
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider [writing-mode:vertical-rl] rotate-180 hover:text-slate-200">
                Taxonomy ({activeStructures.length})
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
