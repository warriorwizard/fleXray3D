import React, { useState, useRef } from 'react';
import { Sliders, Split, Maximize2, ShieldCheck, Download } from 'lucide-react';

export default function VDESSplitViewer({ study }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'strip' | 'soft' | 'bone'
  const isDraggingRef = useRef(false);
  const containerRef = useRef(null);

  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleMouseMove = (e) => {
    if (!isDraggingRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = Math.max(5, Math.min(95, (x / rect.width) * 100));
    setSliderPosition(percent);
  };

  const softImg = study.images.softTissue;
  const boneImg = study.images.bone;
  const stripImg = study.images.strip;

  return (
    <div className="w-full h-full flex flex-col gap-3">
      {/* Top Controls Bar */}
      <div className="flex items-center justify-between bg-surface border border-borderDark px-4 py-2.5 rounded-xl">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
            <Split className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Virtual Dual-Energy Subtraction (VDES)</h3>
            <p className="text-[11px] text-slate-400">Physics-conditioned material decomposition</p>
          </div>
        </div>

        {/* View Mode Tabs */}
        <div className="flex items-center gap-1 bg-surfaceLight border border-borderDark p-1 rounded-lg text-xs">
          {[
            { id: 'split', label: 'Interactive Split Wipe' },
            { id: 'strip', label: '3-Panel Strip' },
            { id: 'soft', label: 'Soft Tissue (Bone Cancelled)' },
            { id: 'bone', label: 'Bone Only (Soft Cancelled)' },
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id)}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                viewMode === mode.id
                  ? 'bg-cyan-500 text-slate-950 font-semibold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Comparison Canvas */}
      <div className="flex-1 min-h-[460px] bg-[#090a0f] border border-borderDark rounded-xl overflow-hidden relative flex items-center justify-center p-3 select-none">
        {viewMode === 'split' && (
          <div
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className="relative h-full max-h-[680px] w-full max-w-[560px] aspect-[4/5] rounded-xl overflow-hidden border border-borderDark/80 shadow-2xl cursor-col-resize bg-black"
          >
            {/* Background Layer: Bone-Suppressed Soft Tissue */}
            <img
              src={softImg}
              alt="Soft Tissue VDES"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none filter contrast-125"
            />

            {/* Foreground Layer with Clip-path: Soft-Suppressed Bone */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ width: `${sliderPosition}%` }}
            >
              <img
                src={boneImg}
                alt="Bone VDES"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none filter contrast-125"
                style={{ width: containerRef.current?.clientWidth || '100%', maxWidth: 'none' }}
              />
            </div>

            {/* Draggable Divider Handle */}
            <div
              onMouseDown={handleMouseDown}
              className="absolute top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.8)] cursor-ew-resize flex items-center justify-center"
              style={{ left: `${sliderPosition}%` }}
            >
              <div className="w-7 h-7 rounded-full bg-[#11131c] border-2 border-cyan-400 flex items-center justify-center shadow-xl text-cyan-300">
                <Sliders className="w-3.5 h-3.5" />
              </div>
            </div>

            {/* Badges */}
            <div className="absolute top-3 left-3 bg-[#11131c]/90 backdrop-blur-md px-2.5 py-1 rounded border border-borderDark text-xs font-semibold text-cyan-300 pointer-events-none">
              Soft-Suppressed Bone
            </div>
            <div className="absolute top-3 right-3 bg-[#11131c]/90 backdrop-blur-md px-2.5 py-1 rounded border border-borderDark text-xs font-semibold text-emerald-300 pointer-events-none">
              Bone-Suppressed Soft View
            </div>

            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-[#11131c]/80 backdrop-blur px-3 py-1 rounded-full border border-borderDark text-[11px] text-slate-300 pointer-events-none">
              Drag slider left or right to wipe
            </div>
          </div>
        )}

        {viewMode === 'strip' && (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <img
              src={stripImg}
              alt="3-Panel Comparison Strip"
              className="max-h-[460px] w-auto object-contain rounded-lg border border-borderDark shadow-2xl"
            />
            <div className="flex justify-between w-full max-w-[850px] mt-2 text-xs font-mono text-slate-400 px-4">
              <span>(A) Total Radiograph</span>
              <span>(B) Soft Tissue (Bone Cancelled)</span>
              <span>(C) Skeletal Bone Only</span>
            </div>
          </div>
        )}

        {viewMode === 'soft' && (
          <div className="relative max-h-[460px] aspect-square">
            <img
              src={softImg}
              alt="Pure Soft Tissue View"
              className="w-full h-full object-contain rounded-lg border border-borderDark shadow-2xl"
            />
            <div className="absolute bottom-3 left-3 bg-[#11131c]/90 backdrop-blur px-3 py-1.5 rounded-lg border border-borderDark text-xs text-emerald-300 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4" />
              Rib & Clavicle Shadows Cancelled — Parenchymal Field Unmasked
            </div>
          </div>
        )}

        {viewMode === 'bone' && (
          <div className="relative max-h-[460px] aspect-square">
            <img
              src={boneImg}
              alt="Pure Bone View"
              className="w-full h-full object-contain rounded-lg border border-borderDark shadow-2xl"
            />
            <div className="absolute bottom-3 left-3 bg-[#11131c]/90 backdrop-blur px-3 py-1.5 rounded-lg border border-borderDark text-xs text-cyan-300 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4" />
              Soft Tissue Attenuation Cancelled — Cortical Trabeculae Isolated
            </div>
          </div>
        )}
      </div>

      {/* Clinical Notes Banner */}
      <div className="bg-surface border border-borderDark p-3 rounded-xl flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className="text-slate-400">Clinical Impact:</span>
          <span className="text-slate-200">
            {study.bodyPart === 'Knee'
              ? 'Distal femoral condyles and proximal tibial plateaus unmasked with 0% soft-tissue attenuation overlap.'
              : 'Overlapping thoracic ribs 1–12 and clavicles mathematically subtracted to unmask occult apical/retrocardiac lung nodules.'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-emerald-400 font-mono font-medium">CNR Gain: +210%</span>
        </div>
      </div>
    </div>
  );
}

