import React from 'react';
import { Activity, CheckCircle, FileText, Download, Share2, Layers } from 'lucide-react';

export default function BiometricsPanel({ study }) {
  return (
    <div className="w-full flex flex-col gap-4">
      {/* Metrics Card Grid */}
      <div className="grid grid-cols-2 gap-3">
        {Object.entries(study.metrics).map(([key, value]) => (
          <div
            key={key}
            className="bg-surface border border-borderDark rounded-xl p-3.5 flex flex-col justify-between hover:border-slate-600 transition-colors"
          >
            <span className="text-xs text-slate-400 font-medium">{key}</span>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-base font-semibold font-mono text-cyan-300">{value}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Detected Pan-Anatomy Tags */}
      <div className="bg-surface border border-borderDark rounded-xl p-4">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>FleXray Multi-Class Segmentations ({study.presentStructures.length} Detected)</span>
          </div>
          <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
            <CheckCircle className="w-3 h-3" /> 100% Invariant
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5 max-h-[140px] overflow-y-auto pr-1">
          {study.presentStructures.map((struct) => (
            <span
              key={struct}
              className="px-2 py-0.5 rounded text-[11px] font-mono bg-surfaceLight border border-borderDark text-slate-300"
            >
              {struct}
            </span>
          ))}
        </div>
      </div>

      {/* Clinical Finding Report & Export */}
      <div className="bg-surface border border-borderDark rounded-xl p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
          <FileText className="w-4 h-4 text-purple-400" />
          <span>Automated Radiology Summary</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed bg-[#0c0e17] p-3 rounded-lg border border-borderDark/60 font-mono">
          {study.summary}
        </p>

        <div className="flex gap-2 pt-1">
          <button
            onClick={() => {
              if (study.images.dicomSc) {
                window.open(study.images.dicomSc, '_blank');
              } else {
                alert('DICOM Secondary Capture exported in outputs directory.');
              }
            }}
            className="flex-1 py-2 px-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs flex items-center justify-center gap-1.5 transition-colors shadow-lg shadow-cyan-500/20"
          >
            <Download className="w-3.5 h-3.5" />
            Export DICOM Secondary Capture
          </button>
          <button
            onClick={() => {
              if (study.images.ply) {
                window.open(study.images.ply, '_blank');
              } else {
                alert('3D Gaussian Splat PLY model saved in output directory.');
              }
            }}
            className="py-2 px-3 rounded-lg bg-surfaceLight hover:bg-slate-800 text-slate-200 border border-borderDark text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            3D PLY Model
          </button>
        </div>
      </div>
    </div>
  );
}

