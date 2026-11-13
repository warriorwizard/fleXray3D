import React, { useState, useRef } from 'react';
import {
  Search,
  Filter,
  CheckCircle2,
  Clock,
  Activity,
  Layers,
  ChevronRight,
  Database,
  ArrowUpDown,
  Sparkles,
  Shield,
  FileCheck2,
  Upload,
  FileUp,
  X,
  AlertCircle,
  Box,
  FolderArchive
} from 'lucide-react';

export default function Worklist({ studies, onSelectStudy, onUploadSuccess }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBodyPart, setFilterBodyPart] = useState('All');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressMsg, setUploadProgressMsg] = useState('');
  const [uploadError, setUploadError] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const filteredStudies = studies.filter((s) => {
    const matchesSearch =
      s.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.mrn.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.bodyPart.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (s.modality && s.modality.toLowerCase().includes(searchTerm.toLowerCase()));

    let matchesFilter = true;
    if (filterBodyPart === '3D Volumes') {
      matchesFilter = !!s.isVolumetric;
    } else if (filterBodyPart !== 'All') {
      matchesFilter = s.bodyPart.toLowerCase().includes(filterBodyPart.toLowerCase());
    }

    return matchesSearch && matchesFilter;
  });

  const handleFileUpload = async (filesList) => {
    if (!filesList || filesList.length === 0) return;
    setIsUploading(true);
    setUploadError(null);

    const isMultiFile = filesList.length > 1;
    const firstFileName = filesList[0].name.toLowerCase();
    const isArchive = firstFileName.endsWith('.zip');
    const isNifti = firstFileName.endsWith('.nii') || firstFileName.endsWith('.nii.gz');
    const isVolumetricUpload = isMultiFile || isArchive || isNifti;

    const formData = new FormData();

    try {
      if (isVolumetricUpload) {
        setUploadProgressMsg(
          `Uploading ${filesList.length} files: Ingesting 3D volume, calibrating HU & generating WebGL proxy...`
        );
        for (let i = 0; i < filesList.length; i++) {
          formData.append('files', filesList[i]);
        }
        const res = await fetch('/api/upload_volume', {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Upload failed with status ${res.status}`);
        }
        const data = await res.json();
        if (data.status === 'success' && data.study) {
          setIsUploading(false);
          setShowUploadModal(false);
          if (onUploadSuccess) onUploadSuccess(data.study);
        } else {
          throw new Error('Invalid response from server');
        }
      } else {
        setUploadProgressMsg('Uploading radiograph: Extracting VOI LUT and running FleXray segmenter...');
        formData.append('file', filesList[0]);
        const res = await fetch('/api/upload_dicom', {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Upload failed with status ${res.status}`);
        }
        const data = await res.json();
        if (data.status === 'success' && data.study) {
          setIsUploading(false);
          setShowUploadModal(false);
          if (onUploadSuccess) onUploadSuccess(data.study);
        } else {
          throw new Error('Invalid response from server');
        }
      }
    } catch (err) {
      console.error('Upload error:', err);
      setUploadError(err.message || 'Failed to process medical scan.');
      setIsUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#08090e] overflow-hidden">
      {/* Top Banner / Navigation Bar */}
      <div className="bg-[#0e111a] border-b border-slate-800/80 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-slate-950 shadow-lg shadow-cyan-500/20">
            <Activity className="w-5 h-5 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-base font-bold text-slate-100 tracking-tight">FleXray Clinical Studio</h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-semibold">
                Universal PACS 2.0
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              MIT CSAIL & MGH Harvard • 60-Anatomical Structures Universal X-Ray & 3D Volumetric DRR Studio
            </p>
          </div>
        </div>

        {/* System Stats Indicators & Upload Button */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>CUDA DRR Accelerated</span>
          </div>

          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-cyan-300 text-xs font-mono">
            <Layers className="w-3.5 h-3.5" />
            <span>60 Anatomy Channels</span>
          </div>

          {/* Upload Button */}
          <button
            onClick={() => setShowUploadModal(true)}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/25 transition-all"
          >
            <Upload className="w-4 h-4 stroke-[2.5]" />
            Upload Volume / Radiograph
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="px-6 py-3.5 bg-[#0e111a]/50 border-b border-slate-800/80 flex items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by Patient Name, MRN, Modality, or Body Region..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-[#090b12] border border-slate-800 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>

        {/* Anatomical Filter Chips */}
        <div className="flex items-center gap-1.5 bg-[#090b12] border border-slate-800 p-1 rounded-xl text-xs">
          {['All', '3D Volumes', 'Torso', 'Chest', 'Pelvis', 'Brain', 'Knee', 'Extremity', 'Spine'].map((part) => (
            <button
              key={part}
              onClick={() => setFilterBodyPart(part)}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                filterBodyPart === part
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {part}
            </button>
          ))}
        </div>
      </div>

      {/* PACS Worklist Table */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="bg-[#0e111a] border border-slate-800/80 rounded-xl overflow-hidden shadow-2xl">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-900/80 text-slate-400 font-semibold border-b border-slate-800">
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Patient Demographics</th>
                <th className="py-3 px-4">MRN</th>
                <th className="py-3 px-4">Modality / View</th>
                <th className="py-3 px-4">Body Region</th>
                <th className="py-3 px-4">Study Date</th>
                <th className="py-3 px-4">Clinical Description</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredStudies.map((study) => (
                <tr
                  key={study.id}
                  onClick={() => onSelectStudy(study)}
                  className="hover:bg-slate-800/40 cursor-pointer transition-colors group"
                >
                  {/* Status Badge */}
                  <td className="py-3.5 px-4 whitespace-nowrap">
                    {study.isVolumetric ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-300 border border-amber-500/30 font-mono">
                        <Box className="w-3.5 h-3.5" />
                        3D Volume Ready
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        2D Radiograph
                      </span>
                    )}
                  </td>

                  {/* Patient Name */}
                  <td className="py-3.5 px-4 font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors">
                    {study.patientName}
                    <span className="block text-[11px] font-normal text-slate-400">
                      Age {study.age}{study.gender} • {study.institution}
                    </span>
                  </td>

                  {/* MRN */}
                  <td className="py-3.5 px-4 font-mono text-slate-300">{study.mrn}</td>

                  {/* Modality / View */}
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-0.5 rounded font-mono font-medium bg-slate-900 border border-slate-800 text-cyan-300">
                      {study.modality}
                    </span>
                    <span className="ml-2 text-slate-300 font-medium">{study.view}</span>
                  </td>

                  {/* Body Region */}
                  <td className="py-3.5 px-4 font-medium text-slate-200">
                    <span className="px-2.5 py-1 rounded-lg bg-slate-800/70 border border-slate-700/60 text-slate-200">
                      {study.bodyPart}
                    </span>
                  </td>

                  {/* Study Date */}
                  <td className="py-3.5 px-4 text-slate-400 font-mono">{study.studyDate}</td>

                  {/* Description */}
                  <td className="py-3.5 px-4 text-slate-300 text-[11px] max-w-sm truncate">
                    {study.description}
                  </td>

                  {/* Action Button */}
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectStudy(study);
                      }}
                      className="px-3.5 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500 text-cyan-300 hover:text-slate-950 font-bold text-xs border border-cyan-500/30 transition-all inline-flex items-center gap-1.5 group-hover:shadow-md group-hover:shadow-cyan-500/20"
                    >
                      {study.isVolumetric ? 'Inspect 3D Volume' : 'Inspect Anatomy'}
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* UPLOAD MODAL */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0e111a] border border-slate-800 rounded-2xl w-full max-w-lg p-6 shadow-2xl relative">
            <button
              onClick={() => {
                if (!isUploading) setShowUploadModal(false);
              }}
              className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <FileUp className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-100">Upload Volumetric Series or Radiograph</h3>
                <p className="text-xs text-slate-400">
                  Accepts DICOM series (folder / multiple .dcm), ZIP archive, NIfTI (.nii, .nii.gz), or planar radiographs (.png/.jpg)
                </p>
              </div>
            </div>

            {/* Drag & Drop Area */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                isDragging
                  ? 'border-cyan-400 bg-cyan-500/10'
                  : 'border-slate-700 hover:border-slate-500 bg-slate-900/60'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".dcm,.dicom,.zip,.nii,.nii.gz,image/png,image/jpeg"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleFileUpload(e.target.files);
                  }
                }}
              />

              <div className="flex flex-col items-center justify-center gap-3">
                <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-cyan-400">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-200">
                    Click to browse or drop your DICOM series, ZIP archive, or NIfTI scan here
                  </p>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Automatically computes slice normal sorting, HU calibration, and WebGL proxy
                  </p>
                </div>
              </div>
            </div>

            {/* Upload Progress Indicator */}
            {isUploading && (
              <div className="mt-4 p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                <div className="text-xs font-mono text-cyan-300">
                  {uploadProgressMsg || 'Processing medical scan...'}
                </div>
              </div>
            )}

            {/* Error Message */}
            {uploadError && (
              <div className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center gap-2.5 text-xs text-red-300">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Modal Footer */}
            <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between">
              <span className="text-[11px] text-slate-500 font-mono">
                Engine: FleXray 60-Anatomy & nanoDRR CUDA
              </span>
              <button
                type="button"
                onClick={() => setShowUploadModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
