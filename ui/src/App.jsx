import React, { useState, useEffect } from 'react';
import Worklist from './components/Worklist';
import Studio from './components/Studio';

export default function App() {
  const [currentView, setCurrentView] = useState('worklist'); // 'worklist' | 'studio'
  const [studies, setStudies] = useState([]);
  const [selectedStudy, setSelectedStudy] = useState(null);
  const [isLoadingStudies, setIsLoadingStudies] = useState(true);

  // Load studies from API
  const loadStudies = () => {
    fetch('/api/studies')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setStudies(data);
        if (data.length > 0 && !selectedStudy) {
          setSelectedStudy(data[0]);
        }
        setIsLoadingStudies(false);
      })
      .catch((err) => {
        console.error('Failed to load studies from API:', err);
        setIsLoadingStudies(false);
      });
  };

  useEffect(() => {
    loadStudies();
  }, []);

  const handleSelectStudy = (study) => {
    setSelectedStudy(study);
    setCurrentView('studio');
  };

  const handleBackToWorklist = () => {
    setCurrentView('worklist');
    loadStudies();
  };

  const handleUploadSuccess = (newStudy) => {
    setStudies((prev) => [newStudy, ...prev]);
    setSelectedStudy(newStudy);
    setCurrentView('studio');
  };

  return (
    <div className="fixed inset-0 flex flex-col bg-[#08090e] text-slate-100 overflow-hidden font-['Plus_Jakarta_Sans',sans-serif]">
      {currentView === 'worklist' ? (
        <Worklist
          studies={studies}
          onSelectStudy={handleSelectStudy}
          onUploadSuccess={handleUploadSuccess}
        />
      ) : (
        selectedStudy && (
          <Studio key={selectedStudy.id} study={selectedStudy} onBackToWorklist={handleBackToWorklist} />
        )
      )}
    </div>
  );
}
