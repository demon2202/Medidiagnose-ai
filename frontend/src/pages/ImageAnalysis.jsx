import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  X,
  Loader2,
  CheckCircle2,
  TriangleAlert,
  Heart,
  Activity,
  Microscope,
  ScanLine,
  ShieldCheck,
  FileUp,
  ClipboardList,
  ScanSearch,
  RotateCcw,
  FileImage,
  Maximize2,
  CircleAlert
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { config } from '../config/config';
import { scrollToId } from '../lib/scroll';
import { cleanResult } from '../lib/text';
import Disclaimer from '../components/common/Disclaimer';
import ImageResultView from '../components/results/ImageResultView';
import {
  PageHeader,
  EmptyState,
  Reveal
} from '../components/ui/ui';
import { easeOut } from '../lib/motion';

const ANALYSIS_TYPES = [
  {
    id: 'skin',
    label: 'Skin',
    title: 'Skin lesion',
    description: 'Melanoma and skin condition screening from photographs.',
    icon: Microscope,
    accepts: 'Color photos of lesions, moles or spots',
    acceptsSignal: false,
    fileAccept: 'image/*'
  },
  {
    id: 'breast',
    label: 'Breast',
    title: 'Breast imaging',
    description: 'Mammogram and ultrasound screening support.',
    icon: Activity,
    accepts: 'Grayscale mammograms or ultrasounds',
    acceptsSignal: false,
    fileAccept: 'image/*'
  },
  {
    id: 'heart',
    label: 'Heart',
    title: 'Cardiac',
    description: 'ECG printouts or raw signal files (.dat, .hea, .csv).',
    icon: Heart,
    accepts: 'ECG images or signal files',
    acceptsSignal: true,
    fileAccept: 'image/*,.dat,.hea,.csv,.edf,.mat'
  },
  {
    id: 'xray',
    label: 'Chest X-ray',
    title: 'Chest X-ray',
    description: 'Pneumonia and lung condition screening.',
    icon: ScanLine,
    accepts: 'Grayscale chest X-rays',
    acceptsSignal: false,
    fileAccept: 'image/*'
  },
];

const extOf = (name = '') => name.split('.').pop().toLowerCase();
const isSignalFile = (file) => ['dat', 'hea', 'csv', 'edf', 'mat'].includes(extOf(file.name));
const isImageFile = (file) =>
  file.type.startsWith('image/') ||
  ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'].includes(extOf(file.name));

const fmtSize = (bytes) =>
  bytes > 1048576 ? `${(bytes / 1048576).toFixed(2)} MB` : `${(bytes / 1024).toFixed(1)} KB`;

export default function ImageAnalysis() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [lightbox, setLightbox] = useState(false);
  const [analysisType, setAnalysisType] = useState('skin');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileType, setFileType] = useState('image');
  const [heaFile, setHeaFile] = useState(null);
  const fileInputRef = useRef(null);
  const heaFileInputRef = useRef(null);

  const current = ANALYSIS_TYPES.find((t) => t.id === analysisType);

  useEffect(
    () => () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  /* ---------------- file handling (logic preserved) ---------------- */

  const resetMessages = () => {
    setError(null);
    setResult(null);
    setValidationError(null);
  };

  const handleFileSelect = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      if (isSignalFile(file)) {
        if (!current?.acceptsSignal) {
          setError(`Signal files (.${extOf(file.name)}) are only supported for cardiac analysis.`);
          showNotification('Wrong file type for this analysis', 'error');
          return;
        }
        if (file.size > config.upload.maxFileSize) {
          setError(`File must be under ${config.upload.maxFileSize / 1048576} MB.`);
          showNotification('File too large', 'error');
          return;
        }
        if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
        setSelectedFile(file);
        setPreview(null);
        setFileType('signal');
        resetMessages();
        showNotification(`Signal file "${file.name}" loaded`, 'success');
        return;
      }

      if (!isImageFile(file)) {
        setError(
          current?.acceptsSignal
            ? 'Select an image (PNG, JPG) or an ECG signal file (.dat, .hea, .csv).'
            : 'Select a valid image file (PNG, JPG).',
        );
        showNotification('Invalid file type', 'error');
        return;
      }
      if (file.size > config.upload.maxFileSize) {
        setError(`File must be under ${config.upload.maxFileSize / 1048576} MB.`);
        showNotification('File too large', 'error');
        return;
      }
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setFileType('image');
      resetMessages();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showNotification, preview, analysisType],
  );

  const clearSelection = useCallback(() => {
    if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
    setSelectedFile(null);
    setPreview(null);
    setFileType('image');
    setResult(null);
    setError(null);
    setValidationError(null);
    setHeaFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (heaFileInputRef.current) heaFileInputRef.current.value = '';
  }, [preview]);

  /* ---------------- analysis (logic preserved) ---------------- */

  const normalizeResult = (data) => {
    const out = { ...data };
    const norm = (p) => {
      const c = { ...p };
      let conf = c.confidence;
      if (typeof conf === 'string') {
        conf = parseFloat(conf.replace('%', ''));
        if (!Number.isNaN(conf) && conf > 1) conf /= 100;
      }
      if (typeof conf !== 'number' || Number.isNaN(conf)) conf = 0;
      c.confidence = Math.max(0, Math.min(1, conf));
      return c;
    };
    if (out.prediction) {
      out.prediction = norm(out.prediction);
      if (!out.prediction.name) out.prediction.name = 'Unknown condition';
    }
    if (Array.isArray(out.all_predictions)) {
      out.all_predictions = out.all_predictions.map((p) => ({
        ...norm(p),
        name: p.name || 'Unknown'
      }));
    }
    if (!out.severity) out.severity = 'low';
    return out;
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Select a file first.');
      showNotification('No file selected', 'error');
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    setValidationError(null);

    const formData = new FormData();
    if (fileType === 'signal') {
      formData.append('signal_file', selectedFile);
      formData.append('file_type', 'signal');
      if (heaFile) formData.append('hea_file', heaFile);
    } else {
      formData.append('image', selectedFile);
      formData.append('file_type', 'image');
    }

    const endpoints = {
      skin: `${config.api.baseURL}/analyze/skin`,
      breast: `${config.api.baseURL}/analyze/breast`,
      heart: `${config.api.baseURL}/analyze/heart`,
      xray: `${config.api.baseURL}/analyze/xray`
    };

    try {
      const response = await axios.post(endpoints[analysisType], formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: config.api.timeout
      });
      const data = response.data;
      if (data.success) {
        const normalized = normalizeResult(cleanResult(data));
        setResult(normalized);
        addToHistory({
          type: `image_${analysisType}`,
          prediction: normalized.prediction?.name || 'Unknown',
          confidence: normalized.prediction?.confidence || 0,
          severity: normalized.severity || 'unknown',
          fileType,
          fileName: selectedFile.name,
          data: normalized,
          timestamp: new Date().toISOString()
        });
        showNotification('Analysis complete', 'success');
        requestAnimationFrame(() => scrollToId('analysis-result'));
      } else if (data.validation_error) {
        setValidationError({
          message: data.message,
          suggestion: data.suggestion,
          expectedType: data.expected_type
        });
        showNotification('Wrong file type detected', 'error');
      } else {
        setError(data.error || data.message || 'Analysis failed');
        showNotification('Analysis failed', 'error');
      }
    } catch (err) {
      if (err.response?.status === 400 && err.response?.data?.validation_error) {
        setValidationError({
          message: err.response.data.message,
          suggestion: err.response.data.suggestion,
          expectedType: err.response.data.expected_type
        });
        showNotification('Wrong file type', 'error');
      } else if (err.response?.data?.error) {
        const msg = err.response.data.error;
        const sug = err.response.data.suggestion;
        setError(sug ? `${msg} — ${sug}` : msg);
        showNotification('Analysis error', 'error');
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server took too long to respond.');
        showNotification('Request timeout', 'error');
      } else {
        setError(`Could not reach the server. Is the backend running on ${config.api.baseURL}?`);
        showNotification('Connection failed', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const switchType = (id) => {
    setAnalysisType(id);
    setResult(null);
    setError(null);
    setValidationError(null);
  };

  /* ---------------- result fragments ---------------- */


  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Medical imaging"
        title="Image analysis"
        description={current.description}
      />

      <Reveal>
        <Disclaimer message="Preliminary AI image and signal analysis for information only — not a substitute for clinical judgment. Always confirm with a qualified professional." />
      </Reveal>

      {/* models — proper cards, not a text row */}
      <Reveal delay={0.04}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" role="tablist" aria-label="Analysis model">
          {ANALYSIS_TYPES.map((t) => {
            const active = t.id === analysisType;
            return (
              <button
                key={t.id}
                role="tab"
                aria-selected={active}
                onClick={() => switchType(t.id)}
                className={`group relative flex flex-col items-start gap-3 rounded-2xl border p-5 text-left transition-all duration-200 ${
                  active
                    ? 'border-accent bg-accent-soft shadow-soft'
                    : 'border-line bg-surface hover:-translate-y-0.5 hover:border-faint/70 hover:shadow-soft'
                }`}
              >
                <span className="flex w-full items-center justify-between">
                  <span
                    className={`flex h-12 w-12 items-center justify-center rounded-xl border transition-colors ${
                      active
                        ? 'border-accent/30 bg-accent/10 text-accent'
                        : 'border-line bg-paper text-muted group-hover:text-ink'
                    }`}
                  >
                    <t.icon size={22} strokeWidth={1.9} />
                  </span>
                  {active && (
                    <span className="rounded-full bg-accent px-2 py-0.5 text-xs font-bold uppercase tracking-wide text-accent-ink">
                      Active
                    </span>
                  )}
                </span>
                <span className="min-w-0">
                  <span className="block text-[17px] font-semibold leading-tight tracking-tight text-ink">
                    {t.label}
                  </span>
                  <span className="mt-0.5 block text-sm font-medium text-faint">{t.title}</span>
                  <span className="mt-1.5 block text-[15px] leading-snug text-muted">{t.description}</span>
                </span>
              </button>
            );
          })}
        </div>
      </Reveal>

      <div className="grid items-stretch gap-5 lg:grid-cols-12 lg:min-h-[520px]">
        {/* ---------- left: upload (sticky) ---------- */}
        <Reveal delay={0.08} className="lg:col-span-5">
          <div className="panel p-5 lg:sticky lg:top-24">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-[15px] font-semibold text-ink">Source {fileType === 'signal' ? 'signal' : 'image'}</h3>
              {selectedFile && (
                <button onClick={clearSelection} className="btn-quiet btn-sm">
                  <RotateCcw size={13} /> New scan
                </button>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={current.fileAccept}
              onChange={handleFileSelect}
              className="hidden"
            />

            <AnimatePresence mode="wait" initial={false}>
              {!selectedFile ? (
                <Motion.button
                  key="drop"
                  type="button"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) handleFileSelect({ target: { files: [f] } });
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                  }}
                  className={`flex w-full flex-col items-center rounded-xl border border-dashed px-5 py-8 text-center transition-all duration-200 ${
                    isDragging
                      ? 'border-accent bg-accent/[0.06] scale-[1.01]'
                      : 'border-line bg-paper hover:border-faint/70 hover:bg-raised/50'
                  }`}
                >
                  <span className="dotgrid mb-3 flex h-12 w-12 items-center justify-center rounded-xl border border-line bg-surface text-muted">
                    <Upload size={20} strokeWidth={1.8} />
                  </span>
                  <span className="text-base font-medium text-ink">
                    {isDragging ? 'Drop it here' : 'Drop a file or browse'}
                  </span>
                  <span className="mt-1 text-sm text-muted">{current.accepts}</span>
                  <span className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[13px] font-medium text-muted">
                    {current.acceptsSignal ? (
                      <><FileUp size={12} /> PNG · JPG · DAT · HEA · CSV · EDF</>
                    ) : (
                      <><FileImage size={12} /> PNG · JPG · WEBP · up to 32 MB</>
                    )}
                  </span>
                </Motion.button>
              ) : fileType === 'signal' ? (
                <Motion.div
                  key="signal"
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2, ease: easeOut }}
                  className="rounded-xl border border-line bg-paper p-5 text-center"
                >
                  <span className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-critical/10 text-critical">
                    <Heart size={22} strokeWidth={1.9} />
                  </span>
                  <p className="truncate text-sm font-medium text-ink">{selectedFile.name}</p>
                  <p className="t-num mt-0.5 text-xs text-muted">{fmtSize(selectedFile.size)} · ECG signal</p>
                  <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-low/10 px-2.5 py-1 text-xs font-medium text-low">
                    <CheckCircle2 size={13} /> Ready for analysis
                  </span>

                  {selectedFile.name.toLowerCase().endsWith('.dat') && (
                    <div className="mt-4 border-t border-line pt-4 text-left">
                      <input
                        ref={heaFileInputRef}
                        type="file"
                        accept=".hea"
                        className="hidden"
                        onChange={(e) => setHeaFile(e.target.files?.[0] || null)}
                      />
                      {heaFile ? (
                        <div className="flex items-center gap-2.5 rounded-lg border border-line bg-surface px-3 py-2.5">
                          <CheckCircle2 size={15} className="shrink-0 text-low" />
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-[13px] font-medium text-ink">{heaFile.name}</p>
                            <p className="text-[11px] text-low">Header attached</p>
                          </div>
                          <button
                            onClick={() => {
                              setHeaFile(null);
                              if (heaFileInputRef.current) heaFileInputRef.current.value = '';
                            }}
                            className="rounded-lg p-1 text-faint hover:bg-raised hover:text-critical"
                            aria-label="Remove header file"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => heaFileInputRef.current?.click()}
                          className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-line px-3 py-2.5 text-xs text-muted transition-colors hover:border-faint/60 hover:text-ink"
                        >
                          <FileUp size={14} />
                          Attach <span className="t-num font-semibold">.hea</span> header (recommended for PTB-XL)
                        </button>
                      )}
                    </div>
                  )}
                </Motion.div>
              ) : (
                <Motion.div
                  key="preview"
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2, ease: easeOut }}
                  className="group relative overflow-hidden rounded-xl border border-line bg-paper"
                >
                  <button
                    onClick={() => setLightbox(true)}
                    className="group/img relative block w-full cursor-zoom-in"
                    aria-label="Enlarge image"
                  >
                    <img src={preview} alt="Selected scan" className="max-h-[260px] w-full object-contain" />
                    <span className="absolute bottom-2.5 right-2.5 flex h-8 w-8 items-center justify-center rounded-lg bg-black/55 text-white opacity-0 backdrop-blur-sm transition-opacity group-hover/img:opacity-100">
                      <Maximize2 size={15} />
                    </span>
                  </button>
                  <div className="flex items-center gap-2.5 border-t border-line bg-surface px-3.5 py-2.5">
                    <FileImage size={15} className="shrink-0 text-faint" />
                    <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{selectedFile.name}</p>
                    <span className="t-num shrink-0 text-xs text-faint">{fmtSize(selectedFile.size)}</span>
                    <button
                      onClick={clearSelection}
                      className="shrink-0 rounded-lg p-1.5 text-faint transition-colors hover:bg-critical/10 hover:text-critical"
                      aria-label="Remove image"
                    >
                      <X size={14} />
                    </button>
                  </div>
                </Motion.div>
              )}
            </AnimatePresence>

            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isLoading}
              className="btn-accent mt-4 w-full py-3.5 text-[15px]"
            >
              {isLoading ? (
                <><Loader2 size={17} className="animate-spin" /> Analyzing…</>
              ) : (
                <><ScanSearch size={17} /> Analyze {current.title.toLowerCase()}</>
              )}
            </button>

            <AnimatePresence>
              {error && (
                <Motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-3 flex items-start gap-2.5 rounded-xl bg-critical/[0.07] px-3.5 py-3">
                    <CircleAlert size={16} className="mt-px shrink-0 text-critical" />
                    <p className="text-[13px] font-medium leading-relaxed text-critical">{error}</p>
                  </div>
                </Motion.div>
              )}
              {validationError && (
                <Motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-3 rounded-xl bg-moderate/[0.09] px-3.5 py-3">
                    <p className="flex items-center gap-2 text-[13px] font-semibold text-moderate">
                      <TriangleAlert size={15} /> Wrong file type
                    </p>
                    <p className="mt-1 text-[13px] leading-relaxed text-muted">{validationError.message}</p>
                    {validationError.suggestion && (
                      <p className="mt-1.5 text-[13px] font-medium text-ink">{validationError.suggestion}</p>
                    )}
                    <button onClick={clearSelection} className="btn-ghost btn-sm mt-2.5 w-full">
                      Upload the correct file
                    </button>
                  </div>
                </Motion.div>
              )}
            </AnimatePresence>
          </div>
        </Reveal>

        {/* ---------- right: result ---------- */}
        <div className="lg:col-span-7" id="analysis-result" style={{ scrollMarginTop: 90 }}>
          <Reveal delay={0.12} className="h-full">
            <AnimatePresence mode="wait" initial={false}>
              {isLoading ? (
                <Motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="panel p-5"
                >
                  <div className="flex items-center gap-5">
                    <div className="flex-1 space-y-3">
                      <div className="skeleton h-3 w-28 rounded-full" />
                      <div className="skeleton h-7 w-3/4 rounded-lg" />
                      <div className="skeleton h-3 w-40 rounded-full" />
                    </div>
                    <div className="skeleton h-24 w-24 shrink-0 rounded-full" />
                  </div>
                  <div className="mt-5 flex items-center gap-2.5 border-t border-line pt-4 text-[13px] text-muted">
                    <Loader2 size={15} className="animate-spin text-accent" />
                    Running {current.title.toLowerCase()} model — this can take a few seconds…
                  </div>
                </Motion.div>
              ) : result ? (
                <Motion.div
                  key="result"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, ease: easeOut }}
                  className="panel overflow-hidden"
                >
                  <ImageResultView result={result} fileName={selectedFile?.name} imageUrl={preview} onImageClick={() => setLightbox(true)} />
                </Motion.div>
              ) : (
                <Motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="panel flex h-full flex-col"
                >
                  <EmptyState
                    icon={ClipboardList}
                    title="No analysis yet"
                    hint="Upload a file on the left and the result will appear here — condition, confidence and guidance in one view."
                    className="flex-1 justify-center"
                  />
                  <div className="grid grid-cols-3 gap-px overflow-hidden rounded-b-2xl border-t border-line bg-line">
                    {[
                      { icon: Upload, t: 'Upload', d: 'Image or signal' },
                      { icon: ScanSearch, t: 'Analyze', d: 'AI screening' },
                      { icon: ShieldCheck, t: 'Review', d: 'Guided next steps' },
                    ].map((s) => (
                      <div key={s.t} className="bg-surface px-3 py-3.5 text-center">
                        <s.icon size={17} className="mx-auto text-faint" strokeWidth={1.9} />
                        <p className="mt-1.5 text-sm font-medium text-ink">{s.t}</p>
                        <p className="text-[13px] text-faint">{s.d}</p>
                      </div>
                    ))}
                  </div>
                </Motion.div>
              )}
            </AnimatePresence>
          </Reveal>
        </div>

        {/* lightbox */}
        <AnimatePresence>
          {lightbox && preview && (
            <Lightbox key="lightbox" src={preview} alt={selectedFile?.name || 'Selected scan'} onClose={() => setLightbox(false)} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function Lightbox({ src, alt, onClose }) {
  React.useEffect(() => {
    const fn = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', fn);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', fn);
      document.body.style.overflow = '';
    };
  }, [onClose]);
  return (
    <Motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
      onClick={onClose}
      className="fixed inset-0 z-[95] flex cursor-zoom-out items-center justify-center bg-black/85 p-4 backdrop-blur-sm sm:p-8"
      role="dialog"
      aria-label="Image preview"
    >
      <Motion.img
        src={src}
        alt={alt}
        initial={{ scale: 0.96 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.97 }}
        transition={{ duration: 0.2, ease: easeOut }}
        onClick={(e) => e.stopPropagation()}
        className="max-h-[88vh] max-w-full cursor-default rounded-xl object-contain shadow-2xl"
      />
      <span className="absolute left-1/2 top-5 max-w-[80vw] -translate-x-1/2 truncate rounded-full bg-black/55 px-3.5 py-1.5 text-xs text-white/90 backdrop-blur-sm">
        {alt}
      </span>
      <span className="absolute right-4 top-4 flex h-9 w-9 items-center justify-center rounded-full bg-black/55 text-white backdrop-blur-sm">
        <X size={17} />
      </span>
    </Motion.div>
  );
}
