import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import {
  Upload,
  Image as ImageIcon,
  X,
  Loader2,
  AlertCircle,
  CheckCircle,
  Camera,
  Info,
  AlertTriangle,
  Heart,
  Activity,
  Microscope,
  Stethoscope,
  Clock,
  Shield,
  ChevronRight,
  XCircle,
  Pill,
  FileText,
  BarChart3,
  TrendingUp,
  FileUp
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { config } from '../config/config';

function ImageAnalysis() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analysisType, setAnalysisType] = useState('skin');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileType, setFileType] = useState('image'); // 'image' or 'signal'
  const [heaFile, setHeaFile] = useState(null); // companion .hea file for .dat ECG files
  const fileInputRef = useRef(null);
  const heaFileInputRef = useRef(null);

  // Cleanup blob URLs on unmount or preview change
  useEffect(() => {
    return () => {
      if (preview && preview.startsWith('blob:')) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const analysisTypes = [
    {
      id: 'skin',
      title: 'Skin Cancer Analysis',
      description: 'Detect skin cancer, melanoma, and other skin conditions',
      icon: Microscope,
      color: 'from-purple-500 to-purple-600',
      accepts: 'Photos of skin lesions, moles, or suspicious spots',
      expectedType: 'Color photograph of skin',
      imageType: 'color',
      acceptsSignal: false,
      fileAccept: 'image/*'
    },
    {
      id: 'breast',
      title: 'Breast Cancer Screening',
      description: 'Analyze mammograms and breast ultrasounds',
      icon: Activity,
      color: 'from-pink-500 to-rose-600',
      accepts: 'Mammogram images, breast ultrasound images',
      expectedType: 'Grayscale mammogram or ultrasound',
      imageType: 'grayscale',
      acceptsSignal: false,
      fileAccept: 'image/*'
    },
    {
      id: 'heart',
      title: 'Heart Condition Analysis',
      description: 'Detect heart conditions from ECG images OR signal files (.dat, .csv)',
      icon: Heart,
      color: 'from-red-500 to-red-600',
      accepts: 'ECG printouts, echocardiogram images, OR signal files (.dat, .hea, .csv)',
      expectedType: 'ECG/EKG printout, scan, or signal data file',
      imageType: 'grayscale',
      acceptsSignal: true,
      fileAccept: 'image/*,.dat,.hea,.csv,.edf,.mat'
    },
    {
      id: 'xray',
      title: 'Chest X-Ray Analysis',
      description: 'Detect pneumonia and lung conditions',
      icon: Stethoscope,
      color: 'from-blue-500 to-blue-600',
      accepts: 'Chest X-ray images',
      expectedType: 'Grayscale chest X-ray',
      imageType: 'grayscale',
      acceptsSignal: false,
      fileAccept: 'image/*'
    }
  ];

  // ================================================================
  //                    FILE HANDLING
  // ================================================================

  const getFileExtension = (filename) => {
    return filename.split('.').pop().toLowerCase();
  };

  const isSignalFile = (file) => {
    const ext = getFileExtension(file.name);
    return ['dat', 'hea', 'csv', 'edf', 'mat'].includes(ext);
  };

  const isImageFile = (file) => {
    return file.type.startsWith('image/') ||
      ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'].includes(getFileExtension(file.name));
  };

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const currentType = analysisTypes.find(t => t.id === analysisType);
    const ext = getFileExtension(file.name);

    // Check if it's a signal file
    if (isSignalFile(file)) {
      if (!currentType?.acceptsSignal) {
        setError(`Signal files (.${ext}) are only supported for Heart Condition Analysis. Please select an image file.`);
        showNotification('Wrong file type for this analysis', 'error');
        return;
      }

      if (file.size > config.upload.maxFileSize) {
        setError(`File size must be less than ${config.upload.maxFileSize / 1024 / 1024}MB`);
        showNotification('File too large', 'error');
        return;
      }

      // Revoke old preview
      if (preview && preview.startsWith('blob:')) {
        URL.revokeObjectURL(preview);
      }

      setSelectedFile(file);
      setPreview(null); // No preview for signal files
      setFileType('signal');
      setError(null);
      setResult(null);
      setValidationError(null);
      showNotification(`ECG signal file "${file.name}" loaded`, 'success');
      return;
    }

    // Image file validation
    if (!isImageFile(file)) {
      if (currentType?.acceptsSignal) {
        setError(`Please select an image file (PNG, JPG) or ECG signal file (.dat, .hea, .csv)`);
      } else {
        setError('Please select a valid image file (PNG, JPG, etc.)');
      }
      showNotification('Invalid file type', 'error');
      return;
    }

    if (file.size > config.upload.maxFileSize) {
      setError(`File size must be less than ${config.upload.maxFileSize / 1024 / 1024}MB`);
      showNotification('File too large', 'error');
      return;
    }

    // Revoke old preview
    if (preview && preview.startsWith('blob:')) {
      URL.revokeObjectURL(preview);
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setFileType('image');
    setError(null);
    setResult(null);
    setValidationError(null);
  }, [showNotification, preview, analysisType]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    // Create a synthetic event to reuse handleFileSelect logic
    const syntheticEvent = {
      target: { files: [file] }
    };
    handleFileSelect(syntheticEvent);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const clearSelection = useCallback(() => {
    if (preview && preview.startsWith('blob:')) {
      URL.revokeObjectURL(preview);
    }
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

  // ================================================================
  //                    ANALYSIS HANDLER
  // ================================================================

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image or signal file first');
      showNotification('No file selected', 'error');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setValidationError(null);

    const formData = new FormData();

    // For signal files, use different form field and endpoint
    if (fileType === 'signal') {
      formData.append('signal_file', selectedFile);
      formData.append('file_type', 'signal');
      // Attach companion .hea header file if provided (required for .dat PTB-XL files)
      if (heaFile) {
        formData.append('hea_file', heaFile);
      }
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
        const normalizedResult = normalizeResult(data);
        setResult(normalizedResult);

        addToHistory({
          type: `image_${analysisType}`,
          prediction: normalizedResult.prediction?.name || 'Unknown',
          confidence: normalizedResult.prediction?.confidence || 0,
          severity: normalizedResult.severity || 'unknown',
          fileType: fileType,
          fileName: selectedFile.name,
          timestamp: new Date().toISOString()
        });

        showNotification('Analysis complete!', 'success');
      } else {
        if (data.validation_error) {
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
      }
    } catch (err) {
      console.error('Analysis error:', err);

      if (err.response?.status === 400 && err.response?.data?.validation_error) {
        setValidationError({
          message: err.response.data.message,
          suggestion: err.response.data.suggestion,
          expectedType: err.response.data.expected_type
        });
        showNotification('Wrong file type', 'error');
      } else if (err.response?.data?.error) {
        const errMsg = err.response.data.error;
        const suggestion = err.response.data.suggestion;
        setError(suggestion ? `${errMsg} — ${suggestion}` : errMsg);
        showNotification('Analysis error', 'error');
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. The server took too long to respond.');
        showNotification('Request timeout', 'error');
      } else {
        setError(`Failed to connect to server. Ensure backend is running on ${config.api.baseURL}`);
        showNotification('Connection failed', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // ================================================================
  //                    DATA NORMALIZATION
  // ================================================================

  const normalizeResult = (data) => {
    const result = { ...data };

    if (result.prediction) {
      result.prediction = { ...result.prediction };
      let conf = result.prediction.confidence;
      if (typeof conf === 'string') {
        conf = parseFloat(conf.replace('%', ''));
        if (!isNaN(conf) && conf > 1) conf = conf / 100;
      }
      if (typeof conf !== 'number' || isNaN(conf)) conf = 0;
      conf = Math.max(0, Math.min(1, conf));
      result.prediction.confidence = conf;
      result.prediction.confidence_percent = `${(conf * 100).toFixed(1)}%`;
      if (!result.prediction.name) result.prediction.name = 'Unknown Condition';
    }

    if (result.all_predictions && Array.isArray(result.all_predictions)) {
      result.all_predictions = result.all_predictions.map(pred => {
        const p = { ...pred };
        let c = p.confidence;
        if (typeof c === 'string') {
          c = parseFloat(c.replace('%', ''));
          if (!isNaN(c) && c > 1) c = c / 100;
        }
        if (typeof c !== 'number' || isNaN(c)) c = 0;
        c = Math.max(0, Math.min(1, c));
        p.confidence = c;
        p.confidence_percent = `${(c * 100).toFixed(1)}%`;
        if (!p.name) p.name = 'Unknown';
        return p;
      });
    }

    if (!result.severity) result.severity = 'low';
    return result;
  };

  // ================================================================
  //                    STYLING HELPERS
  // ================================================================

  const getSeverityStyles = (severity) => {
    const styles = {
      critical: { bg: 'bg-red-50 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400', border: 'border-red-300 dark:border-red-800', badge: 'bg-red-500 text-white', progressBar: 'bg-gradient-to-r from-red-500 to-red-600', glow: 'shadow-lg shadow-red-100 dark:shadow-red-900/30' },
      high: { bg: 'bg-orange-50 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-400', border: 'border-orange-300 dark:border-orange-800', badge: 'bg-orange-500 text-white', progressBar: 'bg-gradient-to-r from-orange-500 to-orange-600', glow: 'shadow-lg shadow-orange-100 dark:shadow-orange-900/30' },
      moderate: { bg: 'bg-yellow-50 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-400', border: 'border-yellow-300 dark:border-yellow-800', badge: 'bg-yellow-500 text-white', progressBar: 'bg-gradient-to-r from-yellow-500 to-yellow-600', glow: 'shadow-lg shadow-yellow-100 dark:shadow-yellow-900/30' },
      low: { bg: 'bg-green-50 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400', border: 'border-green-300 dark:border-green-800', badge: 'bg-green-500 text-white', progressBar: 'bg-gradient-to-r from-green-500 to-green-600', glow: 'shadow-lg shadow-green-100 dark:shadow-green-900/30' },
      healthy: { bg: 'bg-blue-50 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400', border: 'border-blue-300 dark:border-blue-800', badge: 'bg-blue-500 text-white', progressBar: 'bg-gradient-to-r from-blue-500 to-blue-600', glow: 'shadow-lg shadow-blue-100 dark:shadow-blue-900/30' }
    };
    return styles[severity] || styles.low;
  };

  const getSeverityIcon = (sev) => {
    if (sev === 'critical' || sev === 'high') return <AlertTriangle className="text-red-500" size={24} />;
    if (sev === 'moderate') return <AlertCircle className="text-yellow-500" size={24} />;
    if (sev === 'healthy') return <CheckCircle className="text-blue-500" size={24} />;
    return <CheckCircle className="text-green-500" size={24} />;
  };

  const getConfidenceColor = (c) => {
    if (c >= 0.8) return 'text-green-600 dark:text-green-400';
    if (c >= 0.6) return 'text-blue-600 dark:text-blue-400';
    if (c >= 0.4) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-orange-600 dark:text-orange-400';
  };

  const getConfidenceLabel = (c) => {
    if (c >= 0.9) return 'Very High';
    if (c >= 0.75) return 'High';
    if (c >= 0.6) return 'Moderate';
    if (c >= 0.4) return 'Low';
    return 'Very Low';
  };

  const getConfidenceBgColor = (c) => {
    if (c >= 0.8) return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400';
    if (c >= 0.6) return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400';
    if (c >= 0.4) return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400';
    return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400';
  };

  const getUrgencyColor = (color) => {
    const m = { red: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400', orange: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800 text-orange-700 dark:text-orange-400', yellow: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-700 dark:text-yellow-400', green: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400', blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-400' };
    return m[color] || m.blue;
  };

  const getRecommendationColor = (level) => {
    const m = { critical: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800', high: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800', moderate: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800', low: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800', healthy: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' };
    return m[level] || m.low;
  };

  // ================================================================
  //                    RESULT COMPONENTS
  // ================================================================

  const ConfidenceBar = ({ confidence, severity }) => {
    const styles = getSeverityStyles(severity);
    const pct = Math.max(0, Math.min(100, (confidence || 0) * 100));
    return (
      <div className="mt-5 pt-4 border-t border-gray-200/50 dark:border-gray-700/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <BarChart3 size={16} className={styles.text} />
            <span className={`text-sm font-semibold ${styles.text}`}>Confidence Level</span>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${getConfidenceBgColor(confidence)}`}>{getConfidenceLabel(confidence)}</span>
            <span className={`text-xl font-bold ${styles.text}`}>{pct.toFixed(1)}%</span>
          </div>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3.5 overflow-hidden shadow-inner">
          <div className={`h-full rounded-full transition-all duration-1000 ease-out ${styles.progressBar}`} style={{ width: `${pct}%`, minWidth: pct > 0 ? '12px' : '0px' }} />
        </div>
        <div className="flex justify-between mt-1.5 px-0.5">
          {[0, 25, 50, 75, 100].map(v => <span key={v} className="text-[10px] text-gray-400 dark:text-gray-500 font-medium">{v}%</span>)}
        </div>
      </div>
    );
  };

  const renderStaging = (s) => { if (!s || typeof s !== 'object') return null; return (<div className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-xl"><h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2"><Activity size={18} className="text-blue-500" />Staging Information</h4><div className="space-y-2 text-sm">{s.stage && <div className="flex items-start gap-2"><span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Stage:</span><span className="text-gray-900 dark:text-white font-semibold">{s.stage}</span></div>}{s.description && <div className="flex items-start gap-2"><span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Details:</span><span className="text-gray-600 dark:text-gray-400">{s.description}</span></div>}{s.prognosis && <div className="flex items-start gap-2"><span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Prognosis:</span><span className="text-gray-600 dark:text-gray-400">{s.prognosis}</span></div>}</div></div>); };

  const renderUrgency = (u) => { if (!u || typeof u !== 'object') return null; return (<div className={`mb-4 p-4 rounded-xl border ${getUrgencyColor(u.color)}`}><div className="flex items-center gap-2 mb-1"><Clock size={16} /><span className="font-semibold">{u.timeline || 'N/A'}</span></div><p className="text-sm opacity-90">{u.action || 'N/A'}</p></div>); };

  const renderTreatmentOptions = (t) => { if (!t || !Array.isArray(t) || !t.length) return null; return (<div className="mb-4"><h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2"><Pill size={18} className="text-green-500" />Treatment Options</h4><ul className="space-y-2">{t.slice(0, 8).map((item, i) => <li key={i} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400"><ChevronRight className="text-blue-500 mt-0.5 flex-shrink-0" size={14} /><span>{item}</span></li>)}</ul></div>); };

  const renderRecommendations = (r) => {
    if (!r || typeof r !== 'object') return null;
    return (
      <div className={`p-4 rounded-xl border ${getRecommendationColor(r.level)}`}>
        {r.title && <h4 className="font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2"><Shield size={18} />{r.title}</h4>}
        {r.message && <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">{r.message}</p>}
        {r.actions && Array.isArray(r.actions) && <div className="space-y-2"><p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Recommended Actions:</p>{r.actions.map((a, i) => <p key={i} className="text-sm flex items-start gap-2 text-gray-600 dark:text-gray-400"><span className="font-bold text-blue-500 min-w-[20px]">{i + 1}.</span><span>{a}</span></p>)}</div>}
        {r.next_steps && Array.isArray(r.next_steps) && <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700"><p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Next Steps:</p>{r.next_steps.map((s, i) => <p key={i} className="text-sm text-gray-600 dark:text-gray-400">• {s}</p>)}</div>}
        {r.warning_signs && Array.isArray(r.warning_signs) && <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800"><p className="text-xs font-semibold text-red-600 dark:text-red-400 mb-2">⚠️ Warning Signs:</p>{r.warning_signs.map((s, i) => <p key={i} className="text-sm text-red-600 dark:text-red-400">• {s}</p>)}</div>}
        {r.risk_factors && Array.isArray(r.risk_factors) && <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700"><p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Risk Factors:</p>{r.risk_factors.map((f, i) => <p key={i} className="text-sm text-gray-600 dark:text-gray-400">• {f}</p>)}</div>}
        {r.note && <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700"><p className="text-xs text-gray-500 dark:text-gray-400"><strong>Note:</strong> {r.note}</p></div>}
      </div>
    );
  };

  const renderAllPredictions = (preds) => {
    if (!preds || !Array.isArray(preds) || preds.length <= 1) return null;
    return (
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-sm flex items-center gap-2"><TrendingUp size={16} className="text-blue-500" />All Conditions (Ranked)</h4>
        <div className="space-y-3">
          {preds.slice(0, 7).map((pred, idx) => {
            const conf = typeof pred.confidence === 'number' ? pred.confidence : 0;
            const pct = (conf * 100).toFixed(1);
            const isTop = idx === 0;
            return (
              <div key={idx} className={isTop ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3' : ''}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {isTop && <span className="text-[10px] bg-blue-500 text-white px-1.5 py-0.5 rounded font-bold flex-shrink-0">TOP</span>}
                    <span className={`truncate ${isTop ? 'font-semibold text-gray-900 dark:text-white' : 'text-gray-600 dark:text-gray-400'}`}>{pred.name || 'Unknown'}</span>
                    {pred.type && <span className={`text-[10px] px-1.5 py-0.5 rounded-full flex-shrink-0 ${pred.type === 'malignant' ? 'bg-red-100 dark:bg-red-900/30 text-red-600' : pred.type === 'pre-cancerous' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600' : pred.type === 'disease' ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600' : pred.type === 'healthy' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600' : 'bg-green-100 dark:bg-green-900/30 text-green-600'}`}>{pred.type}</span>}
                  </div>
                  <span className={`font-bold min-w-[55px] text-right ${getConfidenceColor(conf)}`}>{pct}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 overflow-hidden">
                  <div className={`h-full rounded-full transition-all duration-700 ${isTop ? 'bg-blue-500' : pred.type === 'malignant' ? 'bg-red-400' : pred.type === 'disease' ? 'bg-orange-400' : 'bg-gray-400'}`} style={{ width: `${Math.max(1, parseFloat(pct))}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const currentAnalysisType = analysisTypes.find(t => t.id === analysisType);

  // ================================================================
  //                    RENDER
  // ================================================================

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
      <div>
        <h1 className="page-title"><ImageIcon className="text-purple-500" />Medical Image Analysis</h1>
        <p className="page-subtitle">Upload medical images or ECG signal files for AI-powered disease detection.</p>
      </div>

      <div className="alert-warning">
        <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold">Important Medical Disclaimer</p>
          <p className="text-sm mt-1">This AI tool provides preliminary analysis only. Always consult qualified healthcare professionals.</p>
        </div>
      </div>

      {/* Analysis Type Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Select Analysis Type</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {analysisTypes.map((type) => (
            <button key={type.id} onClick={() => { setAnalysisType(type.id); setResult(null); setError(null); setValidationError(null); }}
              className={`p-4 rounded-xl border-2 text-left transition-all hover:shadow-md ${analysisType === type.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-md ring-2 ring-blue-200 dark:ring-blue-800' : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'}`}>
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${type.color} flex items-center justify-center mb-3`}>
                <type.icon className="text-white" size={20} />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">{type.title}</h4>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{type.description}</p>
              <div className="flex items-center gap-1 text-xs">
                <span className="font-medium text-blue-600 dark:text-blue-400">Accepts:</span>
                <span className="text-gray-600 dark:text-gray-400">
                  {type.acceptsSignal ? 'Images + Signal files' : type.imageType === 'color' ? 'Color photos' : 'Grayscale images'}
                </span>
              </div>
              {type.acceptsSignal && (
                <div className="mt-1 flex items-center gap-1">
                  <FileUp size={10} className="text-green-500" />
                  <span className="text-[10px] text-green-600 dark:text-green-400 font-medium">.dat .hea .csv supported</span>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-2 gap-6">

        {/* Left - Upload */}
        <div className="space-y-4">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Upload File</h3>

            <input ref={fileInputRef} type="file" accept={currentAnalysisType?.fileAccept || 'image/*'}
              onChange={handleFileSelect} className="hidden" />

            {!preview && fileType !== 'signal' ? (
              <div onClick={() => fileInputRef.current?.click()} onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${isDragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 scale-[1.02]' : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/10'}`}>
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-600 dark:text-gray-400 font-medium">{isDragging ? 'Drop your file here!' : 'Click to upload or drag and drop'}</p>
                <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">{currentAnalysisType?.accepts}</p>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">PNG, JPG, JPEG up to 32MB</p>
                {currentAnalysisType?.acceptsSignal && (
                  <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                    <p className="text-xs text-green-700 dark:text-green-400 font-medium flex items-center gap-1 justify-center">
                      <FileUp size={12} /> ECG Signal Files Supported
                    </p>
                    <p className="text-[10px] text-green-600 dark:text-green-500 mt-1">
                      Upload .dat, .hea, .csv, .edf files from PTB-XL or other ECG datasets
                    </p>
                  </div>
                )}
              </div>
            ) : fileType === 'signal' && selectedFile ? (
              <div className="relative p-6 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-xl border-2 border-red-200 dark:border-red-800">
                <button onClick={clearSelection} className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-all shadow-lg" title="Remove file"><X size={16} /></button>
                <div className="text-center">
                  <div className="w-16 h-16 bg-red-100 dark:bg-red-900/40 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Heart className="text-red-500" size={32} />
                  </div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-1">ECG Signal File Loaded</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                  <div className="mt-3 px-3 py-1.5 bg-green-100 dark:bg-green-900/30 rounded-full inline-flex items-center gap-1">
                    <CheckCircle size={12} className="text-green-500" />
                    <span className="text-xs text-green-700 dark:text-green-400 font-medium">Ready for analysis</span>
                  </div>
                </div>

                {/* .hea companion file upload — required for PTB-XL .dat files */}
                {selectedFile.name.toLowerCase().endsWith('.dat') && (
                  <div className="mt-4 pt-4 border-t border-red-200 dark:border-red-700">
                    <input
                      ref={heaFileInputRef}
                      type="file"
                      accept=".hea"
                      className="hidden"
                      onChange={(e) => setHeaFile(e.target.files?.[0] || null)}
                    />
                    {heaFile ? (
                      <div className="flex items-center gap-2 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
                        <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-green-700 dark:text-green-400 truncate">{heaFile.name}</p>
                          <p className="text-[10px] text-green-600 dark:text-green-500">Header file attached ✓</p>
                        </div>
                        <button
                          onClick={() => { setHeaFile(null); if (heaFileInputRef.current) heaFileInputRef.current.value = ''; }}
                          className="text-green-600 hover:text-red-500 transition-colors"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => heaFileInputRef.current?.click()}
                        className="w-full flex items-center justify-center gap-2 p-2.5 border border-dashed border-amber-400 dark:border-amber-600 rounded-lg text-xs text-amber-700 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
                      >
                        <FileUp size={14} />
                        <span>
                          <span className="font-semibold">Attach .hea header file</span>
                          <span className="text-amber-600 dark:text-amber-500"> (recommended for PTB-XL .dat files)</span>
                        </span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="relative group">
                <img src={preview} alt="Preview" className="w-full h-72 object-contain bg-gray-100 dark:bg-gray-800 rounded-xl" />
                <button onClick={clearSelection} className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-all shadow-lg opacity-80 group-hover:opacity-100" title="Remove"><X size={16} /></button>
              </div>
            )}

            {selectedFile && fileType === 'image' && (
              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg flex items-center gap-3">
                <ImageIcon className="text-gray-400" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
            )}

            <button onClick={handleAnalyze} disabled={!selectedFile || isLoading}
              className="w-full btn-primary mt-4 py-3 disabled:opacity-50 disabled:cursor-not-allowed">
              {isLoading ? (<><Loader2 className="animate-spin" size={20} /> Analyzing...</>) : (
                <><Camera size={20} /> Analyze {fileType === 'signal' ? 'ECG Signal' : 'Image'} for {currentAnalysisType?.title}</>
              )}
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
              <AlertCircle size={18} className="flex-shrink-0 text-red-500 mt-0.5" />
              <span className="text-sm text-red-700 dark:text-red-400">{error}</span>
            </div>
          )}

          {validationError && (
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl">
              <div className="flex items-start gap-3">
                <XCircle className="text-orange-500 flex-shrink-0 mt-0.5" size={24} />
                <div className="flex-1">
                  <h4 className="font-semibold text-orange-800 dark:text-orange-300 flex items-center gap-2"><AlertTriangle size={16} /> Wrong File Type</h4>
                  <p className="text-sm text-orange-700 dark:text-orange-400 mt-1">{validationError.message}</p>
                  <div className="mt-3 p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                    <p className="text-sm font-medium text-orange-800 dark:text-orange-300">💡 {validationError.suggestion}</p>
                  </div>
                  <button onClick={clearSelection} className="mt-3 w-full px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors text-sm font-medium">
                    Upload Correct File
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right - Results */}
        <div className="space-y-4">
          {result ? (
            <div className="card animate-scale-in">
              <div className="flex items-center gap-3 mb-6">
                {getSeverityIcon(result.severity)}
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Analysis Complete</h3>
                  {result.demo_mode && <span className="text-xs text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 rounded-full font-medium">⚠️ Demo Mode</span>}
                  {result.signal_processed && <span className="text-xs text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded-full font-medium ml-1">📊 Signal Analysis</span>}
                </div>
              </div>

              {result.prediction && (
                <div className={`p-5 rounded-xl border-2 mb-4 ${getSeverityStyles(result.severity).bg} ${getSeverityStyles(result.severity).border} ${getSeverityStyles(result.severity).glow}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className={`text-sm font-medium ${getSeverityStyles(result.severity).text}`}>Detected Condition</span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${getSeverityStyles(result.severity).badge}`}>{result.severity}</span>
                  </div>
                  <p className={`text-2xl font-bold ${getSeverityStyles(result.severity).text} mb-1`}>{result.prediction.name}</p>
                  {result.prediction.type && <p className={`text-sm ${getSeverityStyles(result.severity).text} opacity-75 mb-1`}>Type: <span className="font-medium">{result.prediction.type}</span></p>}
                  {result.prediction.code && <p className={`text-xs ${getSeverityStyles(result.severity).text} opacity-60`}>Code: {result.prediction.code}</p>}
                  {result.prediction.birads && <p className={`text-sm ${getSeverityStyles(result.severity).text} opacity-75 mt-1`}>{result.prediction.birads}</p>}
                  <ConfidenceBar confidence={result.prediction.confidence} severity={result.severity} />
                </div>
              )}

              {renderStaging(result.staging)}
              {renderUrgency(result.urgency)}
              {renderTreatmentOptions(result.treatment_options)}
              {renderRecommendations(result.recommendations)}
              {renderAllPredictions(result.all_predictions)}

              {result.note && (
                <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                  <p className="text-sm text-amber-700 dark:text-amber-400"><strong>⚠️ Note:</strong> {result.note}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="card">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2"><Info size={20} className="text-blue-500" />How It Works</h3>
              <div className="space-y-4 text-sm text-gray-600 dark:text-gray-400">
                {[{ n: '1', t: 'Select Analysis Type', d: 'Choose the type of medical analysis' }, { n: '2', t: 'Upload File', d: 'Upload an image or ECG signal file (.dat)' }, { n: '3', t: 'Get Results', d: 'Receive detailed results with confidence levels' }].map(s => (
                  <div key={s.n} className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0"><span className="text-blue-600 dark:text-blue-400 font-bold">{s.n}</span></div>
                    <div><p className="font-medium text-gray-900 dark:text-white">{s.t}</p><p>{s.d}</p></div>
                  </div>
                ))}
              </div>

              {/* Signal file info */}
              <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800">
                <p className="text-sm text-green-800 dark:text-green-300 font-medium mb-2 flex items-center gap-2"><FileUp size={16} />ECG Signal File Support (Heart Analysis)</p>
                <div className="space-y-1 text-xs text-green-700 dark:text-green-400">
                  <p>• <strong>.dat / .hea files</strong> - PTB-XL, PhysioNet WFDB format</p>
                  <p>• <strong>.csv files</strong> - Comma-separated ECG data</p>
                  <p>• <strong>.edf files</strong> - European Data Format</p>
                  <p className="mt-2 text-green-600 dark:text-green-500 italic">Signal files are converted to ECG images server-side for analysis</p>
                </div>
              </div>

              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-800 dark:text-blue-300 font-medium mb-3 flex items-center gap-2"><ImageIcon size={16} />Image Requirements:</p>
                <div className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                  {[{ icon: Microscope, label: 'Skin Cancer', desc: 'Color photos of lesions/moles' }, { icon: Stethoscope, label: 'Chest X-ray', desc: 'Grayscale X-ray images' }, { icon: Activity, label: 'Breast Cancer', desc: 'Grayscale mammograms' }, { icon: Heart, label: 'Heart/ECG', desc: 'ECG images OR .dat signal files' }].map(i => (
                    <div key={i.label} className="flex items-start gap-2"><i.icon size={14} className="mt-0.5 flex-shrink-0" /><div><strong>{i.label}:</strong> {i.desc}</div></div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageAnalysis;