import React, { useState, useRef } from 'react';
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
  CalendarClock,
  FileText
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function ImageAnalysis() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();
  
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analysisType, setAnalysisType] = useState('skin');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  // Analysis types configuration
  const analysisTypes = [
    {
      id: 'skin',
      title: 'Skin Cancer Analysis',
      description: 'Detect skin cancer, melanoma, and other skin conditions from photos',
      icon: Microscope,
      color: 'from-purple-500 to-purple-600',
      accepts: 'Photos of skin lesions, moles, or suspicious spots',
      expectedType: 'Color photograph of skin',
      imageType: 'color'
    },
    {
      id: 'breast',
      title: 'Breast Cancer Screening',
      description: 'Analyze mammograms and breast ultrasounds for abnormalities',
      icon: Activity,
      color: 'from-pink-500 to-rose-600',
      accepts: 'Mammogram images, breast ultrasound images',
      expectedType: 'Grayscale mammogram or ultrasound',
      imageType: 'grayscale'
    },
    {
      id: 'heart',
      title: 'Heart Condition Analysis',
      description: 'Detect heart conditions from ECG images and cardiac scans',
      icon: Heart,
      color: 'from-red-500 to-red-600',
      accepts: 'ECG printouts, echocardiogram images',
      expectedType: 'ECG/EKG printout or scan',
      imageType: 'grayscale'
    },
    {
      id: 'xray',
      title: 'Chest X-Ray Analysis',
      description: 'Detect pneumonia and lung conditions from chest X-rays',
      icon: Stethoscope,
      color: 'from-blue-500 to-blue-600',
      accepts: 'Chest X-ray images',
      expectedType: 'Grayscale chest X-ray',
      imageType: 'grayscale'
    }
  ];

  // ==============================================================================
  //                           FILE HANDLING
  // ==============================================================================

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file (PNG, JPG, etc.)');
        showNotification('Invalid file type', 'error');
        return;
      }
      if (file.size > 32 * 1024 * 1024) {
        setError('File size must be less than 32MB');
        showNotification('File too large', 'error');
        return;
      }
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
      setResult(null);
      setValidationError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
      setResult(null);
      setValidationError(null);
    } else {
      setError('Please drop a valid image file');
      showNotification('Invalid file', 'error');
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setValidationError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      showNotification('No image selected', 'error');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setValidationError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    const endpoints = {
      skin: 'http://localhost:5000/analyze/skin',
      breast: 'http://localhost:5000/analyze/breast',
      heart: 'http://localhost:5000/analyze/heart',
      xray: 'http://localhost:5000/analyze/xray'
    };

    try {
      const response = await axios.post(endpoints[analysisType], formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000 // 60 second timeout
      });

      if (response.data.success) {
        setResult(response.data);
        
        // Add to history
        addToHistory({
          type: `image_${analysisType}`,
          prediction: response.data.prediction?.name || 'Unknown',
          confidence: response.data.prediction?.confidence || 0,
          severity: response.data.severity || 'unknown',
          timestamp: new Date().toISOString()
        });
        
        showNotification('Analysis complete!', 'success');
      } else {
        // Check if it's a validation error
        if (response.data.validation_error) {
          setValidationError({
            message: response.data.message,
            suggestion: response.data.suggestion,
            expectedType: response.data.expected_type
          });
          showNotification('Wrong image type detected', 'error');
        } else {
          setError(response.data.error || 'Analysis failed');
          showNotification('Analysis failed', 'error');
        }
      }
    } catch (err) {
      console.error('Analysis error:', err);
      
      // Handle validation error from 400 response
      if (err.response?.status === 400 && err.response?.data?.validation_error) {
        setValidationError({
          message: err.response.data.message,
          suggestion: err.response.data.suggestion,
          expectedType: err.response.data.expected_type
        });
        showNotification('Wrong image type', 'error');
      } else if (err.response?.data?.error) {
        setError(err.response.data.error);
        showNotification('Analysis error', 'error');
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. The server took too long to respond.');
        showNotification('Request timeout', 'error');
      } else {
        setError('Failed to connect to server. Please ensure backend is running on http://localhost:5000');
        showNotification('Connection failed', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // ==============================================================================
  //                           STYLING HELPERS
  // ==============================================================================

  const getSeverityStyles = (severity) => {
    const styles = {
      critical: {
        bg: 'bg-red-100 dark:bg-red-900/30',
        text: 'text-red-700 dark:text-red-400',
        border: 'border-red-200 dark:border-red-800',
        badge: 'bg-red-500 text-white',
        progressBar: 'bg-red-500'
      },
      high: {
        bg: 'bg-orange-100 dark:bg-orange-900/30',
        text: 'text-orange-700 dark:text-orange-400',
        border: 'border-orange-200 dark:border-orange-800',
        badge: 'bg-orange-500 text-white',
        progressBar: 'bg-orange-500'
      },
      moderate: {
        bg: 'bg-yellow-100 dark:bg-yellow-900/30',
        text: 'text-yellow-700 dark:text-yellow-400',
        border: 'border-yellow-200 dark:border-yellow-800',
        badge: 'bg-yellow-500 text-white',
        progressBar: 'bg-yellow-500'
      },
      low: {
        bg: 'bg-green-100 dark:bg-green-900/30',
        text: 'text-green-700 dark:text-green-400',
        border: 'border-green-200 dark:border-green-800',
        badge: 'bg-green-500 text-white',
        progressBar: 'bg-green-500'
      },
      healthy: {
        bg: 'bg-blue-100 dark:bg-blue-900/30',
        text: 'text-blue-700 dark:text-blue-400',
        border: 'border-blue-200 dark:border-blue-800',
        badge: 'bg-blue-500 text-white',
        progressBar: 'bg-blue-500'
      }
    };
    return styles[severity] || styles.low;
  };

  const getSeverityIcon = (severity) => {
    if (severity === 'critical' || severity === 'high') {
      return <AlertTriangle className="text-red-500" size={24} />;
    } else if (severity === 'moderate') {
      return <AlertCircle className="text-yellow-500" size={24} />;
    } else {
      return <CheckCircle className="text-green-500" size={24} />;
    }
  };

  const getUrgencyColor = (color) => {
    const colors = {
      red: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400',
      orange: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800 text-orange-700 dark:text-orange-400',
      yellow: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-700 dark:text-yellow-400',
      green: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400',
      blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-400'
    };
    return colors[color] || colors.blue;
  };

  const getRecommendationColor = (level) => {
    const colors = {
      critical: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
      high: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
      moderate: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      low: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      healthy: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
    };
    return colors[level] || colors.low;
  };

  // ==============================================================================
  //                           RESULT RENDERING COMPONENTS
  // ==============================================================================

  const renderStaging = (staging) => {
    if (!staging || typeof staging !== 'object') return null;

    return (
      <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
          <Activity size={18} className="text-blue-500" />
          Staging Information
        </h4>
        <div className="space-y-2 text-sm">
          {staging.stage && (
            <div className="flex items-start gap-2">
              <span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Stage:</span>
              <span className="text-gray-900 dark:text-white font-semibold">{staging.stage}</span>
            </div>
          )}
          {staging.description && (
            <div className="flex items-start gap-2">
              <span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Details:</span>
              <span className="text-gray-600 dark:text-gray-400">{staging.description}</span>
            </div>
          )}
          {staging.prognosis && (
            <div className="flex items-start gap-2">
              <span className="font-medium text-gray-700 dark:text-gray-300 min-w-[80px]">Prognosis:</span>
              <span className="text-gray-600 dark:text-gray-400">{staging.prognosis}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderUrgency = (urgency) => {
    if (!urgency || typeof urgency !== 'object') return null;

    return (
      <div className={`mb-4 p-4 rounded-xl border ${getUrgencyColor(urgency.color)}`}>
        <div className="flex items-center gap-2 mb-1">
          <Clock size={16} />
          <span className="font-semibold">{urgency.timeline || 'Timeline not specified'}</span>
        </div>
        <p className="text-sm opacity-90">{urgency.action || 'Action not specified'}</p>
      </div>
    );
  };

  const renderTreatmentOptions = (treatments) => {
    if (!treatments || !Array.isArray(treatments) || treatments.length === 0) return null;

    return (
      <div className="mb-4">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
          <Pill size={18} className="text-green-500" />
          Treatment Options
        </h4>
        <ul className="space-y-2">
          {treatments.slice(0, 8).map((treatment, idx) => (
            <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
              <ChevronRight className="text-blue-500 mt-0.5 flex-shrink-0" size={14} />
              <span>{treatment}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderRecommendations = (recommendations) => {
    if (!recommendations || typeof recommendations !== 'object') return null;

    return (
      <div className={`p-4 rounded-xl border ${getRecommendationColor(recommendations.level)}`}>
        {recommendations.title && (
          <h4 className="font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
            <Shield size={18} />
            {recommendations.title}
          </h4>
        )}
        {recommendations.message && (
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            {recommendations.message}
          </p>
        )}
        {recommendations.actions && Array.isArray(recommendations.actions) && (
          <div className="space-y-2">
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Recommended Actions:</p>
            {recommendations.actions.map((action, idx) => (
              <p key={idx} className="text-sm flex items-start gap-2 text-gray-600 dark:text-gray-400">
                <span className="font-bold text-blue-500 min-w-[20px]">{idx + 1}.</span>
                <span>{action}</span>
              </p>
            ))}
          </div>
        )}
        {recommendations.next_steps && Array.isArray(recommendations.next_steps) && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Next Steps:</p>
            {recommendations.next_steps.map((step, idx) => (
              <p key={idx} className="text-sm text-gray-600 dark:text-gray-400">• {step}</p>
            ))}
          </div>
        )}
        {recommendations.warning_signs && Array.isArray(recommendations.warning_signs) && (
          <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800">
            <p className="text-xs font-semibold text-red-600 dark:text-red-400 mb-2">⚠️ Warning Signs:</p>
            {recommendations.warning_signs.map((sign, idx) => (
              <p key={idx} className="text-sm text-red-600 dark:text-red-400">• {sign}</p>
            ))}
          </div>
        )}
        {recommendations.risk_factors && Array.isArray(recommendations.risk_factors) && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Risk Factors Identified:</p>
            {recommendations.risk_factors.map((factor, idx) => (
              <p key={idx} className="text-sm text-gray-600 dark:text-gray-400">• {factor}</p>
            ))}
          </div>
        )}
        {recommendations.note && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              <strong>Note:</strong> {recommendations.note}
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderAllPredictions = (predictions) => {
    if (!predictions || !Array.isArray(predictions) || predictions.length <= 1) return null;

    return (
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-sm flex items-center gap-2">
          <FileText size={16} />
          All Detected Conditions
        </h4>
        <div className="space-y-2">
          {predictions.slice(0, 5).map((pred, idx) => (
            <div key={idx} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <span className="text-gray-600 dark:text-gray-400">{pred.name || 'Unknown'}</span>
                {pred.type && (
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    pred.type === 'malignant' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' :
                    pred.type === 'pre-cancerous' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400' :
                    'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                  }`}>
                    {pred.type}
                  </span>
                )}
                {pred.birads && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {pred.birads}
                  </span>
                )}
              </div>
              <span className="font-medium text-gray-900 dark:text-white">
                {typeof pred.confidence === 'number' 
                  ? `${(pred.confidence * 100).toFixed(1)}%` 
                  : 'N/A'}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Get current analysis type info
  const currentAnalysisType = analysisTypes.find(t => t.id === analysisType);

  // ==============================================================================
  //                           RENDER MAIN COMPONENT
  // ==============================================================================

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="page-title">
          <ImageIcon className="text-purple-500" />
          Medical Image Analysis
        </h1>
        <p className="page-subtitle">
          Upload medical images for AI-powered disease detection with comprehensive staging and recommendations.
        </p>
      </div>

      {/* Medical Disclaimer */}
      <div className="alert-warning">
        <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold">Important Medical Disclaimer</p>
          <p className="text-sm mt-1">
            This AI tool provides preliminary analysis only and is NOT a substitute for professional medical diagnosis. 
            Results should NOT be used for self-diagnosis or treatment decisions. 
            Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment. 
            This tool is for educational and screening purposes only.
          </p>
        </div>
      </div>

      {/* Analysis Type Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Select Analysis Type
        </h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {analysisTypes.map((type) => (
            <button
              key={type.id}
              onClick={() => {
                setAnalysisType(type.id);
                setResult(null);
                setError(null);
                setValidationError(null);
              }}
              className={`p-4 rounded-xl border-2 text-left transition-all hover:shadow-md ${
                analysisType === type.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-md'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${type.color} flex items-center justify-center mb-3`}>
                <type.icon className="text-white" size={20} />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">{type.title}</h4>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{type.description}</p>
              <div className="flex items-center gap-1 text-xs">
                <span className="font-medium text-blue-600 dark:text-blue-400">Expected:</span>
                <span className="text-gray-600 dark:text-gray-400">{type.imageType === 'color' ? 'Color' : 'Grayscale'}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        
        {/* Left Column - Upload Section */}
        <div className="space-y-4">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Upload Image
            </h3>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {!preview ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors"
              >
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-600 dark:text-gray-400 font-medium">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">
                  {currentAnalysisType?.accepts}
                </p>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                  PNG, JPG, JPEG up to 32MB
                </p>
                <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <Info size={14} className="text-blue-500" />
                  <span className="text-xs text-blue-600 dark:text-blue-400">
                    Must be {currentAnalysisType?.imageType === 'color' ? 'color' : 'grayscale'} image
                  </span>
                </div>
              </div>
            ) : (
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-72 object-contain bg-gray-100 dark:bg-gray-800 rounded-xl"
                />
                <button
                  onClick={clearSelection}
                  className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg"
                  title="Remove image"
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {selectedFile && (
              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg flex items-center gap-3">
                <ImageIcon className="text-gray-400" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isLoading}
              className="w-full btn-primary mt-4 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Analyzing Image...
                </>
              ) : (
                <>
                  <Camera size={20} />
                  Analyze for {currentAnalysisType?.title}
                </>
              )}
            </button>
          </div>

          {/* Regular Error Message */}
          {error && (
            <div className="alert-error animate-shake">
              <AlertCircle size={18} className="flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Validation Error Message - NEW! */}
          {validationError && (
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl animate-shake">
              <div className="flex items-start gap-3">
                <XCircle className="text-orange-500 flex-shrink-0 mt-0.5" size={24} />
                <div className="flex-1">
                  <h4 className="font-semibold text-orange-800 dark:text-orange-300 flex items-center gap-2">
                    <AlertTriangle size={16} />
                    Wrong Image Type Detected
                  </h4>
                  <p className="text-sm text-orange-700 dark:text-orange-400 mt-1">
                    {validationError.message}
                  </p>
                  
                  <div className="mt-3 p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                    <p className="text-sm font-medium text-orange-800 dark:text-orange-300 flex items-center gap-2">
                      💡 Suggestion:
                    </p>
                    <p className="text-sm text-orange-700 dark:text-orange-400 mt-1">
                      {validationError.suggestion}
                    </p>
                  </div>
                  
                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 bg-white dark:bg-gray-800 rounded">
                      <p className="text-gray-500 dark:text-gray-400">You uploaded:</p>
                      <p className="font-semibold text-gray-900 dark:text-white">
                        {selectedFile?.type.includes('png') ? 'PNG Image' : 
                         selectedFile?.type.includes('jpeg') || selectedFile?.type.includes('jpg') ? 'JPEG Image' : 
                         'Image File'}
                      </p>
                    </div>
                    <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                      <p className="text-blue-600 dark:text-blue-400">Expected:</p>
                      <p className="font-semibold text-blue-700 dark:text-blue-300">
                        {validationError.expectedType}
                      </p>
                    </div>
                  </div>
                  
                  <button
                    onClick={clearSelection}
                    className="mt-3 w-full px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors text-sm font-medium"
                  >
                    Upload Correct Image Type
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Results Section */}
        <div className="space-y-4">
          {result ? (
            <div className="card animate-scale-in">
              {/* Result Header */}
              <div className="flex items-center gap-3 mb-6">
                {getSeverityIcon(result.severity)}
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
                    Analysis Complete
                  </h3>
                  {result.demo_mode && (
                    <span className="text-xs text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 rounded-full">
                      Demo Mode - Train models for real predictions
                    </span>
                  )}
                </div>
              </div>

              {/* Main Prediction Card */}
              {result.prediction && (
                <div className={`p-4 rounded-xl border mb-4 ${getSeverityStyles(result.severity).bg} ${getSeverityStyles(result.severity).border}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm font-medium ${getSeverityStyles(result.severity).text}`}>
                      Detected Condition
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${getSeverityStyles(result.severity).badge}`}>
                      {result.severity || 'Unknown'}
                    </span>
                  </div>
                  
                  <p className={`text-2xl font-bold ${getSeverityStyles(result.severity).text} mb-1`}>
                    {result.prediction.name || 'Unknown Condition'}
                  </p>
                  
                  {result.prediction.birads && (
                    <p className={`text-sm ${getSeverityStyles(result.severity).text} opacity-75`}>
                      {result.prediction.birads}
                    </p>
                  )}
                  
                  {result.prediction.type && (
                    <p className={`text-sm ${getSeverityStyles(result.severity).text} opacity-75`}>
                      Type: {result.prediction.type}
                    </p>
                  )}
                  
                  {result.prediction.code && (
                    <p className={`text-xs ${getSeverityStyles(result.severity).text} opacity-60 mt-1`}>
                      Code: {result.prediction.code}
                    </p>
                  )}
                  
                  {/* Confidence Bar */}
                  <div className="mt-4">
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className={`${getSeverityStyles(result.severity).text} opacity-75`}>
                        Confidence Level
                      </span>
                      <span className={`font-medium ${getSeverityStyles(result.severity).text}`}>
                        {result.prediction.confidence_percent || 
                         (typeof result.prediction.confidence === 'number' 
                           ? `${(result.prediction.confidence * 100).toFixed(1)}%` 
                           : 'N/A')}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                      <div 
                        className={`h-2 rounded-full transition-all duration-500 ${getSeverityStyles(result.severity).progressBar}`}
                        style={{ 
                          width: `${(result.prediction.confidence || 0) * 100}%`,
                          minWidth: '2%'
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Staging Information */}
              {renderStaging(result.staging)}

              {/* Urgency Timeline */}
              {renderUrgency(result.urgency)}

              {/* Treatment Options */}
              {renderTreatmentOptions(result.treatment_options)}

              {/* Recommendations */}
              {renderRecommendations(result.recommendations)}

              {/* All Predictions */}
              {renderAllPredictions(result.all_predictions)}

              {/* Note/Warning */}
              {result.note && (
                <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    <strong>Note:</strong> {result.note}
                  </p>
                </div>
              )}
            </div>
          ) : (
            /* Instructions when no result */
            <div className="card">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Info size={20} className="text-blue-500" />
                How It Works
              </h3>
              
              <div className="space-y-4 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 dark:text-blue-400 font-bold">1</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">Select Analysis Type</p>
                    <p>Choose the type of medical image you want to analyze</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 dark:text-blue-400 font-bold">2</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">Upload Correct Image Type</p>
                    <p>The system validates that your image matches the expected type</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 dark:text-blue-400 font-bold">3</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">Get Comprehensive Analysis</p>
                    <p>Receive detailed results with staging, treatment options, and recommendations</p>
                  </div>
                </div>
              </div>

              {/* Image Type Requirements */}
              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-800 dark:text-blue-300 font-medium mb-3 flex items-center gap-2">
                  <ImageIcon size={16} />
                  Image Type Requirements:
                </p>
                <div className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                  <div className="flex items-start gap-2">
                    <Microscope size={14} className="mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Skin Cancer:</strong> Color photos of lesions/moles (RGB)
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Stethoscope size={14} className="mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Chest X-ray:</strong> Grayscale chest X-ray images
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Activity size={14} className="mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Breast Cancer:</strong> Grayscale mammograms or ultrasound
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Heart size={14} className="mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Heart/ECG:</strong> ECG printouts or heart scan images
                    </div>
                  </div>
                </div>
              </div>

              {/* Best Practices */}
              <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
                <p className="text-sm text-amber-800 dark:text-amber-300 font-medium mb-2">
                  📸 For Best Results:
                </p>
                <ul className="space-y-1 text-sm text-amber-700 dark:text-amber-400">
                  <li>• Clear and in focus</li>
                  <li>• Well-lit without shadows</li>
                  <li>• Showing the area of concern clearly</li>
                  <li>• Original medical images (not photos of screens)</li>
                  <li>• Correct image type (color vs grayscale)</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageAnalysis;