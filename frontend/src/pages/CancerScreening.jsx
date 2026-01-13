import React, { useState } from 'react';
import axios from 'axios';
import {
  Microscope,
  Activity,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info,
  Shield,
  AlertTriangle,
  FileText
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function CancerScreening() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();
  const [formData, setFormData] = useState({
    radius_mean: '',
    texture_mean: '',
    perimeter_mean: '',
    area_mean: '',
    smoothness_mean: '',
    compactness_mean: '',
    concavity_mean: '',
    concave_points_mean: '',
    symmetry_mean: '',
    fractal_dimension_mean: ''
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError(null);
  };

  const loadSampleData = (type) => {
    if (type === 'benign') {
      setFormData({
        radius_mean: '12.5',
        texture_mean: '17.2',
        perimeter_mean: '78.5',
        area_mean: '450',
        smoothness_mean: '0.09',
        compactness_mean: '0.07',
        concavity_mean: '0.04',
        concave_points_mean: '0.02',
        symmetry_mean: '0.17',
        fractal_dimension_mean: '0.06'
      });
    } else {
      setFormData({
        radius_mean: '18.5',
        texture_mean: '22.0',
        perimeter_mean: '120.0',
        area_mean: '1050',
        smoothness_mean: '0.11',
        compactness_mean: '0.18',
        concavity_mean: '0.20',
        concave_points_mean: '0.10',
        symmetry_mean: '0.21',
        fractal_dimension_mean: '0.07'
      });
    }
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const emptyFields = Object.entries(formData).filter(([_, value]) => value === '');
    if (emptyFields.length > 0) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const numericData = {};
      for (const [key, value] of Object.entries(formData)) {
        numericData[key] = parseFloat(value);
      }

      const response = await axios.post('http://localhost:5000/predict-cancer', numericData);

      if (response.data.success) {
        setResult(response.data);
        addToHistory({
          type: 'cancer',
          prediction: response.data.prediction,
          probability: response.data.probability,
          confidence: response.data.confidence,
          details: response.data.recommendation
        });
        if (showNotification) {
          showNotification('Cancer screening analysis complete', 'success');
        }
      } else {
        setError(response.data.error || 'Failed to analyze');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please ensure the backend is running.');
      console.error('Cancer screening error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const formFields = [
    { name: 'radius_mean', label: 'Radius (mean)', placeholder: 'e.g., 14.5', hint: '6-30' },
    { name: 'texture_mean', label: 'Texture (mean)', placeholder: 'e.g., 19.0', hint: '9-40' },
    { name: 'perimeter_mean', label: 'Perimeter (mean)', placeholder: 'e.g., 92.0', hint: '40-190' },
    { name: 'area_mean', label: 'Area (mean)', placeholder: 'e.g., 655.0', hint: '140-2500' },
    { name: 'smoothness_mean', label: 'Smoothness (mean)', placeholder: 'e.g., 0.096', hint: '0.05-0.16' },
    { name: 'compactness_mean', label: 'Compactness (mean)', placeholder: 'e.g., 0.104', hint: '0.02-0.35' },
    { name: 'concavity_mean', label: 'Concavity (mean)', placeholder: 'e.g., 0.088', hint: '0-0.43' },
    { name: 'concave_points_mean', label: 'Concave Points (mean)', placeholder: 'e.g., 0.049', hint: '0-0.20' },
    { name: 'symmetry_mean', label: 'Symmetry (mean)', placeholder: 'e.g., 0.181', hint: '0.10-0.30' },
    { name: 'fractal_dimension_mean', label: 'Fractal Dimension (mean)', placeholder: 'e.g., 0.063', hint: '0.05-0.10' }
  ];

  return (
    <div className="space-y-6 animate-fade-in max-w-6xl mx-auto">
      <div>
        <h1 className="page-title flex items-center gap-3">
          <Microscope className="text-purple-500" />
          Breast Cancer Screening
        </h1>
        <p className="page-subtitle">
          AI-powered analysis of tumor characteristics for breast cancer risk assessment.
        </p>
      </div>

      <div className="alert-warning">
        <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Important Medical Disclaimer</p>
          <p className="text-sm mt-1">
            This screening tool is for educational purposes only. It analyzes tumor characteristics 
            typically obtained from fine needle aspirate (FNA) tests. Always consult an oncologist 
            for proper diagnosis and treatment planning.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Form */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                <Activity size={20} className="text-purple-600" />
                Tumor Characteristics
              </h3>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => loadSampleData('benign')}
                  className="btn-sm btn-ghost text-emerald-600"
                >
                  Load Benign Sample
                </button>
                <button
                  type="button"
                  onClick={() => loadSampleData('malignant')}
                  className="btn-sm btn-ghost text-red-600"
                >
                  Load Malignant Sample
                </button>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <label className="input-label">
                      {field.label}
                      <span className="text-xs text-gray-400 ml-2">({field.hint})</span>
                    </label>
                    <input
                      type="number"
                      name={field.name}
                      value={formData[field.name]}
                      onChange={handleChange}
                      placeholder={field.placeholder}
                      step="any"
                      className="input-field"
                    />
                  </div>
                ))}
              </div>

              {error && (
                <div className="alert-error">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="btn-primary w-full py-3 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Microscope size={20} />
                    Analyze Tumor Characteristics
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Results Sidebar */}
        <div className="space-y-4">
          {result ? (
            <div className="card animate-scale-in">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle className="text-emerald-600" size={24} />
                <h3 className="font-semibold text-gray-900 dark:text-white">Analysis Result</h3>
              </div>

              <div className="space-y-4">
                {/* Prediction */}
                <div className="text-center p-6 rounded-xl bg-gray-50 dark:bg-gray-800">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Prediction</p>
                  <span className={`inline-block px-6 py-3 rounded-full text-lg font-bold ${
                    result.prediction === 'Malignant' 
                      ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                      : 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
                  }`}>
                    {result.prediction}
                  </span>
                </div>

                {/* Probability Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-500 dark:text-gray-400">Malignancy Probability</span>
                    <span className="font-medium">{(result.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-bar-fill"
                      style={{ 
                        width: `${result.probability * 100}%`,
                        background: result.probability > 0.5 
                          ? 'linear-gradient(90deg, #ef4444, #dc2626)' 
                          : 'linear-gradient(90deg, #10b981, #059669)'
                      }}
                    />
                  </div>
                </div>

                {/* Confidence */}
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Model Confidence</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Recommendations */}
                {result.recommendation && (
                  <div className={`p-4 rounded-xl ${
                    result.recommendation.level === 'critical' ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800' :
                    result.recommendation.level === 'warning' ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800' :
                    'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800'
                  }`}>
                    <p className={`font-medium mb-2 ${
                      result.recommendation.level === 'critical' ? 'text-red-700 dark:text-red-400' :
                      result.recommendation.level === 'warning' ? 'text-amber-700 dark:text-amber-400' :
                      'text-emerald-700 dark:text-emerald-400'
                    }`}>
                      {result.recommendation.message}
                    </p>
                    {result.recommendation.actions && result.recommendation.actions.length > 0 && (
                      <ul className="space-y-1 mt-3">
                        {result.recommendation.actions.map((action, index) => (
                          <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <span className="text-purple-600 mt-1">•</span>
                            {action}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Info size={20} className="text-purple-600" />
                About This Analysis
              </h3>
              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                <p>
                  This tool analyzes tumor characteristics from Fine Needle Aspirate (FNA) tests 
                  to predict whether a breast mass is benign or malignant.
                </p>
                <p>
                  The model was trained on the Wisconsin Breast Cancer dataset and analyzes 
                  10 features computed from digitized images of cell nuclei.
                </p>
              </div>
            </div>
          )}

          {/* Guidelines */}
          <div className="card bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border-purple-100 dark:border-purple-800">
            <div className="flex items-center gap-3 mb-3">
              <Shield className="text-purple-600" size={20} />
              <span className="font-medium text-gray-900 dark:text-white">Screening Recommendations</span>
            </div>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Monthly breast self-examinations</li>
              <li>• Annual clinical breast exams after age 40</li>
              <li>• Mammography as recommended by your doctor</li>
              <li>• Report any changes to your healthcare provider</li>
              <li>• Know your family history of breast cancer</li>
            </ul>
          </div>

          {/* Dataset Info */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <FileText size={18} className="text-gray-500" />
              <span className="font-medium text-gray-900 dark:text-white text-sm">Dataset Information</span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Model trained on the UCI Wisconsin Breast Cancer Dataset containing 569 samples 
              with 30 features computed from digitized images of FNA tests.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CancerScreening;