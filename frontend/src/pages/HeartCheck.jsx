import React, { useState } from 'react';
import axios from 'axios';
import {
  HeartPulse,
  Activity,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info,
  TrendingUp,
  Shield
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function HeartCheck() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    cp: '',
    trestbps: '',
    chol: '',
    fbs: '',
    restecg: '',
    thalach: '',
    exang: '',
    oldpeak: '',
    slope: '',
    ca: '',
    thal: ''
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate all fields are filled
    const emptyFields = Object.entries(formData).filter(([_, value]) => value === '');
    if (emptyFields.length > 0) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert form data to numbers
      const numericData = {};
      for (const [key, value] of Object.entries(formData)) {
        numericData[key] = parseFloat(value);
      }

      const response = await axios.post('http://localhost:5000/predict-heart', numericData);

      if (response.data.success) {
        setResult(response.data);
        addToHistory({
          type: 'heart',
          prediction: response.data.risk_level,
          confidence: response.data.confidence,
          probability: response.data.probability,
          details: response.data.recommendation
        });
        showNotification('Heart health analysis complete', 'success');
      } else {
        setError(response.data.error || 'Failed to analyze');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please ensure the backend is running.');
      console.error('Heart check error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const formFields = [
    { name: 'age', label: 'Age', type: 'number', placeholder: 'Years', min: 20, max: 100 },
    { 
      name: 'sex', label: 'Sex', type: 'select', 
      options: [{ value: '', label: 'Select' }, { value: '0', label: 'Female' }, { value: '1', label: 'Male' }]
    },
    { 
      name: 'cp', label: 'Chest Pain Type', type: 'select',
      options: [
        { value: '', label: 'Select' },
        { value: '0', label: 'Typical Angina' },
        { value: '1', label: 'Atypical Angina' },
        { value: '2', label: 'Non-anginal Pain' },
        { value: '3', label: 'Asymptomatic' }
      ]
    },
    { name: 'trestbps', label: 'Resting Blood Pressure', type: 'number', placeholder: 'mm Hg', min: 90, max: 200 },
    { name: 'chol', label: 'Cholesterol', type: 'number', placeholder: 'mg/dl', min: 100, max: 600 },
    { 
      name: 'fbs', label: 'Fasting Blood Sugar > 120', type: 'select',
      options: [{ value: '', label: 'Select' }, { value: '0', label: 'No' }, { value: '1', label: 'Yes' }]
    },
    { 
      name: 'restecg', label: 'Resting ECG', type: 'select',
      options: [
        { value: '', label: 'Select' },
        { value: '0', label: 'Normal' },
        { value: '1', label: 'ST-T Wave Abnormality' },
        { value: '2', label: 'Left Ventricular Hypertrophy' }
      ]
    },
    { name: 'thalach', label: 'Max Heart Rate', type: 'number', placeholder: 'bpm', min: 60, max: 220 },
    { 
      name: 'exang', label: 'Exercise Induced Angina', type: 'select',
      options: [{ value: '', label: 'Select' }, { value: '0', label: 'No' }, { value: '1', label: 'Yes' }]
    },
    { name: 'oldpeak', label: 'ST Depression', type: 'number', placeholder: 'Value', step: '0.1', min: 0, max: 7 },
    { 
      name: 'slope', label: 'Slope of ST Segment', type: 'select',
      options: [
        { value: '', label: 'Select' },
        { value: '0', label: 'Upsloping' },
        { value: '1', label: 'Flat' },
        { value: '2', label: 'Downsloping' }
      ]
    },
    { 
      name: 'ca', label: 'Major Vessels Colored', type: 'select',
      options: [
        { value: '', label: 'Select' },
        { value: '0', label: '0' },
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' }
      ]
    },
    { 
      name: 'thal', label: 'Thalassemia', type: 'select',
      options: [
        { value: '', label: 'Select' },
        { value: '0', label: 'Normal' },
        { value: '1', label: 'Fixed Defect' },
        { value: '2', label: 'Reversible Defect' },
        { value: '3', label: 'Unknown' }
      ]
    }
  ];

  const getRiskColor = (level) => {
    switch (level) {
      case 'High': return 'text-red-600 bg-red-100 dark:bg-red-900/30';
      case 'Moderate': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/30';
      case 'Low': return 'text-emerald-600 bg-emerald-100 dark:bg-emerald-900/30';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-6xl mx-auto">
      <div>
        <h1 className="page-title flex items-center gap-3">
          <HeartPulse className="text-red-500" />
          Heart Health Check
        </h1>
        <p className="page-subtitle">
          Assess your cardiovascular health risk using AI-powered analysis.
        </p>
      </div>

    
      <div className="alert-warning">
        <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Medical Disclaimer</p>
          <p className="text-sm mt-1">
            This tool provides educational insights only and should not replace professional 
            medical advice. Always consult a cardiologist for proper heart health evaluation.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
    
        <div className="lg:col-span-2">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <Activity size={20} className="text-blue-600" />
              Enter Your Health Metrics
            </h3>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <label className="input-label">{field.label}</label>
                    {field.type === 'select' ? (
                      <select
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleChange}
                        className="select-field"
                      >
                        {field.options.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={field.type}
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleChange}
                        placeholder={field.placeholder}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        className="input-field"
                      />
                    )}
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
                className="btn-primary w-full py-3"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <HeartPulse size={20} />
                    Analyze Heart Health
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

       
        <div className="space-y-4">
          {result ? (
            <div className="card animate-scale-in">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle className="text-emerald-600" size={24} />
                <h3 className="font-semibold text-gray-900 dark:text-white">Analysis Result</h3>
              </div>

              <div className="space-y-4">
               
                <div className="text-center p-6 rounded-xl bg-gray-50 dark:bg-gray-800">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Risk Level</p>
                  <span className={`inline-block px-4 py-2 rounded-full text-lg font-bold ${getRiskColor(result.risk_level)}`}>
                    {result.risk_level} Risk
                  </span>
                </div>

               
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-500 dark:text-gray-400">Risk Probability</span>
                    <span className="font-medium">{(result.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-bar-fill"
                      style={{ 
                        width: `${result.probability * 100}%`,
                        background: result.probability > 0.7 ? 'linear-gradient(90deg, #ef4444, #dc2626)' :
                                   result.probability > 0.4 ? 'linear-gradient(90deg, #f59e0b, #d97706)' :
                                   'linear-gradient(90deg, #10b981, #059669)'
                      }}
                    />
                  </div>
                </div>

               
                {result.recommendation && (
                  <div className={`p-4 rounded-xl ${
                    result.recommendation.level === 'critical' ? 'bg-red-50 dark:bg-red-900/20' :
                    result.recommendation.level === 'warning' ? 'bg-amber-50 dark:bg-amber-900/20' :
                    'bg-emerald-50 dark:bg-emerald-900/20'
                  }`}>
                    <p className={`font-medium mb-2 ${
                      result.recommendation.level === 'critical' ? 'text-red-700 dark:text-red-400' :
                      result.recommendation.level === 'warning' ? 'text-amber-700 dark:text-amber-400' :
                      'text-emerald-700 dark:text-emerald-400'
                    }`}>
                      {result.recommendation.message}
                    </p>
                    <ul className="space-y-1 mt-3">
                      {result.recommendation.actions.map((action, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <span className="text-blue-600">•</span>
                          {action}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Info size={20} className="text-blue-600" />
                How It Works
              </h3>
              <div className="space-y-4 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 font-bold text-sm">1</span>
                  </div>
                  <p>Enter your health metrics and medical test results</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 font-bold text-sm">2</span>
                  </div>
                  <p>Our AI analyzes your data using trained models</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 font-bold text-sm">3</span>
                  </div>
                  <p>Get personalized risk assessment and recommendations</p>
                </div>
              </div>
            </div>
          )}

          
          <div className="card bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-red-100 dark:border-red-800">
            <div className="flex items-center gap-3 mb-3">
              <Shield className="text-red-600" size={20} />
              <span className="font-medium text-gray-900 dark:text-white">Heart Health Tips</span>
            </div>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Exercise for 30 minutes daily</li>
              <li>• Maintain a heart-healthy diet</li>
              <li>• Monitor blood pressure regularly</li>
              <li>• Avoid smoking and limit alcohol</li>
              <li>• Manage stress effectively</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HeartCheck;