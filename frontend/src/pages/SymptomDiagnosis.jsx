import React, { useState } from 'react';
import axios from 'axios';
import { 
  Search, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Stethoscope,
  Shield,
  Info,
  AlertTriangle
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { symptoms, symptomCategories } from '../data/symptoms';

function SymptomDiagnosis() {
  const { addToHistory, isLoading, setIsLoading } = useApp();
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const filteredSymptoms = symptoms.filter(symptom => {
    const matchesSearch = symptom.label.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || symptom.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const toggleSymptom = (symptomId) => {
    setSelectedSymptoms(prev => 
      prev.includes(symptomId)
        ? prev.filter(id => id !== symptomId)
        : [...prev, symptomId]
    );
    setError(null);
  };

  const handleDiagnose = async () => {
    if (selectedSymptoms.length < 2) {
      setError('Please select at least 2 symptoms for diagnosis');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:5000/predict-disease', {
        symptoms: selectedSymptoms,
      });

      if (response.data.success) {
        const diagnosisData = response.data;
        
        // ─── confidence lives at the TOP LEVEL (server.py now sends it there) ───
        // Fallback: also check inside the nested prediction object for safety
        const rawConf =
          diagnosisData.confidence ??
          diagnosisData.prediction?.confidence ??
          0;
        const confidence = Math.max(0, Math.min(1, parseFloat(rawConf) || 0));
        
        // prediction may be an object {disease, confidence} OR a plain string
        const predictionRaw = diagnosisData.prediction;
        const predictionStr =
          predictionRaw && typeof predictionRaw === 'object'
            ? predictionRaw.disease || String(predictionRaw)
            : String(predictionRaw ?? 'Unknown');

        // confidence_percent: prefer top-level, then nested, then compute
        const confidencePct =
          diagnosisData.confidence_percent ||
          diagnosisData.prediction?.confidence_percent ||
          `${(confidence * 100).toFixed(1)}%`;

        setResult({
          ...diagnosisData,
          prediction: predictionStr,
          confidence,
          confidence_percent: confidencePct,
        });
        
        addToHistory({
          type: 'symptom',
          symptoms: selectedSymptoms.map(id => {
            const symptom = symptoms.find(s => s.id === id);
            return symptom ? symptom.label : id;
          }),
          prediction: predictionStr,
          confidence,
          description: diagnosisData.description,
          precautions: diagnosisData.precautions || [],
          recommendations: diagnosisData.recommendations || [],
          timestamp: new Date().toISOString(),
        });
      } else {
        setError(response.data.error || 'Failed to get diagnosis');
      }
    } catch (err) {
      console.error('Diagnosis error:', err);
      
      if (err.response) {
        // Server responded with error
        setError(err.response.data?.error || 'Server error occurred');
      } else if (err.request) {
        // Request made but no response
        setError('Failed to connect to the server. Please ensure the backend is running on http://localhost:5000');
      } else {
        // Error in request setup
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedSymptoms([]);
    setResult(null);
    setError(null);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    if (confidence >= 0.4) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getConfidenceText = (confidence) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Moderate Confidence';
    if (confidence >= 0.4) return 'Low-Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="page-title">Symptom Diagnosis</h1>
        <p className="text-gray-600 -mt-4 mb-6">
          Select your symptoms below to receive an AI-powered health assessment.
        </p>
      </div>

      <div className="card bg-amber-50 border-amber-200">
        <div className="flex items-start gap-3">
          <AlertCircle className="text-amber-600 flex-shrink-0 mt-0.5" size={20} />
          <div>
            <p className="font-medium text-amber-800">Medical Disclaimer</p>
            <p className="text-sm text-amber-700 mt-1">
              This AI tool provides informational insights only and is not a substitute for professional 
              medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare 
              provider with any questions about your health.
            </p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="card">
            <div className="flex flex-col md:flex-row gap-4 mb-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
                <input
                  type="text"
                  placeholder="Search symptoms..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input-field pl-10"
                />
              </div>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="input-field md:w-48"
              >
                {symptomCategories.map(category => (
                  <option key={category} value={category}>{category}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-96 overflow-y-auto pr-2">
              {filteredSymptoms.map(symptom => (
                <button
                  key={symptom.id}
                  onClick={() => toggleSymptom(symptom.id)}
                  className={selectedSymptoms.includes(symptom.id) 
                    ? 'symptom-checkbox-selected' 
                    : 'symptom-checkbox'
                  }
                >
                  <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                    selectedSymptoms.includes(symptom.id)
                      ? 'bg-primary-600 border-primary-600'
                      : 'border-gray-300'
                  }`}>
                    {selectedSymptoms.includes(symptom.id) && (
                      <CheckCircle className="text-white" size={14} />
                    )}
                  </div>
                  <div className="text-left">
                    <p className="font-medium text-gray-900">{symptom.label}</p>
                    <p className="text-xs text-gray-500">{symptom.category}</p>
                  </div>
                </button>
              ))}
            </div>

            {filteredSymptoms.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No symptoms found matching your search.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <Stethoscope size={20} className="text-primary-600" />
              Selected Symptoms ({selectedSymptoms.length})
            </h3>
            
            {selectedSymptoms.length === 0 ? (
              <p className="text-gray-500 text-sm">No symptoms selected yet</p>
            ) : (
              <div className="space-y-2 mb-4 max-h-60 overflow-y-auto">
                {selectedSymptoms.map(id => {
                  const symptom = symptoms.find(s => s.id === id);
                  return (
                    <div 
                      key={id}
                      className="flex items-center justify-between bg-primary-50 px-3 py-2 rounded-lg"
                    >
                      <span className="text-sm text-primary-700">{symptom?.label}</span>
                      <button
                        onClick={() => toggleSymptom(id)}
                        className="text-primary-600 hover:text-primary-800 font-bold text-lg leading-none"
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={clearSelection}
                disabled={selectedSymptoms.length === 0}
                className="flex-1 btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Clear
              </button>
              <button
                onClick={handleDiagnose}
                disabled={isLoading || selectedSymptoms.length < 2}
                className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={18} />
                    Analyzing...
                  </>
                ) : (
                  'Diagnose'
                )}
              </button>
            </div>

            {selectedSymptoms.length > 0 && selectedSymptoms.length < 2 && (
              <p className="text-xs text-amber-600 mt-2 flex items-center gap-1">
                <AlertTriangle size={12} />
                Select at least {2 - selectedSymptoms.length} more symptom(s)
              </p>
            )}
          </div>

          {error && (
            <div className="card bg-red-50 border-red-200 animate-slide-up">
              <div className="flex items-start gap-2">
                <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={18} />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}

          {result && (
            <div className="card border-green-200 animate-slide-up">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle className="text-green-600" size={24} />
                <h3 className="font-semibold text-gray-900">Diagnosis Result</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Predicted Condition</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.prediction}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-500 mb-2">Confidence Level</p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-3">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${getConfidenceColor(result.confidence)}`}
                          style={{ width: `${Math.max(5, result.confidence * 100)}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium min-w-[60px] text-right">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-gray-600">
                      {getConfidenceText(result.confidence)}
                    </p>
                  </div>
                </div>

                {result.description && (
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Description</p>
                    <p className="text-sm text-gray-700 leading-relaxed">{result.description}</p>
                  </div>
                )}

                {result.alternative_diagnoses && result.alternative_diagnoses.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      <Info size={14} />
                      Alternative Possibilities
                    </p>
                    <div className="space-y-1">
                      {result.alternative_diagnoses.map((alt, index) => (
                        <div key={index} className="flex items-center justify-between bg-gray-50 px-3 py-2 rounded text-sm">
                          <span className="text-gray-700">{alt.disease}</span>
                          <span className="text-gray-500 font-medium">{(alt.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.recommendations && result.recommendations.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      <Shield size={14} />
                      Recommendations
                    </p>
                    <ul className="space-y-1.5">
                      {result.recommendations.map((recommendation, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                          <span className="text-primary-600 mt-0.5">•</span>
                          <span className="flex-1">{recommendation}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.precautions && result.precautions.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      <Shield size={14} />
                      Precautions
                    </p>
                    <ul className="space-y-1.5">
                      {result.precautions.map((precaution, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                          <span className="text-primary-600 mt-0.5">•</span>
                          <span className="flex-1">{precaution}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                  <div className="flex items-start gap-2">
                    <Info className="text-blue-600 flex-shrink-0 mt-0.5" size={16} />
                    <p className="text-xs text-blue-700">
                      This is an AI-generated assessment based on the symptoms you provided. 
                      Please consult a healthcare professional for proper diagnosis and treatment. 
                      Do not use this as a substitute for professional medical advice.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default SymptomDiagnosis;