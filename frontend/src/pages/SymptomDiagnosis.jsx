import React, { useState } from 'react';
import axios from 'axios';
import { 
  Search, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Stethoscope,
  Shield,
  Info
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
    if (selectedSymptoms.length < 3) {
      setError('Please select at least 3 symptoms for accurate diagnosis');
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
        setResult(response.data);
        addToHistory({
          type: 'symptom',
          symptoms: selectedSymptoms,
          prediction: response.data.prediction,
          confidence: response.data.confidence,
          description: response.data.description,
          precautions: response.data.precautions,
        });
      } else {
        setError(response.data.error || 'Failed to get diagnosis');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please ensure the backend is running.');
      console.error('Diagnosis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedSymptoms([]);
    setResult(null);
    setError(null);
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
              Selected Symptoms
            </h3>
            
            {selectedSymptoms.length === 0 ? (
              <p className="text-gray-500 text-sm">No symptoms selected yet</p>
            ) : (
              <div className="space-y-2 mb-4">
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
                        className="text-primary-600 hover:text-primary-800"
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
                className="flex-1 btn-secondary disabled:opacity-50"
              >
                Clear
              </button>
              <button
                onClick={handleDiagnose}
                disabled={isLoading || selectedSymptoms.length < 3}
                className="flex-1 btn-primary disabled:opacity-50 flex items-center justify-center gap-2"
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

            {selectedSymptoms.length > 0 && selectedSymptoms.length < 3 && (
              <p className="text-xs text-amber-600 mt-2">
                Select at least {3 - selectedSymptoms.length} more symptom(s)
              </p>
            )}
          </div>

          {error && (
            <div className="card bg-red-50 border-red-200">
              <div className="flex items-start gap-2">
                <AlertCircle className="text-red-600 flex-shrink-0" size={18} />
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
                  <p className="text-sm text-gray-500">Predicted Condition</p>
                  <p className="text-xl font-bold text-gray-900">
                    {typeof result.prediction === 'string' 
                      ? result.prediction 
                      : result.prediction?.disease || 'Unknown'}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-500">Confidence Level</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          result.confidence >= 0.8 
                            ? 'bg-green-500' 
                            : result.confidence >= 0.5 
                              ? 'bg-yellow-500' 
                              : 'bg-red-500'
                        }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {Math.round(result.confidence * 100)}%
                    </span>
                  </div>
                </div>

                {result.description && (
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Description</p>
                    <p className="text-sm text-gray-700">{result.description}</p>
                  </div>
                )}

                {result.recommendations && result.recommendations.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      <Shield size={14} />
                      Recommendations
                    </p>
                    <ul className="space-y-1">
                      {result.recommendations.map((recommendation, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                          <span className="text-primary-600">•</span>
                          {recommendation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.precautions && result.precautions.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      <Shield size={14} />
                      Recommended Precautions
                    </p>
                    <ul className="space-y-1">
                      {result.precautions.map((precaution, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                          <span className="text-primary-600">•</span>
                          {precaution}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="flex items-start gap-2">
                    <Info className="text-blue-600 flex-shrink-0 mt-0.5" size={16} />
                    <p className="text-xs text-blue-700">
                      This is an AI-generated assessment. Please consult a healthcare 
                      professional for proper diagnosis and treatment.
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