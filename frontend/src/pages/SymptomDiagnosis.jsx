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
import { config } from '../config/config';
import Disclaimer from '../components/common/Disclaimer';

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
      const response = await axios.post(`${config.api.baseURL}/predict-disease`, {
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

        // confidence_tier: computed server-side using both the raw confidence
        // AND the margin over the runner-up prediction — a 42-class model with
        // overlapping symptoms (fever/fatigue/etc show up in a dozen+ diseases)
        // can legitimately peak around 30-50% while still being decisive.
        // Falls back to a local equivalent if an older backend response lacks it.
        const runnerUp = diagnosisData.alternative_diagnoses?.[0]?.confidence ?? 0;
        const confidenceTier = diagnosisData.confidence_tier || getConfidenceTier(confidence, runnerUp);

        setResult({
          ...diagnosisData,
          prediction: predictionStr,
          confidence,
          confidence_percent: confidencePct,
          confidence_tier: confidenceTier,
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
        setError(`Failed to connect to the server. Please ensure the backend is running on ${config.api.baseURL}`);
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

  // Mirrors server.py's get_confidence_tier(): for a 42-class problem with
  // heavy symptom overlap, a flat >80% bar for "High" is unrealistic without
  // overfitting. Margin over the runner-up prediction matters as much as the
  // raw number — 39.6% vs a distant 20.7% runner-up is more decisive than
  // 39.6% vs 38%.
  const getConfidenceTier = (confidence, runnerUp = 0) => {
    const margin = confidence - runnerUp;
    if (confidence > 0.55 || (confidence > 0.35 && margin > 0.20)) return 'High';
    if (confidence > 0.25 || margin > 0.10) return 'Moderate';
    return 'Low';
  };

  const getConfidenceColor = (tier) => {
    if (tier === 'High') return 'bg-green-500';
    if (tier === 'Moderate') return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  const getConfidenceText = (tier) => {
    if (tier === 'High') return 'High Confidence';
    if (tier === 'Moderate') return 'Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="page-title">Symptom Diagnosis</h1>
        <p className="page-subtitle -mt-3">
          Select your symptoms below to receive an AI-powered health assessment.
        </p>
      </div>

      <Disclaimer />

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
                className="select-field md:w-48"
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
                      : 'border-zinc-300 dark:border-zinc-700'
                  }`}>
                    {selectedSymptoms.includes(symptom.id) && (
                      <CheckCircle className="text-white animate-scale-in" size={14} />
                    )}
                  </div>
                  <div className="text-left">
                    <p className="font-medium text-zinc-900 dark:text-zinc-150">{symptom.label}</p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">{symptom.category}</p>
                  </div>
                </button>
              ))}
            </div>

            {filteredSymptoms.length === 0 && (
              <div className="text-center py-8 text-zinc-500">
                No symptoms found matching your search.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="card">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-3 flex items-center gap-2">
              <Stethoscope size={20} className="text-primary-600" />
              Selected Symptoms ({selectedSymptoms.length})
            </h3>
            
            {selectedSymptoms.length === 0 ? (
              <p className="text-zinc-500 text-sm">No symptoms selected yet</p>
            ) : (
              <div className="space-y-2 mb-4 max-h-60 overflow-y-auto">
                {selectedSymptoms.map(id => {
                  const symptom = symptoms.find(s => s.id === id);
                  return (
                    <div 
                      key={id}
                      className="flex items-center justify-between bg-zinc-100 dark:bg-zinc-800/60 px-3 py-2 rounded-lg"
                    >
                      <span className="text-sm text-zinc-800 dark:text-zinc-200">{symptom?.label}</span>
                      <button
                        onClick={() => toggleSymptom(id)}
                        className="text-zinc-500 hover:text-red-500 font-bold text-lg leading-none transition-colors px-1"
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
                className="flex-1 btn-primary"
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
            <div className="card bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800/60 animate-slide-up">
              <div className="flex items-start gap-2">
                <AlertCircle className="text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" size={18} />
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="card border-green-200 dark:border-green-800/40 p-8 shadow-xl animate-slide-up mt-6 bg-white dark:bg-zinc-900/50 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-6">
            <CheckCircle className="text-green-600 dark:text-green-400" size={28} />
            <h3 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">Diagnosis Result</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div>
                <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Predicted Condition</p>
                <p className="text-2xl font-extrabold text-zinc-900 dark:text-zinc-100">
                  {result.prediction}
                </p>
              </div>

              <div>
                <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Confidence Level</p>
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <div className="flex-1 bg-zinc-100 dark:bg-zinc-800 rounded-full h-3.5 overflow-hidden border border-zinc-200/50 dark:border-zinc-700/50">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${getConfidenceColor(result.confidence_tier)}`}
                        style={{ width: `${Math.max(5, result.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100 min-w-[60px] text-right">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-zinc-500 dark:text-zinc-400 font-medium">
                    {getConfidenceText(result.confidence_tier)}
                  </p>
                </div>
              </div>

              {result.description && (
                <div>
                  <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Condition Description</p>
                  <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">{result.description}</p>
                </div>
              )}

              {result.alternative_diagnoses && result.alternative_diagnoses.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                    <Info size={14} />
                    Alternative Possibilities
                  </p>
                  <div className="space-y-2 max-w-md">
                    {result.alternative_diagnoses.map((alt, index) => (
                      <div key={index} className="flex items-center justify-between bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-100 dark:border-zinc-800 px-4 py-2.5 rounded-lg text-sm">
                        <span className="text-zinc-700 dark:text-zinc-300 font-medium">{alt.disease}</span>
                        <span className="text-zinc-900 dark:text-zinc-100 font-bold">{(alt.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-6">
              {result.recommendations && result.recommendations.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Shield size={14} className="text-zinc-500" />
                    Recommendations
                  </p>
                  <ul className="space-y-2 bg-zinc-50 dark:bg-zinc-800/20 border border-zinc-100 dark:border-zinc-800/60 p-4 rounded-xl">
                    {result.recommendations.map((recommendation, index) => (
                      <li key={index} className="flex items-start gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                        <span className="text-zinc-500 mt-1 flex-shrink-0">•</span>
                        <span className="flex-1 leading-relaxed">{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.precautions && result.precautions.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Shield size={14} className="text-zinc-500" />
                    Precautions
                  </p>
                  <ul className="space-y-2 bg-zinc-50 dark:bg-zinc-800/20 border border-zinc-100 dark:border-zinc-800/60 p-4 rounded-xl">
                    {result.precautions.map((precaution, index) => (
                      <li key={index} className="flex items-start gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                        <span className="text-zinc-500 mt-1 flex-shrink-0">•</span>
                        <span className="flex-1 leading-relaxed">{precaution}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default SymptomDiagnosis;