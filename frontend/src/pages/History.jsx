import React, { useState } from 'react';
import { 
  Clock, 
  FileText, 
  Image as ImageIcon, 
  Activity, 
  AlertCircle, 
  CheckCircle,
  AlertTriangle,
  Trash2,
  Filter,
  Download,
  TrendingUp
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function History() {
  const { history, clearHistory } = useApp();
  const [filter, setFilter] = useState('all'); // all, image, symptom

  // Helper function to safely extract prediction text
  const getPredictionText = (item) => {
    if (!item.prediction) return 'Unknown';
    
    // If prediction is a string, return it
    if (typeof item.prediction === 'string') {
      return item.prediction;
    }
    
    // If prediction is an object, extract the name/disease
    if (typeof item.prediction === 'object') {
      // Try different possible fields
      return item.prediction.name || 
             item.prediction.disease || 
             item.prediction.condition || 
             item.prediction.finding || 
             'Unknown';
    }
    
    return 'Unknown';
  };

  // Helper function to get confidence
  const getConfidence = (item) => {
    if (!item.confidence && item.confidence !== 0) return null;
    
    // If confidence is already a number between 0-1, convert to percentage
    if (typeof item.confidence === 'number') {
      if (item.confidence <= 1) {
        return (item.confidence * 100).toFixed(1) + '%';
      }
      return item.confidence.toFixed(1) + '%';
    }
    
    return null;
  };

  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
      case 'high':
        return <AlertTriangle className="text-red-500" size={20} />;
      case 'moderate':
        return <AlertCircle className="text-yellow-500" size={20} />;
      case 'low':
      case 'healthy':
        return <CheckCircle className="text-green-500" size={20} />;
      default:
        return <Activity className="text-gray-500" size={20} />;
    }
  };

  // Get severity badge color
  const getSeverityBadge = (severity) => {
    const colors = {
      critical: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
      high: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
      moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
      low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
      healthy: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
    };
    return colors[severity?.toLowerCase()] || 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400';
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown time';
    
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
      if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
      if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
      
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
      });
    } catch (e) {
      return 'Unknown time';
    }
  };

  // Get type icon
  const getTypeIcon = (type) => {
    if (type?.includes('image')) {
      return <ImageIcon size={16} />;
    }
    return <FileText size={16} />;
  };

  // Filter history
  const filteredHistory = history.filter(item => {
    if (filter === 'all') return true;
    if (filter === 'image') return item.type?.includes('image');
    if (filter === 'symptom') return item.type?.includes('symptom') || item.type?.includes('disease');
    return true;
  });

  // Export history as JSON
  const exportHistory = () => {
    const dataStr = JSON.stringify(history, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `medidiagnose-history-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="page-title">
          <Clock className="text-blue-500" />
          Analysis History
        </h1>
        <p className="page-subtitle">
          View and manage your previous medical analyses
        </p>
      </div>

      {/* Stats & Actions */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
              <FileText className="text-blue-500" size={20} />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Analyses</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{history.length}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
              <ImageIcon className="text-purple-500" size={20} />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Image Analyses</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {history.filter(h => h.type?.includes('image')).length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
              <Activity className="text-green-500" size={20} />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Symptom Checks</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {history.filter(h => h.type?.includes('symptom') || h.type?.includes('disease')).length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
              <TrendingUp className="text-orange-500" size={20} />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">This Week</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {history.filter(h => {
                  const date = new Date(h.timestamp);
                  const weekAgo = new Date();
                  weekAgo.setDate(weekAgo.getDate() - 7);
                  return date > weekAgo;
                }).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters & Actions */}
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Filter size={18} className="text-gray-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Filter:</span>
            <div className="flex gap-2">
              <button
                onClick={() => setFilter('all')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'all'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                All ({history.length})
              </button>
              <button
                onClick={() => setFilter('image')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'image'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                Images ({history.filter(h => h.type?.includes('image')).length})
              </button>
              <button
                onClick={() => setFilter('symptom')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'symptom'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                Symptoms ({history.filter(h => h.type?.includes('symptom') || h.type?.includes('disease')).length})
              </button>
            </div>
          </div>

          <div className="flex gap-2">
            {history.length > 0 && (
              <>
                <button
                  onClick={exportHistory}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2 text-sm font-medium"
                >
                  <Download size={16} />
                  Export
                </button>
                <button
                  onClick={clearHistory}
                  className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2 text-sm font-medium"
                >
                  <Trash2 size={16} />
                  Clear All
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* History List */}
      {filteredHistory.length === 0 ? (
        <div className="card text-center py-12">
          <Clock className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            No History Yet
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            {filter === 'all' 
              ? 'Your analysis history will appear here'
              : `No ${filter} analyses found`}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item, index) => (
            <div key={index} className="card hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                {/* Icon */}
                <div className="flex-shrink-0">
                  {getSeverityIcon(item.severity)}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        {getTypeIcon(item.type)}
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {getPredictionText(item)}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {item.type?.replace('_', ' ').replace('image', 'Image Analysis')}
                      </p>
                    </div>

                    {/* Severity Badge */}
                    {item.severity && (
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold uppercase ${getSeverityBadge(item.severity)}`}>
                        {item.severity}
                      </span>
                    )}
                  </div>

                  {/* Details */}
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                    {getConfidence(item) && (
                      <div className="flex items-center gap-1">
                        <TrendingUp size={14} />
                        <span>Confidence: {getConfidence(item)}</span>
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Clock size={14} />
                      <span>{formatTimestamp(item.timestamp)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Info */}
      {history.length > 0 && (
        <div className="card bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-3">
            <AlertCircle className="text-blue-500 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="text-sm text-blue-800 dark:text-blue-300 font-medium">
                About Your History
              </p>
              <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">
                Your analysis history is stored locally in your browser. It will be cleared if you clear your browser data. 
                Use the Export button to save a copy of your history.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default History;