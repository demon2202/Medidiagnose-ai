import React from 'react';
import { Link } from 'react-router-dom';
import {
  Stethoscope,
  Image as ImageIcon,
  History as HistoryIcon,
  TrendingUp,
  Activity,
  Calendar,
  ArrowRight,
  HeartPulse,
  Microscope,
  Shield,
  Brain
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function Dashboard() {
  const { user, getStats, history } = useApp();
  const stats = getStats();
  const recentHistory = history.slice(0, 5);

  const quickActions = [
    {
      to: '/symptoms',
      icon: Stethoscope,
      title: 'Symptom Diagnosis',
      description: 'Check symptoms and get AI-powered health insights',
      gradient: 'from-blue-500 to-blue-600',
      shadowColor: 'shadow-blue-500/20'
    },
    {
      to: '/image-analysis',
      icon: ImageIcon,
      title: 'Image Analysis',
      description: 'Upload medical images for AI analysis',
      gradient: 'from-purple-500 to-purple-600',
      shadowColor: 'shadow-purple-500/20'
    },
    {
      to: '/heart-check',
      icon: HeartPulse,
      title: 'Heart Health',
      description: 'Assess cardiovascular disease risk',
      gradient: 'from-red-500 to-pink-600',
      shadowColor: 'shadow-red-500/20'
    },
    {
      to: '/cancer-screening',
      icon: Microscope,
      title: 'Cancer Screening',
      description: 'Breast cancer risk assessment',
      gradient: 'from-emerald-500 to-teal-600',
      shadowColor: 'shadow-emerald-500/20'
    }
  ];

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400';
    if (confidence >= 0.5) return 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400';
    return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400';
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'symptom': return Stethoscope;
      case 'image': return ImageIcon;
      case 'heart': return HeartPulse;
      case 'cancer': return Microscope;
      default: return Activity;
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'symptom': return 'bg-blue-100 dark:bg-blue-900/30 text-blue-600';
      case 'image': return 'bg-purple-100 dark:bg-purple-900/30 text-purple-600';
      case 'heart': return 'bg-red-100 dark:bg-red-900/30 text-red-600';
      case 'cancer': return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600';
      default: return 'bg-gray-100 dark:bg-gray-800 text-gray-600';
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
     
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 rounded-2xl p-6 md:p-8 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2 blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full translate-y-1/2 -translate-x-1/2 blur-3xl" />
        
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
              <Brain size={24} />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">
                Welcome back, {user.name.split(' ')[0]}!
              </h1>
              <p className="text-blue-100">Your AI-powered health companion is ready to help.</p>
            </div>
          </div>
        </div>
      </div>


      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-3">
            <Activity className="text-blue-600" size={20} />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.totalDiagnoses}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Diagnoses</p>
        </div>

        <div className="stat-card">
          <div className="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg flex items-center justify-center mb-3">
            <Stethoscope className="text-emerald-600" size={20} />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.symptomDiagnoses}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Symptom Checks</p>
        </div>

        <div className="stat-card">
          <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mb-3">
            <ImageIcon className="text-purple-600" size={20} />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.imageDiagnoses}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Image Analyses</p>
        </div>

        <div className="stat-card">
          <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center mb-3">
            <Calendar className="text-amber-600" size={20} />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.recentDiagnoses}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">This Week</p>
        </div>
      </div>


      <div>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Actions</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.map((action) => (
            <Link
              key={action.to}
              to={action.to}
              className={`bg-gradient-to-br ${action.gradient} rounded-xl p-5 text-white hover:shadow-lg ${action.shadowColor} transition-all duration-300 hover:-translate-y-1 group`}
            >
              <action.icon size={28} className="mb-3 group-hover:scale-110 transition-transform" />
              <h3 className="font-semibold mb-1">{action.title}</h3>
              <p className="text-sm text-white/80">{action.description}</p>
              <ArrowRight size={18} className="mt-3 group-hover:translate-x-1 transition-transform" />
            </Link>
          ))}
        </div>
      </div>


      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Activity</h3>
          <Link to="/history" className="text-blue-600 text-sm font-medium hover:text-blue-700 flex items-center gap-1">
            View All <ArrowRight size={16} />
          </Link>
        </div>

        {recentHistory.length === 0 ? (
          <div className="text-center py-12">
            <HistoryIcon className="mx-auto text-gray-300 dark:text-gray-600 mb-4" size={48} />
            <p className="text-gray-500 dark:text-gray-400 font-medium">No recent activity</p>
            <p className="text-gray-400 dark:text-gray-500 text-sm mt-1">
              Start by using one of the diagnostic tools above
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentHistory.map((item) => {
              const TypeIcon = getTypeIcon(item.type);
              return (
                <div 
                  key={item.id} 
                  className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getTypeColor(item.type)}`}>
                    <TypeIcon size={20} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 dark:text-white truncate">
  {typeof item.prediction === 'string'
    ? item.prediction
    : item.prediction?.disease}
</p>

                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {new Date(item.timestamp).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getConfidenceColor(item.confidence)}`}>
                    {Math.round(item.confidence * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      
      <div className="card bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 border-emerald-100 dark:border-emerald-800">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center flex-shrink-0">
            <Shield className="text-emerald-600" size={24} />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Daily Health Tip</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              Regular health check-ups can help detect potential health issues before they become 
              serious. Schedule annual screenings and stay proactive about your health.
            </p>
            <Link to="/health-tips" className="text-emerald-600 text-sm font-medium hover:text-emerald-700 flex items-center gap-1">
              View More Tips <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;