import React, { useEffect } from 'react';
import { CheckCircle, AlertCircle, Info, X, AlertTriangle } from 'lucide-react';

function Notification({ message, type = 'info', onClose, duration = 4000 }) {
  useEffect(() => {
    if (duration > 0 && onClose) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  };

  const styles = {
    success: 'bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-700',
    error: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700',
    warning: 'bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700',
    info: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-700',
  };

  const iconStyles = {
    success: 'text-emerald-600 dark:text-emerald-400',
    error: 'text-red-600 dark:text-red-400',
    warning: 'text-amber-600 dark:text-amber-400',
    info: 'text-blue-600 dark:text-blue-400',
  };

  const textStyles = {
    success: 'text-emerald-800 dark:text-emerald-200',
    error: 'text-red-800 dark:text-red-200',
    warning: 'text-amber-800 dark:text-amber-200',
    info: 'text-blue-800 dark:text-blue-200',
  };

  const Icon = icons[type] || Info;

  return (
    <div 
      className="fixed top-4 right-4 z-[100] max-w-md animate-fade-in"
      role="alert"
      aria-live="polite"
    >
      <div 
        className={`flex items-start gap-3 px-4 py-3 rounded-xl border shadow-lg backdrop-blur-sm ${styles[type]}`}
      >
        <div className={`flex-shrink-0 mt-0.5 ${iconStyles[type]}`}>
          <Icon size={20} />
        </div>
        
        <p className={`flex-1 font-medium text-sm leading-relaxed ${textStyles[type]}`}>
          {message}
        </p>
        
        <button
          onClick={onClose}
          className={`flex-shrink-0 p-1 hover:bg-black/10 dark:hover:bg-white/10 rounded-lg transition-colors ${textStyles[type]}`}
          aria-label="Close notification"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}

export default Notification;