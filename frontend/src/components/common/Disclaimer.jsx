import React, { useState } from 'react';
import { AlertCircle, ChevronDown, ChevronUp, ShieldAlert } from 'lucide-react';

function Disclaimer({ 
  title = "Medical Disclaimer", 
  message = "This AI tool provides preliminary informational insights only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions about your health." 
}) {
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      return localStorage.getItem('mediDiagnose_disclaimer_collapsed') === 'true';
    } catch {
      return false;
    }
  });

  const toggleCollapse = () => {
    const nextState = !isCollapsed;
    setIsCollapsed(nextState);
    try {
      localStorage.setItem('mediDiagnose_disclaimer_collapsed', String(nextState));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className={`transition-all duration-300 rounded-xl border ${
      isCollapsed 
        ? 'bg-zinc-50/50 dark:bg-zinc-900/30 border-zinc-200/60 dark:border-zinc-800/40 p-3' 
        : 'bg-amber-50/40 dark:bg-amber-950/10 border-amber-200/50 dark:border-amber-800/40 p-4 md:p-5'
    }`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <AlertCircle className={`flex-shrink-0 ${
            isCollapsed ? 'text-zinc-500 dark:text-zinc-400' : 'text-amber-600 dark:text-amber-500'
          }`} size={18} />
          <p className={`font-semibold text-sm truncate ${
            isCollapsed ? 'text-zinc-700 dark:text-zinc-300' : 'text-amber-900 dark:text-amber-200'
          }`}>
            {title} {isCollapsed && <span className="font-normal text-xs text-zinc-500 dark:text-zinc-400 ml-1"> (Collapsed)</span>}
          </p>
        </div>
        <button
          onClick={toggleCollapse}
          type="button"
          className={`p-1 rounded-lg transition-colors hover:bg-black/5 dark:hover:bg-white/5 ${
            isCollapsed ? 'text-zinc-500 dark:text-zinc-400' : 'text-amber-700 dark:text-amber-400'
          }`}
          aria-label={isCollapsed ? "Expand disclaimer" : "Collapse disclaimer"}
        >
          {isCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
        </button>
      </div>

      {!isCollapsed && (
        <div className="mt-2 text-xs md:text-sm leading-relaxed text-amber-800/85 dark:text-amber-300/80 animate-fade-in pl-7">
          {message}
        </div>
      )}
    </div>
  );
}

export default Disclaimer;
