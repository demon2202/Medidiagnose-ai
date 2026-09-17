import React, { useEffect } from 'react';
import { motion as Motion } from 'framer-motion';
import { CheckCircle2, AlertCircle, Info, X, TriangleAlert } from 'lucide-react';

const META = {
  success: { icon: CheckCircle2, cls: 'text-low' },
  error: { icon: AlertCircle, cls: 'text-critical' },
  warning: { icon: TriangleAlert, cls: 'text-moderate' },
  info: { icon: Info, cls: 'text-accent' },
};

export default function Notification({ message, type = 'info', onClose, duration = 4000 }) {
  useEffect(() => {
    if (duration > 0 && onClose) {
      const t = setTimeout(onClose, duration);
      return () => clearTimeout(t);
    }
  }, [duration, onClose, message]);

  const { icon: Icon, cls } = META[type] || META.info;

  return (
    <Motion.div
      role="alert"
      aria-live="polite"
      layout
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 480, damping: 36 }}
      className="panel pointer-events-auto flex w-[min(360px,calc(100vw-2rem))] items-start gap-3 !rounded-xl p-3.5 !shadow-lift"
    >
      <Icon size={18} className={`mt-px shrink-0 ${cls}`} strokeWidth={2} />
      <p className="flex-1 text-[13px] font-medium leading-relaxed text-ink">{message}</p>
      <button
        onClick={onClose}
        aria-label="Dismiss"
        className="-mr-1 -mt-1 rounded-lg p-1.5 text-faint transition-colors hover:bg-raised hover:text-ink"
      >
        <X size={14} />
      </button>
    </Motion.div>
  );
}
