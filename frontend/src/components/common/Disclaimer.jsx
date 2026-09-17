import React, { useState } from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, ChevronDown } from 'lucide-react';

export default function Disclaimer({
  title = 'For information only — not a diagnosis',
  message = 'This tool offers preliminary AI insights and is not a substitute for professional medical advice, diagnosis or treatment. Always consult a qualified clinician about your health.',
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-raised/60"
        aria-expanded={open}
      >
        <ShieldAlert size={15} className="shrink-0 text-moderate" strokeWidth={2} />
        <span className="flex-1 truncate text-sm font-medium text-muted">{title}</span>
        <ChevronDown
          size={14}
          className={`shrink-0 text-faint transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <Motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="border-t border-line px-4 py-3 pl-[42px] text-sm leading-relaxed text-muted">
              {message}
            </p>
          </Motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
