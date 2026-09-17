import React from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import { X, Copy, Check } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { spring, easeOut } from '../../lib/motion';
import { severityMeta } from '../../lib/severity';

export function Reveal({ children, delay = 0, y = 10, className }) {
  return (
    <Motion.div
      className={className}
      initial={{ opacity: 0, y }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: easeOut, delay }}
    >
      {children}
    </Motion.div>
  );
}

/* ---------------- page header ---------------- */
export function PageHeader({ eyebrow, title, description, action }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <h1 className="text-[32px] font-semibold leading-tight tracking-tight text-ink md:text-[38px]">
          {title}
        </h1>
        {description && (
          <p className="mt-1.5 max-w-2xl text-base leading-relaxed text-muted">{description}</p>
        )}
      </div>
      {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
    </div>
  );
}

/* ---------------- segmented control with sliding pill ---------------- */
export function Seg({ options, value, onChange, id = 'seg', size = 'md' }) {
  return (
    <div className="seg" role="tablist" aria-label={id}>
      {options.map((opt) => {
        const active = opt.value === value;
        const Icon = opt.icon;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            data-active={active}
            className={`seg-btn shrink-0 whitespace-nowrap ${size === 'sm' ? '!px-2.5 !py-1 !text-xs' : ''}`}
          >
            {active && (
              <Motion.span
                layoutId={`seg-pill-${id}`}
                transition={spring}
                className="absolute inset-0 rounded-lg border border-line bg-surface shadow-soft"
              />
            )}
            <span className="relative z-10 flex items-center gap-1.5">
              {Icon && <Icon size={size === 'sm' ? 13 : 15} strokeWidth={2} />}
              {opt.label}
              {opt.count != null && (
                <span className="t-num rounded-full bg-ink/5 px-1.5 py-px text-xs font-semibold dark:bg-white/10">
                  {opt.count}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------------- severity ---------------- */
export function SeverityBadge({ level, className = '' }) {
  const meta = severityMeta(level);
  return (
    <span
      className={`badge ${className}`}
      style={{ background: meta.soft, color: meta.color }}
    >
      <span className="dot" style={{ background: meta.color }} />
      {meta.label}
    </span>
  );
}

/* ---------------- confidence ring ---------------- */
export function ConfidenceRing({ value = 0, size = 96, stroke = 9, label }) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const id = React.useId();
  return (
    <div className="flex shrink-0 flex-col items-center gap-1.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            strokeWidth={stroke}
            className="stroke-line"
          />
          <Motion.circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={`url(#${id})`}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: c - c * v }}
            transition={{ duration: 1, ease: easeOut, delay: 0.15 }}
          />
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="rgb(var(--c-accent))" />
              <stop offset="100%" stopColor="rgb(var(--c-accent))" stopOpacity="0.55" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="t-num text-[22px] font-semibold text-ink">
            {(v * 100).toFixed(0)}
            <span className="text-sm font-medium text-faint">%</span>
          </span>
        </div>
      </div>
      {label && <span className="eyebrow !text-[10px]">{label}</span>}
    </div>
  );
}

/* ---------------- slim meter ---------------- */
export function Meter({ value = 0, tone, className = '' }) {
  const v = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-ink/[0.07] dark:bg-white/10 ${className}`}>
      <Motion.div
        className="h-full rounded-full"
        style={{ background: tone || 'rgb(var(--c-accent))' }}
        initial={{ width: 0 }}
        animate={{ width: `${v}%` }}
        transition={{ duration: 0.8, ease: easeOut }}
      />
    </div>
  );
}

/* ---------------- empty state ---------------- */
export function EmptyState({ icon: Icon, title, hint, action, className = '' }) {
  return (
    <div className={`flex flex-col items-center px-6 py-10 text-center ${className}`}>
      {Icon && (
        <div className="dotgrid mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-line bg-raised/50 text-faint">
          <Icon size={26} strokeWidth={1.75} />
        </div>
      )}
      <p className="text-base font-medium text-ink">{title}</p>
      {hint && <p className="mt-1 max-w-sm text-[15px] leading-relaxed text-muted">{hint}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ---------------- fields ---------------- */
export function TextField({ label, hint, ...props }) {
  return (
    <div className="min-w-0">
      {label && (
        <label className="label">
          {label}
          {hint && <span className="ml-1.5 font-normal text-faint">{hint}</span>}
        </label>
      )}
      <input {...props} className={`field t-num ${props.className || ''}`} />
    </div>
  );
}

export function SelectField({ label, options = [], ...props }) {
  return (
    <div className="min-w-0">
      {label && <label className="label">{label}</label>}
      <select {...props} className={`field ${props.className || ''}`}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/* ---------------- modal shell ---------------- */
export function Modal({ onClose, children, wide, size }) {
  const maxW = size === 'xl' ? 'max-w-3xl' : size === 'lg' || wide ? 'max-w-lg' : 'max-w-md';
  React.useEffect(() => {
    const fn = (e) => e.key === 'Escape' && onClose?.();
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [onClose]);

  return (
    <Motion.div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-black/45 p-4 backdrop-blur-[3px]"
      role="dialog"
      aria-modal="true"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
      onClick={onClose}
    >
      <Motion.div
        initial={{ opacity: 0, scale: 0.96, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.97, y: 8 }}
        transition={{ duration: 0.22, ease: easeOut }}
        onClick={(e) => e.stopPropagation()}
        className={`panel flex w-full flex-col overflow-hidden !shadow-lift ${maxW} ${size === 'xl' ? 'max-h-[88vh]' : ''}`}
      >
        {children}
      </Motion.div>
    </Motion.div>
  );
}

export function ModalHeader({ title, subtitle, onClose }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
      <div>
        <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
        {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
      </div>
      {onClose && (
        <button onClick={onClose} className="icon-btn -mr-1 -mt-1 !h-8 !w-8" aria-label="Close">
          <X size={16} />
        </button>
      )}
    </div>
  );
}

/* ---------------- tabbed panel (kills the endless scroll) ---------------- */
export function TabPanel({ tabs, value, onChange, id }) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto border-b border-line no-scrollbar" role="tablist">
      {tabs.map((t) => {
        const active = t.value === value;
        const Icon = t.icon;
        return (
          <button
            key={t.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(t.value)}
            className={`relative flex shrink-0 items-center gap-1.5 px-4 py-2.5 text-[15px] font-medium transition-colors ${
              active ? 'text-ink' : 'text-faint hover:text-muted'
            }`}
          >
            {Icon && <Icon size={14} strokeWidth={2} />}
            {t.label}
            {t.count != null && (
              <span className="t-num rounded-full bg-ink/5 px-1.5 text-[11px] font-semibold text-muted dark:bg-white/10">
                {t.count}
              </span>
            )}
            {active && (
              <Motion.span
                layoutId={`tab-underline-${id}`}
                transition={spring}
                className="absolute inset-x-2 -bottom-px h-[2px] rounded-full bg-ink dark:bg-white"
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

export function TabBody({ tabKey, children, className = '' }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <Motion.div
        key={tabKey}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.18, ease: easeOut }}
        className={className}
      >
        {children}
      </Motion.div>
    </AnimatePresence>
  );
}

/* ---------------- list row ---------------- */
export function Row({ icon: Icon, title, subtitle, right, onClick, className = '' }) {
  const Cmp = onClick ? 'button' : 'div';
  return (
    <Cmp
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
        onClick ? 'hover:bg-raised/70' : ''
      } ${className}`}
    >
      {Icon && (
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-line bg-surface text-muted">
          <Icon size={16} strokeWidth={1.9} />
        </span>
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[15px] font-medium text-ink">{title}</span>
        {subtitle && <span className="block truncate text-[13px] text-muted">{subtitle}</span>}
      </span>
      {right}
    </Cmp>
  );
}

/* ---------------- copy-to-clipboard ---------------- */
export function CopyButton({ text, label = 'Copy summary', className = '' }) {
  const { showNotification } = useApp();
  const [copied, setCopied] = React.useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    setCopied(true);
    showNotification('Summary copied to clipboard', 'success');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button onClick={copy} className={`btn-quiet btn-sm ${className}`}>
      {copied ? <Check size={13} className="text-low" /> : <Copy size={13} />}
      {copied ? 'Copied' : label}
    </button>
  );
}
