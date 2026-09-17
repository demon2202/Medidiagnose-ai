import React, { useState, useMemo, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import {
  Search,
  LayoutDashboard,
  Stethoscope,
  ScanLine,
  HeartPulse,
  Microscope,
  History as HistoryIcon,
  Sprout,
  Settings,
  Moon,
  Sun,
  Download,
  LogOut,
  CornerDownLeft,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { easeOut } from '../../lib/motion';
import {
  diagnosisTypeMeta,
  entryTitle,
  timeAgo,
} from '../../lib/diagnosis';
import { downloadHistory } from '../../lib/diagnosis';

const PAGES = [
  { path: '/', icon: LayoutDashboard, label: 'Overview', hint: 'Dashboard' },
  { path: '/symptoms', icon: Stethoscope, label: 'Symptom diagnosis', hint: 'New assessment' },
  { path: '/image-analysis', icon: ScanLine, label: 'Image analysis', hint: 'New scan' },
  { path: '/heart-check', icon: HeartPulse, label: 'Heart health', hint: 'Screening' },
  { path: '/cancer-screening', icon: Microscope, label: 'Cancer screening', hint: 'Screening' },
  { path: '/history', icon: HistoryIcon, label: 'History', hint: 'Library' },
  { path: '/health-tips', icon: Sprout, label: 'Health tips', hint: 'Library' },
  { path: '/settings', icon: Settings, label: 'Settings', hint: 'Workspace' },
];

export default function CommandPalette() {
  const {
    paletteOpen,
    setPaletteOpen,
    history,
    theme,
    toggleTheme,
    signOut,
    openInspector,
    showNotification,
  } = useApp();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    if (paletteOpen) {
      setQuery('');
      setIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [paletteOpen ]);

  const close = () => setPaletteOpen(false);

  const actions = useMemo(
    () => [
      {
        icon: theme === 'dark' ? Sun : Moon,
        label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
        hint: 'Theme',
        run: () => toggleTheme(),
      },
      {
        icon: Download,
        label: 'Export history as JSON',
        hint: `${history.length} records`,
        run: () => {
          if (!history.length) {
            showNotification('Nothing to export yet', 'info');
            return;
          }
          downloadHistory(history);
          showNotification('History exported', 'success');
        },
      },
      {
        icon: LogOut,
        label: 'Sign out',
        hint: '',
        run: () => {
          signOut();
          navigate('/login');
        },
      },
    ],
    [theme, toggleTheme, history, signOut, navigate, showNotification],
  );

  const sections = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (label, extra = '') =>
      !q || `${label} ${extra}`.toLowerCase().includes(q);
    const pages = PAGES.filter((p) => match(p.label, p.hint)).map((p) => ({
      ...p,
      run: () => navigate(p.path),
    }));
    const acts = actions.filter((a) => match(a.label, a.hint));
    const recent = history
      .filter((h) => match(entryTitle(h), h.type))
      .slice(0, 5)
      .map((h) => {
        const m = diagnosisTypeMeta(h.type);
        return {
          icon: m.icon,
          label: entryTitle(h),
          hint: timeAgo(h.timestamp),
          run: () => openInspector(h),
        };
      });
    const out = [];
    if (pages.length) out.push({ title: 'Go to', items: pages });
    if (acts.length) out.push({ title: 'Actions', items: acts });
    if (recent.length) out.push({ title: 'Recent results', items: recent });
    return out;
  }, [query, actions, history, navigate, openInspector]);

  const flat = useMemo(() => sections.flatMap((s) => s.items), [sections]);
  const active = flat[Math.min(index, Math.max(0, flat.length - 1))];

  useEffect(() => setIndex(0), [query]);
  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-idx="${Math.min(index, flat.length - 1)}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [index, flat.length]);

  if (!paletteOpen) return null;

  return (
    <Motion.div
      className="fixed inset-0 z-[95] flex items-start justify-center bg-black/45 px-4 pt-[12vh] backdrop-blur-[3px]"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      onClick={close}
    >
      <Motion.div
        initial={{ opacity: 0, scale: 0.98, y: -8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.98, y: -6 }}
        transition={{ duration: 0.18, ease: easeOut }}
        onClick={(e) => e.stopPropagation()}
        className="panel w-full max-w-lg overflow-hidden !shadow-lift"
        role="dialog"
        aria-label="Command palette"
      >
        <div className="flex items-center gap-2.5 border-b border-line px-4">
          <Search size={16} className="shrink-0 text-faint" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault();
                setIndex((i) => (i + 1) % Math.max(1, flat.length));
              } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setIndex((i) => (i - 1 + flat.length) % Math.max(1, flat.length));
              } else if (e.key === 'Enter' && active) {
                e.preventDefault();
                close();
                active.run();
              } else if (e.key === 'Escape') {
                close();
              }
            }}
            placeholder="Jump to a page, run an action, reopen a result…"
            className="w-full bg-transparent py-3.5 text-sm text-ink placeholder:text-faint focus:outline-none"
          />
          <kbd className="kbd">esc</kbd>
        </div>

        <div ref={listRef} className="max-h-[340px] overflow-y-auto p-1.5">
          {flat.length === 0 && (
            <p className="px-3 py-8 text-center text-sm text-faint">
              Nothing matches “{query}”.
            </p>
          )}
          {(() => {
            let n = 0;
            return sections.map((s) => (
              <div key={s.title} className="mb-1">
                <p className="eyebrow !text-[10px] px-3 pb-1 pt-2">{s.title}</p>
                {s.items.map((item) => {
                  const idx = n++;
                  const isActive = idx === Math.min(index, flat.length - 1);
                  return (
                    <button
                      key={`${s.title}-${item.label}`}
                      data-idx={idx}
                      onMouseEnter={() => setIndex(idx)}
                      onClick={() => {
                        close();
                        item.run();
                      }}
                      className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                        isActive ? 'bg-ink/[0.06] dark:bg-white/10' : ''
                      }`}
                    >
                      <item.icon
                        size={16}
                        strokeWidth={1.9}
                        className={isActive ? 'text-ink' : 'text-faint'}
                      />
                      <span className={`flex-1 truncate font-medium ${isActive ? 'text-ink' : 'text-muted'}`}>
                        {item.label}
                      </span>
                      {item.hint && (
                        <span className="shrink-0 text-xs text-faint">{item.hint}</span>
                      )}
                      {isActive && <CornerDownLeft size={13} className="shrink-0 text-faint" />}
                    </button>
                  );
                })}
              </div>
            ));
          })()}
        </div>

        <div className="flex items-center gap-3 border-t border-line px-4 py-2.5 text-[11px] text-faint">
          <span className="flex items-center gap-1"><kbd className="kbd">↑↓</kbd> navigate</span>
          <span className="flex items-center gap-1"><kbd className="kbd">↵</kbd> open</span>
          <span className="ml-auto">MediDiagnose command</span>
        </div>
      </Motion.div>
    </Motion.div>
  );
}
