import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Loader2,
  Stethoscope,
  TriangleAlert,
  CircleAlert,
  Check,
  X,
  ArrowRight,
  Info
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { symptoms, symptomCategories } from '../data/symptoms';
import { config } from '../config/config';
import { scrollToId } from '../lib/scroll';
import { cleanResult } from '../lib/text';
import Disclaimer from '../components/common/Disclaimer';
import SymptomResultView from '../components/results/SymptomResultView';
import {
  PageHeader,
  ConfidenceRing,
  Meter,
  Reveal
} from '../components/ui/ui';
import { easeOut } from '../lib/motion';

/* Mirrors the backend tiering: for a 42-class problem with heavy symptom
   overlap, margin over the runner-up matters as much as the raw number. */
const getConfidenceTier = (confidence, runnerUp = 0) => {
  const margin = confidence - runnerUp;
  if (confidence > 0.55 || (confidence > 0.35 && margin > 0.2)) return 'High';
  if (confidence > 0.25 || margin > 0.1) return 'Moderate';
  return 'Low';
};

export default function SymptomDiagnosis() {
  const searchRef = useRef(null);

  /* press / anywhere to jump to the search box */
  useEffect(() => {
    const fn = (e) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      e.preventDefault();
      searchRef.current?.focus();
    };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, []);
  const { addToHistory, isLoading, setIsLoading } = useApp();
  const [selected, setSelected] = useState([]);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('All');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const filtered = symptoms.filter((s) => {
    const q = s.label.toLowerCase().includes(query.toLowerCase());
    const c = category === 'All' || s.category === category;
    return q && c;
  });

  const toggle = (id) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
    setError(null);
  };

  const handleDiagnose = async () => {
    if (selected.length < 2) {
      setError('Select at least 2 symptoms for an assessment.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post(`${config.api.baseURL}/predict-disease`, {
        symptoms: selected
      });
      if (response.data.success) {
        const d = cleanResult(response.data);
        const rawConf = d.confidence ?? d.prediction?.confidence ?? 0;
        const confidence = Math.max(0, Math.min(1, parseFloat(rawConf) || 0));
        const raw = d.prediction;
        const prediction =
          raw && typeof raw === 'object' ? raw.disease || String(raw) : String(raw ?? 'Unknown');
        const runnerUp = d.alternative_diagnoses?.[0]?.confidence ?? 0;
        const out = {
          ...d,
          prediction,
          confidence,
          confidence_percent:
            d.confidence_percent ||
            d.prediction?.confidence_percent ||
            `${(confidence * 100).toFixed(1)}%`,
          confidence_tier: d.confidence_tier || getConfidenceTier(confidence, runnerUp)
        };
        setResult(out);
        addToHistory({
          type: 'symptom',
          symptoms: selected.map((id) => symptoms.find((s) => s.id === id)?.label || id),
          prediction,
          confidence,
          description: d.description,
          precautions: d.precautions || [],
          recommendations: d.recommendations || [],
          data: out,
          timestamp: new Date().toISOString()
        });
        requestAnimationFrame(() => scrollToId('symptom-result'));
      } else {
        setError(response.data.error || 'Assessment failed');
      }
    } catch (err) {
      if (err.response) setError(err.response.data?.error || 'Server error occurred');
      else if (err.request)
        setError(`Could not reach the server. Is the backend running on ${config.api.baseURL}?`);
      else setError('An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Triage"
        title="Symptom diagnosis"
        description="Select what you're experiencing — the model weighs combinations, not single symptoms."
      />

      <Reveal>
        <Disclaimer />
      </Reveal>

      <div className="grid items-start gap-5 lg:grid-cols-12">
        {/* picker */}
        <Reveal delay={0.05} className="lg:col-span-7">
          <div className="panel p-5">
            <div className="relative">
              <Search size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                ref={searchRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search symptoms — e.g. fever, cough…"
                className="field !pl-10 !pr-10"
              />
              <span className="kbd pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2">/</span>
            </div>

            <div className="mt-3 flex gap-1.5 overflow-x-auto pb-1 no-scrollbar" style={{ maskImage: 'linear-gradient(to right, black 92%, transparent)' }}>
              {symptomCategories.map((c) => (
                <button
                  key={c}
                  onClick={() => setCategory(c)}
                  data-on={category === c}
                  className="chip shrink-0 !py-2"
                >
                  {c}
                </button>
              ))}
            </div>

            <div className="mt-3 max-h-[520px] min-h-[340px] overflow-y-auto pr-1">
              {filtered.length === 0 ? (
                <p className="py-10 text-center text-sm text-faint">
                  No symptoms match “{query}”.
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {filtered.map((s) => {
                    const on = selected.includes(s.id);
                    return (
                      <button key={s.id} onClick={() => toggle(s.id)} data-on={on} className="chip">
                        <span
                          className={`flex h-4 w-4 items-center justify-center rounded-full border transition-all ${
                            on ? 'border-transparent bg-current' : 'border-line'
                          }`}
                        >
                          {on && <Check size={10} strokeWidth={3.5} className="text-paper dark:text-black" />}
                        </span>
                        {s.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </Reveal>

        {/* selection rail */}
        <Reveal delay={0.1} className="lg:col-span-5">
          <div className="panel p-5 lg:sticky lg:top-24">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-[15px] font-semibold text-ink">
                <Stethoscope size={16} className="text-muted" />
                Selected
                <span className="t-num rounded-full bg-ink/[0.06] px-2 py-0.5 text-xs font-semibold text-ink dark:bg-white/10">
                  {selected.length}
                </span>
              </h3>
              {selected.length > 0 && (
                <button
                  onClick={() => {
                    setSelected([]);
                    setResult(null);
                    setError(null);
                  }}
                  className="btn-quiet btn-sm"
                >
                  Clear
                </button>
              )}
            </div>

            <div className="mt-3 min-h-[112px]">
              {selected.length === 0 ? (
                <p className="rounded-xl border border-dashed border-line bg-paper px-4 py-6 text-center text-sm text-faint">
                  Nothing selected yet — pick at least two.
                </p>
              ) : (
                <div className="flex max-h-[180px] flex-wrap gap-1.5 overflow-y-auto">
                  <AnimatePresence initial={false}>
                    {selected.map((id) => (
                      <Motion.span
                        key={id}
                        layout
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ duration: 0.15 }}
                        className="inline-flex items-center gap-1.5 rounded-full bg-ink/[0.06] py-1.5 pl-3 pr-1.5 text-sm font-medium text-ink dark:bg-white/10"
                      >
                        {symptoms.find((s) => s.id === id)?.label}
                        <button
                          onClick={() => toggle(id)}
                          className="rounded-full p-1 text-faint transition-colors hover:bg-black/10 hover:text-ink dark:hover:bg-white/15 dark:hover:text-white"
                          aria-label="Remove"
                        >
                          <X size={12} />
                        </button>
                      </Motion.span>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>

            <button
              onClick={handleDiagnose}
              disabled={isLoading || selected.length < 2}
              className="btn-accent mt-4 w-full py-3"
            >
              {isLoading ? (
                <><Loader2 size={17} className="animate-spin" /> Analyzing…</>
              ) : (
                <>Run assessment <ArrowRight size={16} /></>
              )}
            </button>
            {selected.length === 1 && (
              <p className="mt-2.5 flex items-center justify-center gap-1.5 text-[13px] text-moderate">
                <TriangleAlert size={13} /> One more symptom needed
              </p>
            )}

            <AnimatePresence>
              {error && (
                <Motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-3 flex items-start gap-2.5 rounded-xl bg-critical/[0.07] px-3.5 py-3">
                    <CircleAlert size={16} className="mt-px shrink-0 text-critical" />
                    <p className="text-sm font-medium leading-relaxed text-critical">{error}</p>
                  </div>
                </Motion.div>
              )}
            </AnimatePresence>
          </div>
        </Reveal>
      </div>

      {/* result — compact 12-col card, everything visible at once */}
      <AnimatePresence>
        {(result || isLoading) && (
          <Motion.div
            id="symptom-result"
            style={{ scrollMarginTop: 90 }}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: easeOut }}
            className="panel overflow-hidden"
          >
            {isLoading && !result ? (
              <div className="flex items-center gap-3 p-6 text-sm text-muted">
                <Loader2 size={17} className="animate-spin text-accent" />
                Weighing symptom combinations across 42 conditions…
              </div>
            ) : result ? (
              <SymptomResultView
                result={result}
                symptoms={selected.map((id) => symptoms.find((x) => x.id === id)?.label || id)}
              />
            ) : null}
          </Motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
