import React, { useState } from 'react';
import axios from 'axios';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  Microscope,
  Loader2,
  CircleAlert,
  Activity,
  ArrowRight
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { config } from '../config/config';
import { scrollToId } from '../lib/scroll';
import { cleanResult } from '../lib/text';
import Disclaimer from '../components/common/Disclaimer';
import ScreeningResultView from '../components/results/ScreeningResultView';
import {
  PageHeader,
  TextField,
  EmptyState,
  Reveal
} from '../components/ui/ui';
import { easeOut } from '../lib/motion';

const FIELDS = [
  { name: 'radius_mean', label: 'Radius', hint: '6 – 30', placeholder: '14.5' },
  { name: 'texture_mean', label: 'Texture', hint: '9 – 40', placeholder: '19.0' },
  { name: 'perimeter_mean', label: 'Perimeter', hint: '40 – 190', placeholder: '92.0' },
  { name: 'area_mean', label: 'Area', hint: '140 – 2500', placeholder: '655.0' },
  { name: 'smoothness_mean', label: 'Smoothness', hint: '0.05 – 0.16', placeholder: '0.096' },
  { name: 'compactness_mean', label: 'Compactness', hint: '0.02 – 0.35', placeholder: '0.104' },
  { name: 'concavity_mean', label: 'Concavity', hint: '0 – 0.43', placeholder: '0.088' },
  { name: 'concave_points_mean', label: 'Concave points', hint: '0 – 0.20', placeholder: '0.049' },
  { name: 'symmetry_mean', label: 'Symmetry', hint: '0.10 – 0.30', placeholder: '0.181' },
  { name: 'fractal_dimension_mean', label: 'Fractal dimension', hint: '0.05 – 0.10', placeholder: '0.063' },
];

const EMPTY = Object.fromEntries(FIELDS.map((f) => [f.name, '']));

export default function CancerScreening() {
  const { addToHistory, isLoading, setIsLoading, showNotification } = useApp();
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const filled = FIELDS.filter((f) => form[f.name] !== '').length;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (filled < FIELDS.length) {
      setError(`Fill in all fields — ${FIELDS.length - filled} remaining.`);
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const numeric = Object.fromEntries(Object.entries(form).map(([k, v]) => [k, parseFloat(v)]));
      const response = await axios.post(`${config.api.baseURL}/predict-cancer`, numeric);
      if (response.data.success) {
        setResult(cleanResult(response.data));
        addToHistory({
          type: 'cancer',
          prediction: response.data.prediction,
          probability: response.data.probability,
          confidence: response.data.confidence,
          details: response.data.recommendation,
          data: cleanResult(response.data),
        });
        showNotification('Screening complete', 'success');
        requestAnimationFrame(() => scrollToId('cancer-result'));
      } else {
        setError(response.data.error || 'Failed to analyze');
      }
    } catch (err) {
      if (err.response)
        setError(`Analysis failed: ${err.response.data?.error || err.response.data?.message || `server error ${err.response.status}`}`);
      else if (err.request)
        setError(`Could not reach the server. Is the backend running on ${config.api.baseURL}?`);
      else setError(`Request error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Screening"
        title="Breast cancer screening"
        description="Ten cell-nucleus features from fine-needle aspirate (FNA) tests, scored by a Wisconsin-dataset model."
      />

      <Reveal>
        <Disclaimer message="Educational screening support only — never a diagnosis. Any concerning result needs prompt review by an oncologist." />
      </Reveal>

      <div className="grid items-start gap-5 lg:grid-cols-12">
        <Reveal delay={0.05} className="lg:col-span-7">
          <form onSubmit={handleSubmit} className="panel p-5 sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-base font-semibold text-ink">
                <Activity size={16} className="text-muted" /> Tumor characteristics
              </h3>
              <span className="t-num text-[13px] text-faint">{filled}/{FIELDS.length} filled</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {FIELDS.map((f) => (
                <TextField
                  key={f.name}
                  label={f.label}
                  hint={f.hint}
                  type="number"
                  step="any"
                  placeholder={f.placeholder}
                  value={form[f.name]}
                  onChange={(e) => {
                    setForm({ ...form, [f.name]: e.target.value });
                    setError(null);
                  }}
                />
              ))}
            </div>

            <AnimatePresence>
              {error && (
                <Motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-critical/[0.07] px-3.5 py-3">
                    <CircleAlert size={16} className="mt-px shrink-0 text-critical" />
                    <p className="text-sm font-medium leading-relaxed text-critical">{error}</p>
                  </div>
                </Motion.div>
              )}
            </AnimatePresence>

            <button type="submit" disabled={isLoading} className="btn-accent mt-4 w-full py-3">
              {isLoading ? (
                <><Loader2 size={17} className="animate-spin" /> Analyzing…</>
              ) : (
                <><Microscope size={17} /> Score characteristics</>
              )}
            </button>

            <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-line bg-paper px-4 py-3">
              <Activity size={15} className="mt-0.5 shrink-0 text-faint" />
              <p className="text-sm leading-relaxed text-muted">
                Ten cell-nucleus measurements from a fine-needle aspirate — the model weighs
                them together and returns a benign or malignant read-out.
              </p>
            </div>
          </form>
        </Reveal>

        <div className="lg:col-span-5" id="cancer-result" style={{ scrollMarginTop: 90 }}>
          <Reveal delay={0.1}>
            <div className="panel overflow-hidden lg:sticky lg:top-24">
              <AnimatePresence mode="wait" initial={false}>
                {isLoading ? (
                  <Motion.div key="l" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="p-6">
                    <div className="flex items-center gap-4">
                      <div className="flex-1 space-y-3">
                        <div className="skeleton h-3 w-24 rounded-full" />
                        <div className="skeleton h-6 w-2/3 rounded-lg" />
                      </div>
                      <div className="skeleton h-20 w-20 rounded-full" />
                    </div>
                    <p className="mt-4 flex items-center gap-2 text-sm text-muted">
                      <Loader2 size={14} className="animate-spin text-accent" /> Scoring characteristics…
                    </p>
                  </Motion.div>
                ) : result ? (
                  <Motion.div
                    key="r"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: easeOut }}
                  >
                    <ScreeningResultView kind="cancer" result={result} />
                  </Motion.div>
                ) : (
                  <Motion.div key="e" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <EmptyState
                      icon={Microscope}
                      title="Awaiting characteristics"
                      hint="Enter the ten FNA features and the benign / malignant read-out appears here."
                    />
                    <div className="border-t border-line bg-paper/60 p-5">
                      <p className="eyebrow mb-3">Your result will include</p>
                      <ul className="space-y-2.5">
                        {[
                          'Benign or malignant prediction',
                          'A confidence score for the read',
                          'Recommended next steps and review',
                        ].map((t) => (
                          <li key={t} className="flex items-center gap-2.5 text-sm text-muted">
                            <span className="dot !bg-accent" /> {t}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </Motion.div>
                )}
              </AnimatePresence>
            </div>
          </Reveal>
        </div>
      </div>
    </div>
  );
}
