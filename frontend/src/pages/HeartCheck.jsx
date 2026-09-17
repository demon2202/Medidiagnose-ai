import React, { useState } from 'react';
import axios from 'axios';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  HeartPulse,
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
  SelectField,
  EmptyState,
  Reveal
} from '../components/ui/ui';
import { easeOut } from '../lib/motion';

const FIELDS = [
  { name: 'age', label: 'Age', kind: 'number', placeholder: 'Years', min: 20, max: 100 },
  {
    name: 'sex', label: 'Sex', kind: 'select',
    options: [{ value: '', label: 'Select' }, { value: '0', label: 'Female' }, { value: '1', label: 'Male' }]
  },
  {
    name: 'cp', label: 'Chest pain type', kind: 'select',
    options: [
      { value: '', label: 'Select' },
      { value: '0', label: 'Typical angina' },
      { value: '1', label: 'Atypical angina' },
      { value: '2', label: 'Non-anginal pain' },
      { value: '3', label: 'Asymptomatic' },
    ]
  },
  { name: 'trestbps', label: 'Resting BP', kind: 'number', placeholder: 'mm Hg', min: 90, max: 200 },
  { name: 'chol', label: 'Cholesterol', kind: 'number', placeholder: 'mg/dl', min: 100, max: 600 },
  {
    name: 'fbs', label: 'Fasting sugar > 120', kind: 'select',
    options: [{ value: '', label: 'Select' }, { value: '0', label: 'No' }, { value: '1', label: 'Yes' }]
  },
  {
    name: 'restecg', label: 'Resting ECG', kind: 'select',
    options: [
      { value: '', label: 'Select' },
      { value: '0', label: 'Normal' },
      { value: '1', label: 'ST-T abnormality' },
      { value: '2', label: 'LV hypertrophy' },
    ]
  },
  { name: 'thalach', label: 'Max heart rate', kind: 'number', placeholder: 'bpm', min: 60, max: 220 },
  {
    name: 'exang', label: 'Exercise angina', kind: 'select',
    options: [{ value: '', label: 'Select' }, { value: '0', label: 'No' }, { value: '1', label: 'Yes' }]
  },
  { name: 'oldpeak', label: 'ST depression', kind: 'number', placeholder: 'Value', step: '0.1', min: 0, max: 7 },
  {
    name: 'slope', label: 'ST slope', kind: 'select',
    options: [
      { value: '', label: 'Select' },
      { value: '0', label: 'Upsloping' },
      { value: '1', label: 'Flat' },
      { value: '2', label: 'Downsloping' },
    ]
  },
  {
    name: 'ca', label: 'Major vessels', kind: 'select',
    options: [
      { value: '', label: 'Select' },
      { value: '0', label: '0' }, { value: '1', label: '1' },
      { value: '2', label: '2' }, { value: '3', label: '3' },
    ]
  },
  {
    name: 'thal', label: 'Thalassemia', kind: 'select',
    options: [
      { value: '', label: 'Select' },
      { value: '0', label: 'Normal' },
      { value: '1', label: 'Fixed defect' },
      { value: '2', label: 'Reversible defect' },
      { value: '3', label: 'Unknown' },
    ]
  },
];

const EMPTY = Object.fromEntries(FIELDS.map((f) => [f.name, '']));

export default function HeartCheck() {
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
      const response = await axios.post(`${config.api.baseURL}/predict-heart`, numeric);
      if (response.data.success) {
        setResult(cleanResult(response.data));
        addToHistory({
          type: 'heart',
          prediction: response.data.risk_level || response.data.prediction,
          confidence: response.data.confidence,
          probability: response.data.probability,
          details: response.data.recommendation
        });
        showNotification('Heart analysis complete', 'success');
        requestAnimationFrame(() => scrollToId('heart-result'));
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
        title="Heart health check"
        description="Thirteen clinical metrics, one risk read-out. Values come from routine cardiac workups."
      />

      <Reveal>
        <Disclaimer message="Risk screening for information only — not a diagnosis. Discuss results with a cardiologist before acting on them." />
      </Reveal>

      <div className="grid items-start gap-5 lg:grid-cols-12">
        <Reveal delay={0.05} className="lg:col-span-7">
          <form onSubmit={handleSubmit} className="panel p-5 sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-base font-semibold text-ink">
                <Activity size={16} className="text-muted" /> Health metrics
              </h3>
              <span className="t-num text-[13px] text-faint">{filled}/{FIELDS.length} filled</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {FIELDS.map((f) =>
                f.kind === 'select' ? (
                  <SelectField
                    key={f.name}
                    label={f.label}
                    value={form[f.name]}
                    options={f.options}
                    onChange={(e) => {
                      setForm({ ...form, [f.name]: e.target.value });
                      setError(null);
                    }}
                  />
                ) : (
                  <TextField
                    key={f.name}
                    label={f.label}
                    type="number"
                    placeholder={f.placeholder}
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={form[f.name]}
                    onChange={(e) => {
                      setForm({ ...form, [f.name]: e.target.value });
                      setError(null);
                    }}
                  />
                ),
              )}
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
                <><HeartPulse size={17} /> Assess risk</>
              )}
            </button>

            <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-line bg-paper px-4 py-3">
              <Activity size={15} className="mt-0.5 shrink-0 text-faint" />
              <p className="text-sm leading-relaxed text-muted">
                Thirteen metrics from a routine cardiac workup — the model estimates your
                cardiovascular risk and suggests sensible next steps.
              </p>
            </div>
          </form>
        </Reveal>

        {/* sticky result rail */}
        <div className="lg:col-span-5" id="heart-result" style={{ scrollMarginTop: 90 }}>
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
                      <Loader2 size={14} className="animate-spin text-accent" /> Scoring cardiovascular risk…
                    </p>
                  </Motion.div>
                ) : result ? (
                  <Motion.div
                    key="r"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: easeOut }}
                  >
                    <ScreeningResultView kind="heart" result={result} />
                  </Motion.div>
                ) : (
                  <Motion.div key="e" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <EmptyState
                      icon={HeartPulse}
                      title="Awaiting your metrics"
                      hint="Complete the form and your risk read-out with tailored guidance appears here."
                    />
                    <div className="border-t border-line bg-paper/60 p-5">
                      <p className="eyebrow mb-3">Your result will include</p>
                      <ul className="space-y-2.5">
                        {[
                          'Risk level — low, moderate or high',
                          'A probability score for the prediction',
                          'Tailored next steps for your profile',
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
