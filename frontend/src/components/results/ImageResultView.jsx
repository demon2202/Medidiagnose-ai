import React, { useState } from 'react';
import {
  AlertCircle,
  TriangleAlert,
  Activity,
  Stethoscope,
  Clock,
  ShieldCheck,
  Pill,
  ListOrdered,
} from 'lucide-react';
import {
  SeverityBadge,
  ConfidenceRing,
  Meter,
  TabPanel,
  TabBody,
  CopyButton,
} from '../ui/ui';
import { severityMeta } from '../../lib/severity';
import { buildImageSummary } from '../../lib/summary';

/* Full image-analysis result: hero + tabbed detail.
   Rendered both on the analysis page and in the history inspector. */
export default function ImageResultView({ result, fileName, imageUrl, onImageClick }) {
  const [tab, setTab] = useState('overview');

  const ranked = result?.all_predictions?.length > 1 ? result.all_predictions.slice(0, 6) : null;
  const rec =
    result?.recommendations && typeof result.recommendations === 'object'
      ? result.recommendations
      : null;
  const guidanceCount =
    (rec?.actions?.length || 0) + (rec?.next_steps?.length || 0) + (rec?.warning_signs?.length || 0);
  const hasClinical = Boolean(
    result?.staging || result?.urgency || result?.treatment_options?.length,
  );

  const tabs = [
    { value: 'overview', label: 'Overview', icon: ListOrdered },
    { value: 'clinical', label: 'Clinical detail', icon: Stethoscope },
    { value: 'guidance', label: 'Guidance', icon: ShieldCheck, count: guidanceCount || undefined },
  ];

  return (
    <>
      {/* hero: condition + ring side by side */}
      <div className="flex items-center gap-5 bg-gradient-to-br from-accent/[0.06] via-transparent to-transparent p-5 pb-4 sm:gap-7 sm:p-6 sm:pb-5">
        {imageUrl && (
          <button
            onClick={onImageClick}
            aria-label="Enlarge image"
            className="hidden h-[112px] w-[112px] shrink-0 cursor-zoom-in self-start overflow-hidden rounded-2xl border border-line bg-surface transition-transform hover:scale-[1.03] sm:block"
          >
            <img src={imageUrl} alt="" className="h-full w-full object-cover" />
          </button>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <SeverityBadge level={result.severity} />
            {result.signal_processed && (
              <span className="badge bg-low/10 text-low">Signal analysis</span>
            )}
            {result.prediction?.type && (
              <span className="badge border border-line bg-paper font-medium !text-muted">
                {result.prediction.type}
              </span>
            )}
            <span className="ml-auto hidden sm:inline-flex">
              <CopyButton text={buildImageSummary(result, fileName)} />
            </span>
          </div>
          <p className="eyebrow mt-3">Detected condition</p>
          <h3 className="mt-1 text-balance font-display text-[34px] leading-tight tracking-normal text-ink sm:text-[38px]">
            {result.prediction?.name}
          </h3>
          <div className="t-num mt-2 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[13px] text-faint">
            {result.prediction?.code && <span>Code {result.prediction.code}</span>}
            {result.prediction?.birads && <span>{result.prediction.birads}</span>}
            {fileName && <span className="truncate">{fileName}</span>}
          </div>
          <span className="mt-3 inline-flex sm:hidden">
            <CopyButton text={buildImageSummary(result, fileName)} />
          </span>
        </div>
        <div className="hidden h-28 w-px shrink-0 bg-line sm:block" />
        <ConfidenceRing value={result.prediction?.confidence} size={112} label="Confidence" />
      </div>

      <div className="border-t border-line px-2 pt-1 sm:px-4">
        <TabPanel tabs={tabs} value={tab} onChange={setTab} id="img-result" />
      </div>
      <div className="min-h-[240px] p-5 pt-4 sm:p-6 sm:pt-4">
        {tab === 'overview' && (
          <TabBody tabKey="overview">
            {ranked ? (
              <div className="space-y-1">
                {ranked.map((p, i) => {
                  const meta = severityMeta(i === 0 ? result.severity : 'unknown');
                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-3 rounded-lg px-2.5 py-2 ${i === 0 ? 'bg-raised/70' : ''}`}
                    >
                      <span className="t-num w-5 shrink-0 text-xs font-semibold text-faint">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline justify-between gap-3">
                          <p className={`truncate text-sm ${i === 0 ? 'font-semibold text-ink' : 'text-muted'}`}>
                            {p.name}
                          </p>
                          <span className="t-num shrink-0 text-sm font-semibold text-ink">
                            {(p.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <Meter value={p.confidence} tone={i === 0 ? meta.color : undefined} className="mt-1.5" />
                      </div>
                      {p.type && (
                        <span className="hidden shrink-0 text-xs font-medium text-faint sm:block">
                          {p.type}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="py-8 text-center text-[13px] text-faint">
                Single-condition result — see Clinical detail and Guidance.
              </p>
            )}
            {result.note && (
              <p className="mt-3 flex items-start gap-2 rounded-lg bg-moderate/[0.08] px-3 py-2.5 text-sm leading-relaxed text-muted">
                <AlertCircle size={14} className="mt-0.5 shrink-0 text-moderate" />
                {result.note}
              </p>
            )}
          </TabBody>
        )}

        {tab === 'clinical' && (
          <TabBody tabKey="clinical" className="space-y-4">
            {!hasClinical && (
              <p className="py-8 text-center text-[13px] text-faint">
                No additional clinical detail for this result.
              </p>
            )}
            {result.staging && typeof result.staging === 'object' && (
              <section>
                <p className="eyebrow mb-2 flex items-center gap-1.5">
                  <Activity size={12} /> Staging
                </p>
                <dl className="grid gap-2 text-sm sm:grid-cols-3">
                  {['stage', 'description', 'prognosis'].map(
                    (k) =>
                      result.staging[k] && (
                        <div key={k} className="inset px-3 py-2.5">
                          <dt className="text-xs font-medium uppercase tracking-wide text-faint">{k}</dt>
                          <dd className="mt-0.5 font-medium leading-relaxed text-ink">{result.staging[k]}</dd>
                        </div>
                      ),
                  )}
                </dl>
              </section>
            )}
            {result.urgency && typeof result.urgency === 'object' && (
              <section className="flex items-start gap-3 rounded-xl border border-line bg-paper px-4 py-3">
                <Clock size={16} className="mt-0.5 shrink-0 text-accent" />
                <div>
                  <p className="text-[15px] font-semibold text-ink">{result.urgency.timeline}</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-muted">{result.urgency.action}</p>
                </div>
              </section>
            )}
            {Array.isArray(result.treatment_options) && result.treatment_options.length > 0 && (
              <section>
                <p className="eyebrow mb-2 flex items-center gap-1.5">
                  <Pill size={12} /> Treatment options
                </p>
                <ul className="grid gap-1.5 sm:grid-cols-2">
                  {result.treatment_options.slice(0, 8).map((t, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm leading-relaxed text-muted"
                    >
                      <span className="t-num mt-px shrink-0 text-xs font-semibold text-faint">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      {t}
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </TabBody>
        )}

        {tab === 'guidance' && (
          <TabBody tabKey="guidance" className="space-y-4">
            {!rec && (
              <p className="py-8 text-center text-[13px] text-faint">
                No structured guidance for this result.
              </p>
            )}
            {rec?.title && <p className="text-base font-semibold text-ink">{rec.title}</p>}
            {rec?.message && <p className="text-[15px] leading-relaxed text-muted">{rec.message}</p>}
            {Array.isArray(rec?.actions) && rec.actions.length > 0 && (
              <section>
                <p className="eyebrow mb-2">Recommended actions</p>
                <ol className="space-y-1.5">
                  {rec.actions.map((a, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-[15px] leading-relaxed text-ink">
                      <span className="t-num mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-ink/[0.06] text-xs font-semibold text-muted dark:bg-white/10">
                        {i + 1}
                      </span>
                      {a}
                    </li>
                  ))}
                </ol>
              </section>
            )}
            {Array.isArray(rec?.next_steps) && rec.next_steps.length > 0 && (
              <section>
                <p className="eyebrow mb-2">Next steps</p>
                <ul className="space-y-1 text-[15px] leading-relaxed text-muted">
                  {rec.next_steps.map((s, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="text-faint">—</span> {s}
                    </li>
                  ))}
                </ul>
              </section>
            )}
            {Array.isArray(rec?.warning_signs) && rec.warning_signs.length > 0 && (
              <section className="rounded-xl bg-critical/[0.06] px-4 py-3">
                <p className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-critical">
                  <TriangleAlert size={14} /> Warning signs — seek care if these appear
                </p>
                <ul className="space-y-1 text-sm leading-relaxed text-critical/90">
                  {rec.warning_signs.map((s, i) => (
                    <li key={i}>• {s}</li>
                  ))}
                </ul>
              </section>
            )}
            {Array.isArray(rec?.risk_factors) && rec.risk_factors.length > 0 && (
              <section>
                <p className="eyebrow mb-2">Risk factors</p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.risk_factors.map((f, i) => (
                    <span key={i} className="rounded-full border border-line bg-paper px-2.5 py-1 text-[13px] text-muted">
                      {f}
                    </span>
                  ))}
                </div>
              </section>
            )}
            {rec?.note && (
              <p className="border-t border-line pt-3 text-[13px] leading-relaxed text-faint">{rec.note}</p>
            )}
          </TabBody>
        )}
      </div>
    </>
  );
}
