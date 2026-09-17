import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';
import { ConfidenceRing, Meter, CopyButton } from '../ui/ui';
import { buildSymptomSummary } from '../../lib/summary';

const TIER_TONE = { High: '#059669', Moderate: '#ca8a04', Low: '#ea580c' };

/* Compact 3-column symptom result. Shared by the page and the inspector. */
export default function SymptomResultView({ result, symptoms = [] }) {
  if (!result) return null;
  return (
    <div className="grid lg:grid-cols-12">
      <div className="flex items-center gap-5 bg-gradient-to-br from-accent/[0.06] via-transparent to-transparent p-5 sm:p-6 lg:col-span-5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="eyebrow">Predicted condition</p>
            <CopyButton text={buildSymptomSummary(result, symptoms)} className="!px-2" />
          </div>
          <h3 className="mt-1 text-balance font-display text-[30px] leading-tight tracking-normal text-ink sm:text-[34px]">
            {result.prediction}
          </h3>
          {result.description && (
            <p className="mt-2 line-clamp-3 text-[15px] leading-relaxed text-muted">
              {result.description}
            </p>
          )}
        </div>
        <ConfidenceRing value={result.confidence} size={100} label={result.confidence_tier} />
      </div>

      {result.alternative_diagnoses?.length > 0 && (
        <div className="border-t border-line p-5 sm:p-6 lg:col-span-3 lg:border-l lg:border-t-0">
          <p className="eyebrow mb-3 flex items-center gap-1.5">
            <Info size={12} /> Also considered
          </p>
          <div className="space-y-2.5">
            {result.alternative_diagnoses.slice(0, 4).map((alt, i) => (
              <div key={i}>
                <div className="flex items-baseline justify-between gap-2">
                  <p className="truncate text-sm text-muted">{alt.disease}</p>
                  <span className="t-num shrink-0 text-sm font-semibold text-ink">
                    {(alt.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <Meter value={alt.confidence} tone={TIER_TONE[result.confidence_tier]} className="mt-1" />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-line bg-paper/60 p-5 sm:p-6 lg:col-span-4 lg:border-l lg:border-t-0">
        <p className="eyebrow mb-3 flex items-center gap-1.5">
          <ShieldCheck size={12} /> Guidance
        </p>
        <ul className="space-y-1.5">
          {[...(result.recommendations || []), ...(result.precautions || [])]
            .slice(0, 5)
            .map((r, i) => (
              <li key={i} className="flex gap-2 text-[15px] leading-relaxed text-muted">
                <span className="mt-[9px] h-1 w-1 shrink-0 rounded-full bg-faint" />
                {r}
              </li>
            ))}
        </ul>
      </div>
    </div>
  );
}
