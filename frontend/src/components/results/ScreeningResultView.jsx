import React from 'react';
import { ShieldCheck, ArrowRight } from 'lucide-react';
import { SeverityBadge, ConfidenceRing, CopyButton } from '../ui/ui';
import { buildScreeningSummary } from '../../lib/summary';

const RISK_TO_SEVERITY = { High: 'high', Moderate: 'moderate', Low: 'low' };

/* Shared heart / cancer screening result. */
export default function ScreeningResultView({ kind, result }) {
  if (!result) return null;
  const isHeart = kind === 'heart';
  const malignant = result.prediction === 'Malignant';

  const level = isHeart
    ? RISK_TO_SEVERITY[result.risk_level] || 'unknown'
    : malignant
      ? 'critical'
      : 'healthy';
  const title = isHeart ? `${result.risk_level} risk` : result.prediction;
  const eyebrow = isHeart ? 'Risk level' : 'Prediction';
  const ringLabel = isHeart ? 'Probability' : 'Malignancy';

  return (
    <>
      <div className="flex items-center gap-4 bg-gradient-to-br from-accent/[0.06] via-transparent to-transparent p-5 pb-4 sm:p-6 sm:pb-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <SeverityBadge level={level} />
            <span className="ml-auto">
              <CopyButton text={buildScreeningSummary(kind, result)} className="!px-2" />
            </span>
          </div>
          <p className="eyebrow mt-2.5">{eyebrow}</p>
          <h3 className="font-display text-[30px] leading-tight tracking-normal text-ink sm:text-[32px]">{title}</h3>
          {!isHeart && typeof result.confidence === 'number' && (
            <p className="t-num mt-1 text-[13px] text-faint">
              Model confidence {(result.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>
        <ConfidenceRing value={result.probability} size={104} label={ringLabel} />
      </div>
      {result.recommendation && (
        <div className="border-t border-line bg-paper/60 p-5 sm:p-6">
          <p className="eyebrow mb-2 flex items-center gap-1.5">
            <ShieldCheck size={12} /> Recommendation
          </p>
          <p className="text-[15px] font-medium leading-relaxed text-ink">
            {result.recommendation.message}
          </p>
          {result.recommendation.actions?.length > 0 && (
            <ul className="mt-2.5 space-y-1.5">
              {result.recommendation.actions.map((a, i) => (
                <li key={i} className="flex gap-2 text-sm leading-relaxed text-muted">
                  <ArrowRight size={13} className="mt-1 shrink-0 text-faint" /> {a}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </>
  );
}
