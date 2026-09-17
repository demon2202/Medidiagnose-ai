/* Single source of truth for severity colour. Used sparingly: dots + badges only. */
export const SEVERITY = {
  critical: { label: 'Critical', color: '#dc2626', soft: 'rgba(220,38,38,0.10)' },
  high: { label: 'High', color: '#ea580c', soft: 'rgba(234,88,12,0.10)' },
  moderate: { label: 'Moderate', color: '#ca8a04', soft: 'rgba(202,138,4,0.12)' },
  low: { label: 'Low', color: '#059669', soft: 'rgba(5,150,105,0.10)' },
  healthy: { label: 'Healthy', color: '#0f766e', soft: 'rgba(15,118,110,0.10)' },
  unknown: { label: 'Unknown', color: '#71717a', soft: 'rgba(113,113,122,0.12)' },
};

export const severityMeta = (s) =>
  SEVERITY[String(s || '').toLowerCase()] || SEVERITY.unknown;
