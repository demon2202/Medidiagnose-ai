import {
  Stethoscope,
  ScanLine,
  HeartPulse,
  Microscope,
  Activity,
} from 'lucide-react';

const IMAGE_LABELS = { xray: 'X-ray', skin: 'Skin', breast: 'Breast', heart: 'Heart' };

const META = {
  symptom: { icon: Stethoscope, label: 'Symptom check' },
  heart: { icon: HeartPulse, label: 'Heart screening' },
  cancer: { icon: Microscope, label: 'Cancer screening' },
};

/* Icon + human label for a history entry type (e.g. 'image_xray'). */
export function diagnosisTypeMeta(type) {
  if (type?.startsWith('image_')) {
    const key = type.replace('image_', '');
    return { icon: ScanLine, label: `Image · ${IMAGE_LABELS[key] || key}` };
  }
  return META[type] || { icon: Activity, label: type || 'Check' };
}

/* Display title for a history entry (prediction may be string or object). */
export function entryTitle(entry) {
  const p = entry?.prediction;
  if (!p) return 'Unknown';
  if (typeof p === 'string') return p;
  return p.name || p.disease || p.condition || p.finding || 'Unknown';
}

/* '82%' or null. */
export function entryConfidence(entry) {
  const c = entry?.confidence;
  if (typeof c !== 'number') return null;
  return `${(c <= 1 ? c * 100 : c).toFixed(0)}%`;
}

/* Relative time + full date helpers. */
export function timeAgo(ts) {
  if (!ts) return 'Unknown time';
  const mins = Math.floor((Date.now() - new Date(ts)) / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function dayGroup(ts) {
  const d = new Date(ts);
  const now = new Date();
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate());
  const diff = Math.round((startOf(now) - startOf(d)) / 864e5);
  if (diff <= 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  if (diff < 7) return 'Previous 7 days';
  return 'Earlier';
}

export function fullDate(ts) {
  return new Date(ts).toLocaleDateString('en-US', {
    weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

/* Export history as a JSON file. */
export function downloadHistory(history) {
  const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `medidiagnose-history-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
