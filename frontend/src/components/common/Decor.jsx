import React from 'react';

/* Animated ECG waveform — the app's single recurring visual motif.
   Pure SVG, no assets, theme-aware via the `tone` prop. */

const ECG_PATH =
  'M0 40 H100 c8 -6 16 -6 24 0 H200 l6 -10 l10 -22 l10 22 l6 10 c16 -10 32 -10 48 0 ' +
  'H400 c8 -6 16 -6 24 0 H500 l6 -10 l10 -22 l10 22 l6 10 c16 -10 32 -10 48 0 ' +
  'H700 c8 -6 16 -6 24 0 H800 l6 -10 l10 -22 l10 22 l6 10 c16 -10 32 -10 48 0 H1000';

export function EcgLine({ className = '', tone = 'rgb(var(--c-accent))' }) {
  return (
    <svg
      viewBox="0 0 1000 80"
      preserveAspectRatio="none"
      className={className}
      aria-hidden="true"
      fill="none"
    >
      {/* static trace */}
      <path
        d={ECG_PATH}
        stroke={tone}
        strokeWidth={1.5}
        strokeOpacity={0.22}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      {/* travelling pulse */}
      <path
        d={ECG_PATH}
        className="ecg-travel"
        stroke={tone}
        strokeWidth={1.5}
        strokeOpacity={0.7}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
