import React from 'react';
import { Link } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import { Stethoscope, ScanLine, ShieldCheck } from 'lucide-react';
import { easeOut } from '../../lib/motion';

const POINTS = [
  { icon: Stethoscope, t: 'Symptom triage', d: 'Ranked assessments across 42 conditions.' },
  { icon: ScanLine, t: 'Image screening', d: 'Skin, chest X-ray, mammogram and ECG.' },
  { icon: ShieldCheck, t: 'Private by design', d: 'History lives on your device, nowhere else.' },
];

export default function AuthShell({ children, title, subtitle, footer }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      {/* brand side */}
      <div className="relative hidden flex-col justify-between overflow-hidden bg-[#0d0d0c] p-12 lg:flex">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.5]"
          style={{
            backgroundImage: 'radial-gradient(rgba(255,255,255,0.14) 1px, transparent 1px)',
            backgroundSize: '22px 22px',
            maskImage: 'radial-gradient(ellipse 70% 60% at 30% 40%, black, transparent)'
          }}
        />
        <div className="pointer-events-none absolute -left-32 top-1/3 h-96 w-96 rounded-full bg-[#0f766e]/25 blur-[120px]" />
        <div className="relative flex items-center gap-3">
          <img src="/mark.svg" alt="" className="h-9 w-9 rounded-[10px]" />
          <div>
            <p className="text-[15px] font-semibold tracking-tight text-white">MediDiagnose</p>
            <p className="text-xs text-white/50">AI health companion</p>
          </div>
        </div>

        <div className="relative">
          <Motion.h2
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: easeOut }}
            className="max-w-md text-balance font-display text-[42px] font-normal leading-[1.06] tracking-tight text-white"
          >
            Quiet, precise answers about your health.
          </Motion.h2>
          <div className="mt-8 space-y-2.5">
            {POINTS.map((p, i) => (
              <Motion.div
                key={p.t}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: easeOut, delay: 0.1 + i * 0.08 }}
                className="flex items-center gap-3.5 rounded-xl border border-white/10 bg-white/[0.04] p-3.5 backdrop-blur-sm"
              >
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.06] text-white/80">
                  <p.icon size={17} strokeWidth={1.9} />
                </span>
                <span>
                  <span className="block text-sm font-medium text-white">{p.t}</span>
                  <span className="block text-[13px] text-white/55">{p.d}</span>
                </span>
              </Motion.div>
            ))}
          </div>
        </div>

        <p className="relative text-xs text-white/35">
          For information only — not a medical device.
        </p>
      </div>

      {/* form side */}
      <div className="flex items-center justify-center px-5 py-10 sm:px-10">
        <Motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: easeOut }}
          className="w-full max-w-[400px]"
        >
          <Link to="/login" className="mb-8 flex items-center gap-2.5 lg:hidden">
            <img src="/mark.svg" alt="" className="h-8 w-8 rounded-lg" />
            <span className="text-[15px] font-semibold tracking-tight text-ink">MediDiagnose</span>
          </Link>
          <h1 className="font-display text-[30px] font-normal tracking-tight text-ink">{title}</h1>
          <p className="mb-7 mt-1 text-sm text-muted">{subtitle}</p>
          {children}
          {footer && (
            <p className="mt-7 text-center text-sm text-muted">
              {footer}
            </p>
          )}
        </Motion.div>
      </div>
    </div>
  );
}
