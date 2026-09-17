import React, { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import {
  Stethoscope,
  ScanLine,
  HeartPulse,
  Microscope,
  Activity,
  CalendarClock,
  ArrowRight,
  ArrowUpRight,
  History as HistoryIcon,
  Sprout,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { Reveal } from '../components/ui/ui';
import { EcgLine } from '../components/common/Decor';
import { diagnosisTypeMeta as typeMeta, entryTitle, entryConfidence } from '../lib/diagnosis';
import { useCountUp } from '../lib/useCountUp';

const TIPS = [
  'Annual screenings catch issues early — book yours before symptoms appear.',
  'Eight glasses of water a day keeps kidneys and focus in shape.',
  'Thirty minutes of moderate movement, most days, transforms cardiac risk.',
  'Half your plate plants, a quarter protein, a quarter whole grains.',
  'Seven to nine hours of sleep — non-negotiable for immunity and memory.',
  'Ten minutes of stillness a day measurably lowers resting pressure.',
  'Sunscreen daily, even indoors by a window. Your skin keeps score.',
];

/* soft tint per identity — one hue per tool/metric, kept quiet */
const TINT = {
  accent: 'border-accent/25 bg-accent-soft text-accent',
  ink: 'border-ink/10 bg-ink/[0.05] text-ink dark:border-white/10 dark:bg-white/10 dark:text-white',
  low: 'border-low/20 bg-low/10 text-low',
  moderate: 'border-moderate/20 bg-moderate/10 text-moderate',
  critical: 'border-critical/20 bg-critical/10 text-critical',
};

const TOOLS = [
  { to: '/symptoms', icon: Stethoscope, title: 'Symptom check', desc: 'Describe symptoms, get a ranked assessment.', tint: 'accent' },
  { to: '/image-analysis', icon: ScanLine, title: 'Image analysis', desc: 'Skin, X-ray, mammogram and ECG scans.', tint: 'low' },
  { to: '/heart-check', icon: HeartPulse, title: 'Heart screening', desc: 'Cardiovascular risk from 13 metrics.', tint: 'critical' },
  { to: '/cancer-screening', icon: Microscope, title: 'Cancer screening', desc: 'FNA characteristics, benign or malignant.', tint: 'moderate' },
];

const greeting = () => {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
};

function StatCard({ icon: Icon, value, label, tint }) {
  const v = useCountUp(value);
  return (
    <div className="panel flex h-full flex-col p-5">
      {Icon && (
        <span className={`flex h-11 w-11 items-center justify-center rounded-xl border ${TINT[tint]}`}>
          <Icon size={20} strokeWidth={1.9} />
        </span>
      )}
      <p className="t-num mt-4 text-[36px] font-semibold leading-none tracking-tight text-ink">{v}</p>
      <p className="mt-1.5 text-sm text-muted">{label}</p>
    </div>
  );
}

/* last-14-days check volume, dependency-free bars */
function ActivityChart({ history }) {
  const days = useMemo(() => {
    const counts = new Map();
    for (const h of history) {
      const k = new Date(h.timestamp).toDateString();
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    const arr = [];
    const now = new Date();
    for (let i = 13; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
      arr.push({ date: d, count: counts.get(d.toDateString()) || 0, today: i === 0 });
    }
    return arr;
  }, [history]);
  const max = Math.max(1, ...days.map((d) => d.count));
  const total = days.reduce((n, d) => n + d.count, 0);

  return (
    <div className="panel flex h-full flex-col p-6">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold text-ink">Activity</h3>
        <p className="t-num text-[13px] text-faint">
          {total} check{total === 1 ? '' : 's'} · 14 days
        </p>
      </div>
      <div className="mt-5 flex flex-1 items-end gap-[6px]" role="img" aria-label={`${total} checks in the last 14 days`}>
        {days.map((d) => (
          <div
            key={d.date.toISOString()}
            title={`${d.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} — ${d.count}`}
            className="group flex h-44 min-w-0 flex-1 flex-col items-center justify-end gap-2"
          >
            <span
              className={`w-full max-w-[26px] rounded-[5px] transition-all duration-300 ${
                d.count === 0
                  ? 'bg-ink/[0.07] dark:bg-white/10'
                  : d.today
                    ? 'bg-ink dark:bg-white'
                    : 'bg-ink/30 group-hover:bg-ink/50 dark:bg-white/30 dark:group-hover:bg-white/60'
              }`}
              style={{ height: d.count === 0 ? 5 : `${Math.max(12, (d.count / max) * 100)}%` }}
            />
            <span className={`text-[11px] leading-none ${d.today ? 'font-bold text-ink' : 'text-faint'}`}>
              {d.date.toLocaleDateString('en-US', { weekday: 'narrow' })}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { user, getStats, history, openInspector } = useApp();
  const stats = getStats();
  const recent = history.slice(0, 5);
  const tip = TIPS[new Date().getDay() % TIPS.length];
  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });

  const cells = [
    { value: stats.totalDiagnoses, label: 'Total checks', icon: Activity, tint: 'accent' },
    { value: stats.symptomDiagnoses, label: 'Symptoms', icon: Stethoscope, tint: 'ink' },
    { value: stats.imageDiagnoses, label: 'Images', icon: ScanLine, tint: 'low' },
    { value: stats.recentDiagnoses, label: 'This week', icon: CalendarClock, tint: 'moderate' },
  ];

  return (
    <div className="space-y-5">
      {/* greeting hero */}
      <Reveal>
        <div className="panel relative overflow-hidden">
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-accent/[0.07] via-transparent to-transparent" />
          <div className="dotgrid pointer-events-none absolute inset-y-0 right-0 w-2/5 opacity-40 [mask-image:radial-gradient(70%_70%_at_80%_20%,black,transparent)]" />
          <div className="relative flex flex-wrap items-center justify-between gap-6 p-6 sm:p-8">
            <div className="min-w-0">
              <p className="eyebrow mb-2 flex items-center gap-2">
                <span className="dot !bg-accent" /> {today}
              </p>
              <h1 className="text-balance font-display text-[34px] font-normal leading-tight tracking-tight text-ink md:text-[44px]">
                {greeting()}, {user?.name?.split(' ')[0] || 'there'}.
              </h1>
              <p className="mt-2 text-base text-muted">What would you like to check today?</p>
            </div>
            <div className="flex shrink-0 items-center gap-2.5">
              <Link to="/symptoms" className="btn-accent px-5 py-3 text-[15px]">
                New assessment <ArrowRight size={16} />
              </Link>
              <Link to="/history" className="btn-ghost px-4 py-3">
                <HistoryIcon size={16} /> History
              </Link>
            </div>
          </div>
          <EcgLine className="h-12 w-full" />
        </div>
      </Reveal>

      {/* stats + activity */}
      <div className="grid items-stretch gap-5 lg:grid-cols-12">
        <Reveal delay={0.05} className="lg:col-span-7">
          <div className="grid h-full grid-cols-2 gap-4">
            {cells.map((c) => (
              <StatCard key={c.label} {...c} />
            ))}
          </div>
        </Reveal>
        <Reveal delay={0.08} className="lg:col-span-5">
          <ActivityChart history={history} />
        </Reveal>
      </div>

      {/* tools */}
      <Reveal delay={0.1}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {TOOLS.map((t, i) => (
            <Motion.div
              key={t.to}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.05, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              <Link
                to={t.to}
                className="panel group flex h-full flex-col p-6 transition-all duration-200 hover:-translate-y-0.5 hover:!shadow-lift"
              >
                <span className={`flex h-12 w-12 items-center justify-center rounded-xl border ${TINT[t.tint]} transition-transform group-hover:scale-105`}>
                  <t.icon size={21} strokeWidth={1.9} />
                </span>
                <span className="mt-4 flex items-center justify-between gap-2">
                  <span className="text-[17px] font-semibold tracking-tight text-ink">{t.title}</span>
                  <ArrowUpRight
                    size={17}
                    className="shrink-0 text-faint transition-all duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-ink"
                  />
                </span>
                <span className="mt-1 text-[15px] leading-relaxed text-muted">{t.desc}</span>
              </Link>
            </Motion.div>
          ))}
        </div>
      </Reveal>

      <div className="grid items-stretch gap-5 lg:grid-cols-12">
        {/* recent */}
        <Reveal delay={0.15} className="h-full lg:col-span-7">
          <div className="panel flex h-full flex-col p-2">
            <div className="flex items-center justify-between px-3.5 pb-1 pt-3">
              <h3 className="text-base font-semibold text-ink">Recent activity</h3>
              <Link to="/history" className="btn-quiet btn-sm !px-2">
                View all <ArrowRight size={13} />
              </Link>
            </div>
            {recent.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center px-6 py-14 text-center">
                <span className="dotgrid mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-raised/50 text-faint">
                  <HistoryIcon size={24} strokeWidth={1.8} />
                </span>
                <p className="text-base font-medium text-ink">No checks yet</p>
                <p className="mt-1 text-sm text-muted">Run an assessment and it will collect here.</p>
              </div>
            ) : (
              <div className="pb-1.5">
                {recent.map((item) => {
                  const m = typeMeta(item.type);
                  const conf = entryConfidence(item);
                  const reopenable = Boolean(item.data);
                  return (
                    <button
                      key={item.id}
                      onClick={() => reopenable && openInspector(item)}
                      disabled={!reopenable}
                      title={reopenable ? 'Open full result' : undefined}
                      className={`flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-left transition-colors ${
                        reopenable ? 'hover:bg-raised/60' : 'cursor-default'
                      }`}
                    >
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-line bg-surface text-muted">
                        <m.icon size={16} strokeWidth={1.9} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[15px] font-medium text-ink">{entryTitle(item)}</span>
                        <span className="block text-[13px] capitalize text-faint">
                          {m.label} ·{' '}
                          {new Date(item.timestamp).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      </span>
                      {conf != null && (
                        <span className="t-num shrink-0 rounded-full bg-ink/[0.05] px-2.5 py-1 text-[13px] font-semibold text-ink dark:bg-white/10">
                          {conf}%
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </Reveal>

        {/* tip */}
        <Reveal delay={0.2} className="h-full lg:col-span-5">
          <div className="panel relative flex h-full flex-col overflow-hidden p-6">
            <div className="dotgrid pointer-events-none absolute inset-x-0 top-0 h-28 opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
            <div className="relative flex h-full flex-col">
              <p className="eyebrow flex items-center gap-1.5">
                <Sprout size={14} /> Daily note
              </p>
              <p className="mt-3 flex-1 text-balance font-display text-[24px] font-normal leading-snug tracking-tight text-ink">
                “{tip}”
              </p>
              <EcgLine tone="rgb(var(--c-ink))" className="mt-4 h-8 w-full opacity-40" />
              <Link to="/health-tips" className="btn-quiet btn-sm mt-4 !px-0">
                Browse all tips <ArrowRight size={13} />
              </Link>
            </div>
          </div>
        </Reveal>
      </div>
    </div>
  );
}
