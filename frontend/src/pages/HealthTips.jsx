import React, { useState } from 'react';
import { motion as Motion } from 'framer-motion';
import {
  Apple,
  Moon,
  Dumbbell,
  Brain,
  Droplets,
  LayoutGrid,
  ShieldCheck,
  ArrowUpRight,
} from 'lucide-react';
import { PageHeader, Reveal } from '../components/ui/ui';

const CATS = [
  { id: 'all', label: 'All', icon: LayoutGrid },
  { id: 'nutrition', label: 'Nutrition', icon: Apple },
  { id: 'sleep', label: 'Sleep', icon: Moon },
  { id: 'exercise', label: 'Exercise', icon: Dumbbell },
  { id: 'mental', label: 'Mind', icon: Brain },
  { id: 'hydration', label: 'Hydration', icon: Droplets },
];

const TIPS = [
  { id: 1, cat: 'nutrition', icon: Apple, title: 'Eat the rainbow', desc: 'Different colors signal different nutrients. Cover red, orange, green and purple most days.' },
  { id: 2, cat: 'sleep', icon: Moon, title: 'Keep a fixed wake time', desc: 'Same wake-up daily — weekends included. It anchors your circadian rhythm better than bedtime does.' },
  { id: 3, cat: 'exercise', icon: Dumbbell, title: '150 minutes a week', desc: 'WHO baseline: moderate cardio like brisk walking or cycling, plus two strength sessions.' },
  { id: 4, cat: 'mental', icon: Brain, title: 'Ten mindful minutes', desc: 'A short daily practice measurably lowers perceived stress within two weeks.' },
  { id: 5, cat: 'hydration', icon: Droplets, title: 'Two litres, roughly', desc: 'Pale-yellow urine is the target. More if you train, sweat, or drink coffee.' },
  { id: 6, cat: 'nutrition', icon: Apple, title: 'Crowd out ultra-processed', desc: 'Don’t diet — add. More whole foods naturally displaces the packaged stuff.' },
  { id: 7, cat: 'sleep', icon: Moon, title: 'Screens off, wind down', desc: 'One screen-free hour before bed. Read, stretch, dim the lights.' },
  { id: 8, cat: 'exercise', icon: Dumbbell, title: 'Lift twice weekly', desc: 'Strength work preserves muscle and bone density — the longevity reserve.' },
  { id: 9, cat: 'mental', icon: Brain, title: 'Maintain your people', desc: 'A regular call with someone you trust rivals exercise for mood.' },
  { id: 10, cat: 'hydration', icon: Droplets, title: 'Front-load water', desc: 'A large glass on waking offsets overnight loss and morning fog.' },
  { id: 11, cat: 'nutrition', icon: Apple, title: 'Slow the plate', desc: 'Smaller plates, slower bites. Satiety needs ~20 minutes to register.' },
  { id: 12, cat: 'exercise', icon: Dumbbell, title: 'Break up sitting', desc: 'Five moving minutes per seated hour. Set the timer, thank your spine.' },
];

export default function HealthTips() {
  const [cat, setCat] = useState('all');
  const list = cat === 'all' ? TIPS : TIPS.filter((t) => t.cat === cat);

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Library"
        title="Health tips"
        description="Small, evidence-based habits. Pick one and repeat it until it's boring."
      />

      <Reveal delay={0.05}>
        <div className="panel flex items-start gap-4 p-5">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-ink text-paper dark:bg-white dark:text-black">
            <ShieldCheck size={20} strokeWidth={1.9} />
          </span>
          <div>
            <p className="eyebrow">Featured</p>
            <p className="mt-1 text-base font-semibold tracking-tight text-ink">
              Prevention beats cure — schedule the check-up.
            </p>
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted">
              Early detection changes outcomes. Annual screenings matched to your age and risk
              factors are the highest-leverage health habit there is.
            </p>
          </div>
        </div>
      </Reveal>

      <Reveal delay={0.08}>
        <div className="flex gap-1.5 overflow-x-auto pb-1 no-scrollbar">
          {CATS.map((c) => (
            <button
              key={c.id}
              onClick={() => setCat(c.id)}
              data-on={cat === c.id}
              className="chip shrink-0"
            >
              <c.icon size={14} strokeWidth={2} />
              {c.label}
            </button>
          ))}
        </div>
      </Reveal>

      <Motion.div layout className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {list.map((t, i) => (
          <Motion.article
            layout
            key={t.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(i * 0.04, 0.3), duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="panel group p-5 transition-all duration-200 hover:-translate-y-0.5 hover:!shadow-lift"
          >
            <div className="flex items-start justify-between">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-line bg-paper text-ink">
                <t.icon size={18} strokeWidth={1.9} />
              </span>
              <ArrowUpRight size={15} className="text-faint opacity-0 transition-opacity group-hover:opacity-100" />
            </div>
            <h3 className="mt-3.5 text-base font-semibold tracking-tight text-ink">{t.title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-muted">{t.desc}</p>
            <p className="mt-3 text-xs font-medium uppercase tracking-[0.08em] text-faint">{t.cat}</p>
          </Motion.article>
        ))}
      </Motion.div>
    </div>
  );
}
