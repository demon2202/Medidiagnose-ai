import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  Trash2,
  Download,
  History as HistoryIcon,
  ArrowUpRight,
  Stethoscope,
  ScanLine,
  HeartPulse,
  Microscope
} from 'lucide-react';
import {
  diagnosisTypeMeta as typeMeta,
  entryTitle,
  entryConfidence,
  timeAgo,
  dayGroup,
  downloadHistory
} from '../lib/diagnosis';
import { useApp } from '../context/AppContext';
import {
  PageHeader,
  Seg,
  EmptyState,
  Modal,
  ModalHeader,
  Reveal
} from '../components/ui/ui';
import { severityMeta } from '../lib/severity';

const matches = (h, filter) => {
  if (filter === 'all') return true;
  if (filter === 'image') return h.type?.startsWith('image');
  if (filter === 'symptom') return h.type === 'symptom';
  if (filter === 'screening') return h.type === 'heart' || h.type === 'cancer';
  return true;
};

export default function History() {
  const { history, clearHistory, removeFromHistory, openInspector } = useApp();
  const [filter, setFilter] = useState('all');
  const [confirmClear, setConfirmClear] = useState(false);

  const counts = useMemo(() => ({
    all: history.length,
    image: history.filter((h) => matches(h, 'image')).length,
    symptom: history.filter((h) => matches(h, 'symptom')).length,
    screening: history.filter((h) => matches(h, 'screening')).length
  }), [history]);

  /* newest-first, bucketed by calendar day */
  const groups = useMemo(() => {
    const list = history.filter((h) => matches(h, filter));
    const buckets = new Map();
    for (const h of list) {
      const g = dayGroup(h.timestamp);
      if (!buckets.has(g)) buckets.set(g, []);
      buckets.get(g).push(h);
    }
    return [...buckets.entries()];
  }, [history, filter]);

  const shown = groups.reduce((n, [, items]) => n + items.length, 0);

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Library"
        title="History"
        description="Every check you've run, stored locally on this device. Select any row to revisit the full result."
        action={
          history.length > 0 && (
            <>
              <button onClick={() => downloadHistory(history)} className="btn-ghost btn-sm">
                <Download size={14} /> Export
              </button>
              <button onClick={() => setConfirmClear(true)} className="btn-danger-quiet btn-sm">
                <Trash2 size={14} /> Clear
              </button>
            </>
          )
        }
      />

      <Reveal delay={0.05}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Seg
            id="history-filter"
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'all', label: 'All', count: counts.all },
              { value: 'image', label: 'Images', count: counts.image },
              { value: 'symptom', label: 'Symptoms', count: counts.symptom },
              { value: 'screening', label: 'Screenings', count: counts.screening },
            ]}
          />
          <p className="t-num text-sm text-faint">
            {shown} of {history.length} shown
          </p>
        </div>
      </Reveal>

      <Reveal delay={0.1}>
        {shown === 0 ? (
          <div className="panel flex min-h-[calc(100vh-320px)] flex-col overflow-hidden">
            <EmptyState
              icon={HistoryIcon}
              title={filter === 'all' ? 'No history yet' : `No ${filter} checks yet`}
              hint="Run an assessment and it will be logged here automatically."
              className="flex-1 justify-center"
            />
            {filter === 'all' && (
              <div className="grid grid-cols-2 gap-px border-t border-line bg-line sm:grid-cols-4">
                {[
                  { to: '/symptoms', icon: Stethoscope, t: 'Symptom check' },
                  { to: '/image-analysis', icon: ScanLine, t: 'Image analysis' },
                  { to: '/heart-check', icon: HeartPulse, t: 'Heart screening' },
                  { to: '/cancer-screening', icon: Microscope, t: 'Cancer screening' },
                ].map((s) => (
                  <Link
                    key={s.to}
                    to={s.to}
                    className="group flex flex-col items-center gap-2 bg-surface px-4 py-8 text-center transition-colors hover:bg-raised/60"
                  >
                    <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-paper text-muted transition-colors group-hover:text-ink">
                      <s.icon size={20} strokeWidth={1.9} />
                    </span>
                    <span className="text-sm font-medium text-ink">{s.t}</span>
                    <span className="flex items-center gap-1 text-[13px] text-faint transition-colors group-hover:text-ink">
                      Start <ArrowUpRight size={12} />
                    </span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {groups.map(([label, items]) => (
              <section key={label} aria-label={label}>
                <div className="mb-2 flex items-center gap-3 px-1">
                  <h3 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-faint">{label}</h3>
                  <span className="h-px flex-1 bg-line" />
                  <span className="t-num text-[11px] text-faint">{items.length}</span>
                </div>
                <div className="panel divide-y divide-line overflow-hidden">
                  <AnimatePresence initial={false}>
                    {items.map((item) => {
                      const m = typeMeta(item.type);
                      const sev = item.severity ? severityMeta(item.severity) : null;
                      const conf = entryConfidence(item);
                      const reopenable = Boolean(item.data);
                      return (
                        <Motion.div
                          key={item.id}
                          layout
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0, height: 0, paddingTop: 0, paddingBottom: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div
                            className={`group flex w-full items-center gap-1 px-4 py-3.5 transition-colors sm:px-5 ${
                              reopenable ? 'hover:bg-raised/60' : 'opacity-70'
                            }`}
                          >
                            <button
                              onClick={() => reopenable && openInspector(item)}
                              disabled={!reopenable}
                              title={reopenable ? 'Open full result' : 'Logged before full results were saved — re-run to revisit'}
                              className={`flex min-w-0 flex-1 items-center gap-3.5 text-left ${reopenable ? '' : 'cursor-default'}`}
                            >
                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-line bg-surface text-muted">
                              <m.icon size={17} strokeWidth={1.9} />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="flex items-center gap-2">
                                <span className="truncate text-[15px] font-medium text-ink">{entryTitle(item)}</span>
                                {sev && (
                                  <span
                                    className="hidden shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold sm:inline-flex"
                                    style={{ background: sev.soft, color: sev.color }}
                                  >
                                    {sev.label}
                                  </span>
                                )}
                              </span>
                              <span className="mt-0.5 block truncate text-[13px] text-faint">
                                <span className="capitalize">{m.label}</span>
                                <span aria-hidden> · </span>
                                {timeAgo(item.timestamp)}
                                {item.fileName && (
                                  <>
                                    <span aria-hidden> · </span>
                                    <span>{item.fileName}</span>
                                  </>
                                )}
                              </span>
                            </span>
                            {conf != null && (
                              <span className="t-num hidden shrink-0 text-sm font-semibold text-muted sm:block">
                                {conf}%
                              </span>
                            )}
                            {reopenable && (
                              <ArrowUpRight
                                size={15}
                                className="shrink-0 text-faint opacity-0 transition-all group-hover:translate-x-[1px] group-hover:opacity-100 max-lg:opacity-60"
                              />
                            )}
                            </button>
                            <button
                              aria-label="Delete entry"
                              onClick={() => removeFromHistory(item.id)}
                              className="shrink-0 rounded-lg p-2 text-faint opacity-0 transition-all hover:bg-critical/10 hover:text-critical group-hover:opacity-100 max-lg:opacity-100"
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </Motion.div>
                      );
                    })}
                  </AnimatePresence>
                </div>
              </section>
            ))}
          </div>
        )}
      </Reveal>

      <AnimatePresence>
        {confirmClear && (
          <Modal onClose={() => setConfirmClear(false)}>
            <ModalHeader
              title="Clear all history?"
              subtitle={`${history.length} records will be permanently removed.`}
              onClose={() => setConfirmClear(false)}
            />
            <div className="flex gap-2.5 p-5">
              <button onClick={() => setConfirmClear(false)} className="btn-ghost flex-1">
                Keep
              </button>
              <button
                onClick={() => {
                  clearHistory();
                  setConfirmClear(false);
                }}
                className="btn flex-1 bg-critical px-4 py-2.5 text-white hover:brightness-110"
              >
                Delete all
              </button>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  );
}
