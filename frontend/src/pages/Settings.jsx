import React, { useState } from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  UserRound,
  Lock,
  SlidersHorizontal,
  Database,
  Trash2,
  Check,
  Eye,
  EyeOff,
  Moon,
  Sun,
  Bell,
  History as HistoryIcon,
  ShieldCheck
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { PageHeader, Modal, ModalHeader, Reveal } from '../components/ui/ui';
import { spring } from '../lib/motion';

const SECTIONS = [
  { id: 'profile', label: 'Profile', icon: UserRound },
  { id: 'security', label: 'Password', icon: Lock },
  { id: 'preferences', label: 'Preferences', icon: SlidersHorizontal },
  { id: 'data', label: 'Data', icon: Database },
];

function Switch({ on, onToggle, label, desc, icon: Icon }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="flex min-w-0 items-center gap-3">
        {Icon && <Icon size={17} className="shrink-0 text-muted" />}
        <div className="min-w-0">
          <p className="text-[15px] font-medium text-ink">{label}</p>
          <p className="truncate text-sm text-muted">{desc}</p>
        </div>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={onToggle}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors duration-200 ${
          on ? 'bg-ink dark:bg-white' : 'bg-ink/15 dark:bg-white/15'
        }`}
      >
        <Motion.span
          animate={{ x: on ? 22 : 2 }}
          transition={spring}
          className={`absolute top-[3px] h-[18px] w-[18px] rounded-full ${
            on ? 'bg-paper dark:bg-black' : 'bg-white dark:bg-white/60'
          }`}
        />
      </button>
    </div>
  );
}

export default function Settings() {
  const {
    user,
    updateProfile,
    settings,
    updateSettings,
    theme,
    toggleTheme,
    clearHistory,
    history,
    changePassword
  } = useApp();

  const [section, setSection] = useState('profile');
  const [form, setForm] = useState({ name: user?.name || '', email: user?.email || '' });
  const [saved, setSaved] = useState(false);
  const [pw, setPw] = useState({ current: '', next: '', confirm: '' });
  const [show, setShow] = useState({ current: false, next: false, confirm: false });
  const [pwError, setPwError] = useState('');
  const [pwDone, setPwDone] = useState(false);
  const [pwBusy, setPwBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const saveProfile = async () => {
    await updateProfile(form);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const submitPassword = async (e) => {
    e.preventDefault();
    setPwError('');
    setPwDone(false);
    if (!pw.current || !pw.next || !pw.confirm) {
      setPwError('Fill in all three fields.');
      return;
    }
    if (pw.next !== pw.confirm) {
      setPwError('New passwords do not match.');
      return;
    }
    setPwBusy(true);
    const res = await changePassword(pw.current, pw.next);
    setPwBusy(false);
    if (res.success) {
      setPwDone(true);
      setPw({ current: '', next: '', confirm: '' });
      setTimeout(() => setPwDone(false), 3000);
    } else {
      setPwError(res.error || 'Could not change password.');
    }
  };

  const pwField = (key, label) => (
    <div>
      <label className="label">{label}</label>
      <div className="relative">
        <input
          type={show[key] ? 'text' : 'password'}
          value={pw[key]}
          onChange={(e) => setPw({ ...pw, [key]: e.target.value })}
          placeholder="••••••••"
          autoComplete="new-password"
          className="field !pr-11"
        />
        <button
          type="button"
          onClick={() => setShow({ ...show, [key]: !show[key] })}
          className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-faint hover:text-ink"
          aria-label="Toggle visibility"
        >
          {show[key] ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
    </div>
  );

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Workspace"
        title="Settings"
        description="Profile, security and how the app behaves."
      />

      <div className="grid items-stretch gap-5 lg:grid-cols-12">
        <Reveal className="lg:col-span-3">
          <nav className="panel flex gap-1 overflow-x-auto p-1.5 lg:h-full lg:flex-col no-scrollbar">
            {SECTIONS.map((s) => {
              const active = section === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => setSection(s.id)}
                  className={`relative flex shrink-0 items-center gap-2.5 rounded-lg px-3.5 py-2.5 text-sm transition-colors lg:w-full ${
                    active ? 'font-medium text-ink' : 'text-muted hover:text-ink'
                  }`}
                >
                  {active && (
                    <Motion.span
                      layoutId="settings-active"
                      transition={spring}
                      className="absolute inset-0 rounded-lg bg-ink/[0.06] dark:bg-white/[0.08]"
                    />
                  )}
                  <s.icon size={16} className="relative z-10" />
                  <span className="relative z-10">{s.label}</span>
                </button>
              );
            })}
          </nav>
        </Reveal>

        <Reveal delay={0.06} className="h-full lg:col-span-9">
          <div className="panel min-h-[calc(100vh-240px)] p-6 sm:p-8">
            <AnimatePresence mode="wait" initial={false}>
              <Motion.div
                key={section}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
              >
                {section === 'profile' && (
                  <div className="max-w-lg">
                    <h3 className="text-base font-semibold text-ink">Profile</h3>
                    <p className="mb-5 mt-0.5 text-sm text-muted">Stored locally on this device.</p>
                    <div className="mb-6 flex items-center gap-4">
                      <span className="t-num flex h-16 w-16 items-center justify-center rounded-2xl bg-ink text-xl text-paper font-semibold dark:bg-white dark:text-black">
                        {(form.name || 'U')
                          .split(' ')
                          .map((n) => n[0])
                          .join('')
                          .toUpperCase()
                          .slice(0, 2)}
                      </span>
                      <div>
                        <p className="text-base font-medium text-ink">{form.name || 'Your name'}</p>
                        <p className="text-[13px] text-faint">{form.email || 'No email set'}</p>
                        {user?.createdAt && (
                          <p className="mt-0.5 text-[13px] text-faint">
                            Member since{' '}
                            {new Date(user.createdAt).toLocaleDateString('en-US', {
                              month: 'short',
                              year: 'numeric',
                            })}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <label className="label">Full name</label>
                        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="field" />
                      </div>
                      <div>
                        <label className="label">Email</label>
                        <input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} className="field" />
                      </div>
                    </div>
                    <button onClick={saveProfile} className="btn-primary mt-4">
                      {saved ? <><Check size={16} /> Saved</> : 'Save changes'}
                    </button>
                  </div>
                )}

                {section === 'security' && (
                  <form onSubmit={submitPassword} className="max-w-md">
                    <h3 className="text-base font-semibold text-ink">Change password</h3>
                    <p className="mb-5 mt-0.5 text-sm text-muted">Hashed with bcrypt before it ever touches storage.</p>
                    {pwError && (
                      <p className="mb-3 rounded-xl bg-critical/[0.07] px-3.5 py-2.5 text-sm font-medium text-critical">
                        {pwError}
                      </p>
                    )}
                    {pwDone && (
                      <p className="mb-3 rounded-xl bg-low/[0.09] px-3.5 py-2.5 text-sm font-medium text-low">
                        Password changed.
                      </p>
                    )}
                    <div className="space-y-3">
                      {pwField('current', 'Current password')}
                      {pwField('next', 'New password')}
                      {pwField('confirm', 'Confirm new password')}
                    </div>
                    <button type="submit" disabled={pwBusy} className="btn-primary mt-4">
                      {pwBusy ? 'Updating…' : 'Update password'}
                    </button>
                  </form>
                )}

                {section === 'preferences' && (
                  <div className="max-w-xl">
                    <h3 className="text-base font-semibold text-ink">Preferences</h3>
                    <p className="mb-5 mt-0.5 text-sm text-muted">How the app looks and behaves.</p>
                    <div className="space-y-2.5">
                      <Switch
                        on={theme === 'dark'}
                        onToggle={toggleTheme}
                        label="Dark theme"
                        desc={theme === 'dark' ? 'Charcoal surfaces, low glare' : 'Warm paper surfaces'}
                        icon={theme === 'dark' ? Moon : Sun}
                      />
                      <Switch
                        on={!!settings.notifications}
                        onToggle={() => updateSettings({ notifications: !settings.notifications })}
                        label="Notifications"
                        desc="Toast confirmations for actions and results"
                        icon={Bell}
                      />
                      <Switch
                        on={settings.autoSaveHistory !== false}
                        onToggle={() => updateSettings({ autoSaveHistory: settings.autoSaveHistory === false })}
                        label="Auto-save history"
                        desc="Log every check to this device automatically"
                        icon={HistoryIcon}
                      />
                    </div>
                  </div>
                )}

                {section === 'data' && (
                  <div className="max-w-xl">
                    <h3 className="text-base font-semibold text-ink">Privacy & data</h3>
                    <p className="mb-5 mt-0.5 text-sm text-muted">Everything stays in your browser.</p>
                    <div className="flex items-start gap-3 rounded-xl border border-line bg-paper px-4 py-3.5">
                      <ShieldCheck size={17} className="mt-0.5 shrink-0 text-low" />
                      <div>
                        <p className="text-[15px] font-medium text-ink">Local-only storage</p>
                        <p className="mt-0.5 text-sm leading-relaxed text-muted">
                          Diagnosis history and credentials live in this browser's local storage.
                          Clearing site data erases them. Export a backup from History first.
                        </p>
                      </div>
                    </div>
                    <div className="mt-2.5 flex items-center justify-between gap-4 rounded-xl border border-line px-4 py-3.5">
                      <div>
                        <p className="text-[15px] font-medium text-ink">Diagnosis history</p>
                        <p className="t-num text-sm text-muted">{history.length} records stored</p>
                      </div>
                      <button
                        onClick={() => setConfirmClear(true)}
                        disabled={history.length === 0}
                        className="btn-danger-quiet btn-sm"
                      >
                        <Trash2 size={14} /> Clear all
                      </button>
                    </div>
                  </div>
                )}
              </Motion.div>
            </AnimatePresence>
          </div>
        </Reveal>
      </div>

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
