import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, Eye, EyeOff, Loader2, CircleAlert, ArrowRight, ArrowLeft, CheckCircle2 } from 'lucide-react';
import { useApp } from '../context/AppContext';
import AuthShell from '../components/layout/AuthShell';

export default function ForgotPassword() {
  const navigate = useNavigate();
  const { resetPassword, isLoading } = useApp();
  const [email, setEmail] = useState('');
  const [step, setStep] = useState(1);
  const [pw, setPw] = useState({ next: '', confirm: '' });
  const [show, setShow] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);

  const verify = async (e) => {
    e.preventDefault();
    setError('');
    if (!email) {
      setError('Enter your account email.');
      return;
    }
    const res = await resetPassword(email);
    if (res.success && res.verified) setStep(2);
    else if (!res.success) setError(res.error || 'Verification failed.');
  };

  const reset = async (e) => {
    e.preventDefault();
    setError('');
    if (!pw.next || !pw.confirm) {
      setError('Fill in both password fields.');
      return;
    }
    if (pw.next !== pw.confirm) {
      setError('Passwords do not match.');
      return;
    }
    const res = await resetPassword(email, pw.next);
    if (res.success) {
      setDone(true);
      setTimeout(() => navigate('/login'), 1600);
    } else {
      setError(res.error || 'Reset failed.');
    }
  };

  return (
    <AuthShell
      title={step === 1 ? 'Reset password' : 'Choose a new one'}
      subtitle={step === 1 ? 'We’ll verify your local account first.' : `Resetting password for ${email || 'your account'}.`}
      footer={
        <Link to="/login" className="inline-flex items-center gap-1.5 font-medium text-ink underline-offset-4 hover:underline">
          <ArrowLeft size={14} /> Back to sign in
        </Link>
      }
    >
      {error && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl bg-critical/[0.07] px-3.5 py-3">
          <CircleAlert size={16} className="mt-px shrink-0 text-critical" />
          <p className="text-sm font-medium leading-relaxed text-critical">{error}</p>
        </div>
      )}

      {done ? (
        <div className="flex flex-col items-center rounded-2xl border border-line bg-surface px-6 py-10 text-center">
          <CheckCircle2 size={30} className="text-low" />
          <p className="mt-3 text-base font-medium text-ink">Password updated</p>
          <p className="mt-1 text-sm text-muted">Taking you to sign in…</p>
        </div>
      ) : step === 1 ? (
        <form onSubmit={verify} className="space-y-3.5">
          <div>
            <label className="label">Account email</label>
            <div className="relative">
              <Mail size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError('');
                }}
                placeholder="you@example.com"
                autoComplete="email"
                className="field !pl-10"
              />
            </div>
          </div>
          <button type="submit" disabled={isLoading} className="btn-accent w-full !py-3.5 text-[15px]">
            {isLoading ? (
              <><Loader2 size={17} className="animate-spin" /> Verifying…</>
            ) : (
              <>Verify identity <ArrowRight size={16} /></>
            )}
          </button>
        </form>
      ) : (
        <form onSubmit={reset} className="space-y-3.5">
          <div>
            <label className="label">New password</label>
            <div className="relative">
              <Lock size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                type={show ? 'text' : 'password'}
                value={pw.next}
                onChange={(e) => setPw({ ...pw, next: e.target.value })}
                placeholder="8+ chars, a number, an uppercase"
                autoComplete="new-password"
                className="field !pl-10 !pr-11"
              />
              <button
                type="button"
                onClick={() => setShow((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-faint hover:text-ink"
                aria-label={show ? 'Hide password' : 'Show password'}
              >
                {show ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <div>
            <label className="label">Confirm new password</label>
            <input
              type={show ? 'text' : 'password'}
              value={pw.confirm}
              onChange={(e) => setPw({ ...pw, confirm: e.target.value })}
              placeholder="Repeat it"
              autoComplete="new-password"
              className="field"
            />
          </div>
          <button type="submit" disabled={isLoading} className="btn-accent w-full !py-3.5 text-[15px]">
            {isLoading ? (
              <><Loader2 size={17} className="animate-spin" /> Updating…</>
            ) : (
              <>Set new password <ArrowRight size={16} /></>
            )}
          </button>
        </form>
      )}
    </AuthShell>
  );
}
