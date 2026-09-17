import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { UserRound, Mail, Lock, Eye, EyeOff, Loader2, CircleAlert, ArrowRight, Check } from 'lucide-react';
import { useApp } from '../context/AppContext';
import AuthShell from '../components/layout/AuthShell';

const RULES = [
  { label: '8+ characters', test: (p) => p.length >= 8 },
  { label: 'A number', test: (p) => /\d/.test(p) },
  { label: 'An uppercase letter', test: (p) => /[A-Z]/.test(p) },
];

export default function Signup() {
  const navigate = useNavigate();
  const { signUp, isLoading } = useApp();
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [show, setShow] = useState(false);
  const [error, setError] = useState('');

  const set = (k, v) => {
    setForm((p) => ({ ...p, [k]: v }));
    setError('');
  };

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.name.trim() || !form.email || !form.password) {
      setError('Fill in every field.');
      return;
    }
    if (form.password !== form.confirm) {
      setError('Passwords do not match.');
      return;
    }
    const res = await signUp(form.name, form.email, form.password);
    if (res.success) navigate('/');
    else setError(res.error || 'Could not create the account.');
  };

  return (
    <AuthShell
      title="Create your account"
      subtitle="Local-first. Your data never leaves this device."
      footer={
        <>
          Have an account?{' '}
          <Link to="/login" className="font-medium text-ink underline-offset-4 hover:underline">
            Sign in
          </Link>
        </>
      }
    >
      {error && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl bg-critical/[0.07] px-3.5 py-3">
          <CircleAlert size={16} className="mt-px shrink-0 text-critical" />
          <p className="text-[13px] font-medium leading-relaxed text-critical">{error}</p>
        </div>
      )}
      <form onSubmit={submit} className="space-y-3.5">
        <div>
          <label className="label">Full name</label>
          <div className="relative">
            <UserRound size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
            <input
              value={form.name}
              onChange={(e) => set('name', e.target.value)}
              placeholder="Jane Cooper"
              autoComplete="name"
              className="field !pl-10"
            />
          </div>
        </div>
        <div>
          <label className="label">Email</label>
          <div className="relative">
            <Mail size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
            <input
              type="email"
              value={form.email}
              onChange={(e) => set('email', e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              className="field !pl-10"
            />
          </div>
        </div>
        <div className="grid gap-3.5 sm:grid-cols-2">
          <div>
            <label className="label">Password</label>
            <div className="relative">
              <Lock size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                type={show ? 'text' : 'password'}
                value={form.password}
                onChange={(e) => set('password', e.target.value)}
                placeholder="Create one"
                autoComplete="new-password"
                className="field !pl-10 !pr-10"
              />
              <button
                type="button"
                onClick={() => setShow((v) => !v)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-faint hover:text-ink"
                aria-label={show ? 'Hide password' : 'Show password'}
              >
                {show ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          </div>
          <div>
            <label className="label">Confirm</label>
            <input
              type={show ? 'text' : 'password'}
              value={form.confirm}
              onChange={(e) => set('confirm', e.target.value)}
              placeholder="Repeat it"
              autoComplete="new-password"
              className="field"
            />
          </div>
        </div>
        {form.password && (
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {RULES.map((r) => {
              const ok = r.test(form.password);
              return (
                <span key={r.label} className={`flex items-center gap-1 text-xs ${ok ? 'text-low' : 'text-faint'}`}>
                  <Check size={12} strokeWidth={ok ? 3 : 2} /> {r.label}
                </span>
              );
            })}
          </div>
        )}
        <button type="submit" disabled={isLoading} className="btn-primary w-full !py-3">
          {isLoading ? (
            <><Loader2 size={17} className="animate-spin" /> Creating…</>
          ) : (
            <>Create account <ArrowRight size={16} /></>
          )}
        </button>
      </form>
    </AuthShell>
  );
}
