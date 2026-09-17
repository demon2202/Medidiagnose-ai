import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, Eye, EyeOff, Loader2, CircleAlert, ArrowRight } from 'lucide-react';
import { useApp } from '../context/AppContext';
import AuthShell from '../components/layout/AuthShell';

export default function Login() {
  const navigate = useNavigate();
  const { signIn, isLoading } = useApp();
  const [form, setForm] = useState({ email: '', password: '', rememberMe: false });
  const [show, setShow] = useState(false);
  const [error, setError] = useState('');

  const set = (k, v) => {
    setForm((p) => ({ ...p, [k]: v }));
    setError('');
  };

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.email || !form.password) {
      setError('Fill in both email and password.');
      return;
    }
    const res = await signIn(form.email, form.password, form.rememberMe);
    if (res.success) navigate('/');
    else setError(res.error || 'Invalid email or password.');
  };

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to your local workspace."
      footer={
        <>
          New here?{' '}
          <Link to="/signup" className="font-medium text-ink underline-offset-4 hover:underline">
            Create an account
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
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label className="text-[13px] font-medium text-muted">Password</label>
            <Link to="/forgot-password" className="text-xs text-faint transition-colors hover:text-ink">
              Forgot password?
            </Link>
          </div>
          <div className="relative">
            <Lock size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
            <input
              type={show ? 'text' : 'password'}
              value={form.password}
              onChange={(e) => set('password', e.target.value)}
              placeholder="Your password"
              autoComplete="current-password"
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
        <label className="flex cursor-pointer items-center gap-2.5 pt-0.5">
          <input
            type="checkbox"
            checked={form.rememberMe}
            onChange={(e) => set('rememberMe', e.target.checked)}
            className="h-4 w-4 rounded border-line accent-[#0f766e]"
          />
          <span className="text-[13px] text-muted">Remember this device</span>
        </label>
        <button type="submit" disabled={isLoading} className="btn-primary w-full !py-3">
          {isLoading ? (
            <><Loader2 size={17} className="animate-spin" /> Signing in…</>
          ) : (
            <>Sign in <ArrowRight size={16} /></>
          )}
        </button>
      </form>
    </AuthShell>
  );
}
