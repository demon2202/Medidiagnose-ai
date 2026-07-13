import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  ArrowRight,
  Heart,
  Shield,
  Activity
} from 'lucide-react';
import { useApp } from '../context/AppContext';

function Login() {
  const navigate = useNavigate();
  const { signIn, isLoading } = useApp();
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false,
  });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!formData.email || !formData.password) {
      setError('Please fill in all fields.');
      return;
    }

    const result = await signIn(formData.email, formData.password, formData.rememberMe);
    
    if (result.success) {
      navigate('/');
    } else {
      setError(result.error || 'Invalid email or password.');
    }
  };

  const features = [
    { icon: Activity, text: 'Precision AI-Powered Diagnosis' },
    { icon: Shield, text: 'Strict Local Security & Privacy' },
    { icon: Heart, text: 'Tailored Professional Recommendations' },
  ];

  return (
    <div className="min-h-screen flex bg-zinc-50 dark:bg-zinc-950 font-sans selection:bg-zinc-800 selection:text-white">
      {/* Left Column - Premium Brand Panel */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-b from-zinc-800 via-zinc-900 to-black p-16 flex-col justify-between relative overflow-hidden border-r border-zinc-800">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute -top-40 -left-40 w-96 h-96 bg-zinc-700 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -right-40 w-[30rem] h-[30rem] bg-zinc-800 rounded-full blur-3xl" />
        </div>
        
        <div className="relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-zinc-800/80 border border-zinc-700/50 rounded-xl flex items-center justify-center shadow-lg">
              <Activity className="text-zinc-100 animate-pulse" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">MediDiagnose</h1>
              <p className="text-zinc-400 text-xs">Clinical AI Assistant</p>
            </div>
          </div>
        </div>

        <div className="relative z-10 space-y-10">
          <div>
            <h2 className="text-4xl font-extrabold text-white tracking-tight leading-tight">
              Clinical intelligence,<br />right at your fingertips.
            </h2>
            <p className="text-zinc-400 mt-4 text-base max-w-md leading-relaxed">
              Get instant AI-driven health evaluations, scan medical imaging datasets, 
              and explore structured diagnostic insights locally.
            </p>
          </div>

          <div className="space-y-4">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="flex items-center gap-3 text-zinc-300 bg-zinc-900/40 border border-zinc-800/40 rounded-xl p-3 backdrop-blur-sm hover:border-zinc-700/50 transition-all duration-300"
              >
                <div className="w-8 h-8 bg-zinc-800 border border-zinc-700 rounded-lg flex items-center justify-center text-zinc-300 shadow">
                  <feature.icon size={16} />
                </div>
                <span className="text-sm font-medium">{feature.text}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="relative z-10">
          <p className="text-zinc-500 text-xs">
            © 2026 MediDiagnose-AI. Designed for educational and demonstrative use.
          </p>
        </div>
      </div>

      {/* Right Column - Login Form */}
      <div className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md space-y-8">
          {/* Mobile Navigation Header */}
          <div className="lg:hidden text-center mb-6">
            <div className="inline-flex items-center gap-3">
              <div className="w-10 h-10 bg-zinc-900 dark:bg-zinc-100 rounded-xl flex items-center justify-center shadow-md">
                <Activity className="text-white dark:text-zinc-950" size={20} />
              </div>
              <div className="text-left">
                <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">MediDiagnose</h1>
                <p className="text-zinc-500 text-xs">Clinical AI Assistant</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800/80 rounded-2xl p-8 shadow-xl backdrop-blur-md">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">
                Welcome Back
              </h2>
              <p className="text-zinc-500 text-sm mt-1">
                Access your local diagnosis dashboard
              </p>
            </div>

            {error && (
              <div className="flex items-start gap-3 p-4 mb-6 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800/60 rounded-xl">
                <AlertCircle size={18} className="text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-red-700 dark:text-red-300 font-medium leading-relaxed">{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400 dark:text-zinc-500" size={18} />
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="name@example.com"
                    className="w-full pl-11 pr-4 py-2.5 text-zinc-900 dark:text-zinc-100 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg focus:ring-2 focus:ring-zinc-500 focus:border-transparent transition-all outline-none"
                    autoComplete="email"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                    Password
                  </label>
                  <Link
                    to="/forgot-password"
                    className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors font-medium"
                  >
                    Forgot password?
                  </Link>
                </div>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400 dark:text-zinc-500" size={18} />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="••••••••"
                    className="w-full pl-11 pr-11 py-2.5 text-zinc-900 dark:text-zinc-100 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg focus:ring-2 focus:ring-zinc-500 focus:border-transparent transition-all outline-none"
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-400 dark:text-zinc-500 hover:text-zinc-600 dark:hover:text-zinc-400"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              <div className="flex items-center">
                <label className="flex items-center gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    name="rememberMe"
                    checked={formData.rememberMe}
                    onChange={handleChange}
                    className="w-4 h-4 text-zinc-900 dark:text-zinc-100 bg-zinc-50 dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 rounded focus:ring-2 focus:ring-zinc-500 accent-zinc-900 dark:accent-zinc-100"
                  />
                  <span className="text-sm text-zinc-600 dark:text-zinc-400 select-none">Remember this device</span>
                </label>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-zinc-200 text-white dark:text-zinc-950 font-semibold rounded-lg transition-all shadow-md active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={18} />
                    Authenticating...
                  </>
                ) : (
                  <>
                    Sign In
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </form>

            <p className="text-center text-sm text-zinc-500 dark:text-zinc-400 mt-8">
              New to MediDiagnose?{' '}
              <Link to="/signup" className="text-zinc-900 dark:text-zinc-100 hover:underline font-semibold">
                Create an account
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;