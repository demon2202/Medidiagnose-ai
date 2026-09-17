import React, { useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation
} from 'react-router-dom';
import { AnimatePresence, motion as Motion } from 'framer-motion';
import Lenis from 'lenis';
import { AppProvider, useApp } from './context/AppContext';
import { setLenis, scrollTop } from './lib/scroll';
import CommandPalette from './components/common/CommandPalette';
import ResultInspector from './components/common/ResultInspector';
import { easeOut } from './lib/motion';

import Sidebar from './components/layout/Sidebar';
import MobileNav from './components/layout/MobileNav';
import Notification from './components/common/Notification';

import Dashboard from './pages/Dashboard';
import SymptomDiagnosis from './pages/SymptomDiagnosis';
import ImageAnalysis from './pages/ImageAnalysis';
import HeartCheck from './pages/HeartCheck';
import CancerScreening from './pages/CancerScreening';
import History from './pages/History';
import HealthTips from './pages/HealthTips';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';

import './index.css';

function ProtectedRoute({ children }) {
  const { isAuthenticated } = useApp();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
}

function PublicRoute({ children }) {
  const { isAuthenticated } = useApp();
  if (isAuthenticated) return <Navigate to="/" replace />;
  return children;
}

function Page({ children }) {
  return (
    <Motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.28, ease: easeOut }}
    >
      {children}
    </Motion.div>
  );
}

function MainLayout({ children }) {
  return (
    <div className="flex min-h-screen items-start">
      <Sidebar />
      <div className="min-w-0 flex-1">
        <main className="mx-auto w-full max-w-[1720px] px-4 pb-10 pt-6 md:px-8 md:pb-12 md:pt-8">
          {children}
        </main>
        <MobileNav />
      </div>
    </div>
  );
}

function AnimatedRoutes() {
  const location = useLocation();

  useEffect(() => {
    scrollTop(true);
  }, [location.pathname]);

  /* NOTE: no AnimatePresence around routes on purpose. The layout tree
     contains nested AnimatePresence instances (sidebar/header), and an outer
     presence — in any mode — never completes its exit here: mode="wait"
     blocks the new page forever, popLayout/sync leak ghost trees. Pages
     animate in via <Page>; that transition is instant and cannot get stuck. */
  return (
    <Routes location={location} key={location.pathname}>
        <Route path="/login" element={<PublicRoute><Page><Login /></Page></PublicRoute>} />
        <Route path="/signup" element={<PublicRoute><Page><Signup /></Page></PublicRoute>} />
        <Route path="/forgot-password" element={<PublicRoute><Page><ForgotPassword /></Page></PublicRoute>} />

        <Route path="/" element={<ProtectedRoute><MainLayout><Page><Dashboard /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/symptoms" element={<ProtectedRoute><MainLayout><Page><SymptomDiagnosis /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/image-analysis" element={<ProtectedRoute><MainLayout><Page><ImageAnalysis /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/heart-check" element={<ProtectedRoute><MainLayout><Page><HeartCheck /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/cancer-screening" element={<ProtectedRoute><MainLayout><Page><CancerScreening /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/history" element={<ProtectedRoute><MainLayout><Page><History /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/health-tips" element={<ProtectedRoute><MainLayout><Page><HealthTips /></Page></MainLayout></ProtectedRoute>} />
        <Route path="/settings" element={<ProtectedRoute><MainLayout><Page><Settings /></Page></MainLayout></ProtectedRoute>} />

        <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function AppContent() {
  const { notification, hideNotification, paletteOpen, setPaletteOpen } = useApp();

  /* command-palette shortcut (app-wide) */
  useEffect(() => {
    const fn = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [setPaletteOpen]);

  /* Lenis smooth scrolling (desktop) */
  useEffect(() => {
    const fine = window.matchMedia('(pointer: fine)').matches;
    if (!fine) return;
    const lenis = new Lenis({ duration: 1.05, smoothWheel: true });
    setLenis(lenis);
    let raf;
    const loop = (time) => {
      lenis.raf(time);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
      setLenis(null);
    };
  }, []);

  return (
    <Router>
      <div className="pointer-events-none fixed bottom-24 right-4 z-[100] flex flex-col items-end gap-2 md:bottom-6 md:right-6">
        <AnimatePresence>
          {notification && (
            <Notification
              key={notification.id}
              message={notification.message}
              type={notification.type}
              onClose={hideNotification}
            />
          )}
        </AnimatePresence>
      </div>
      <AnimatedRoutes />
      <AnimatePresence>
        {paletteOpen && <CommandPalette key="palette" />}
      </AnimatePresence>
      <AnimatePresence>
        <ResultInspector key="inspector" />
      </AnimatePresence>
    </Router>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
