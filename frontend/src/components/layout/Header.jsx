import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  Menu,
  Moon,
  Sun,
  UserRound,
  Settings,
  LogOut,
  ChevronDown,
  Search
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { Modal, ModalHeader } from '../ui/ui';
import { easeOut } from '../../lib/motion';
import ProfileModal from '../modals/ProfileModal';

const TITLES = {
  '/': 'Overview',
  '/symptoms': 'Symptom diagnosis',
  '/image-analysis': 'Image analysis',
  '/heart-check': 'Heart health',
  '/cancer-screening': 'Cancer screening',
  '/history': 'History',
  '/health-tips': 'Health tips',
  '/settings': 'Settings'
};

const initials = (name = '') =>
  name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || 'U';

export default function Header() {
  const {
    user,
    setSidebarOpen,
    profileModalOpen,
    setProfileModalOpen,
    signOut,
    theme,
    toggleTheme,
    setPaletteOpen
  } = useApp();
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmOut, setConfirmOut] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const close = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, []);

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric'
  });

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-line bg-paper/85 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-[1720px] items-center justify-between gap-3 px-4 md:px-8">
          <div className="flex min-w-0 items-center gap-2.5">
            <button
              onClick={() => setSidebarOpen(true)}
              className="icon-btn md:hidden"
              aria-label="Open menu"
            >
              <Menu size={19} />
            </button>
            <div className="min-w-0">
              <h2 className="truncate text-[15px] font-semibold tracking-tight text-ink">
                {TITLES[location.pathname] || 'MediDiagnose'}
              </h2>
              <p className="hidden text-xs text-faint sm:block">{today}</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPaletteOpen(true)}
              className="mr-0.5 hidden items-center gap-2 rounded-xl border border-line bg-surface px-2.5 py-[7px] text-[12.5px] text-faint transition-colors hover:border-faint hover:text-muted sm:flex"
              aria-label="Search (Ctrl+K)"
            >
              <Search size={15} />
              <span className="hidden lg:inline">Search or jump to&hellip;</span>
              <span className="kbd">Ctrl K</span>
            </button>
            <button onClick={() => setPaletteOpen(true)} className="icon-btn sm:hidden" aria-label="Search">
              <Search size={18} />
            </button>

            <button onClick={toggleTheme} className="icon-btn" aria-label="Toggle theme">
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>

            {/* profile */}
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setMenuOpen((v) => !v)}
                className="flex items-center gap-2 rounded-xl p-1.5 transition-colors hover:bg-raised"
              >
                <span className="t-num flex h-8 w-8 items-center justify-center rounded-lg bg-ink text-[12px] font-semibold text-paper dark:bg-white dark:text-black">
                  {initials(user?.name)}
                </span>
                <ChevronDown
                  size={14}
                  className={`hidden text-faint transition-transform sm:block ${menuOpen ? 'rotate-180' : ''}`}
                />
              </button>

              <AnimatePresence>
                {menuOpen && (
                  <Motion.div
                    initial={{ opacity: 0, y: -4, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -4, scale: 0.98 }}
                    transition={{ duration: 0.16, ease: easeOut }}
                    className="panel absolute right-0 mt-2 w-60 overflow-hidden !rounded-xl p-1.5"
                  >
                    <div className="px-3 pb-2.5 pt-2">
                      <p className="truncate text-sm font-medium text-ink">{user?.name}</p>
                      <p className="truncate text-xs text-muted">{user?.email}</p>
                    </div>
                    <div className="hairline-t pt-1.5">
                      <button
                        onClick={() => {
                          setMenuOpen(false);
                          setProfileModalOpen(true);
                        }}
                        className="btn-quiet w-full !justify-start"
                      >
                        <UserRound size={15} /> Edit profile
                      </button>
                      <Link
                        to="/settings"
                        onClick={() => setMenuOpen(false)}
                        className="btn-quiet w-full !justify-start"
                      >
                        <Settings size={15} /> Settings
                      </Link>
                    </div>
                    <div className="hairline-t mt-1.5 pt-1.5">
                      <button
                        onClick={() => {
                          setMenuOpen(false);
                          setConfirmOut(true);
                        }}
                        className="btn-danger-quiet w-full !justify-start"
                      >
                        <LogOut size={15} /> Sign out
                      </button>
                    </div>
                  </Motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </header>

      <AnimatePresence>{profileModalOpen && <ProfileModal />}</AnimatePresence>

      <AnimatePresence>
        {confirmOut && (
          <Modal onClose={() => setConfirmOut(false)}>
            <ModalHeader title="Sign out?" subtitle="Your history stays saved on this device." onClose={() => setConfirmOut(false)} />
            <div className="flex gap-2.5 p-5">
              <button onClick={() => setConfirmOut(false)} className="btn-ghost flex-1">
                Stay
              </button>
              <button
                onClick={() => {
                  setConfirmOut(false);
                  signOut();
                  navigate('/login');
                }}
                className="btn flex-1 bg-critical px-4 py-2.5 text-white hover:brightness-110"
              >
                <LogOut size={16} /> Sign out
              </button>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </>
  );
}
