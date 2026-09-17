import React, { useState, useRef, useEffect } from 'react';
import { NavLink, useLocation, Link, useNavigate } from 'react-router-dom';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Stethoscope,
  ScanLine,
  HeartPulse,
  Microscope,
  History,
  Sprout,
  Settings,
  X,
  Menu,
  Moon,
  Sun,
  UserRound,
  LogOut,
  ChevronDown
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { Modal, ModalHeader } from '../ui/ui';
import { spring, easeOut } from '../../lib/motion';
import ProfileModal from '../modals/ProfileModal';

const GROUPS = [
  {
    label: 'Assess',
    items: [
      { path: '/', icon: LayoutDashboard, label: 'Overview', end: true },
      { path: '/symptoms', icon: Stethoscope, label: 'Symptoms' },
      { path: '/image-analysis', icon: ScanLine, label: 'Image analysis' },
    ]
  },
  {
    label: 'Screening',
    items: [
      { path: '/heart-check', icon: HeartPulse, label: 'Heart' },
      { path: '/cancer-screening', icon: Microscope, label: 'Cancer' },
    ]
  },
  {
    label: 'Library',
    items: [
      { path: '/history', icon: History, label: 'History' },
      { path: '/health-tips', icon: Sprout, label: 'Health tips' },
      { path: '/settings', icon: Settings, label: 'Settings' },
    ]
  },
];

const initials = (name = '') =>
  name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || 'U';

function NavItem({ item, expanded, onNavigate, delay = 0 }) {
  return (
    <NavLink
      to={item.path}
      end={item.end}
      onClick={onNavigate}
      title={item.label}
      className={({ isActive }) =>
        `relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors duration-150 ${
          isActive ? 'font-medium text-ink' : 'text-muted hover:text-ink'
        } ${expanded ? '' : 'justify-center !px-0'}`
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <Motion.span
              layoutId="sidebar-active"
              transition={spring}
              className="absolute inset-0 rounded-xl bg-ink/[0.06] dark:bg-white/[0.08]"
            />
          )}
          <span className="relative z-10 flex shrink-0 items-center justify-center">
            <item.icon size={19} strokeWidth={isActive ? 2.1 : 1.8} />
          </span>
          <AnimatePresence initial={false}>
            {expanded && (
              <Motion.span
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -6 }}
                transition={{ duration: 0.16, ease: easeOut, delay: isActive ? 0 : delay }}
                className="relative z-10 whitespace-nowrap"
              >
                {item.label}
              </Motion.span>
            )}
          </AnimatePresence>
        </>
      )}
    </NavLink>
  );
}

function Brand({ expanded }) {
  return (
    <div className={`flex items-center gap-3 px-3 ${expanded ? '' : 'justify-center !px-0'}`}>
      <img src="/mark.svg" alt="MediDiagnose" className="h-9 w-9 shrink-0 rounded-[10px] shadow-soft" />
      <AnimatePresence initial={false}>
        {expanded && (
          <Motion.div
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -6 }}
            transition={{ duration: 0.16, ease: easeOut }}
            className="overflow-hidden whitespace-nowrap"
          >
            <p className="text-[15px] font-semibold leading-none tracking-tight text-ink">
              MediDiagnose
            </p>
            <p className="mt-1 text-[11px] leading-none text-faint">AI health companion</p>
          </Motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Sidebar() {
  const {
    user,
    sidebarOpen,
    setSidebarOpen,
    profileModalOpen,
    setProfileModalOpen,
    signOut,
    theme,
    toggleTheme
  } = useApp();
  const [expanded, setExpanded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmOut, setConfirmOut] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  /* close the drawer on navigation */
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]); // eslint-disable-line react-hooks/exhaustive-deps

  /* close the profile menu on outside click */
  useEffect(() => {
    const close = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, []);

  /* profile dropdown — align 'right' on desktop (opens beside the rail),
     align 'up' in the drawer (opens above the trigger) */
  const profileMenu = (align) => (
    <Motion.div
      initial={{ opacity: 0, scale: 0.97, y: align === 'up' ? 6 : -4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97, y: align === 'up' ? 6 : -4 }}
      transition={{ duration: 0.15, ease: easeOut }}
      className={`panel absolute z-[60] w-60 overflow-hidden !rounded-xl p-1.5 ${
        align === 'up' ? 'bottom-full left-0 right-0 mb-2' : 'bottom-0 left-full ml-2'
      }`}
    >
      <div className="px-3 pb-2.5 pt-2">
        <p className="truncate text-sm font-medium text-ink">{user?.name}</p>
        <p className="truncate text-[13px] text-muted">{user?.email}</p>
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
  );

  return (
    <>
      {/* ------- mobile: floating menu button (replaces the top bar) ------- */}
      <button
        onClick={() => setSidebarOpen(true)}
        aria-label="Open menu"
        className="fixed left-3 top-3 z-40 flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-surface/90 text-ink shadow-soft backdrop-blur-md md:hidden"
      >
        <Menu size={20} />
      </button>

      {/* ------- desktop: hover-expanding rail ------- */}
      <Motion.aside
        onMouseEnter={() => setExpanded(true)}
        onMouseLeave={() => {
          if (!menuOpen) setExpanded(false);
        }}
        initial={false}
        animate={{ width: expanded ? 240 : 78 }}
        transition={spring}
        className="sticky top-0 z-40 hidden h-screen shrink-0 flex-col border-r border-line bg-surface/80 pt-5 backdrop-blur-md md:flex"
      >
        <div className="px-2">
          <Brand expanded={expanded} />
        </div>

        <nav className="mt-7 flex-1 space-y-6 overflow-y-hidden px-2.5 no-scrollbar">
          {GROUPS.map((group) => (
            <div key={group.label}>
              <div className={`mb-1.5 px-3 ${expanded ? '' : 'px-0 text-center'}`}>
                {expanded ? (
                  <p className="eyebrow !text-[10px]">{group.label}</p>
                ) : (
                  <span className="mx-auto block h-px w-6 bg-line" />
                )}
              </div>
              <div className="space-y-0.5">
                {group.items.map((item, i) => (
                  <NavItem key={item.path} item={item} expanded={expanded} delay={i * 0.02} />
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* ------- bottom cluster: search · theme · profile ------- */}
        <div className="border-t border-line px-2.5 pb-4 pt-3">
          <div className="space-y-1">
            <button
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-muted transition-colors hover:bg-raised hover:text-ink ${
                expanded ? '' : 'justify-center !px-0'
              }`}
            >
              {theme === 'dark' ? <Sun size={18} className="shrink-0" /> : <Moon size={18} className="shrink-0" />}
              <AnimatePresence initial={false}>
                {expanded && (
                  <Motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.16 }}
                    className="whitespace-nowrap"
                  >
                    {theme === 'dark' ? 'Light mode' : 'Dark mode'}
                  </Motion.span>
                )}
              </AnimatePresence>
            </button>

            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setMenuOpen((v) => !v)}
                title="Account"
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors hover:bg-raised ${
                  expanded ? '' : 'justify-center !px-0'
                }`}
              >
                <span className="t-num flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink text-[12px] font-semibold text-paper dark:bg-white dark:text-black">
                  {initials(user?.name)}
                </span>
                <AnimatePresence initial={false}>
                  {expanded && (
                    <Motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.16 }}
                      className="flex min-w-0 flex-1 items-center justify-between gap-2 whitespace-nowrap"
                    >
                      <span className="min-w-0 flex-1 truncate text-left text-ink">{user?.name}</span>
                      <ChevronDown
                        size={14}
                        className={`shrink-0 text-faint transition-transform ${menuOpen ? 'rotate-180' : ''}`}
                      />
                    </Motion.span>
                  )}
                </AnimatePresence>
              </button>
              <AnimatePresence>{menuOpen && profileMenu('right')}</AnimatePresence>
            </div>
          </div>
        </div>
      </Motion.aside>

      {/* ------- mobile: slide-over drawer ------- */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            <Motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              onClick={() => setSidebarOpen(false)}
              className="fixed inset-0 z-50 bg-black/45 backdrop-blur-[2px] md:hidden"
            />
            <Motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={spring}
              className="fixed inset-y-0 left-0 z-50 flex w-[260px] flex-col border-r border-line bg-surface pt-5 md:hidden"
            >
              <div className="flex items-center justify-between px-2 pr-3">
                <Brand expanded />
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="icon-btn !h-8 !w-8"
                  aria-label="Close menu"
                >
                  <X size={17} />
                </button>
              </div>
              <nav className="mt-6 flex-1 space-y-6 overflow-y-auto px-2.5">
                {GROUPS.map((group) => (
                  <div key={group.label}>
                    <p className="eyebrow mb-1.5 !text-[10px] px-3">{group.label}</p>
                    <div className="space-y-0.5">
                      {group.items.map((item) => (
                        <NavItem
                          key={item.path}
                          item={item}
                          expanded
                          onNavigate={() => setSidebarOpen(false)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </nav>

              <div className="border-t border-line px-2.5 pb-5 pt-3">
                <div className="space-y-1">
                  <button
                    onClick={toggleTheme}
                    className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-muted transition-colors hover:bg-raised hover:text-ink"
                  >
                    {theme === 'dark' ? <Sun size={18} className="shrink-0" /> : <Moon size={18} className="shrink-0" />}
                    <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
                  </button>
                  <div className="relative" ref={menuRef}>
                    <button
                      onClick={() => setMenuOpen((v) => !v)}
                      className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors hover:bg-raised"
                    >
                      <span className="t-num flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink text-[12px] font-semibold text-paper dark:bg-white dark:text-black">
                        {initials(user?.name)}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-left text-ink">{user?.name}</span>
                      <ChevronDown
                        size={14}
                        className={`shrink-0 text-faint transition-transform ${menuOpen ? 'rotate-180' : ''}`}
                      />
                    </button>
                    <AnimatePresence>{menuOpen && profileMenu('up')}</AnimatePresence>
                  </div>
                </div>
              </div>
            </Motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* ------- modals (moved up from the old top bar) ------- */}
      <AnimatePresence>{profileModalOpen && <ProfileModal />}</AnimatePresence>

      <AnimatePresence>
        {confirmOut && (
          <Modal onClose={() => setConfirmOut(false)}>
            <ModalHeader
              title="Sign out?"
              subtitle="Your history stays saved on this device."
              onClose={() => setConfirmOut(false)}
            />
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
