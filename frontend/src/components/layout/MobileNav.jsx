import React from 'react';
import { NavLink } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import {
  LayoutDashboard,
  Stethoscope,
  ScanLine,
  History,
  Settings
} from 'lucide-react';
import { } from '../ui/ui';
import { spring } from '../../lib/motion';

const ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'Home', end: true },
  { path: '/symptoms', icon: Stethoscope, label: 'Symptoms' },
  { path: '/image-analysis', icon: ScanLine, label: 'Scan' },
  { path: '/history', icon: History, label: 'History' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

export default function MobileNav() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 px-4 pb-4 pt-8 md:hidden [background:linear-gradient(to_top,rgb(var(--c-bg))_55%,transparent)]">
      <div className="panel flex items-center justify-around !rounded-2xl px-1.5 py-1.5 !shadow-lift">
        {ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.end}
            className={({ isActive }) =>
              `relative flex flex-1 flex-col items-center gap-0.5 rounded-xl px-2 py-1.5 text-[10px] font-medium transition-colors ${
                isActive ? 'text-ink' : 'text-faint'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <Motion.span
                    layoutId="mobile-active"
                    transition={spring}
                    className="absolute inset-0 rounded-xl bg-ink/[0.06] dark:bg-white/[0.09]"
                  />
                )}
                <item.icon size={19} strokeWidth={isActive ? 2.1 : 1.8} className="relative z-10" />
                <span className="relative z-10">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
