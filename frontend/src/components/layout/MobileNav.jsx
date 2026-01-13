import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Stethoscope, 
  Image as ImageIcon, 
  History as HistoryIcon, 
  Settings as SettingsIcon 
} from 'lucide-react';

function MobileNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 md:hidden z-40 shadow-lg">
      <div className="flex items-center justify-around py-2 px-2">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-all ${
              isActive 
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`
          }
        >
          <LayoutDashboard size={20} />
          <span className="text-xs font-medium">Home</span>
        </NavLink>

        <NavLink
          to="/symptoms"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-all ${
              isActive 
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`
          }
        >
          <Stethoscope size={20} />
          <span className="text-xs font-medium">Symptoms</span>
        </NavLink>

        <NavLink
          to="/image-analysis"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-all ${
              isActive 
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`
          }
        >
          <ImageIcon size={20} />
          <span className="text-xs font-medium">Scan</span>
        </NavLink>

        <NavLink
          to="/history"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-all ${
              isActive 
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`
          }
        >
          <HistoryIcon size={20} />
          <span className="text-xs font-medium">History</span>
        </NavLink>

        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-all ${
              isActive 
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`
          }
        >
          <SettingsIcon size={20} />
          <span className="text-xs font-medium">Settings</span>
        </NavLink>
      </div>
    </nav>
  );
}

export default MobileNav;