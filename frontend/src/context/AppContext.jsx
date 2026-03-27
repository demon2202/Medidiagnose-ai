import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import bcrypt from 'bcryptjs';

const AppContext = createContext(null);

// Storage keys
const STORAGE_KEYS = {
  USER: 'mediDiagnose_user',
  REGISTERED_USERS: 'mediDiagnose_registeredUsers',
  THEME: 'mediDiagnose_theme',
  HISTORY: 'mediDiagnose_history',
  SETTINGS: 'mediDiagnose_settings',
};

// Helper functions for localStorage
const getRegisteredUsers = () => {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.REGISTERED_USERS);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveRegisteredUsers = (users) => {
  try {
    // Never store plain passwords - only hashed
    const sanitized = users.map(u => {
      const { password, ...safe } = u; // Remove plain password if exists
      return safe;
    });
    localStorage.setItem(STORAGE_KEYS.REGISTERED_USERS, JSON.stringify(sanitized));
  } catch (e) {
    console.error('Failed to save users:', e);
  }
};

const getSavedUser = () => {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.USER);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
};

const saveUser = (userData) => {
  try {
    if (userData) {
      // Never save password-related fields to session
      const { passwordHash, password, ...safeData } = userData;
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(safeData));
    } else {
      localStorage.removeItem(STORAGE_KEYS.USER);
    }
  } catch (e) {
    console.error('Failed to save user:', e);
  }
};

const getHistory = () => {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.HISTORY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveHistory = (history) => {
  try {
    localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(history));
  } catch (e) {
    console.error('Failed to save history:', e);
  }
};

const getSettings = () => {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.SETTINGS);
    return data ? JSON.parse(data) : {};
  } catch {
    return {};
  }
};

const saveSettings = (settings) => {
  try {
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
};

export function AppProvider({ children }) {
  // Auth state
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // App state
  const [theme, setThemeState] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEYS.THEME) || 'light';
    } catch {
      return 'light';
    }
  });
  const [history, setHistory] = useState(() => getHistory());
  const [settings, setSettingsState] = useState(() => getSettings());

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);

  // Notification state
  const [notification, setNotification] = useState(null);
  const notificationTimer = useRef(null);

  // Initialize - check for saved session
  useEffect(() => {
    const savedUser = getSavedUser();
    if (savedUser && savedUser.isAuthenticated) {
      setUser(savedUser);
      setIsAuthenticated(true);
    }
  }, []);

  // Theme effect
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try {
      localStorage.setItem(STORAGE_KEYS.THEME, theme);
    } catch {}
  }, [theme]);

  // ============================================================
  //                    NOTIFICATIONS
  // ============================================================

  const showNotification = useCallback((message, type = 'info', duration = 4000) => {
    if (notificationTimer.current) {
      clearTimeout(notificationTimer.current);
    }
    setNotification({ message, type, id: Date.now() });
    notificationTimer.current = setTimeout(() => {
      setNotification(null);
    }, duration);
  }, []);

  const dismissNotification = useCallback(() => {
    if (notificationTimer.current) {
      clearTimeout(notificationTimer.current);
    }
    setNotification(null);
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (notificationTimer.current) {
        clearTimeout(notificationTimer.current);
      }
    };
  }, []);

  // ============================================================
  //                    AUTH - SIGN IN (BCRYPT)
  // ============================================================

  const signIn = useCallback(async (email, password, rememberMe = false) => {
    setIsLoading(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 800));

      if (!email || !password) {
        throw new Error('Email and password are required');
      }

      if (!/\S+@\S+\.\S+/.test(email)) {
        throw new Error('Please enter a valid email address');
      }

      if (password.length < 6) {
        throw new Error('Password must be at least 6 characters');
      }

      const registeredUsers = getRegisteredUsers();
      const foundUser = registeredUsers.find(
        u => u.email.toLowerCase() === email.toLowerCase()
      );

      if (!foundUser) {
        throw new Error('No account found with this email. Please sign up first.');
      }

      // Verify password with bcrypt
      let isValidPassword = false;

      if (foundUser.passwordHash) {
        // New secure format
        isValidPassword = await bcrypt.compare(password, foundUser.passwordHash);
      } else if (foundUser.password) {
        // Legacy plain text - migrate to hashed
        isValidPassword = foundUser.password === password;
        if (isValidPassword) {
          // Migrate: hash the password and remove plain text
          const hash = await bcrypt.hash(password, 10);
          const userIndex = registeredUsers.findIndex(u => u.id === foundUser.id);
          if (userIndex !== -1) {
            registeredUsers[userIndex].passwordHash = hash;
            delete registeredUsers[userIndex].password;
            saveRegisteredUsers(registeredUsers);
            console.log('Migrated user password to bcrypt hash');
          }
        }
      }

      if (!isValidPassword) {
        throw new Error('Incorrect password. Please try again.');
      }

      const userData = {
        id: foundUser.id,
        name: foundUser.name,
        email: foundUser.email,
        avatar: foundUser.avatar || null,
        isAuthenticated: true,
        createdAt: foundUser.createdAt,
        rememberMe,
      };

      setUser(userData);
      setIsAuthenticated(true);
      saveUser(userData);

      showNotification(`Welcome back, ${foundUser.name.split(' ')[0]}!`, 'success');
      return { success: true };

    } catch (error) {
      showNotification(error.message || 'Sign in failed', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, [showNotification]);

  // ============================================================
  //                    AUTH - SIGN UP (BCRYPT)
  // ============================================================

  const signUp = useCallback(async (name, email, password) => {
    setIsLoading(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      if (!name || !email || !password) {
        throw new Error('All fields are required');
      }

      if (name.trim().length < 2) {
        throw new Error('Name must be at least 2 characters');
      }

      if (!/\S+@\S+\.\S+/.test(email)) {
        throw new Error('Please enter a valid email address');
      }

      if (password.length < 8) {
        throw new Error('Password must be at least 8 characters');
      }

      if (!/\d/.test(password)) {
        throw new Error('Password must contain at least one number');
      }

      if (!/[A-Z]/.test(password)) {
        throw new Error('Password must contain at least one uppercase letter');
      }

      const registeredUsers = getRegisteredUsers();

      const emailExists = registeredUsers.some(
        u => u.email.toLowerCase() === email.toLowerCase()
      );
      if (emailExists) {
        throw new Error('An account with this email already exists. Please sign in instead.');
      }

      // Hash password with bcrypt (salt rounds = 10)
      const passwordHash = await bcrypt.hash(password, 10);

      const newUser = {
        id: Date.now().toString(),
        name: name.trim(),
        email: email.toLowerCase().trim(),
        passwordHash: passwordHash, // Stored hashed, never plain text
        avatar: null,
        createdAt: new Date().toISOString(),
      };

      registeredUsers.push(newUser);
      saveRegisteredUsers(registeredUsers);

      const userData = {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
        avatar: null,
        isAuthenticated: true,
        createdAt: newUser.createdAt,
      };

      setUser(userData);
      setIsAuthenticated(true);
      saveUser(userData);

      showNotification(`Welcome to MediDiagnose, ${newUser.name.split(' ')[0]}!`, 'success');
      return { success: true };

    } catch (error) {
      showNotification(error.message || 'Sign up failed', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, [showNotification]);

  // ============================================================
  //                    AUTH - SIGN OUT
  // ============================================================

  const signOut = useCallback(() => {
    setUser(null);
    setIsAuthenticated(false);
    saveUser(null);
    showNotification('Signed out successfully', 'info');
  }, [showNotification]);

  // ============================================================
  //                    AUTH - CHANGE PASSWORD (BCRYPT)
  // ============================================================

  const changePassword = useCallback(async (currentPassword, newPassword) => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 800));

      if (!currentPassword || !newPassword) {
        throw new Error('Please fill in all fields');
      }

      if (newPassword.length < 8) {
        throw new Error('New password must be at least 8 characters');
      }

      if (!/\d/.test(newPassword)) {
        throw new Error('New password must contain at least one number');
      }

      if (!/[A-Z]/.test(newPassword)) {
        throw new Error('New password must contain at least one uppercase letter');
      }

      if (currentPassword === newPassword) {
        throw new Error('New password must be different from current password');
      }

      const registeredUsers = getRegisteredUsers();
      const userIndex = registeredUsers.findIndex(u => u.id === user.id);

      if (userIndex === -1) {
        throw new Error('User not found');
      }

      // Verify current password
      let isValid = false;
      const storedUser = registeredUsers[userIndex];

      if (storedUser.passwordHash) {
        isValid = await bcrypt.compare(currentPassword, storedUser.passwordHash);
      } else if (storedUser.password) {
        // Legacy format
        isValid = storedUser.password === currentPassword;
      }

      if (!isValid) {
        throw new Error('Current password is incorrect');
      }

      // Hash new password
      const newPasswordHash = await bcrypt.hash(newPassword, 10);
      registeredUsers[userIndex].passwordHash = newPasswordHash;
      delete registeredUsers[userIndex].password; // Remove legacy field

      saveRegisteredUsers(registeredUsers);

      showNotification('Password changed successfully', 'success');
      return { success: true };

    } catch (error) {
      showNotification(error.message || 'Failed to change password', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, [user, showNotification]);

  // ============================================================
  //                    AUTH - UPDATE PROFILE
  // ============================================================

  const updateProfile = useCallback(async (updates) => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const registeredUsers = getRegisteredUsers();
      const userIndex = registeredUsers.findIndex(u => u.id === user.id);

      if (userIndex === -1) {
        throw new Error('User not found');
      }

      // Only allow safe fields to be updated
      const allowedFields = ['name', 'avatar', 'email'];
      const safeUpdates = {};
      for (const key of allowedFields) {
        if (updates[key] !== undefined) {
          safeUpdates[key] = updates[key];
        }
      }

      // If email is changing, check for duplicates
      if (safeUpdates.email && safeUpdates.email !== user.email) {
        const emailExists = registeredUsers.some(
          (u, idx) => idx !== userIndex && u.email.toLowerCase() === safeUpdates.email.toLowerCase()
        );
        if (emailExists) {
          throw new Error('This email is already in use');
        }
      }

      // Update user in registered users
      registeredUsers[userIndex] = { ...registeredUsers[userIndex], ...safeUpdates };
      saveRegisteredUsers(registeredUsers);

      // Update current session
      const updatedUser = { ...user, ...safeUpdates };
      setUser(updatedUser);
      saveUser(updatedUser);

      showNotification('Profile updated successfully', 'success');
      return { success: true };

    } catch (error) {
      showNotification(error.message || 'Failed to update profile', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, [user, showNotification]);

  // ============================================================
  //                    HISTORY
  // ============================================================

  const addToHistory = useCallback((entry) => {
    setHistory(prev => {
      const newHistory = [
        {
          ...entry,
          id: Date.now().toString(),
          timestamp: entry.timestamp || new Date().toISOString()
        },
        ...prev
      ].slice(0, 100); // Keep last 100 entries
      saveHistory(newHistory);
      return newHistory;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
    showNotification('History cleared', 'info');
  }, [showNotification]);

  const removeFromHistory = useCallback((id) => {
    setHistory(prev => {
      const newHistory = prev.filter(item => item.id !== id);
      saveHistory(newHistory);
      return newHistory;
    });
  }, []);

  // ============================================================
  //                    THEME
  // ============================================================

  const setTheme = useCallback((newTheme) => {
    setThemeState(newTheme);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  const toggleDarkMode = useCallback(() => {
    setThemeState(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  // ============================================================
  //                    SETTINGS
  // ============================================================

  const updateSettings = useCallback((newSettings) => {
    setSettingsState(prev => {
      const updated = { ...prev, ...newSettings };
      saveSettings(updated);
      return updated;
    });
  }, []);

  // ============================================================
  //                    CONTEXT VALUE
  // ============================================================

  const value = {
    // Auth
    user,
    isAuthenticated,
    isLoading,
    setIsLoading,
    signIn,
    signUp,
    signOut,
    changePassword,
    updateProfile,

    // Theme
    theme,
    setTheme,
    toggleTheme,
    toggleDarkMode,

    // UI State
    sidebarOpen,
    setSidebarOpen,
    profileModalOpen,
    setProfileModalOpen,

    // History
    history,
    addToHistory,
    clearHistory,
    removeFromHistory,

    // Settings
    settings,
    updateSettings,

    // Notifications
    notification,
    showNotification,
    dismissNotification,
    hideNotification: dismissNotification, // alias used in App.jsx

    // Stats (computed from history)
    getStats: () => {
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      return {
        totalDiagnoses: history.length,
        symptomDiagnoses: history.filter(h => h.type === 'symptom').length,
        imageDiagnoses: history.filter(h => h.type === 'image').length,
        heartDiagnoses: history.filter(h => h.type === 'heart').length,
        cancerDiagnoses: history.filter(h => h.type === 'cancer').length,
        recentDiagnoses: history.filter(h => new Date(h.timestamp) >= oneWeekAgo).length,
      };
    },
  };

  return (
    <AppContext.Provider value={value}>
      {children}

      {/* Notification Toast */}
      {notification && (
        <div className="fixed top-4 right-4 z-[9999] animate-slide-in-right max-w-sm">
          <div className={`
            px-4 py-3 rounded-xl shadow-2xl border flex items-start gap-3
            ${notification.type === 'success' ? 'bg-green-50 dark:bg-green-900/40 border-green-200 dark:border-green-800 text-green-800 dark:text-green-300' :
              notification.type === 'error' ? 'bg-red-50 dark:bg-red-900/40 border-red-200 dark:border-red-800 text-red-800 dark:text-red-300' :
              notification.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900/40 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300' :
              'bg-blue-50 dark:bg-blue-900/40 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-300'
            }
          `}>
            <span className="text-lg flex-shrink-0">
              {notification.type === 'success' ? '✅' :
               notification.type === 'error' ? '❌' :
               notification.type === 'warning' ? '⚠️' : 'ℹ️'}
            </span>
            <p className="text-sm font-medium flex-1">{notification.message}</p>
            <button
              onClick={dismissNotification}
              className="text-current opacity-50 hover:opacity-100 transition-opacity flex-shrink-0"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

export default AppContext;