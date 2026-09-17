import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
} from 'react';
import bcrypt from 'bcryptjs';

const AppContext = createContext(null);

const STORAGE_KEYS = {
  USER: 'mediDiagnose_user',
  REGISTERED_USERS: 'mediDiagnose_registeredUsers',
  THEME: 'mediDiagnose_theme',
  HISTORY: 'mediDiagnose_history',
  SETTINGS: 'mediDiagnose_settings',
};

/* ---------------- storage helpers ---------------- */

const readJSON = (key, fallback) => {
  try {
    const data = localStorage.getItem(key);
    return data ? JSON.parse(data) : fallback;
  } catch {
    return fallback;
  }
};

const writeJSON = (key, value) => {
  try {
    if (value === null || value === undefined) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.error(`Failed to persist ${key}:`, e);
  }
};

const getRegisteredUsers = () => readJSON(STORAGE_KEYS.REGISTERED_USERS, []);

const saveRegisteredUsers = (users) =>
  writeJSON(
    STORAGE_KEYS.REGISTERED_USERS,
    // strip any legacy plain-text password before persisting
    users.map(({ password: _dropped, ...safe }) => safe), // eslint-disable-line no-unused-vars
  );

const getSavedUser = () => readJSON(STORAGE_KEYS.USER, null);

const saveUser = (userData) => {
  if (!userData) return writeJSON(STORAGE_KEYS.USER, null);
  const { passwordHash: _h, password: _p, ...safeData } = userData;
  writeJSON(STORAGE_KEYS.USER, safeData);
};

const EMAIL_RE = /\S+@\S+\.\S+/;

function validatePassword(password) {
  if (!password || password.length < 8)
    throw new Error('Password must be at least 8 characters');
  if (!/\d/.test(password))
    throw new Error('Password must contain at least one number');
  if (!/[A-Z]/.test(password))
    throw new Error('Password must contain at least one uppercase letter');
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------------- provider ---------------- */

export function AppProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [theme, setThemeState] = useState(() => {
    const stored = readJSON(STORAGE_KEYS.THEME, null);
    if (stored === 'light' || stored === 'dark') return stored;
    try {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } catch {
      return 'light';
    }
  });
  const [history, setHistory] = useState(() => readJSON(STORAGE_KEYS.HISTORY, []));
  const [settings, setSettingsState] = useState(() => ({
    darkMode: readJSON(STORAGE_KEYS.THEME, 'light') === 'dark',
    notifications: true,
    autoSaveHistory: true,
    ...readJSON(STORAGE_KEYS.SETTINGS, {}),
  }));

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [inspectorEntry, setInspectorEntry] = useState(null);
  const [notification, setNotification] = useState(null);
  const notificationTimer = useRef(null);

  useEffect(() => {
    const savedUser = getSavedUser();
    if (savedUser?.isAuthenticated) {
      setUser(savedUser);
      setIsAuthenticated(true);
    }
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    writeJSON(STORAGE_KEYS.THEME, theme);
    setSettingsState((prev) =>
      prev.darkMode === (theme === 'dark')
        ? prev
        : { ...prev, darkMode: theme === 'dark' },
    );
  }, [theme]);

  /* ---------------- notifications (rendered once in App.jsx) ---------------- */

  const showNotification = useCallback((message, type = 'info', duration = 4000) => {
    if (notificationTimer.current) clearTimeout(notificationTimer.current);
    setNotification({ message, type, id: Date.now() });
    notificationTimer.current = setTimeout(() => setNotification(null), duration);
  }, []);

  const dismissNotification = useCallback(() => {
    if (notificationTimer.current) clearTimeout(notificationTimer.current);
    setNotification(null);
  }, []);

  useEffect(
    () => () => {
      if (notificationTimer.current) clearTimeout(notificationTimer.current);
    },
    [],
  );

  /* ---------------- auth ---------------- */

  const signIn = useCallback(
    async (email, password, rememberMe = false) => {
      setIsLoading(true);
      try {
        await wait(700);
        if (!email || !password) throw new Error('Email and password are required');
        if (!EMAIL_RE.test(email)) throw new Error('Please enter a valid email address');
        if (password.length < 6) throw new Error('Password must be at least 6 characters');

        const registeredUsers = getRegisteredUsers();
        const foundUser = registeredUsers.find(
          (u) => u.email.toLowerCase() === email.toLowerCase(),
        );
        if (!foundUser)
          throw new Error('No account found with this email. Please sign up first.');

        let isValidPassword = false;
        if (foundUser.passwordHash) {
          isValidPassword = await bcrypt.compare(password, foundUser.passwordHash);
        } else if (foundUser.password) {
          isValidPassword = foundUser.password === password;
          if (isValidPassword) {
            const hash = await bcrypt.hash(password, 10);
            const idx = registeredUsers.findIndex((u) => u.id === foundUser.id);
            if (idx !== -1) {
              registeredUsers[idx].passwordHash = hash;
              delete registeredUsers[idx].password;
              saveRegisteredUsers(registeredUsers);
            }
          }
        }
        if (!isValidPassword) throw new Error('Incorrect password. Please try again.');

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
        showNotification(`Welcome back, ${foundUser.name.split(' ')[0]}`, 'success');
        return { success: true };
      } catch (error) {
        showNotification(error.message || 'Sign in failed', 'error');
        return { success: false, error: error.message };
      } finally {
        setIsLoading(false);
      }
    },
    [showNotification],
  );

  const signUp = useCallback(
    async (name, email, password) => {
      setIsLoading(true);
      try {
        await wait(900);
        if (!name || !email || !password) throw new Error('All fields are required');
        if (name.trim().length < 2) throw new Error('Name must be at least 2 characters');
        if (!EMAIL_RE.test(email)) throw new Error('Please enter a valid email address');
        validatePassword(password);

        const registeredUsers = getRegisteredUsers();
        if (
          registeredUsers.some((u) => u.email.toLowerCase() === email.toLowerCase())
        )
          throw new Error('An account with this email already exists. Please sign in instead.');

        const passwordHash = await bcrypt.hash(password, 10);
        const newUser = {
          id: Date.now().toString(),
          name: name.trim(),
          email: email.toLowerCase().trim(),
          passwordHash,
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
        showNotification(`Welcome to MediDiagnose, ${newUser.name.split(' ')[0]}`, 'success');
        return { success: true };
      } catch (error) {
        showNotification(error.message || 'Sign up failed', 'error');
        return { success: false, error: error.message };
      } finally {
        setIsLoading(false);
      }
    },
    [showNotification],
  );

  const signOut = useCallback(() => {
    setUser(null);
    setIsAuthenticated(false);
    saveUser(null);
    showNotification('Signed out', 'info');
  }, [showNotification]);

  const changePassword = useCallback(
    async (currentPassword, newPassword) => {
      setIsLoading(true);
      try {
        await wait(700);
        if (!currentPassword || !newPassword) throw new Error('Please fill in all fields');
        validatePassword(newPassword);
        if (currentPassword === newPassword)
          throw new Error('New password must be different from the current one');

        const registeredUsers = getRegisteredUsers();
        const idx = registeredUsers.findIndex((u) => u.id === user.id);
        if (idx === -1) throw new Error('User not found');

        const stored = registeredUsers[idx];
        const isValid = stored.passwordHash
          ? await bcrypt.compare(currentPassword, stored.passwordHash)
          : stored.password === currentPassword;
        if (!isValid) throw new Error('Current password is incorrect');

        registeredUsers[idx].passwordHash = await bcrypt.hash(newPassword, 10);
        delete registeredUsers[idx].password;
        saveRegisteredUsers(registeredUsers);

        showNotification('Password changed', 'success');
        return { success: true };
      } catch (error) {
        showNotification(error.message || 'Failed to change password', 'error');
        return { success: false, error: error.message };
      } finally {
        setIsLoading(false);
      }
    },
    [user, showNotification],
  );

  const updateProfile = useCallback(
    async (updates) => {
      setIsLoading(true);
      try {
        await wait(400);
        const registeredUsers = getRegisteredUsers();
        const idx = registeredUsers.findIndex((u) => u.id === user.id);
        if (idx === -1) throw new Error('User not found');

        const safeUpdates = {};
        for (const key of ['name', 'avatar', 'email']) {
          if (updates[key] !== undefined) safeUpdates[key] = updates[key];
        }
        if (safeUpdates.email && safeUpdates.email !== user.email) {
          const taken = registeredUsers.some(
            (u, i) => i !== idx && u.email.toLowerCase() === safeUpdates.email.toLowerCase(),
          );
          if (taken) throw new Error('This email is already in use');
        }

        registeredUsers[idx] = { ...registeredUsers[idx], ...safeUpdates };
        saveRegisteredUsers(registeredUsers);

        const updatedUser = { ...user, ...safeUpdates };
        setUser(updatedUser);
        saveUser(updatedUser);
        showNotification('Profile updated', 'success');
        return { success: true };
      } catch (error) {
        showNotification(error.message || 'Failed to update profile', 'error');
        return { success: false, error: error.message };
      } finally {
        setIsLoading(false);
      }
    },
    [user, showNotification],
  );

  const resetPassword = useCallback(
    async (email, newPassword = null) => {
      setIsLoading(true);
      try {
        await wait(900);
        const registeredUsers = getRegisteredUsers();
        const idx = registeredUsers.findIndex(
          (u) => u.email.toLowerCase() === email.toLowerCase(),
        );
        if (idx === -1) throw new Error('No account found with this email.');

        if (newPassword) {
          validatePassword(newPassword);
          registeredUsers[idx].passwordHash = await bcrypt.hash(newPassword, 10);
          saveRegisteredUsers(registeredUsers);
          showNotification('Password reset. You can sign in now.', 'success');
          return { success: true };
        }
        showNotification('Identity verified. Set a new password.', 'success');
        return { success: true, verified: true };
      } catch (error) {
        showNotification(error.message || 'Failed to reset password', 'error');
        return { success: false, error: error.message };
      } finally {
        setIsLoading(false);
      }
    },
    [showNotification],
  );

  /* ---------------- history ---------------- */

  const addToHistory = useCallback((entry) => {
    setHistory((prev) => {
      const next = [
        { ...entry, id: Date.now().toString(), timestamp: entry.timestamp || new Date().toISOString() },
        ...prev,
      ].slice(0, 100);
      writeJSON(STORAGE_KEYS.HISTORY, next);
      return next;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    writeJSON(STORAGE_KEYS.HISTORY, []);
    showNotification('History cleared', 'info');
  }, [showNotification]);

  const removeFromHistory = useCallback((id) => {
    setHistory((prev) => {
      const next = prev.filter((item) => item.id !== id);
      writeJSON(STORAGE_KEYS.HISTORY, next);
      return next;
    });
  }, []);

  /* ---------------- theme / settings ---------------- */

  const setTheme = useCallback((newTheme) => setThemeState(newTheme), []);
  const toggleTheme = useCallback(
    () => setThemeState((prev) => (prev === 'light' ? 'dark' : 'light')),
    [],
  );

  const updateSettings = useCallback((newSettings) => {
    setSettingsState((prev) => {
      const updated = { ...prev, ...newSettings };
      writeJSON(STORAGE_KEYS.SETTINGS, updated);
      return updated;
    });
  }, []);

  const getStats = useCallback(() => {
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const isImage = (h) => h.type?.startsWith('image');
    return {
      totalDiagnoses: history.length,
      symptomDiagnoses: history.filter((h) => h.type === 'symptom').length,
      imageDiagnoses: history.filter(isImage).length,
      heartDiagnoses: history.filter((h) => h.type === 'heart').length,
      cancerDiagnoses: history.filter((h) => h.type === 'cancer').length,
      recentDiagnoses: history.filter((h) => new Date(h.timestamp) >= weekAgo).length,
    };
  }, [history]);

  const value = {
    user,
    isAuthenticated,
    isLoading,
    setIsLoading,
    signIn,
    signUp,
    signOut,
    changePassword,
    resetPassword,
    updateProfile,
    theme,
    setTheme,
    toggleTheme,
    toggleDarkMode: toggleTheme,
    sidebarOpen,
    setSidebarOpen,
    profileModalOpen,
    setProfileModalOpen,
    paletteOpen,
    setPaletteOpen,
    inspectorEntry,
    setInspectorEntry,
    openInspector: (entry) => setInspectorEntry(entry),
    closeInspector: () => setInspectorEntry(null),
    history,
    addToHistory,
    clearHistory,
    removeFromHistory,
    settings,
    updateSettings,
    notification,
    showNotification,
    dismissNotification,
    hideNotification: dismissNotification,
    getStats,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within an AppProvider');
  return context;
}

export default AppContext;
