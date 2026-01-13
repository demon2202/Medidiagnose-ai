import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AppContext = createContext();

const getStoredData = (key, defaultValue) => {
  try {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : defaultValue;
  } catch {
    return defaultValue;
  }
};

const setStoredData = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error('Failed to save to localStorage:', error);
  }
};

const DEFAULT_USER = {
  id: null,
  name: 'Guest User',
  email: '',
  avatar: null,
  isAuthenticated: false,
  createdAt: null,
};

const DEFAULT_SETTINGS = {
  darkMode: false,
  notifications: true,
  language: 'en',
  autoSaveHistory: true,
};

export function AppProvider({ children }) {
  const [user, setUser] = useState(() => getStoredData('medidiagnose_user', DEFAULT_USER));
  const [isAuthenticated, setIsAuthenticated] = useState(() => getStoredData('medidiagnose_auth', false));
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState(null);
  
  const [settings, setSettings] = useState(() => getStoredData('medidiagnose_settings', DEFAULT_SETTINGS));
  
  const [history, setHistory] = useState(() => getStoredData('medidiagnose_history', []));

  useEffect(() => {
    const root = document.documentElement;
    if (settings.darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [settings.darkMode]);

  useEffect(() => {
    setStoredData('medidiagnose_user', user);
    setStoredData('medidiagnose_auth', isAuthenticated);
  }, [user, isAuthenticated]);

  useEffect(() => {
    setStoredData('medidiagnose_settings', settings);
  }, [settings]);

  useEffect(() => {
    if (settings.autoSaveHistory) {
      setStoredData('medidiagnose_history', history);
    }
  }, [history, settings.autoSaveHistory]);

  const getRegisteredUsers = () => {
    return getStoredData('medidiagnose_registered_users', []);
  };

  const saveRegisteredUsers = (users) => {
    setStoredData('medidiagnose_registered_users', users);
  };

  const signIn = useCallback(async (email, password, rememberMe = false) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
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
      
      const foundUser = registeredUsers.find(u => u.email.toLowerCase() === email.toLowerCase());
      
      if (!foundUser) {
        throw new Error('No account found with this email. Please sign up first.');
      }

      if (foundUser.password !== password) {
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
      showNotification(`Welcome back, ${foundUser.name.split(' ')[0]}!`, 'success');
      
      return { success: true };
    } catch (error) {
      showNotification(error.message || 'Sign in failed', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const signUp = useCallback(async (name, email, password) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1200));
      
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
      
      const emailExists = registeredUsers.some(u => u.email.toLowerCase() === email.toLowerCase());
      if (emailExists) {
        throw new Error('An account with this email already exists. Please sign in instead.');
      }
      
      const newUser = {
        id: Date.now().toString(),
        name: name.trim(),
        email: email.toLowerCase().trim(),
        password: password, 
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
      showNotification(`Welcome to MediDiagnose, ${newUser.name.split(' ')[0]}!`, 'success');
      
      return { success: true };
    } catch (error) {
      showNotification(error.message || 'Sign up failed', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const signOut = useCallback(() => {
    setUser(DEFAULT_USER);
    setIsAuthenticated(false);
    setSidebarOpen(false);
    setProfileModalOpen(false);
    showNotification('Signed out successfully', 'info');
  }, []);

  const updateUser = useCallback((updates) => {
    setUser(prev => {
      const updated = { ...prev, ...updates };
      
      if (updates.email || updates.name) {
        const registeredUsers = getRegisteredUsers();
        const userIndex = registeredUsers.findIndex(u => u.id === prev.id);
        if (userIndex !== -1) {
          registeredUsers[userIndex] = {
            ...registeredUsers[userIndex],
            name: updates.name || registeredUsers[userIndex].name,
            email: updates.email || registeredUsers[userIndex].email,
          };
          saveRegisteredUsers(registeredUsers);
        }
      }
      
      return updated;
    });
    showNotification('Profile updated successfully', 'success');
  }, []);

  const resetPassword = useCallback(async (email) => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      if (!email) {
        throw new Error('Please enter your email address');
      }

      if (!/\S+@\S+\.\S+/.test(email)) {
        throw new Error('Please enter a valid email address');
      }

      const registeredUsers = getRegisteredUsers();
      const userExists = registeredUsers.some(u => u.email.toLowerCase() === email.toLowerCase());
      
      if (!userExists) {
        throw new Error('No account found with this email address');
      }

      showNotification('Password reset email sent! Check your inbox.', 'success');
      return { success: true };
    } catch (error) {
      showNotification(error.message || 'Failed to send reset email', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, []);

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

      const registeredUsers = getRegisteredUsers();
      const userIndex = registeredUsers.findIndex(u => u.id === user.id);
      
      if (userIndex === -1) {
        throw new Error('User not found');
      }

      if (registeredUsers[userIndex].password !== currentPassword) {
        throw new Error('Current password is incorrect');
      }

      registeredUsers[userIndex].password = newPassword;
      saveRegisteredUsers(registeredUsers);

      showNotification('Password changed successfully', 'success');
      return { success: true };
    } catch (error) {
      showNotification(error.message || 'Failed to change password', 'error');
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  }, [user]);


  const updateSettings = useCallback((newSettings) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const toggleDarkMode = useCallback(() => {
    setSettings(prev => ({ ...prev, darkMode: !prev.darkMode }));
  }, []);

 
  const addToHistory = useCallback((item) => {
    const newItem = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      ...item,
    };
    setHistory(prev => [newItem, ...prev]);
  }, []);

  const deleteHistoryItem = useCallback((id) => {
    setHistory(prev => prev.filter(item => item.id !== id));
    showNotification('Item deleted', 'info');
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setStoredData('medidiagnose_history', []);
    showNotification('History cleared', 'info');
  }, []);

  const getStats = useCallback(() => {
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    
    return {
      totalDiagnoses: history.length,
      symptomDiagnoses: history.filter(h => h.type === 'symptom').length,
      imageDiagnoses: history.filter(h => h.type === 'image').length,
      recentDiagnoses: history.filter(h => new Date(h.timestamp) >= weekAgo).length,
      averageConfidence: history.length > 0
        ? (history.reduce((sum, h) => sum + (h.confidence || 0), 0) / history.length * 100).toFixed(1)
        : 0,
    };
  }, [history]);

 
  const showNotification = useCallback((message, type = 'info', duration = 4000) => {
    const id = Date.now();
    setNotification({ id, message, type });
    
    if (duration > 0) {
      setTimeout(() => {
        setNotification(prev => prev?.id === id ? null : prev);
      }, duration);
    }
  }, []);

  const hideNotification = useCallback(() => {
    setNotification(null);
  }, []);

  const value = {
    user,
    isAuthenticated,
    signIn,
    signUp,
    signOut,
    updateUser,
    resetPassword,
    changePassword,
    
    sidebarOpen,
    setSidebarOpen,
    profileModalOpen,
    setProfileModalOpen,
    isLoading,
    setIsLoading,
    
   
    settings,
    updateSettings,
    toggleDarkMode,
    

    history,
    addToHistory,
    deleteHistoryItem,
    clearHistory,
    getStats,
    
    
    notification,
    showNotification,
    hideNotification,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
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