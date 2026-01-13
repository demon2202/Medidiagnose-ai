import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider, useApp } from './context/AppContext';

import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
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
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
}

function PublicRoute({ children }) {
  const { isAuthenticated } = useApp();
  
  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }
  
  return children;
}


function MainLayout({ children }) {
  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Header />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 pb-24 md:pb-6">
          {children}
        </main>
        <MobileNav />
      </div>
    </div>
  );
}

function AppContent() {
  const { notification, hideNotification } = useApp();

  return (
    <Router>
      {notification && (
        <Notification
          message={notification.message}
          type={notification.type}
          onClose={hideNotification}
        />
      )}

      <Routes>
       
        <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
        <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
        <Route path="/forgot-password" element={<PublicRoute><ForgotPassword /></PublicRoute>} />

        
        <Route path="/" element={<ProtectedRoute><MainLayout><Dashboard /></MainLayout></ProtectedRoute>} />
        <Route path="/symptoms" element={<ProtectedRoute><MainLayout><SymptomDiagnosis /></MainLayout></ProtectedRoute>} />
        <Route path="/image-analysis" element={<ProtectedRoute><MainLayout><ImageAnalysis /></MainLayout></ProtectedRoute>} />
        <Route path="/heart-check" element={<ProtectedRoute><MainLayout><HeartCheck /></MainLayout></ProtectedRoute>} />
        <Route path="/cancer-screening" element={<ProtectedRoute><MainLayout><CancerScreening /></MainLayout></ProtectedRoute>} />
        <Route path="/history" element={<ProtectedRoute><MainLayout><History /></MainLayout></ProtectedRoute>} />
        <Route path="/health-tips" element={<ProtectedRoute><MainLayout><HealthTips /></MainLayout></ProtectedRoute>} />
        <Route path="/settings" element={<ProtectedRoute><MainLayout><Settings /></MainLayout></ProtectedRoute>} />

        
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;