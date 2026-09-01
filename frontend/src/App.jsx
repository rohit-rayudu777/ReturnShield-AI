import React, { useState, useEffect } from 'react';

import Header from './components/Header';
import Footer from './components/Footer';
import LandingPage from './components/LandingPage';
import RiskConsole from './components/RiskConsole';
import AuthModal from './components/AuthModal';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  // Navigation & View state: 'landing' or 'console'
  const [activeView, setActiveView] = useState('landing');
  
  // Auth state & Modal
  const [user, setUser] = useState(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState('signin'); // 'signin' or 'signup'

  // API Backend Health state
  const [apiHealth, setApiHealth] = useState({ online: false, checked: false });

  // On initial mount, load saved user session if available and check backend health
  useEffect(() => {
    const savedUser = localStorage.getItem('returnshield_user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (e) {
        localStorage.removeItem('returnshield_user');
      }
    }
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiHealth({ online: data.model_loaded, checked: true });
      } else {
        setApiHealth({ online: false, checked: true });
      }
    } catch (err) {
      setApiHealth({ online: false, checked: true });
    }
  };

  const handleOpenAuth = (mode = 'signin') => {
    setAuthMode(mode);
    setAuthModalOpen(true);
  };

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    setAuthModalOpen(false);
    setActiveView('console'); // Automatically navigate to Risk Console after login
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('returnshield_user');
    setActiveView('landing');
  };

  const handleOpenConsole = () => {
    setActiveView('console');
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <Header
        activeView={activeView}
        setActiveView={setActiveView}
        user={user}
        onLogout={handleLogout}
        onOpenAuth={handleOpenAuth}
        apiHealth={apiHealth}
      />

      {/* Main View Area */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeView === 'landing' ? (
          <LandingPage
            onOpenConsole={handleOpenConsole}
            onOpenAuth={handleOpenAuth}
            user={user}
          />
        ) : (
          <RiskConsole
            apiHealth={apiHealth}
            onCheckHealth={checkHealth}
          />
        )}
      </main>

      {/* Footer */}
      <Footer setActiveView={setActiveView} />

      {/* Demo Auth Modal */}
      {authModalOpen && (
        <AuthModal
          initialMode={authMode}
          onClose={() => setAuthModalOpen(false)}
          onLoginSuccess={handleLoginSuccess}
        />
      )}
    </div>
  );
}
