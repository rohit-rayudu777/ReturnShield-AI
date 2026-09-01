import React from 'react';

export default function Header({
  activeView,
  setActiveView,
  user,
  onLogout,
  onOpenAuth,
  apiHealth
}) {
  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Brand logo & name */}
        <div className="brand-container" onClick={() => setActiveView('landing')}>
          <div className="brand-mark">RS</div>
          <div className="brand-title-group">
            <span className="brand-name">ReturnShield AI</span>
            <span className="brand-sub">Defense-Only Risk Intelligence</span>
          </div>
        </div>

        {/* Navigation links */}
        <nav>
          <ul className="nav-links">
            <li>
              <button
                className={`nav-link-btn ${activeView === 'landing' ? 'active' : ''}`}
                onClick={() => setActiveView('landing')}
              >
                Home
              </button>
            </li>
            <li>
              <button
                className={`nav-link-btn ${activeView === 'console' ? 'active' : ''}`}
                onClick={() => setActiveView('console')}
              >
                Risk Console
              </button>
            </li>
            <li>
              <button
                className="nav-link-btn"
                onClick={() => {
                  setActiveView('landing');
                  setTimeout(() => {
                    document.getElementById('methodology-section')?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                }}
              >
                Methodology
              </button>
            </li>
            <li>
              <button
                className="nav-link-btn"
                onClick={() => {
                  setActiveView('landing');
                  setTimeout(() => {
                    document.getElementById('safety-section')?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                }}
              >
                About & Safety
              </button>
            </li>
          </ul>
        </nav>

        {/* Right side actions */}
        <div className="header-actions">
          {/* API Health badge */}
          {apiHealth.checked && (
            <div
              className={`api-status-badge ${apiHealth.online ? 'online' : 'offline'}`}
              title={apiHealth.online ? 'FastAPI model backend connected' : 'API backend disconnected'}
            >
              <span className="status-dot"></span>
              API: {apiHealth.online ? 'Online' : 'Offline'}
            </div>
          )}

          {/* User Auth state */}
          {user ? (
            <div className="user-menu">
              <span className="user-name-tag">👤 {user.name}</span>
              <button className="btn btn-outline btn-sm" onClick={onLogout}>
                Log Out
              </button>
            </div>
          ) : (
            <div className="header-auth-btns">
              <button className="btn btn-outline btn-sm" onClick={() => onOpenAuth('signin')}>
                Sign In
              </button>
              <button className="btn btn-primary btn-sm" onClick={() => onOpenAuth('signup')}>
                Create Account
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
