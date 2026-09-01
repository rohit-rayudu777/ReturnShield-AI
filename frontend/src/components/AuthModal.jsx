import React, { useState } from 'react';

export default function AuthModal({ initialMode = 'signin', onClose, onLoginSuccess }) {
  const [mode, setMode] = useState(initialMode); // 'signin' or 'signup'
  
  // Form fields
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError(null);

    if (!email || !password) {
      setError('Please fill in all required fields.');
      return;
    }

    if (mode === 'signup') {
      if (!name) {
        setError('Please enter your full name.');
        return;
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match.');
        return;
      }
    }

    // Demo Authentication Success
    const userData = {
      name: mode === 'signup' ? name : (email.split('@')[0] || 'Risk Operator'),
      email: email
    };

    if (rememberMe) {
      localStorage.setItem('returnshield_user', JSON.stringify(userData));
    }

    onLoginSuccess(userData);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close">
          ×
        </button>

        <div className="auth-header">
          <h2>{mode === 'signin' ? 'Sign In to Risk Console' : 'Create Operator Account'}</h2>
          <p>
            {mode === 'signin'
              ? 'Access ReturnShield AI risk intelligence portal.'
              : 'Register for decision-support risk management access.'}
          </p>
        </div>

        <div className="demo-auth-banner">
          ℹ️ <strong>Demo Auth Mode:</strong> Frontend session authentication for buildathon evaluation. No credentials stored on server.
        </div>

        {error && (
          <div className="error-summary-box" style={{ color: 'var(--color-high)', marginBottom: '16px', fontSize: '13px' }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'signup' && (
            <div className="form-control">
              <label>Full Name</label>
              <input
                type="text"
                placeholder="Jane Doe"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
          )}

          <div className="form-control">
            <label>Work Email</label>
            <input
              type="email"
              placeholder="operator@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-control">
            <label>Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {mode === 'signup' && (
            <div className="form-control">
              <label>Confirm Password</label>
              <input
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          )}

          {mode === 'signin' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
              <input
                type="checkbox"
                id="remember"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label htmlFor="remember" style={{ color: 'var(--text-secondary)', cursor: 'pointer' }}>
                Remember session on this device
              </label>
            </div>
          )}

          <button type="submit" className="btn btn-primary btn-lg" style={{ marginTop: '8px' }}>
            {mode === 'signin' ? 'Sign In to Console' : 'Complete Account Registration'}
          </button>
        </form>

        <div className="auth-toggle-text">
          {mode === 'signin' ? (
            <>
              Don't have an account?
              <button className="auth-toggle-btn" onClick={() => setMode('signup')}>
                Create Account
              </button>
            </>
          ) : (
            <>
              Already have an account?
              <button className="auth-toggle-btn" onClick={() => setMode('signin')}>
                Sign In
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
