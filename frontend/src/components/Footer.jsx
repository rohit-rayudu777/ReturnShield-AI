import React from 'react';

export default function Footer({ setActiveView }) {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-top">
          <div className="footer-brand">
            <h4>ReturnShield AI</h4>
            <p>Defense-only AI risk intelligence & decision support for e-commerce return management.</p>
          </div>

          <div className="footer-links">
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
            <button
              className="nav-link-btn"
              onClick={() => setActiveView('console')}
            >
              Risk Console
            </button>
            <button
              className="nav-link-btn"
              onClick={() => {
                setActiveView('landing');
                setTimeout(() => {
                  document.getElementById('safety-section')?.scrollIntoView({ behavior: 'smooth' });
                }, 100);
              }}
            >
              Safety Guidelines
            </button>
          </div>
        </div>

        <div className="footer-bottom">
          <span>© 2026 ReturnShield AI. All rights reserved.</span>
          <span>Built for the Razorpay Buildathon — AI Risk Manager Track</span>
        </div>
      </div>
    </footer>
  );
}
