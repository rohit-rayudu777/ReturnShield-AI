import React from 'react';

export default function LandingPage({ onOpenConsole, onOpenAuth, user }) {
  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-inner">
          <span className="eyebrow">DEFENSE-ONLY AI RISK INTELLIGENCE</span>
          <h1 className="hero-title">
            Return Risk <span className="hero-title-accent">Intelligence.</span>
          </h1>
          <p className="hero-subtitle">
            ReturnShield AI provides defense-only AI-powered return-risk assessment to assist human review, protect business margins, and preserve trustworthy customer relationships.
          </p>
          <div className="hero-cta-group">
            <button className="btn btn-primary btn-lg" onClick={onOpenConsole}>
              Explore Risk Console →
            </button>
            <button className="btn btn-secondary btn-lg" onClick={() => scrollToSection('methodology-section')}>
              Learn Methodology
            </button>
          </div>
        </div>
      </section>

      {/* Core Capabilities */}
      <section className="landing-section">
        <div className="section-header">
          <h2 className="section-title">Core Capabilities</h2>
          <p className="section-desc">
            Designed for institutional risk operations requiring transparent, decision-support machine learning.
          </p>
        </div>

        <div className="capabilities-grid">
          <div className="capability-card">
            <span className="capability-number">01</span>
            <h3>Return Risk Scoring</h3>
            <p>
              Statistical probability and calibrated 0–100 risk scores derived from transaction history, customer age, velocity, and payment parameters.
            </p>
          </div>

          <div className="capability-card">
            <span className="capability-number">02</span>
            <h3>Explainable Risk Signals</h3>
            <p>
              Exact SHAP feature attributions that explain every prediction by isolating positive risk drivers and mitigating factors in plain language.
            </p>
          </div>

          <div className="capability-card">
            <span className="capability-number">03</span>
            <h3>Cost-Aware Thresholding</h3>
            <p>
              Configurable probability thresholds calibrated against false-positive customer friction costs versus false-negative fraud losses.
            </p>
          </div>

          <div className="capability-card">
            <span className="capability-number">04</span>
            <h3>Human-in-the-Loop Review</h3>
            <p>
              Defense-only decision recommendations (ALLOW / REVIEW) structured to assist risk analysts without automated denial or account bans.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="methodology-section" className="landing-section-alt">
        <div className="section-inner">
          <div className="section-header">
            <h2 className="section-title">How ReturnShield AI Works</h2>
            <p className="section-desc">
              End-to-end data processing and inference architecture for decision-support risk intelligence.
            </p>
          </div>

          <div className="flow-container">
            <div className="flow-step">
              <span className="flow-step-num">Step 01</span>
              <span className="flow-step-title">Transaction Data</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="flow-step-num">Step 02</span>
              <span className="flow-step-title">Feature Engineering</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="flow-step-num">Step 03</span>
              <span className="flow-step-title">ML Risk Model</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="flow-step-num">Step 04</span>
              <span className="flow-step-title">Risk Score & SHAP</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="flow-step-num">Step 05</span>
              <span className="flow-step-title">Human Review</span>
            </div>
          </div>
        </div>
      </section>

      {/* Model Performance & Metrics */}
      <section className="landing-section">
        <div className="section-header">
          <h2 className="section-title">Empirical Performance Metrics</h2>
          <p className="section-desc">
            Baseline model performance measured on held-out evaluation dataset.
          </p>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-val">77.36%</div>
            <div className="metric-lbl">Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-val">88.38%</div>
            <div className="metric-lbl">Precision</div>
          </div>
          <div className="metric-card">
            <div className="metric-val">74.16%</div>
            <div className="metric-lbl">Recall</div>
          </div>
          <div className="metric-card">
            <div className="metric-val">80.65%</div>
            <div className="metric-lbl">F1 Score</div>
          </div>
          <div className="metric-card">
            <div className="metric-val">86.15%</div>
            <div className="metric-lbl">ROC-AUC</div>
          </div>
        </div>
        <p className="metrics-disclaimer">
          📊 Evaluation on held-out 20,000-sample synthetic test dataset (Phase 2C evaluation).
        </p>
      </section>

      {/* Safety & Defense-Only Principles */}
      <section id="safety-section" className="landing-section-alt">
        <div className="section-inner">
          <div className="section-header">
            <h2 className="section-title">Safety & Responsible AI Governance</h2>
            <p className="section-desc">
              Strict governance safeguards built into the architecture of ReturnShield AI.
            </p>
          </div>

          <div className="safety-grid">
            <div className="safety-card">
              <h3>Defense-Only Decision Support</h3>
              <p>
                The system recommends ALLOW or REVIEW. It never automatically denies returns, cancels transactions, or suspends customer accounts. Human review remains strictly required.
              </p>
            </div>

            <div className="safety-card">
              <h3>Synthetic Data Guarantee</h3>
              <p>
                All model training, evaluations, and demonstration inputs rely entirely on synthetic datasets. No real customer identities, payment data, or PII are used.
              </p>
            </div>

            <div className="safety-card">
              <h3>Transparent SHAP Explainability</h3>
              <p>
                Every risk score is paired with grounded SHAP factor attributions so human risk operators can audit the exact features influencing model probability.
              </p>
            </div>

            <div className="safety-card">
              <h3>Human-in-the-Loop Operation</h3>
              <p>
                Automated punishments are strictly prohibited. Decision boundaries are configurable display recommendations intended to prioritize review queues for fraud analysts.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
