import React, { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL !== undefined ? import.meta.env.VITE_API_URL : 'http://localhost:8000';

const PRESETS = {
  low: {
    order_id: 'ORD_LOW_001',
    customer_id: 'CUST_LOW_001',
    timestamp: '2025-06-15 14:30:00',
    order_amount: 120.00,
    product_category: 'Books',
    payment_method: 'Credit Card',
    customer_age_days: 365,
    previous_orders: 10,
    previous_returns: 0,
    customer_return_rate: 0.0,
    orders_last_7_days: 0,
    orders_last_30_days: 2,
    returns_last_7_days: 0,
    returns_last_30_days: 0,
    average_order_value: 115.00,
    discount_percentage: 0.0,
    delivery_days: 2,
    return_days_after_delivery: 14,
    address_change_count: 0,
    payment_failures: 0,
    previous_chargebacks: 0,
    is_first_order: 0,
    is_high_value_order: 0,
    review_threshold: 0.30
  },
  medium: {
    order_id: 'ORD_MED_001',
    customer_id: 'CUST_MED_001',
    timestamp: '2025-06-15 14:30:00',
    order_amount: 89.99,
    product_category: 'Clothing',
    payment_method: 'UPI',
    customer_age_days: 550,
    previous_orders: 6,
    previous_returns: 1,
    customer_return_rate: 0.15,
    orders_last_7_days: 1,
    orders_last_30_days: 2,
    returns_last_7_days: 0,
    returns_last_30_days: 0,
    average_order_value: 95.00,
    discount_percentage: 10.0,
    delivery_days: 4,
    return_days_after_delivery: 8,
    address_change_count: 1,
    payment_failures: 0,
    previous_chargebacks: 0,
    is_first_order: 0,
    is_high_value_order: 0,
    review_threshold: 0.30
  },
  high: {
    order_id: 'ORD_HIGH_001',
    customer_id: 'CUST_HIGH_001',
    timestamp: '2025-06-15 14:30:00',
    order_amount: 399.99,
    product_category: 'Electronics',
    payment_method: 'COD',
    customer_age_days: 500,
    previous_orders: 5,
    previous_returns: 2,
    customer_return_rate: 0.35,
    orders_last_7_days: 1,
    orders_last_30_days: 2,
    returns_last_7_days: 0,
    returns_last_30_days: 1,
    average_order_value: 200.00,
    discount_percentage: 25.0,
    delivery_days: 4,
    return_days_after_delivery: 3,
    address_change_count: 1,
    payment_failures: 0,
    previous_chargebacks: 0,
    is_first_order: 0,
    is_high_value_order: 1,
    review_threshold: 0.30
  }
};

export default function RiskConsole({ apiHealth, onCheckHealth }) {
  const [formData, setFormData] = useState({ ...PRESETS.low });
  const [activePreset, setActivePreset] = useState('low');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    let parsedValue = value;
    if (type === 'number') {
      parsedValue = value === '' ? '' : parseFloat(value);
    } else if (name === 'is_first_order' || name === 'is_high_value_order') {
      parsedValue = parseInt(value, 10);
    }
    setFormData((prev) => ({ ...prev, [name]: parsedValue }));
    setActivePreset(null);
  };

  const loadPreset = (presetKey) => {
    const now = new Date();
    const formatted = now.toISOString().replace('T', ' ').substring(0, 19);
    setFormData({
      ...PRESETS[presetKey],
      timestamp: formatted
    });
    setActivePreset(presetKey);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Operational threshold is controlled server-side; omit review_threshold from API payload
    const { review_threshold, ...payload } = formData;

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          Array.isArray(errorData.detail)
            ? errorData.detail.map((d) => `${d.loc.join('.')}: ${d.msg}`).join(', ')
            : errorData.detail || 'Prediction request failed'
        );
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (band) => {
    switch (band) {
      case 'Low': return 'var(--color-low)';
      case 'Medium': return 'var(--accent-gold)';
      case 'High': return 'var(--color-high)';
      case 'Very High': return 'var(--color-veryhigh)';
      default: return 'var(--text-muted)';
    }
  };

  return (
    <div className="console-wrapper">
      {/* Console Title Bar */}
      <div className="console-header">
        <div className="console-title-group">
          <h1>Risk Assessment Console</h1>
          <p>Institutional decision-support for e-commerce return evaluation.</p>
        </div>

        {/* Demo Preset Bar */}
        <div className="preset-controls">
          <span className="preset-label">Demo Cases:</span>
          <button
            type="button"
            className={`preset-btn ${activePreset === 'low' ? 'active-low' : ''}`}
            onClick={() => loadPreset('low')}
          >
            Low Risk
          </button>
          <button
            type="button"
            className={`preset-btn ${activePreset === 'medium' ? 'active-med' : ''}`}
            onClick={() => loadPreset('medium')}
          >
            Medium Risk
          </button>
          <button
            type="button"
            className={`preset-btn ${activePreset === 'high' ? 'active-high' : ''}`}
            onClick={() => loadPreset('high')}
          >
            High Risk
          </button>
        </div>
      </div>

      {/* SECTION 1: Two Column Layout (Form Left | Risk Summary Right) */}
      <div className="console-section-1">
        {/* Left Column: Form */}
        <div className="editorial-card">
          <div className="card-title-bar">
            <h2>Evaluate Return Record</h2>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--accent-gold)' }}>
              POST /predict
            </span>
          </div>

          <form onSubmit={handleSubmit} className="eval-form">
            {/* Identifiers */}
            <div className="form-group-section">
              <span className="form-group-title">TRANSACTION IDENTIFIERS</span>
              <div className="form-grid-3">
                <div className="form-control">
                  <label>Order ID</label>
                  <input type="text" name="order_id" value={formData.order_id} onChange={handleInputChange} required />
                </div>
                <div className="form-control">
                  <label>Customer ID</label>
                  <input type="text" name="customer_id" value={formData.customer_id} onChange={handleInputChange} required />
                </div>
                <div className="form-control">
                  <label>Timestamp</label>
                  <input type="text" name="timestamp" value={formData.timestamp} onChange={handleInputChange} required />
                </div>
              </div>
            </div>

            {/* Transaction Basics */}
            <div className="form-group-section">
              <span className="form-group-title">ORDER INFORMATION</span>
              <div className="form-grid-4">
                <div className="form-control">
                  <label>Amount (₹)</label>
                  <input type="number" name="order_amount" value={formData.order_amount} onChange={handleInputChange} min="0" step="0.01" required />
                </div>
                <div className="form-control">
                  <label>Discount (%)</label>
                  <input type="number" name="discount_percentage" value={formData.discount_percentage} onChange={handleInputChange} min="0" max="100" step="0.1" required />
                </div>
                <div className="form-control">
                  <label>Category</label>
                  <select name="product_category" value={formData.product_category} onChange={handleInputChange}>
                    <option value="Electronics">Electronics</option>
                    <option value="Clothing">Clothing</option>
                    <option value="Beauty">Beauty</option>
                    <option value="Home">Home</option>
                    <option value="Books">Books</option>
                  </select>
                </div>
                <div className="form-control">
                  <label>Payment</label>
                  <select name="payment_method" value={formData.payment_method} onChange={handleInputChange}>
                    <option value="Credit Card">Credit Card</option>
                    <option value="Debit Card">Debit Card</option>
                    <option value="UPI">UPI</option>
                    <option value="Netbanking">Netbanking</option>
                    <option value="COD">COD</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Customer History */}
            <div className="form-group-section">
              <span className="form-group-title">CUSTOMER PROFILE & HISTORY</span>
              <div className="form-grid-4">
                <div className="form-control">
                  <label>Account Days</label>
                  <input type="number" name="customer_age_days" value={formData.customer_age_days} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Prev. Orders</label>
                  <input type="number" name="previous_orders" value={formData.previous_orders} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Prev. Returns</label>
                  <input type="number" name="previous_returns" value={formData.previous_returns} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Return Rate</label>
                  <input type="number" name="customer_return_rate" value={formData.customer_return_rate} onChange={handleInputChange} min="0" max="1" step="0.01" required />
                </div>
              </div>
            </div>

            {/* Activity & Velocity */}
            <div className="form-group-section">
              <span className="form-group-title">RECENT ACTIVITY</span>
              <div className="form-grid-4">
                <div className="form-control">
                  <label>Orders (7d)</label>
                  <input type="number" name="orders_last_7_days" value={formData.orders_last_7_days} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Orders (30d)</label>
                  <input type="number" name="orders_last_30_days" value={formData.orders_last_30_days} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Returns (7d)</label>
                  <input type="number" name="returns_last_7_days" value={formData.returns_last_7_days} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Returns (30d)</label>
                  <input type="number" name="returns_last_30_days" value={formData.returns_last_30_days} onChange={handleInputChange} min="0" required />
                </div>
              </div>
            </div>

            {/* Risk Flags & Policy */}
            <div className="form-group-section">
              <span className="form-group-title">RISK FLAGS & THRESHOLD</span>
              <div className="form-grid-4">
                <div className="form-control">
                  <label>Return Delay</label>
                  <input type="number" name="return_days_after_delivery" value={formData.return_days_after_delivery} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Addr Changes</label>
                  <input type="number" name="address_change_count" value={formData.address_change_count} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Pay Failures</label>
                  <input type="number" name="payment_failures" value={formData.payment_failures} onChange={handleInputChange} min="0" required />
                </div>
                <div className="form-control">
                  <label>Threshold</label>
                  <input type="number" name="review_threshold" value={formData.review_threshold} onChange={handleInputChange} min="0" max="1" step="0.01" required />
                </div>
              </div>
            </div>

            <button type="submit" className="btn btn-primary btn-submit-eval" disabled={loading}>
              {loading ? 'Evaluating Model Parameters...' : 'Analyze Return Risk'}
            </button>
          </form>
        </div>

        {/* Right Column: Risk Summary Panel */}
        <div className="editorial-card summary-panel">
          <div className="card-title-bar">
            <h2>Risk Assessment Summary</h2>
          </div>

          {loading && (
            <div className="loading-summary-box">
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--accent-gold)' }}>
                Running Logistic Regression & SHAP explainer...
              </div>
            </div>
          )}

          {error && (
            <div className="error-summary-box">
              <span style={{ fontSize: '24px', marginBottom: '8px' }}>⚠️</span>
              <p style={{ fontWeight: 600, marginBottom: '6px', color: 'var(--color-high)' }}>Evaluation Failed</p>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>{error}</p>
              <button className="btn btn-outline btn-sm" onClick={onCheckHealth}>
                Re-check API Connection
              </button>
            </div>
          )}

          {!loading && !error && !result && (
            <div className="empty-summary-box">
              <div className="empty-icon">🛡️</div>
              <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', marginBottom: '8px', color: 'var(--text-primary)' }}>
                Awaiting Evaluation
              </h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '280px' }}>
                Select a demo preset above or edit transaction fields, then click <strong>Analyze Return Risk</strong>.
              </p>
            </div>
          )}

          {!loading && !error && result && (
            <div>
              {/* Ring Gauge */}
              <div className="gauge-section">
                <div className="ring-gauge-container">
                  <svg viewBox="0 0 100 100" className="ring-gauge">
                    <circle cx="50" cy="50" r="42" className="ring-bg" />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      className="ring-val"
                      style={{
                        strokeDasharray: '264',
                        strokeDashoffset: 264 - (264 * result.risk_score) / 100,
                        stroke: getRiskColor(result.risk_band)
                      }}
                    />
                  </svg>
                  <div className="gauge-center-text">
                    <span className="gauge-score-num">{result.risk_score}</span>
                    <span className="gauge-score-lbl">Risk Score</span>
                  </div>
                </div>
              </div>

              {/* Recommendation Callout */}
              <div className={`decision-callout ${result.decision}`}>
                <span className="decision-lbl">Decision Recommendation</span>
                <span className="decision-val">{result.decision}</span>
              </div>

              {/* Key Metrics */}
              <div className="summary-metrics-row">
                <div className="summary-metric-box">
                  <span className="sm-lbl">Risk Band</span>
                  <span className="sm-val" style={{ color: getRiskColor(result.risk_band) }}>
                    {result.risk_band}
                  </span>
                </div>
                <div className="summary-metric-box">
                  <span className="sm-lbl">Abuse Probability</span>
                  <span className="sm-val">
                    {(result.risk_probability * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* SECTION 2: Assessment Summary Narrative (Full Width Report Card) */}
      {!loading && !error && result && (
        <>
          <div className="console-section-2">
            <div className="narrative-card">
              <h3>Formal Assessment Report</h3>
              <p className="narrative-text">"{result.summary}"</p>
            </div>
          </div>

          {/* SECTION 3: Two Equal Columns (Increasing Risk | Mitigating Risk) */}
          <div className="console-section-3">
            {/* Increasing Risk */}
            <div className="editorial-card">
              <h3 className="factor-card-title pos">
                <span>Factors Increasing Risk</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>+SHAP</span>
              </h3>

              {result.positive_factors.length === 0 ? (
                <p className="no-factors-msg">No individual features significantly increased risk.</p>
              ) : (
                <div className="factors-list">
                  {result.positive_factors.map((f, i) => (
                    <div className="factor-row" key={i}>
                      <div className="factor-meta">
                        <span className="factor-name">{f.display_name}</span>
                        <span className="factor-val">val: {f.value}</span>
                      </div>
                      <div className="factor-bar-wrapper">
                        <div className="factor-bar-track">
                          <div
                            className="factor-bar-fill pos-fill"
                            style={{ width: `${Math.min(100, f.contribution * 40)}%` }}
                          ></div>
                        </div>
                        <span className="factor-contrib pos">+{f.contribution.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Mitigating Risk */}
            <div className="editorial-card">
              <h3 className="factor-card-title neg">
                <span>Factors Mitigating Risk</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>-SHAP</span>
              </h3>

              {result.negative_factors.length === 0 ? (
                <p className="no-factors-msg">No individual features significantly mitigated risk.</p>
              ) : (
                <div className="factors-list">
                  {result.negative_factors.map((f, i) => (
                    <div className="factor-row" key={i}>
                      <div className="factor-meta">
                        <span className="factor-name">{f.display_name}</span>
                        <span className="factor-val">val: {f.value}</span>
                      </div>
                      <div className="factor-bar-wrapper">
                        <div className="factor-bar-track">
                          <div
                            className="factor-bar-fill neg-fill"
                            style={{ width: `${Math.min(100, Math.abs(f.contribution) * 40)}%` }}
                          ></div>
                        </div>
                        <span className="factor-contrib neg">{f.contribution.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* SECTION 4: Governance Disclaimer */}
      <div className="console-section-4">
        🛡️ <strong>Defense-Only Operational Notice:</strong> ReturnShield AI risk scores and decision recommendations (ALLOW / REVIEW) are generated strictly for human fraud analyst decision support. Automated return rejection, payment withholding, or account suspensions are strictly forbidden.
      </div>
    </div>
  );
}
