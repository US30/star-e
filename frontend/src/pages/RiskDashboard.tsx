import React, { useState } from 'react';
import VaRChart from '../components/VaRChart';

const RiskDashboard: React.FC = () => {
  const [returns] = useState(() => {
    const data = [];
    for (let i = 0; i < 500; i++) {
      data.push((Math.random() - 0.5) * 0.06);
    }
    return data;
  });

  const [riskMetrics] = useState({
    var95: 0.023,
    var99: 0.035,
    cvar95: 0.031,
    cvar99: 0.045,
    maxDrawdown: 0.082,
    volatility: 0.182,
    beta: 1.12,
    trackingError: 0.045,
  });

  const [stressTests] = useState([
    { scenario: 'Market Crash (-20%)', portfolioImpact: -18.5, var: 0.185 },
    { scenario: '2x Volatility', portfolioImpact: -5.2, var: 0.046 },
    { scenario: 'Correlation Spike', portfolioImpact: -7.8, var: 0.055 },
    { scenario: 'Liquidity Crisis', portfolioImpact: -12.3, var: 0.098 },
  ]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-100">Risk Dashboard</h1>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="metric-label">VaR (95%)</div>
          <div className="metric-value text-orange-500">
            {(riskMetrics.var95 * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">Daily</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">VaR (99%)</div>
          <div className="metric-value text-red-500">
            {(riskMetrics.var99 * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">Daily</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">CVaR (95%)</div>
          <div className="metric-value text-orange-500">
            {(riskMetrics.cvar95 * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">Expected Shortfall</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Max Drawdown</div>
          <div className="metric-value text-red-500">
            {(riskMetrics.maxDrawdown * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500">Historical</div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="metric-label">Annualized Volatility</div>
          <div className="metric-value">
            {(riskMetrics.volatility * 100).toFixed(1)}%
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Beta</div>
          <div className="metric-value">{riskMetrics.beta.toFixed(2)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Tracking Error</div>
          <div className="metric-value">
            {(riskMetrics.trackingError * 100).toFixed(2)}%
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Information Ratio</div>
          <div className="metric-value text-green-500">0.85</div>
        </div>
      </div>

      {/* VaR Distribution Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Return Distribution & VaR
        </h2>
        <VaRChart
          returns={returns}
          var95={riskMetrics.var95}
          var99={riskMetrics.var99}
          cvar95={riskMetrics.cvar95}
          width={900}
          height={300}
        />
      </div>

      {/* Stress Testing */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Stress Test Scenarios</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-slate-700">
                <th className="py-3 px-4 text-slate-400 font-medium">Scenario</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">
                  Portfolio Impact
                </th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">
                  Stressed VaR
                </th>
                <th className="py-3 px-4 text-slate-400 font-medium">Severity</th>
              </tr>
            </thead>
            <tbody>
              {stressTests.map((test, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-3 px-4 text-slate-200">{test.scenario}</td>
                  <td className="py-3 px-4 text-right text-red-500">
                    {test.portfolioImpact.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(test.var * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center">
                      <div className="h-2 bg-slate-700 rounded-full w-24 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            Math.abs(test.portfolioImpact) > 15
                              ? 'bg-red-500'
                              : Math.abs(test.portfolioImpact) > 8
                              ? 'bg-orange-500'
                              : 'bg-yellow-500'
                          }`}
                          style={{ width: `${Math.min(Math.abs(test.portfolioImpact) * 5, 100)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monte Carlo Simulation Results */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Monte Carlo Simulation (10,000 paths)
        </h2>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <div className="text-slate-400 text-sm mb-1">5th Percentile</div>
            <div className="text-2xl font-bold text-red-500">-8.2%</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm mb-1">Expected Return</div>
            <div className="text-2xl font-bold text-slate-200">+12.5%</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm mb-1">95th Percentile</div>
            <div className="text-2xl font-bold text-green-500">+28.3%</div>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-700">
          <div className="text-sm text-slate-400">
            Simulation uses correlated multivariate normal distribution based on historical returns
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskDashboard;
