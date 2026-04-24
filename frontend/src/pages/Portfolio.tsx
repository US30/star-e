import React, { useState } from 'react';
import AllocationPie from '../components/AllocationPie';
import CorrelationHeatmap from '../components/CorrelationHeatmap';

const Portfolio: React.FC = () => {
  const [allocation] = useState([
    { ticker: 'AAPL', weight: 0.22 },
    { ticker: 'MSFT', weight: 0.18 },
    { ticker: 'GOOGL', weight: 0.15 },
    { ticker: 'AMZN', weight: 0.12 },
    { ticker: 'META', weight: 0.10 },
    { ticker: 'NVDA', weight: 0.08 },
    { ticker: 'TSLA', weight: 0.08 },
    { ticker: 'Cash', weight: 0.07 },
  ]);

  const [correlationMatrix] = useState([
    [1.0, 0.72, 0.68, 0.65, 0.58, 0.45, 0.52, -0.12],
    [0.72, 1.0, 0.75, 0.62, 0.55, 0.48, 0.45, -0.08],
    [0.68, 0.75, 1.0, 0.70, 0.62, 0.42, 0.48, -0.10],
    [0.65, 0.62, 0.70, 1.0, 0.58, 0.38, 0.55, -0.15],
    [0.58, 0.55, 0.62, 0.58, 1.0, 0.35, 0.42, -0.05],
    [0.45, 0.48, 0.42, 0.38, 0.35, 1.0, 0.58, -0.18],
    [0.52, 0.45, 0.48, 0.55, 0.42, 0.58, 1.0, -0.20],
    [-0.12, -0.08, -0.10, -0.15, -0.05, -0.18, -0.20, 1.0],
  ]);

  const labels = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'Cash'];

  const [cointegrationPairs] = useState([
    { pair: 'AAPL - MSFT', spread_zscore: 1.23, hedge_ratio: 0.85, half_life: 15.2 },
    { pair: 'GOOGL - META', spread_zscore: -0.87, hedge_ratio: 1.12, half_life: 22.5 },
    { pair: 'AMZN - TSLA', spread_zscore: 2.15, hedge_ratio: 0.72, half_life: 18.8 },
  ]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-100">Portfolio Management</h1>
        <div className="flex space-x-2">
          <button className="btn-primary">Optimize</button>
          <button className="btn-secondary">Rebalance</button>
        </div>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="metric-card">
          <div className="metric-label">Expected Return</div>
          <div className="metric-value text-green-500">14.2%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Volatility</div>
          <div className="metric-value">18.5%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Sharpe Ratio</div>
          <div className="metric-value text-blue-500">1.42</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Sortino Ratio</div>
          <div className="metric-value text-blue-500">1.85</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Positions</div>
          <div className="metric-value">8</div>
        </div>
      </div>

      {/* Allocation and Correlation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Current Allocation</h2>
          <AllocationPie data={allocation} width={400} height={400} />
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Correlation Matrix</h2>
          <CorrelationHeatmap
            correlationMatrix={correlationMatrix}
            labels={labels}
            width={420}
            height={420}
          />
        </div>
      </div>

      {/* Holdings Table */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Holdings</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-slate-700">
                <th className="py-3 px-4 text-slate-400 font-medium">Ticker</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Weight</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Return (YTD)</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Volatility</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Beta</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Contribution</th>
              </tr>
            </thead>
            <tbody>
              {allocation.map((holding, i) => (
                <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/50">
                  <td className="py-3 px-4 font-medium text-slate-200">{holding.ticker}</td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(holding.weight * 100).toFixed(1)}%
                  </td>
                  <td className={`py-3 px-4 text-right ${
                    Math.random() > 0.3 ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {(Math.random() * 30 - 5).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(15 + Math.random() * 20).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(0.8 + Math.random() * 0.6).toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(holding.weight * (Math.random() * 30 - 5)).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cointegration Pairs */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Cointegrated Pairs (Johansen + Engle-Granger)
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-slate-700">
                <th className="py-3 px-4 text-slate-400 font-medium">Pair</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Spread Z-Score</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Hedge Ratio</th>
                <th className="py-3 px-4 text-slate-400 font-medium text-right">Half-Life</th>
                <th className="py-3 px-4 text-slate-400 font-medium">Signal</th>
              </tr>
            </thead>
            <tbody>
              {cointegrationPairs.map((pair, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-3 px-4 font-medium text-slate-200">{pair.pair}</td>
                  <td className={`py-3 px-4 text-right font-medium ${
                    Math.abs(pair.spread_zscore) > 2 ? 'text-yellow-500' :
                    Math.abs(pair.spread_zscore) > 1.5 ? 'text-orange-500' : 'text-slate-300'
                  }`}>
                    {pair.spread_zscore.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {pair.hedge_ratio.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {pair.half_life.toFixed(1)} days
                  </td>
                  <td className="py-3 px-4">
                    {Math.abs(pair.spread_zscore) > 2 ? (
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        pair.spread_zscore > 0 ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'
                      }`}>
                        {pair.spread_zscore > 0 ? 'Short Spread' : 'Long Spread'}
                      </span>
                    ) : (
                      <span className="px-2 py-1 rounded text-xs font-medium bg-slate-700 text-slate-400">
                        No Signal
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
