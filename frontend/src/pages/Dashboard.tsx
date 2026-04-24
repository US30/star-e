import React, { useState, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import RegimeChart from '../components/RegimeChart';
import AllocationPie from '../components/AllocationPie';

const MetricCard: React.FC<{
  label: string;
  value: string | number;
  change?: number;
  format?: string;
}> = ({ label, value, change, format }) => {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {typeof value === 'number' ? value.toFixed(2) : value}
        {format && <span className="text-sm text-slate-400 ml-1">{format}</span>}
      </div>
      {change !== undefined && (
        <div className={`metric-change ${change >= 0 ? 'positive' : 'negative'}`}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </div>
      )}
    </div>
  );
};

const Dashboard: React.FC = () => {
  const { selectedTickers } = useStore();
  const [loading, setLoading] = useState(true);

  // Mock data for demonstration
  const [metrics] = useState({
    totalReturn: 23.4,
    sharpeRatio: 1.42,
    maxDrawdown: -8.2,
    currentRegime: 'Bull',
    volatility: 0.182,
    var95: 0.023,
  });

  const [priceData] = useState(() => {
    const data = [];
    let price = 100;
    let regime = 2;
    const today = new Date();

    for (let i = 365; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);

      if (Math.random() < 0.03) {
        regime = Math.floor(Math.random() * 3);
      }

      const regimeReturn = regime === 0 ? -0.002 : regime === 1 ? 0 : 0.002;
      const dailyReturn = regimeReturn + (Math.random() - 0.5) * 0.02;
      price *= (1 + dailyReturn);

      data.push({ date, price, regime });
    }
    return data;
  });

  const [allocation] = useState([
    { ticker: 'AAPL', weight: 0.25 },
    { ticker: 'MSFT', weight: 0.20 },
    { ticker: 'GOOGL', weight: 0.18 },
    { ticker: 'AMZN', weight: 0.15 },
    { ticker: 'META', weight: 0.12 },
    { ticker: 'Others', weight: 0.10 },
  ]);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-100">Portfolio Dashboard</h1>
        <div className="flex items-center space-x-2">
          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
            metrics.currentRegime === 'Bull' ? 'regime-bull' :
            metrics.currentRegime === 'Bear' ? 'regime-bear' : 'regime-sideways'
          }`}>
            {metrics.currentRegime} Market
          </span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard
          label="Total Return"
          value={`${metrics.totalReturn.toFixed(1)}%`}
          change={2.1}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={metrics.sharpeRatio}
          change={0.12}
        />
        <MetricCard
          label="Max Drawdown"
          value={`${metrics.maxDrawdown.toFixed(1)}%`}
          change={-1.3}
        />
        <MetricCard
          label="Volatility"
          value={`${(metrics.volatility * 100).toFixed(1)}%`}
        />
        <MetricCard
          label="VaR (95%)"
          value={`${(metrics.var95 * 100).toFixed(2)}%`}
        />
        <MetricCard
          label="Active Positions"
          value={selectedTickers.length}
        />
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">
            Portfolio Performance with Regime Overlay
          </h2>
          <RegimeChart data={priceData} width={700} height={350} />
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">
            Current Allocation
          </h2>
          <AllocationPie data={allocation} width={350} height={350} />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Recent Signals</h2>
        <div className="space-y-2">
          {[
            { time: '2h ago', signal: 'Regime shift detected: Sideways → Bull', type: 'info' },
            { time: '1d ago', signal: 'Rebalancing executed: 5 trades', type: 'success' },
            { time: '2d ago', signal: 'VaR breach warning: Consider reducing exposure', type: 'warning' },
            { time: '3d ago', signal: 'New cointegration pair detected: AAPL-MSFT', type: 'info' },
          ].map((item, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-slate-700 last:border-0">
              <div className="flex items-center space-x-3">
                <span className={`w-2 h-2 rounded-full ${
                  item.type === 'success' ? 'bg-green-500' :
                  item.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`}></span>
                <span className="text-slate-300">{item.signal}</span>
              </div>
              <span className="text-slate-500 text-sm">{item.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
