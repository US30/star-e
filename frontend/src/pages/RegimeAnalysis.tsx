import React, { useState } from 'react';
import TransitionMatrix from '../components/TransitionMatrix';
import RegimeChart from '../components/RegimeChart';

const RegimeAnalysis: React.FC = () => {
  const [transitionMatrix] = useState([
    [0.92, 0.06, 0.02],
    [0.08, 0.84, 0.08],
    [0.03, 0.07, 0.90],
  ]);

  const [stateProbabilities] = useState({
    Bear: 0.05,
    Sideways: 0.15,
    Bull: 0.80,
  });

  const [expectedDurations] = useState({
    Bear: 12.5,
    Sideways: 6.3,
    Bull: 10.0,
  });

  const [priceData] = useState(() => {
    const data = [];
    let price = 100;
    let regime = 2;
    const today = new Date();

    for (let i = 180; i >= 0; i--) {
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

  const regimeColors = {
    Bear: 'text-red-500',
    Sideways: 'text-slate-400',
    Bull: 'text-green-500',
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-100">Regime Analysis</h1>

      {/* Current State */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Current Regime</h2>
          <div className="text-center py-6">
            <div className="text-4xl font-bold text-green-500 mb-2">Bull</div>
            <div className="text-slate-400">Probability: 80%</div>
          </div>
        </div>

        <div className="card col-span-2">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">State Probabilities</h2>
          <div className="space-y-4">
            {Object.entries(stateProbabilities).map(([state, prob]) => (
              <div key={state} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className={regimeColors[state as keyof typeof regimeColors]}>{state}</span>
                  <span className="text-slate-400">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      state === 'Bull' ? 'bg-green-500' :
                      state === 'Bear' ? 'bg-red-500' : 'bg-slate-500'
                    }`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Transition Matrix and Duration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Transition Matrix</h2>
          <TransitionMatrix matrix={transitionMatrix} />
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Expected Duration</h2>
          <div className="space-y-6 py-4">
            {Object.entries(expectedDurations).map(([state, duration]) => (
              <div key={state} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded ${
                    state === 'Bull' ? 'bg-green-500' :
                    state === 'Bear' ? 'bg-red-500' : 'bg-slate-500'
                  }`} />
                  <span className={regimeColors[state as keyof typeof regimeColors]}>{state}</span>
                </div>
                <div className="text-right">
                  <div className="text-xl font-bold text-slate-200">{duration.toFixed(1)}</div>
                  <div className="text-xs text-slate-500">days avg</div>
                </div>
              </div>
            ))}
          </div>
          <div className="text-xs text-slate-500 mt-4 pt-4 border-t border-slate-700">
            Expected duration = 1 / (1 - P(stay))
          </div>
        </div>
      </div>

      {/* Historical Regimes */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Historical Regime Detection</h2>
        <RegimeChart data={priceData} width={900} height={350} />
      </div>

      {/* Model Details */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">HMM Model Details</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-slate-400 text-sm">States</div>
            <div className="text-xl font-bold text-slate-200">3</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm">Covariance Type</div>
            <div className="text-xl font-bold text-slate-200">Full</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm">Log-Likelihood</div>
            <div className="text-xl font-bold text-slate-200">-1234.5</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm">BIC</div>
            <div className="text-xl font-bold text-slate-200">2521.3</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegimeAnalysis;
