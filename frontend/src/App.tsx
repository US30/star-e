import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import RegimeAnalysis from './pages/RegimeAnalysis';
import RiskDashboard from './pages/RiskDashboard';
import ModelComparison from './pages/ModelComparison';
import Portfolio from './pages/Portfolio';

const NavLink: React.FC<{ to: string; children: React.ReactNode }> = ({ to, children }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`nav-link ${isActive ? 'active' : ''}`}
    >
      {children}
    </Link>
  );
};

const Navigation: React.FC = () => {
  return (
    <nav className="bg-slate-900 border-b border-slate-800 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-2xl font-bold text-blue-500">StAR-E</span>
          <span className="text-slate-400 text-sm">Statistical Arbitrage & Risk Engine</span>
        </div>
        <div className="flex items-center space-x-2">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/regime">Regime</NavLink>
          <NavLink to="/portfolio">Portfolio</NavLink>
          <NavLink to="/risk">Risk</NavLink>
          <NavLink to="/models">Models</NavLink>
        </div>
      </div>
    </nav>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-slate-900">
        <Navigation />
        <main className="max-w-7xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/regime" element={<RegimeAnalysis />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/risk" element={<RiskDashboard />} />
            <Route path="/models" element={<ModelComparison />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
