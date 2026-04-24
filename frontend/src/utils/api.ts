import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface RegimeData {
  current_state: string;
  state_probabilities: Record<string, number>;
  expected_duration: Record<string, number>;
  transition_matrix: number[][];
}

export interface PortfolioData {
  weights: Record<string, number>;
  expected_return: number;
  volatility: number;
  sharpe_ratio: number;
  regime: string;
  var_95: number;
  cvar_95: number;
}

export interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  max_drawdown: number;
  volatility: number;
  annualized_volatility: number;
}

export interface BacktestResult {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  trades: number;
}

export const fetchRegime = async (): Promise<RegimeData> => {
  const response = await api.get('/regime/current');
  return response.data;
};

export const optimizePortfolio = async (
  tickers: string[],
  method: string = 'max_sharpe',
  regime?: number
): Promise<PortfolioData> => {
  const response = await api.post('/portfolio/optimize', {
    tickers,
    method,
    current_regime: regime,
  });
  return response.data;
};

export const fetchRiskMetrics = async (
  tickers: string[],
  weights?: number[]
): Promise<RiskMetrics> => {
  const response = await api.post('/risk/calculate', {
    tickers,
    weights,
  });
  return response.data;
};

export const runBacktest = async (
  tickers: string[],
  startDate: string,
  endDate: string,
  strategy: string = 'regime_aware'
): Promise<BacktestResult> => {
  const response = await api.post('/backtest/run', {
    tickers,
    start_date: startDate,
    end_date: endDate,
    strategy,
  });
  return response.data;
};

export const fetchPriceHistory = async (
  tickers: string[],
  startDate?: string,
  endDate?: string
): Promise<any> => {
  const response = await api.get('/data/prices', {
    params: { tickers: tickers.join(','), start_date: startDate, end_date: endDate },
  });
  return response.data;
};

export const fetchCointegrationPairs = async (
  tickers: string[]
): Promise<any[]> => {
  const response = await api.post('/cointegration/find', { tickers });
  return response.data;
};

export const fetchModelComparison = async (): Promise<any> => {
  const response = await api.get('/models/comparison');
  return response.data;
};

export default api;
