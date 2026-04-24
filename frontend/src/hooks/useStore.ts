import { create } from 'zustand';

interface RegimeState {
  currentRegime: string;
  stateProbabilities: Record<string, number>;
  transitionMatrix: number[][];
  expectedDurations: Record<string, number>;
}

interface PortfolioState {
  weights: Record<string, number>;
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
}

interface RiskState {
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  maxDrawdown: number;
}

interface AppState {
  selectedTickers: string[];
  dateRange: { start: string; end: string };
  regime: RegimeState | null;
  portfolio: PortfolioState | null;
  risk: RiskState | null;
  isLoading: boolean;
  error: string | null;

  setSelectedTickers: (tickers: string[]) => void;
  setDateRange: (start: string, end: string) => void;
  setRegime: (regime: RegimeState) => void;
  setPortfolio: (portfolio: PortfolioState) => void;
  setRisk: (risk: RiskState) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useStore = create<AppState>((set) => ({
  selectedTickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
  dateRange: {
    start: '2022-01-01',
    end: new Date().toISOString().split('T')[0],
  },
  regime: null,
  portfolio: null,
  risk: null,
  isLoading: false,
  error: null,

  setSelectedTickers: (tickers) => set({ selectedTickers: tickers }),
  setDateRange: (start, end) => set({ dateRange: { start, end } }),
  setRegime: (regime) => set({ regime }),
  setPortfolio: (portfolio) => set({ portfolio }),
  setRisk: (risk) => set({ risk }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
