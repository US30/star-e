# StAR-E React Dashboard

Interactive data visualization dashboard built with React, TypeScript, and D3.js.

## Features

- **Real-time Portfolio Monitoring**: Track performance metrics and allocation
- **Regime Analysis**: Visualize HMM state transitions and probabilities
- **Risk Dashboard**: Interactive VaR/CVaR distributions and stress tests
- **Model Comparison**: Compare forecasting models side-by-side
- **D3.js Visualizations**: Custom interactive charts
  - Regime overlay on price charts
  - Transition matrix heatmaps
  - Portfolio allocation pie charts
  - VaR distribution histograms
  - Correlation heatmaps

## Tech Stack

- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **D3.js 7**: Data visualization
- **Tailwind CSS**: Styling
- **Zustand**: State management
- **Axios**: API communication
- **React Router**: Navigation

## Getting Started

### Prerequisites

- Node.js 20+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will open at http://localhost:3000

### Build for Production

```bash
# Create optimized build
npm run build

# Test production build locally
npx serve -s build
```

## Project Structure

```
frontend/
├── public/
│   └── index.html              # HTML template
├── src/
│   ├── components/             # Reusable components
│   │   ├── RegimeChart.tsx     # D3 regime overlay
│   │   ├── TransitionMatrix.tsx # HMM transition viz
│   │   ├── AllocationPie.tsx   # Portfolio pie chart
│   │   ├── VaRChart.tsx        # Risk distribution
│   │   └── CorrelationHeatmap.tsx # Asset correlation
│   ├── pages/                  # Main pages
│   │   ├── Dashboard.tsx       # Overview page
│   │   ├── RegimeAnalysis.tsx  # Regime details
│   │   ├── Portfolio.tsx       # Portfolio management
│   │   ├── RiskDashboard.tsx   # Risk analysis
│   │   └── ModelComparison.tsx # Model metrics
│   ├── hooks/
│   │   └── useStore.ts         # Zustand state
│   ├── utils/
│   │   └── api.ts              # API client
│   ├── App.tsx                 # Main app component
│   └── index.tsx               # Entry point
├── package.json
├── tsconfig.json
└── tailwind.config.js
```

## Available Scripts

```bash
# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build

# Eject (one-way operation)
npm run eject
```

## API Integration

The dashboard connects to the FastAPI backend at http://localhost:8000 by default.

Configure the API URL via environment variable:

```bash
# .env.local
REACT_APP_API_URL=http://localhost:8000
```

### API Endpoints Used

- `GET /regime/current` - Current market regime
- `POST /portfolio/optimize` - Optimize portfolio
- `POST /risk/calculate` - Calculate risk metrics
- `GET /data/prices` - Historical prices
- `GET /models/comparison` - Model performance

## Customization

### Adding New Visualizations

Create a new D3.js component:

```typescript
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface MyChartProps {
  data: any[];
  width?: number;
  height?: number;
}

const MyChart: React.FC<MyChartProps> = ({ data, width = 600, height = 400 }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    // D3 visualization code here
  }, [data, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default MyChart;
```

### Adding New Pages

1. Create component in `src/pages/`
2. Add route in `App.tsx`:

```typescript
<Route path="/new-page" element={<NewPage />} />
```

3. Add navigation link:

```typescript
<NavLink to="/new-page">New Page</NavLink>
```

## Styling

### Tailwind CSS Classes

The project uses custom color scheme:

```css
--color-bull: #22c55e    /* Green for bull market */
--color-bear: #ef4444    /* Red for bear market */
--color-sideways: #64748b /* Gray for sideways */
```

Common classes:
- `.card` - Card container
- `.metric-card` - Metric display
- `.regime-bull/bear/sideways` - Regime badges
- `.btn-primary/secondary` - Buttons

## Performance Optimization

### Code Splitting

Routes are automatically code-split using React.lazy:

```typescript
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
```

### Memoization

Use React.memo for expensive components:

```typescript
const ExpensiveChart = React.memo(({ data }) => {
  // Render logic
});
```

### D3 Performance

- Limit SVG elements for large datasets
- Use canvas for >1000 points
- Debounce resize handlers

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test
npm test MyComponent.test.tsx
```

## Deployment

### Docker

```bash
# Build Docker image
docker build -t star-e-frontend .

# Run container
docker run -p 3000:3000 star-e-frontend
```

### Static Hosting

Deploy the `build/` directory to:
- Vercel
- Netlify
- GitHub Pages
- AWS S3 + CloudFront

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari 14+

## Troubleshooting

### D3.js TypeScript errors

Install type definitions:
```bash
npm install --save-dev @types/d3
```

### CORS errors

Configure API CORS settings or use proxy in `package.json`:
```json
"proxy": "http://localhost:8000"
```

### Build fails

Clear cache and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the root directory.

## License

MIT License - see [LICENSE](../LICENSE)
