# Contributing to StAR-E

Thank you for your interest in contributing to StAR-E!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/star-e.git
   cd star-e
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=star_e --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/star_e
```

## Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

## Commit Messages

Follow the conventional commits format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Build process or auxiliary tool changes

Example:
```
feat: add Monte Carlo VaR simulation

- Implement multivariate normal sampling
- Add stress testing scenarios
- Include path-dependent analysis
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Test results
   - Screenshots (for UI changes)

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for all public functions
- Keep functions focused and small
- Add comments for complex logic

## Adding New Models

When adding a new forecasting model:

1. Inherit from `BaseForecaster` in `src/star_e/models/base.py`
2. Implement `fit()` and `forecast()` methods
3. Add MLflow logging
4. Write unit tests
5. Update documentation

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Documentation improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
