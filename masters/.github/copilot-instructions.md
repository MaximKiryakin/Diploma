# Copilot Instructions: Credit Portfolio Management Research Project

## Project Overview
This is an academic research project on **credit portfolio management modeling with macroeconomic factors** for Russian banks. It implements the Merton model for probability of default (PD) calculation and portfolio optimization with macroeconomic shock analysis.

## Core Architecture

### Main Components
- **`utils/portfolio.py`**: Central `Portfolio` class - the main computational engine
- **`utils/load_data.py`**: Data fetching from Finam API and Central Bank of Russia  
- **`utils/logger.py`**: Shared logging with duplicate prevention (critical for Jupyter)
- **`main.ipynb`**: Primary research notebook with method chaining workflow
- **`portfolio_management_example.py`**: Standalone demo for portfolio optimization features

### Data Structure
```
data/
├── multiplicators/    # Financial ratios per company (P/E, debt, etc.)
├── macro/            # Macroeconomic indicators (inflation, USD/RUB, etc.)
└── backup/           # Cached market data to avoid API limits
```

## Key Development Patterns

### Method Chaining Workflow
The `Portfolio` class uses fluent interface for data pipeline:
```python
calc = (Portfolio(dt_calc='2025-05-31', dt_start='2019-11-03', ...)
    .load_stock_data(use_backup_data=True)
    .add_macro_data()
    .add_merton_pd()
    .optimize_portfolio_weights(method='max_sharpe')
    .plot_optimal_weights())
```

### Logging Anti-Pattern Prevention
**CRITICAL**: The logger prevents handler duplication in Jupyter notebooks:
- Uses `Logger._configured_loggers` class tracking
- Sets `logger.propagate = False` to prevent double output
- Console handler uses `sys.stdnull` to avoid Jupyter conflicts

### Russian Financial Data Integration
- **Tickers**: Russian stocks (GAZP, SBER, LKOH, etc.) via `utils/LabelsDict.py`
- **Finam API**: Historical price data with bucket-based pagination
- **CBR API**: Official exchange rates and macroeconomic data via `pycbrf`

### Sector-Based Analysis
Companies grouped by sectors for risk analysis:
```python
# Oil & Gas: GAZP, LKOH, ROSN
# Banking: SBER, VTBR, MOEX  
# Metals: GMKN, NLMK, RUAL
# Telecom: MTSS, RTKM, TTLK
# Retail: MGNT, LNTA, FESH
```

## Development Workflows

### Data Loading Strategy
1. **Backup-first approach**: Always use `use_backup_data=True` for development
2. **API rate limiting**: Use small `bucket_size` for intraday data
3. **Date formats**: Finam uses `DD.MM.YYYY`, internal processing uses `YYYY-MM-DD`

### Model Calculation Pipeline
```python
# Standard research workflow:
.add_merton_pd()                    # Calculate probability of default
.calc_irf()                         # Impulse response functions  
.plot_correlation_matrix()          # Cross-sector correlations
.calc_macro_connections()           # Regression analysis
```

### Portfolio Optimization Methods
- `method='min_variance'`: Minimum risk portfolio
- `method='max_sharpe'`: Maximum Sharpe ratio (requires `risk_free_rate`)
- `method='risk_parity'`: Equal risk contribution

### Jupyter Notebook Patterns
- **Module reloading**: Use Cell #2 pattern to prevent logging duplication
- **Logger reset**: `Logger._configured_loggers.clear()` before imports
- **Commented pipelines**: Large method chains with extensive comments for selective execution

## Key Dependencies & Integration Points

### External APIs
- **Finam Export API**: Real-time and historical Russian stock data
- **pycbrf**: Central Bank of Russia official data (exchange rates, rates)
- **Financial data**: IFRS multipliers stored in CSV format per company

### Statistical Methods
- **VAR models**: `statsmodels.tsa.api.VAR` for impulse response analysis
- **Ridge regression**: Macroeconomic factor significance testing  
- **Merton model**: Custom implementation for PD calculation using `scipy.optimize`

### Visualization Conventions
- Graphs saved to `logs/graphs/` with descriptive names
- Russian labels and sector color coding
- Correlation heatmaps with custom ordering by sector

## Common Tasks

### Adding New Companies
1. Add ticker to `utils/LabelsDict.py` 
2. Create multiplier CSV in `data/multiplicators/{TICKER}.csv`
3. Update sector groupings in analysis methods

### Running Full Analysis
```bash
# Complete research pipeline
python portfolio_management_example.py

# Or use main.ipynb with method chaining
```

### Debugging Logging Issues
If you see duplicate log output, check:
- Logger handler count: `len(logger.handlers)`
- Reset before imports: `Logger._configured_loggers.clear()`
- Use Cell #2 reload pattern in notebooks

### Working with Financial Data
- **Quarterly data**: Format as `{year}_{quarter}` (e.g., `2024_1`)
- **Missing values**: Use `.fill_missing_values()` before calculations
- **Date alignment**: Always use ISO format internally (`YYYY-MM-DD`)

## Research-Specific Notes
This project analyzes **Russian market systemic risks** during 2019-2025 period, focusing on:
- Macroeconomic shock transmission (inflation, USD/RUB, unemployment)  
- Sector-level credit risk differences
- Portfolio optimization under macroeconomic stress
- Academic presentation with LaTeX integration (`text/` directory)