"""Project-wide configuration constants.

All magic numbers, default values, and file paths used across the codebase
are declared here. Import this module as ``import utils.config as cfg``
and reference every constant via the ``cfg.`` prefix.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# Number of trading days in a calendar year; used as annualisation factor
# for volatility, covariance matrices, and expected loss scaling.
TRADING_DAYS_PER_YEAR: int = 252

# Approximate number of trading days in one calendar month (21 business days);
# used as the minimum look-back period in EWMA volatility estimation.
TRADING_DAYS_PER_MONTH: int = 21

# Number of calendar months in a year; used to convert monthly to annual
# rates and in walk-forward backtest horizon calculations.
MONTHS_PER_YEAR: int = 12

# Conversion factor from raw financial statement units to billions
# (e.g. total debt reported in thousands → billions of RUB).
BILLION: float = 1e9

# Small positive floor added to denominators and log arguments to prevent
# division-by-zero and log(0) numerical errors throughout all modules.
EPSILON: float = 1e-6

# Default Loss Given Default (LGD) ratio assumed when no application-level
# LGD is provided; represents the fraction of exposure lost upon default.
DEFAULT_LGD: float = 0.4

# Default risk-aversion coefficient λ for the mean-EL portfolio objective:
# J(w) = λ·vol(w) + (1−λ)·EL(w).  A value of 0.5 balances both equally.
DEFAULT_LAMBDA_RISK: float = 0.5

# Fallback annualised volatility (40%) applied to a ticker when the EWMA
# estimator cannot produce a valid result (e.g. insufficient history).
DEFAULT_VOLATILITY: float = 0.4

# EWM span in trading days for historical volatility estimation.
# 63 trading days ≈ 3 calendar months, giving a responsive but stable estimate.
ROLLING_VOL_WINDOW: int = 63

# ---------------------------------------------------------------------------
# Macroeconomic factor columns
# ---------------------------------------------------------------------------

# Names of the macroeconomic factor columns present in the portfolio DataFrame
# and used in all credit-risk regression and VAR/SARIMAX/Prophet models.
MACRO_COLS: list[str] = ["inflation", "interest_rate", "unemployment_rate", "rubusd_exchange_rate"]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

# Pickle cache for raw OHLCV stock price data downloaded from Finam.
# Used to avoid repeated API calls during development.
BACKUP_STOCKS_PATH: str = "data/backup/stocks.pkl"

# Pickle cache for financial multipliers (P/E, Debt/Equity, EV/EBITDA, etc.)
# per ticker, sourced from Smart-Lab and saved locally.
BACKUP_MULTIPLIERS_PATH: str = "data/backup/multipliers.pkl"

# Directory containing per-company financial multiplier CSV files
# (one file per ticker, e.g. data/multiplicators/GAZP.csv).
MULTIPLICATORS_DIR: str = "data/multiplicators"

# CSV file with historical RUB/USD exchange rate data fetched from the CBR.
# Updated automatically when new dates are requested; cached locally.
MACRO_RUBUSD_PATH: str = "data/macro/rubusd.csv"

# Excel file with monthly CPI (inflation) data loaded from the CBR website.
# This file is read to build the inflation macroeconomic factor series.
MACRO_INFLATION_PATH: str = "data/macro/inflation.xlsx"

# Excel file with monthly unemployment rate data (Rosstat / CBR).
# Used as macroeconomic factor input for credit risk models.
MACRO_UNEMPLOYMENT_PATH: str = "data/macro/unemployment.xlsx"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

# Root directory where all auto-generated plot images (PNG files) are saved.
# Created automatically on first write if it does not exist.
GRAPHS_DIR: Path = Path("logs/graphs")

# Default save path for the macroeconomic factor significance summary chart
# (bar chart of OLS/Granger p-values across tickers and macro variables).
MACRO_SIGNIFICANCE_PLOT_PATH: str = "logs/graphs/macro_significance_summary.png"
