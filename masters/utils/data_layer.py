"""Data loading and preprocessing functions for the Portfolio class.

All public functions accept a ``Portfolio`` instance as the first argument and
mutate ``self.d`` in place, returning ``self`` for method chaining.  Private
helpers are module-level functions that operate on plain data structures.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm
from tqdm import tqdm

from utils.load_data import (
    get_cbr_inflation_data,
    get_rubusd_exchange_rate,
    get_unemployment_data,
    load_multipliers,
    load_pickle_object,
    load_stock_data,
    update_pickle_object,
)
import utils.config as cfg
from utils.logger import Logger

if TYPE_CHECKING:
    from utils.portfolio import Portfolio

log = Logger(__name__)


# ---------------------------------------------------------------------------
# Stock data
# ---------------------------------------------------------------------------


def load_stock_data_fn(
    self: "Portfolio",
    tickers_list: list[str] = None,
    use_backup_data: bool = True,
    update_backup: bool = False,
    backup_path: str = cfg.BACKUP_STOCKS_PATH,
) -> "Portfolio":
    """Loads stock data for the given tickers.

    Args:
        self: Portfolio instance.
        tickers_list: List of tickers. If None, uses self.tickers_list.
        use_backup_data: If True, loads from backup file. Defaults to True.
        update_backup: If True, downloads new data and updates backup. Defaults to False.
        backup_path: Path to the backup file.

    Returns:
        Portfolio: self with self.d['stocks'] populated.
    """
    target_tickers = self.tickers_list if tickers_list is None else tickers_list
    calc_date = pd.to_datetime(self.dt_calc)
    start_date = pd.to_datetime(self.dt_start)

    if use_backup_data and not update_backup:
        if not os.path.isfile(backup_path):
            log.error(f"Stocks backup file not found at {backup_path}")
            return self

        data = load_pickle_object(backup_path)

        if data is not None and not data.empty:
            max_date = pd.to_datetime(data["<DATE>"], format="%Y%m%d").max()
            data = data[
                (pd.to_datetime(data["<DATE>"], format="%Y%m%d") >= start_date)
                & (pd.to_datetime(data["<DATE>"], format="%Y%m%d") <= calc_date)
            ]

            if max_date < calc_date:
                days_gap = (calc_date - max_date).days
                log.warning(
                    f"Stocks backup data incomplete: ends on {max_date.date()}, but calculation "
                    f"date is {self.dt_calc} ({days_gap} days gap). Using available data."
                )
            else:
                log.info(f"Using stocks backup data from {start_date.date()} up to {calc_date.date()}")
        else:
            log.error("Stocks backup file is empty")
            return self
    else:
        log.info(f"Downloading all stock data from {self.dt_start} to {self.dt_calc}")
        data = load_stock_data(
            tickers_list=target_tickers,
            start_date=self.dt_start,
            end_date=self.dt_calc,
            step=self.stocks_step,
        )
        if update_backup:
            update_pickle_object(backup_path, data)
            log.info(f"Stocks backup updated: {backup_path}")

    data = (
        data.assign(date_col=lambda x: pd.to_datetime(x["<DATE>"], format="%Y%m%d"))
        .loc[lambda x: (x["date_col"] >= start_date) & (x["date_col"] <= calc_date)]
        .drop(columns=["date_col"])
        .rename(columns={col: col[1:-1].lower() for col in data.columns})
        .assign(date=lambda x: pd.to_datetime(x["date"]).dt.normalize())
        .assign(quarter=lambda x: pd.to_datetime(x["date"]).dt.quarter)
        .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
        .drop(columns=["per", "vol"])
    )

    min_date = data["date"].min().strftime("%Y-%m-%d")
    max_date = data["date"].max().strftime("%Y-%m-%d")
    period_df = pd.DataFrame([{"Start Date": min_date, "End Date": max_date}])
    log.log_dataframe(period_df, title="Loaded Stock Data Period")
    log.log_missing_values_summary(data, title="Stock Data Missing Values")

    self.d["stocks"] = data
    return self


# ---------------------------------------------------------------------------
# Multipliers data
# ---------------------------------------------------------------------------


def load_multipliers_fn(
    self: "Portfolio",
    tickers_list: list[str] = None,
    use_backup: bool = True,
    update_backup: bool = False,
    backup_path: str = cfg.BACKUP_MULTIPLIERS_PATH,
) -> "Portfolio":
    """Loads multipliers data for the given tickers.

    Args:
        self: Portfolio instance.
        tickers_list: List of tickers. If None, uses self.tickers_list.
        use_backup: If True, loads from backup file. Defaults to True.
        update_backup: If True, updates the backup with new data. Defaults to False.
        backup_path: Path to the backup file.

    Returns:
        Portfolio: self with self.d['multipliers'] populated.
    """
    target_tickers = self.tickers_list if tickers_list is None else tickers_list
    multipliers_df = None
    calc_date = pd.to_datetime(self.dt_calc)
    start_date = pd.to_datetime(self.dt_start)

    if use_backup and os.path.isfile(backup_path):
        multipliers_df = load_pickle_object(backup_path)

        max_date = pd.to_datetime(multipliers_df["date"]).max()

        if max_date < calc_date:
            if update_backup:
                log.warning(
                    f"Backup outdated (Last: {max_date.strftime('%Y-%m-%d')}, "
                    f"Required: {calc_date.strftime('%Y-%m-%d')}). Downloading fresh data..."
                )
                multipliers_df = None
            else:
                log.warning(
                    f"Backup outdated (Last: {max_date.strftime('%Y-%m-%d')}, "
                    f"Required: {calc_date.strftime('%Y-%m-%d')}). Using outdated backup."
                )
        else:
            log.info(f"Using multipliers backup data from {start_date.date()} up to {calc_date.date()}")

    if multipliers_df is None:
        log.info(f"Downloading all multipliers data from {self.dt_start} to {self.dt_calc}")

        multipliers_raw = load_multipliers(companies_list=target_tickers, update_backup=False)

        multipliers_df = (
            pd.melt(
                multipliers_raw,
                id_vars=["company", "characteristic"],
                var_name="year_quarter",
                value_name="value",
            )
            .assign(
                temp_year=lambda x: x["year_quarter"].str.split("_", expand=True)[0].astype(int),
                temp_quarter=lambda x: x["year_quarter"].str.split("_", expand=True)[1].astype(int),
            )
            .assign(
                date=lambda x: pd.to_datetime(
                    x["temp_year"].astype(str) + "-" + (x["temp_quarter"] * 3).astype(str) + "-01"
                )
                + pd.offsets.QuarterEnd(0)
            )
            .assign(year=lambda x: x["date"].dt.year, quarter=lambda x: x["date"].dt.quarter)
            .drop(columns=["year_quarter", "temp_year", "temp_quarter"])
            .set_index(["company", "date", "year", "quarter", "characteristic"])["value"]
            .unstack()
            .reset_index()
            .rename(columns={"company": "ticker"})
        )

        update_pickle_object(backup_path, multipliers_df)
        log.info(f"Multipliers backup updated: {backup_path}")

    if multipliers_df is not None and not multipliers_df.empty:
        multipliers_df = multipliers_df[
            (multipliers_df["date"] >= start_date) & (multipliers_df["date"] <= calc_date)
        ].copy()

        if not multipliers_df.empty:
            min_date = multipliers_df["date"].min().strftime("%Y-%m-%d")
            max_date = multipliers_df["date"].max().strftime("%Y-%m-%d")
            period_df = pd.DataFrame([{"Start Date": min_date, "End Date": max_date}])
            log.log_dataframe(period_df, title="Loaded Multipliers Data Period")

            if max_date < calc_date.strftime("%Y-%m-%d"):
                log.warning(
                    f"Loaded multipliers data ends at {max_date}, but calculation date requires {self.dt_calc}."
                )
        else:
            log.error(f"No multipliers data found for the period {self.dt_start} - {self.dt_calc}")

    self.d["multipliers"] = multipliers_df
    log.log_missing_values_summary(self.d["multipliers"], title="Multipliers Data Missing Values")
    return self


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------


def _log_data_period(df: pd.DataFrame, date_col: str, title: str) -> None:
    """Logs the start and end dates of a dataframe."""
    if df is not None and not df.empty and date_col in df.columns:
        min_date = pd.to_datetime(df[date_col]).min().strftime("%Y-%m-%d")
        max_date = pd.to_datetime(df[date_col]).max().strftime("%Y-%m-%d")
        period_df = pd.DataFrame([{"Start Date": min_date, "End Date": max_date}])
        log.log_dataframe(period_df, title=title)
    else:
        log.warning(f"Could not log period for {title}: DataFrame is empty or missing '{date_col}' column.")


def load_macro_data_fn(
    self: "Portfolio",
    update_inflation: bool = False,
    update_rub_usd: bool = False,
    update_unemployment: bool = False,
    inflation_path: str = cfg.MACRO_INFLATION_PATH,
    rub_usd_path: str = cfg.MACRO_RUBUSD_PATH,
    unemployment_path: str = cfg.MACRO_UNEMPLOYMENT_PATH,
) -> "Portfolio":
    """Adds macroeconomic data to the portfolio.

    Args:
        self: Portfolio instance.
        update_inflation: If True, downloads fresh inflation data from CBR. Defaults to False.
        update_rub_usd: If True, downloads fresh USD/RUB exchange rate data. Defaults to False.
        update_unemployment: If True, downloads fresh unemployment data. Defaults to False.
        inflation_path: Path to inflation data file.
        rub_usd_path: Path to USD/RUB exchange rate file.
        unemployment_path: Path to unemployment data file.

    Returns:
        Portfolio: self with macro data populated in self.d.
    """
    calc_date = pd.to_datetime(self.dt_calc)
    start_date = pd.to_datetime(self.dt_start)

    # 1. Unemployment
    if update_unemployment or not os.path.isfile(unemployment_path):
        unemployment = get_unemployment_data(unemployment_path, update_backup=update_unemployment)
        log.info("Downloaded fresh unemployment data" + (" and updated backup" if update_unemployment else ""))
    else:
        unemployment = pd.read_excel(unemployment_path)
        log.info("Loaded Unemployment data from backup")

    self.d["macro_unemployment"] = (
        unemployment.rename(columns={"Unemployment": "unemployment_rate", "Date": "date"})
        .assign(dtReportLast=lambda x: (pd.to_datetime(x["date"]) + pd.offsets.MonthEnd(0)).dt.normalize())
        .assign(unemployment_rate=lambda x: x.unemployment_rate / 100)
        .loc[
            lambda x: (x["dtReportLast"] >= start_date) & (x["dtReportLast"] <= calc_date),
            ["dtReportLast", "unemployment_rate"],
        ]
    )
    _log_data_period(self.d["macro_unemployment"], "dtReportLast", "Loaded Unemployment Data Period")

    # 2. Inflation & Interest Rate
    if update_inflation or not os.path.isfile(inflation_path):
        inflation = get_cbr_inflation_data(inflation_path, self.dt_start, self.dt_calc, update_backup=update_inflation)
        log.info("Downloaded fresh inflation data" + (" and updated backup" if update_inflation else ""))
    else:
        inflation = pd.read_excel(inflation_path)
        log.info("Loaded inflation data from backup")

    self.d["macro_inflation"] = (
        inflation.assign(dtReportLast=lambda x: (pd.to_datetime(x["Дата"]) + pd.offsets.MonthEnd(0)).dt.normalize())
        .rename(columns={"Ключевая ставка, % годовых": "interest_rate", "Инфляция, % г/г": "inflation"})
        .assign(interest_rate=lambda x: x["interest_rate"] / 100, inflation=lambda x: x["inflation"] / 100)
        .loc[
            lambda x: (x["dtReportLast"] >= start_date) & (x["dtReportLast"] <= calc_date),
            ["dtReportLast", "interest_rate", "inflation"],
        ]
    )
    _log_data_period(self.d["macro_inflation"], "dtReportLast", "Loaded Inflation Data Period")

    # 3. USD/RUB Exchange Rate
    if update_rub_usd or not os.path.isfile(rub_usd_path):
        rub_usd = get_rubusd_exchange_rate(dt_calc=self.dt_calc, dt_start=self.dt_start, update_backup=update_rub_usd)
        log.info("Downloaded fresh USD/RUB exchange rate" + (" and updated backup" if update_rub_usd else ""))
    else:
        rub_usd = pd.read_csv(rub_usd_path)
        log.info("Loaded USD/RUB exchange rate from backup")

    self.d["macro_rub_usd"] = rub_usd.assign(date=lambda x: pd.to_datetime(x["date"]).dt.normalize()).loc[
        lambda x: (x["date"] >= start_date) & (x["date"] <= calc_date), ["date", "rubusd_exchange_rate"]
    ]
    _log_data_period(self.d["macro_rub_usd"], "date", "Loaded USD/RUB Exchange Rate Period")

    return self


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def create_portfolio_fn(self: "Portfolio") -> "Portfolio":
    """Merges stocks, multipliers, and macro data into self.d['portfolio'].

    Args:
        self: Portfolio instance.

    Returns:
        Portfolio: self with self.d['portfolio'] populated.
    """
    self.d["portfolio"] = (
        self.d["stocks"]
        .copy()
        .assign(dtReportLast=lambda x: (x["date"] + pd.offsets.MonthEnd(0)).dt.normalize())
        .merge(self.d["multipliers"].drop(columns=["date"]), on=["ticker", "year", "quarter"], how="left")
        .merge(self.d["macro_inflation"], on="dtReportLast", how="left")
        .merge(self.d["macro_unemployment"], on="dtReportLast", how="left")
        .merge(self.d["macro_rub_usd"], on="date", how="left")
        .drop(columns=["EV/EBITDA", "P/BV", "P/S", "Долг/EBITDA", "P/FCF", "time"])
    )

    columns_new_names = {"Капитализация, млрд руб": "capitalization"}

    column_to_adjust = [
        "Долг, млрд руб",
        "Капитализация, млрд руб",
        "P/E",
        "open",
        "high",
        "close",
        "Чистый долг, млрд руб",
        "low",
    ]

    for col in column_to_adjust:
        if col in self.d["portfolio"].columns:
            if self.d["portfolio"][col].dtype == "object":
                self.d["portfolio"][col] = self.d["portfolio"][col].str.replace(" ", "", regex=False)
            self.d["portfolio"][col] = pd.to_numeric(self.d["portfolio"][col], errors="coerce")

            if col in ["Капитализация, млрд руб", "Долг, млрд руб"]:
                self.d["portfolio"][col] = self.d["portfolio"][col].replace(0, np.nan)

            if "млрд руб" in col:
                self.d["portfolio"][col] *= cfg.BILLION

    self.d["portfolio"]["debt"] = np.select(
        [
            (self.d["portfolio"]["Долг, млрд руб"].notna()) & (self.d["portfolio"]["Долг, млрд руб"].ne(0)),
            (self.d["portfolio"]["Долг, млрд руб"].isna())
            & (
                (self.d["portfolio"]["Чистый долг, млрд руб"].isna())
                | (self.d["portfolio"]["Чистый долг, млрд руб"].eq(0))
            ),
            (self.d["portfolio"]["Долг, млрд руб"].isna())
            & (self.d["portfolio"]["Чистый долг, млрд руб"].notna())
            & (self.d["portfolio"]["Чистый долг, млрд руб"].ne(0)),
        ],
        [
            self.d["portfolio"]["Долг, млрд руб"],
            self.d["portfolio"]["Долг, млрд руб"],
            self.d["portfolio"]["Чистый долг, млрд руб"],
        ],
    )

    log.log_missing_values_summary(self.d["portfolio"], title="Portfolio Missing Values Before Filling")

    self.d["portfolio"] = (
        self.d["portfolio"]
        .assign(debt=lambda x: np.abs(x["debt"]))
        .assign(debt=lambda x: x.groupby("ticker")["debt"].transform(lambda g: g.ffill().bfill()))
        .assign(debt=lambda x: x["debt"].fillna(0))
        .assign(inflation=lambda x: x["inflation"].ffill().bfill())
        .assign(unemployment_rate=lambda x: x["unemployment_rate"].ffill().bfill())
        .rename(columns=columns_new_names)
        .assign(capitalization=lambda x: x.groupby("ticker")["capitalization"].transform(lambda g: g.ffill().bfill()))
        .assign(capitalization=lambda x: x["capitalization"].fillna(0))
    )

    self.d["portfolio"] = self.d["portfolio"].sort_values(["ticker", "date"])
    num_rows = len(self.d["portfolio"])
    portfolio_info = pd.DataFrame(
        [
            {
                "Total Rows": num_rows,
                "Unique Companies": len(self.d["portfolio"].ticker.unique()),
                "Date Range": (
                    f"{self.d['portfolio']['date'].min().strftime('%Y-%m-%d')} to"
                    f" {self.d['portfolio']['date'].max().strftime('%Y-%m-%d')}"
                ),
            }
        ]
    )

    log.log_missing_values_summary(self.d["portfolio"], title="Portfolio Missing Values After Filling")
    log.log_dataframe(portfolio_info, title="Portfolio Dimensions")
    return self


# ---------------------------------------------------------------------------
# Dynamic features
# ---------------------------------------------------------------------------


def add_dynamic_features_fn(self: "Portfolio") -> "Portfolio":
    """Adds EWMA annualized volatility and log-returns to self.d['portfolio'].

    Args:
        self: Portfolio instance.

    Returns:
        Portfolio: self with 'volatility' column added.
    """
    df = self.d["portfolio"].sort_values(["ticker", "date"]).copy()

    df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))

    df["volatility"] = df.groupby("ticker")["log_return"].transform(
        lambda x: x.ewm(span=cfg.ROLLING_VOL_WINDOW, min_periods=cfg.TRADING_DAYS_PER_MONTH).std()
    ) * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)

    df["volatility"] = df.groupby("ticker")["volatility"].transform(lambda x: x.bfill())
    global_avg_vol = df["volatility"].mean()
    df["volatility"] = df["volatility"].fillna(global_avg_vol).fillna(cfg.DEFAULT_VOLATILITY)

    df = df.drop(columns=["log_return"])
    self.d["portfolio"] = df

    log.log_missing_values_summary(self.d["portfolio"], title="Portfolio Missing Values After Adding dynamic features")
    return self


# ---------------------------------------------------------------------------
# Merton model
# ---------------------------------------------------------------------------


def _solve_merton_vectorized_fn(self: "Portfolio", T: float = 1) -> "Portfolio":
    """Solves Merton system of equations to estimate asset value V and sigma_V.

    Args:
        self: Portfolio instance.
        T: Time horizon in years. Defaults to 1.

    Returns:
        Portfolio: self with 'V' and 'sigma_V' columns in self.d['portfolio'].
    """
    E = self.d["portfolio"]["capitalization"].values.astype(float)
    D = self.d["portfolio"]["debt"].values.astype(float)
    sigma_E = self.d["portfolio"]["volatility"].values.astype(float)

    def equations(vars, E_i, D_i, r_i, sigma_E_i, T_i):
        V, sigma_V = vars
        d1 = np.log(V / D_i if D_i != 0 else cfg.EPSILON) + (r_i + 0.5 * sigma_V**2) * T_i
        d1 /= sigma_V * np.sqrt(T_i)
        N_d1 = norm.cdf(d1)
        eq1 = V * N_d1 - D_i * np.exp(-r_i * T_i) * norm.cdf(d1 - sigma_V * np.sqrt(T_i)) - E_i
        eq2 = N_d1 * sigma_V * V - sigma_E_i * E_i
        return [eq1, eq2]

    initial_guess = np.vstack([E + D, sigma_E]).T
    log.info(f"Starting Merton model calculations for {len(initial_guess)} rows...")
    results = np.array(
        [
            root(
                equations,
                guess,
                args=(E[i], D[i], self.d["portfolio"]["interest_rate"][i], sigma_E[i], T),
            ).x
            for i, guess in enumerate(tqdm(initial_guess, desc="Solving Merton equations"))
        ]
    )

    self.d["portfolio"]["V"] = np.where(results[:, 0] <= 0, cfg.EPSILON, results[:, 0])
    self.d["portfolio"]["sigma_V"] = results[:, 1]
    log.info("Capital cost and capital volatility calculated.")
    return self


def _merton_pd_fn(self: "Portfolio", T: float = 1) -> "Portfolio":
    """Calculates PD and DD from solved Merton V and sigma_V.

    Args:
        self: Portfolio instance.
        T: Time horizon for the default event in years. Defaults to 1.

    Returns:
        Portfolio: self with 'PD' and 'DD' columns in self.d['portfolio'].
    """
    V = self.d["portfolio"]["V"].values.astype(float)
    D = self.d["portfolio"]["debt"].values.astype(float)
    sigma_V = self.d["portfolio"]["sigma_V"]

    d2 = (
        np.log(V / np.where(D != 0, D, cfg.EPSILON)) + (self.d["portfolio"]["interest_rate"] - 0.5 * sigma_V**2) * T
    ) / (sigma_V * np.sqrt(T))
    self.d["portfolio"]["PD"] = norm.cdf(-d2)
    self.d["portfolio"]["DD"] = d2
    log.info("Merton's probabilities of default and distance to default calculated.")
    return self


def add_merton_pd_fn(self: "Portfolio") -> "Portfolio":
    """Computes PD and DD via Merton model and adds them to the portfolio.

    Args:
        self: Portfolio instance.

    Returns:
        Portfolio: self with 'PD' and 'DD' columns added and V/sigma_V removed.
    """
    self = _solve_merton_vectorized_fn(self)
    self = _merton_pd_fn(self)
    self.d["portfolio"] = self.d["portfolio"].drop(columns=["V", "sigma_V"])
    return self
