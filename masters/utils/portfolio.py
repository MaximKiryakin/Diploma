from utils.load_data import (
    load_stock_data,
    load_multipliers,
    get_rubusd_exchange_rate,
    get_cbr_inflation_data,
    get_unemployment_data,
    load_pickle_object,
    update_pickle_object,
)
from utils.logger import Logger
from typing import List, Tuple, Dict, Union
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import os
from scipy.optimize import root, minimize, linprog
from scipy.stats import norm
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import utils.plots as plots


log = Logger(__name__)


class Portfolio:
    # --- Constants ---
    TRADING_DAYS_PER_YEAR: int = 252
    TRADING_DAYS_PER_MONTH: int = 21
    MONTHS_PER_YEAR: int = 12
    BILLION: float = 1e9
    EPSILON: float = 1e-6
    DEFAULT_LGD: float = 0.4
    DEFAULT_LAMBDA_RISK: float = 0.5
    DEFAULT_VOLATILITY: float = 0.4
    ROLLING_VOL_WINDOW: int = 63

    def __init__(self, dt_calc: str, dt_start: str, stocks_step: int, tickers_list: list[str]):
        self.dt_calc = dt_calc
        self.dt_start = dt_start
        self.stocks_step = stocks_step
        self.tickers_list = tickers_list

        # Dictionary to store all dataframes
        self.d = {"stocks": None, "multipliers": None, "portfolio": None, "macro_connection_summary": None}

        self.end_time = None
        self.start_time = None

    def log_system_info(self):
        """
        Logs system information.

        Returns:
            Portfolio: Updated portfolio with logged system information.
        """
        self.start_time = datetime.now()

        # Ensure propagation is disabled to prevent root logger interference
        log.propagate = False

        # Log Configuration Parameters
        params = [
            {"Parameter": "Calculation Date", "Value": self.dt_calc},
            {"Parameter": "Start Date", "Value": self.dt_start},
            {"Parameter": "Stocks Step", "Value": self.stocks_step},
            {"Parameter": "Tickers Count", "Value": len(self.tickers_list)},
            {"Parameter": "Tickers", "Value": ", ".join(self.tickers_list)},
        ]
        df_params = pd.DataFrame(params)
        log.log_dataframe(df_params, title="Configuration Parameters")

        return self

    def load_stock_data(
        self,
        tickers_list: list[str] = None,
        use_backup_data: bool = True,
        update_backup: bool = False,
        backup_path: str = "data/backup/stocks.pkl",
    ) -> "Portfolio":
        """
        Loads stock data for the given tickers.

        Args:
            tickers_list (list[str], optional): List of tickers. If not specified, uses the default tickers list.
            use_backup_data (bool, optional): If True, loads stock data from backup file only. Defaults to True.
            update_backup (bool, optional): If True, downloads new data and updates backup. Defaults to False.
            backup_path (str, optional): Path to the backup file. Defaults to "data/backup/stocks.pkl".

        Returns:
            Portfolio: Updated portfolio with loaded stock data.
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

    def load_multipliers(
        self,
        tickers_list: list[str] = None,
        use_backup: bool = True,
        update_backup: bool = False,
        backup_path: str = "data/backup/multipliers.pkl",
    ) -> "Portfolio":
        """
        Loads multipliers data for the given tickers.

        Args:
            tickers_list (list[str], optional): List of tickers. If not specified, uses the default tickers list.
            use_backup (bool, optional): If True, loads multipliers from backup file. Defaults to True.
            update_backup (bool, optional): If True, updates the backup file with new data. Defaults to False.
            backup_path (str, optional): Path to the backup file. Defaults to "data/backup/multipliers.pkl".

        Returns:
            Portfolio: Updated portfolio with loaded multipliers data.
        """

        target_tickers = self.tickers_list if tickers_list is None else tickers_list
        multipliers_df = None
        calc_date = pd.to_datetime(self.dt_calc)
        start_date = pd.to_datetime(self.dt_start)

        if use_backup and os.path.isfile(backup_path):
            multipliers_df = load_pickle_object(backup_path)

            max_date = pd.to_datetime(multipliers_df["date"]).max()
            min_date = pd.to_datetime(multipliers_df["date"]).min()

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

    def _log_data_period(self, df: pd.DataFrame, date_col: str, title: str):
        """
        Helper method to log the start and end dates of a dataframe.
        """
        if df is not None and not df.empty and date_col in df.columns:
            min_date = pd.to_datetime(df[date_col]).min().strftime("%Y-%m-%d")
            max_date = pd.to_datetime(df[date_col]).max().strftime("%Y-%m-%d")
            period_df = pd.DataFrame([{"Start Date": min_date, "End Date": max_date}])
            log.log_dataframe(period_df, title=title)
        else:
            log.warning(f"Could not log period for {title}: DataFrame is empty or missing '{date_col}' column.")

    def load_macro_data(
        self,
        update_inflation: bool = False,
        update_rub_usd: bool = False,
        update_unemployment: bool = False,
        inflation_path: str = "data/macro/inflation.xlsx",
        rub_usd_path: str = "data/macro/rubusd.csv",
        unemployment_path: str = "data/macro/unemployment.xlsx",
    ) -> "Portfolio":
        """
        Adds macroeconomic data to the portfolio data.

        Args:
            update_inflation (bool): If True, downloads fresh inflation data from CBR and updates backup.
            Defaults to False.
            update_rub_usd (bool): If True, downloads fresh USD/RUB exchange rate data. Defaults to False.
            update_unemployment (bool): If True, downloads fresh unemployment data from TradingView. Defaults to False.
            inflation_path (str): Path to inflation data file. Defaults to "data/macro/inflation.xlsx".
            rub_usd_backup_path (str): Path to USD/RUB exchange rate backup. Defaults to "data/backup/rub_usd.pkl".
            unemployment_path (str): Path to unemployment data file. Defaults to "data/macro/unemployment.xlsx".

        Returns:
            Portfolio: Updated portfolio with added macroeconomic data.
        """

        calc_date = pd.to_datetime(self.dt_calc)
        start_date = pd.to_datetime(self.dt_start)

        # 1. Load Unemployment Data
        if update_unemployment or not os.path.isfile(unemployment_path):
            unemployment = get_unemployment_data(unemployment_path, update_backup=update_unemployment)
            if update_unemployment:
                log.info("Downloaded fresh unemployment data and updated backup")
            else:
                log.info("Downloaded fresh unemployment data")
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

        self._log_data_period(self.d["macro_unemployment"], "dtReportLast", "Loaded Unemployment Data Period")

        # 2. Load and process Inflation & Interest Rate
        if update_inflation or not os.path.isfile(inflation_path):
            inflation = get_cbr_inflation_data(
                inflation_path, self.dt_start, self.dt_calc, update_backup=update_inflation
            )
            if update_inflation:
                log.info("Downloaded fresh inflation data and updated backup")
            else:
                log.info("Downloaded fresh inflation data")
        else:
            inflation = pd.read_excel(inflation_path)
            log.info("Loaded inflation data from backup")

        self.d["macro_inflation"] = (
            inflation.assign(dtReportLast=lambda x: (pd.to_datetime(x["Дата"]) + pd.offsets.MonthEnd(0)).dt.normalize())
            .rename(
                columns={
                    "Ключевая ставка, % годовых": "interest_rate",
                    "Инфляция, % г/г": "inflation",
                }
            )
            .assign(interest_rate=lambda x: x["interest_rate"] / 100, inflation=lambda x: x["inflation"] / 100)
            .loc[
                lambda x: (x["dtReportLast"] >= start_date) & (x["dtReportLast"] <= calc_date),
                ["dtReportLast", "interest_rate", "inflation"],
            ]
        )

        self._log_data_period(self.d["macro_inflation"], "dtReportLast", "Loaded Inflation Data Period")

        # 3. Load and process USD/RUB Exchange Rate
        if update_rub_usd or not os.path.isfile(rub_usd_path):
            rub_usd = get_rubusd_exchange_rate(
                dt_calc=self.dt_calc, dt_start=self.dt_start, update_backup=update_rub_usd
            )
            if update_rub_usd:
                log.info("Downloaded fresh USD/RUB exchange rate and updated backup")
            else:
                log.info("Downloaded fresh USD/RUB exchange rate")
        else:
            rub_usd = pd.read_csv(rub_usd_path)
            log.info("Loaded USD/RUB exchange rate from backup")

        self.d["macro_rub_usd"] = rub_usd.assign(date=lambda x: pd.to_datetime(x["date"]).dt.normalize()).loc[
            lambda x: (x["date"] >= start_date) & (x["date"] <= calc_date), ["date", "rubusd_exchange_rate"]
        ]

        self._log_data_period(self.d["macro_rub_usd"], "date", "Loaded USD/RUB Exchange Rate Period")

        return self

    def create_portfolio(self):
        """
        Creates a portfolio by merging stocks, multipliers, and macro data.

        Returns:
            Portfolio: Created portfolio.
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

        columns_new_names = {
            "Капитализация, млрд руб": "capitalization",
        }

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

                # Handle zero-values as missing data to avoid ffill issues and PD spikes
                if col in ["Капитализация, млрд руб", "Долг, млрд руб"]:
                    self.d["portfolio"][col] = self.d["portfolio"][col].replace(0, np.nan)

                if "млрд руб" in col:
                    self.d["portfolio"][col] *= self.BILLION

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
            .assign(
                capitalization=lambda x: x.groupby("ticker")["capitalization"].transform(lambda g: g.ffill().bfill())
            )
            .assign(capitalization=lambda x: x["capitalization"].fillna(0))
        )

        self.d["portfolio"] = self.d["portfolio"].sort_values(["ticker", "date"])
        num_rows = len(self.d["portfolio"])
        portfolio_info = pd.DataFrame(
            [
                {
                    "Total Rows": num_rows,
                    "Unique Companies": len(self.d["portfolio"].ticker.unique()),
                    "Date Range": f"{self.d['portfolio']['date'].min().strftime('%Y-%m-%d')} to"
                    + f" {self.d['portfolio']['date'].max().strftime('%Y-%m-%d')}",
                }
            ]
        )

        log.log_missing_values_summary(self.d["portfolio"], title="Portfolio Missing Values After Filling")
        log.log_dataframe(portfolio_info, title="Portfolio Dimensions")

        return self

    def _solve_merton_vectorized(self, T: float = 1) -> "Portfolio":
        """
        Solves the system of equations to estimate V and sigma_V.

        Args:
            T (float): Time horizon.

        Returns:
            Portfolio: Updated portfolio with calculated capital cost and capital volatility.
        """

        E = self.d["portfolio"]["capitalization"].values.astype(float)
        D = self.d["portfolio"]["debt"].values.astype(float)
        sigma_E = self.d["portfolio"]["volatility"].values.astype(float)

        def equations(vars, E_i, D_i, r_i, sigma_E_i, T_i):
            V, sigma_V = vars
            d1 = np.log(V / D_i if D_i != 0 else Portfolio.EPSILON) + (r_i + 0.5 * sigma_V**2) * T_i
            d1 /= sigma_V * np.sqrt(T_i)
            N_d1 = norm.cdf(d1)
            eq1 = V * N_d1 - D_i * np.exp(-r_i * T_i) * norm.cdf(d1 - sigma_V * np.sqrt(T_i)) - E_i
            eq2 = N_d1 * sigma_V * V - sigma_E_i * E_i
            return [eq1, eq2]

        # Initial guesses for all elements
        initial_guess = np.vstack([E + D, sigma_E]).T

        # Solve for each element with progress monitoring
        log.info(f"Starting Merton model calculations for {len(initial_guess)} rows...")
        results = np.array(
            [
                root(
                    equations,
                    guess,
                    args=(
                        E[i],
                        D[i],
                        self.d["portfolio"]["interest_rate"][i],
                        sigma_E[i],
                        T,
                    ),
                ).x
                for i, guess in enumerate(tqdm(initial_guess, desc="Solving Merton equations"))
            ]
        )

        self.d["portfolio"]["V"] = np.where(results[:, 0] <= 0, self.EPSILON, results[:, 0])
        self.d["portfolio"]["sigma_V"] = results[:, 1]

        log.info("Capital cost and capital volatility calculated.")

        return self

    def _merton_pd(self, T: float = 1) -> "Portfolio":
        """
        Calculates the probability of default (PD) and distance to default (DD) using the Merton model.

        Args:
            T (float): Time horizon for the default event (in years).

        Returns:
            Portfolio: Updated portfolio with calculated probabilities of default and distance to default.
        """

        V = self.d["portfolio"]["V"].values.astype(float)
        D = self.d["portfolio"]["debt"].values.astype(float)
        sigma_V = self.d["portfolio"]["sigma_V"]

        d2 = (
            np.log(V / np.where(D != 0, D, self.EPSILON))
            + (self.d["portfolio"]["interest_rate"] - 0.5 * sigma_V**2) * T
        ) / (sigma_V * np.sqrt(T))
        self.d["portfolio"]["PD"] = norm.cdf(-d2)
        self.d["portfolio"]["DD"] = d2

        log.info("Merton's probabilities of default and distance to default calculated.")

        return self

    def add_merton_pd(self) -> "Portfolio":
        """
        Adds the probability of default (PD) and distance to default (DD)
        calculated using the Merton model to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added probabilities of default and distance to default.
        """
        self = self._solve_merton_vectorized()._merton_pd()

        self.d["portfolio"] = self.d["portfolio"].drop(columns=["V", "sigma_V"])

        return self

    def add_dynamic_features(self):
        """
        Adds dynamic features to the portfolio data.

        Computes EWMA (Exponentially Weighted Moving Average) annualized
        volatility per ticker with span=ROLLING_VOL_WINDOW trading days.
        EWMA gives more weight to recent observations, making the
        volatility estimate more reactive to stress periods (e.g., Feb 2022)
        compared to a simple rolling window.

        Returns:
            Portfolio: Updated portfolio with added dynamic features.
        """
        df = self.d["portfolio"].sort_values(["ticker", "date"]).copy()

        # Daily log-returns per ticker
        df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))

        # EWMA annualized volatility: ewm_std(daily) * sqrt(252)
        # EWMA reacts faster to volatility shocks than simple rolling std
        df["volatility"] = df.groupby("ticker")["log_return"].transform(
            lambda x: x.ewm(span=self.ROLLING_VOL_WINDOW, min_periods=self.TRADING_DAYS_PER_MONTH).std()
        ) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Fill leading NaNs (before the first full window) with the ticker's first valid value, then global mean
        df["volatility"] = df.groupby("ticker")["volatility"].transform(lambda x: x.bfill())
        global_avg_vol = df["volatility"].mean()
        df["volatility"] = df["volatility"].fillna(global_avg_vol).fillna(self.DEFAULT_VOLATILITY)

        df = df.drop(columns=["log_return"])
        self.d["portfolio"] = df

        log.log_missing_values_summary(
            self.d["portfolio"], title="Portfolio Missing Values After Adding dynamic features"
        )

        return self

    def predict_macro_factors(
        self, horizon: int = 1, training_offset: int = 0, model_type: str = "var"
    ) -> pd.DataFrame:
        """
        Predicts macroeconomic factors using specified model type.

        Args:
            horizon (int): Number of months to predict.
            training_offset (int): Months of history to hide.
            model_type (str): 'var', 'sarimax', or 'prophet'.

        Returns:
            pd.DataFrame: Forecasted macro variables.
        """
        if horizon < 0:
            training_offset = abs(horizon)
            horizon = abs(horizon)

        macro_cols = ["inflation", "interest_rate", "unemployment_rate", "rubusd_exchange_rate"]

        macro_df = (
            self.d["portfolio"][["date"] + macro_cols]
            .drop_duplicates("date")
            .set_index("date")
            .resample("ME")
            .mean()
            .dropna()
        )

        if training_offset > 0:
            macro_df = macro_df.iloc[:-training_offset]

        future_dates = pd.date_range(start=macro_df.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")

        if model_type.lower() == "var":
            results = VAR(macro_df).fit(maxlags=max(1, min(3, len(macro_df) // 2 - 1)))
            fc = results.forecast(y=macro_df.values[-results.k_ar :], steps=horizon)
            return pd.DataFrame(fc, index=future_dates, columns=macro_df.columns)

        if model_type.lower() == "sarimax":
            forecast_results = {}
            for col in macro_df.columns:
                d = 1 if adfuller(macro_df[col])[1] > 0.05 else 0
                exog_train = macro_df.drop(columns=[col]).shift(1).bfill()

                # Scaling is crucial because macro variables (e.g. rubusd ~100 vs inflation ~0.1)
                # have different scales, which breaks MLE optimization.
                sy, sx = StandardScaler(), StandardScaler()
                y_scaled = sy.fit_transform(macro_df[[col]])
                x_scaled = sx.fit_transform(exog_train)

                # Using (1, d, 0) for better stability on small datasets, and default optimizer
                results = SARIMAX(
                    y_scaled, exog=x_scaled, order=(1, d, 0), enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False, maxiter=500)

                exog_fc = pd.DataFrame([macro_df.drop(columns=[col]).iloc[-1]] * horizon, columns=exog_train.columns)
                exog_fc_scaled = sx.transform(exog_fc)

                fc_scaled = results.forecast(steps=horizon, exog=exog_fc_scaled)
                forecast_results[col] = sy.inverse_transform(fc_scaled.reshape(-1, 1)).flatten()
            return pd.DataFrame(forecast_results, index=future_dates)

        if model_type.lower() == "prophet":
            forecast_results = {}
            for col in macro_df.columns:
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                )
                m.fit(macro_df[[col]].reset_index().rename(columns={"date": "ds", col: "y"}))
                future = m.make_future_dataframe(periods=horizon, freq="ME")
                forecast_results[col] = m.predict(future)["yhat"].tail(horizon).values
            return pd.DataFrame(forecast_results, index=future_dates)

        return pd.DataFrame()

    def predict_pd(self, horizon: int = 1, training_offset: int = 0, model_type: str = "var") -> "Portfolio":
        """
        Predicts Probability of Default (PD) for portfolio assets based on macro forecasts.

        Args:
            horizon (int): Forecasting horizon in months.
            training_offset (int): Months of history to hide for backtesting.
            model_type (str): Type of macro model to use.

        Returns:
            Portfolio: Self with results stored in self.d['pd_forecast'].
        """
        if horizon < 0:
            training_offset = abs(horizon)
            horizon = abs(horizon)

        macro_forecast = self.predict_macro_factors(
            horizon=horizon, training_offset=training_offset, model_type=model_type
        )

        pd_daily = self.d["portfolio"][["date", "ticker", "PD"]]
        pd_pivot = pd_daily.pivot(index="date", columns="ticker", values="PD")
        pd_monthly = pd_pivot.resample("ME").last()
        pd_monthly.index = pd_monthly.index.normalize() + pd.offsets.MonthEnd(0)

        macro_cols = macro_forecast.columns.tolist()
        macro_hist = (
            self.d["portfolio"][["date"] + macro_cols]
            .drop_duplicates("date")
            .set_index("date")
            .resample("ME")
            .mean()
            .dropna()
        )
        macro_hist.index = macro_hist.index.normalize() + pd.offsets.MonthEnd(0)

        combined_data = pd.concat([pd_monthly, macro_hist], axis=1).dropna()
        train_combined = combined_data.iloc[:-training_offset] if training_offset > 0 else combined_data

        predictions = {}
        for ticker in pd_monthly.columns:
            if ticker not in train_combined.columns:
                continue

            y = train_combined[ticker]
            x = sm.add_constant(train_combined[macro_cols])
            model = sm.OLS(y, x).fit()

            x_pred = sm.add_constant(macro_forecast, has_constant="add")
            if "const" not in x_pred.columns:
                x_pred["const"] = 1.0

            y_pred = model.predict(x_pred)
            predictions[ticker] = y_pred.iloc[-1]

        result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["predicted_pd"])
        result_df.index.name = "ticker"

        target_date = macro_forecast.index[-1]
        comparison_pd = pd_monthly.loc[target_date] if target_date in pd_monthly.index else pd_pivot.ffill().iloc[-1]

        result_df["reference_pd"] = comparison_pd
        result_df["delta"] = result_df["predicted_pd"] - result_df["reference_pd"]
        result_df["predicted_pd"] = result_df["predicted_pd"].clip(0, 1)
        result_df["model"] = model_type

        self.d["pd_forecast"] = result_df
        return self

    def predict_dd(self, horizon: int = 1, training_offset: int = 0, model_type: str = "var") -> "Portfolio":
        """
        Predicts Distance to Default (DD) for portfolio assets based on macro forecasts.

        Args:
            horizon (int): Forecasting horizon in months.
            training_offset (int): Months of history to hide for backtesting.
            model_type (str): Type of macro model to use.

        Returns:
            Portfolio: Self with results stored in self.d['dd_forecast'].
        """
        if horizon < 0:
            training_offset = abs(horizon)
            horizon = abs(horizon)

        macro_forecast = self.predict_macro_factors(
            horizon=horizon, training_offset=training_offset, model_type=model_type
        )

        dd_daily = self.d["portfolio"][["date", "ticker", "DD"]]
        dd_pivot = dd_daily.pivot(index="date", columns="ticker", values="DD")
        dd_monthly = dd_pivot.resample("ME").last()
        dd_monthly.index = dd_monthly.index.normalize() + pd.offsets.MonthEnd(0)

        macro_cols = macro_forecast.columns.tolist()
        macro_hist = (
            self.d["portfolio"][["date"] + macro_cols]
            .drop_duplicates("date")
            .set_index("date")
            .resample("ME")
            .mean()
            .dropna()
        )
        macro_hist.index = macro_hist.index.normalize() + pd.offsets.MonthEnd(0)

        combined_data = pd.concat([dd_monthly, macro_hist], axis=1).dropna()
        train_combined = combined_data.iloc[:-training_offset] if training_offset > 0 else combined_data

        predictions = {}
        for ticker in dd_monthly.columns:
            if ticker not in train_combined.columns:
                continue

            y = train_combined[ticker]
            # OLS on DD is more stable than on PD
            x = sm.add_constant(train_combined[macro_cols])
            model = sm.OLS(y, x).fit()

            x_pred = sm.add_constant(macro_forecast, has_constant="add")
            if "const" not in x_pred.columns:
                x_pred["const"] = 1.0

            y_pred = model.predict(x_pred)
            predictions[ticker] = y_pred.iloc[-1]

        result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["predicted_dd"])
        result_df.index.name = "ticker"

        target_date = macro_forecast.index[-1]
        comparison_dd = dd_monthly.loc[target_date] if target_date in dd_monthly.index else dd_pivot.ffill().iloc[-1]

        result_df["reference_dd"] = comparison_dd
        result_df["delta"] = result_df["predicted_dd"] - result_df["reference_dd"]
        result_df["model"] = model_type

        # Convert back to PD for visualization consistency
        result_df["predicted_pd"] = norm.cdf(-result_df["predicted_dd"].astype(float))
        result_df["reference_pd"] = norm.cdf(-result_df["reference_dd"].astype(float))

        self.d["dd_forecast"] = result_df
        return self

    def backtest_pd(self, n_months: int = 12, models: List[str] = None) -> "Portfolio":
        """
        Performs backtesting of PD predictions across multiple macro models.

        Args:
            n_months (int): Number of months for the backtest window.
            models (List[str], optional): List of models to test. Defaults to ['var'].

        Returns:
            Portfolio: Self with backtest results in self.d['pd_backtest'].
        """
        if models is None:
            models = ["var"]

        all_results = []
        for m_type in models:
            log.info(f"Backtesting PD using {m_type} model...")
            for offset in range(n_months, 0, -1):
                self.predict_pd(horizon=1, training_offset=offset, model_type=m_type)

                # Get the date being predicted for labeling
                macro_fc = self.predict_macro_factors(horizon=1, training_offset=offset, model_type=m_type)
                target_date = macro_fc.index[-1]

                res = self.d["pd_forecast"].copy().reset_index()
                res["date"] = target_date
                all_results.append(res)

        self.d["pd_backtest"] = pd.concat(all_results, ignore_index=True)
        return self

    def backtest_dd(self, n_months: int = 12, models: List[str] = None) -> "Portfolio":
        """
        Performs backtesting of DD predictions across multiple macro models.

        Args:
            n_months (int): Number of months for the backtest window.
            models (List[str], optional): List of models to test. Defaults to ['var'].

        Returns:
            Portfolio: Self with backtest results in self.d['dd_backtest'].
        """
        if models is None:
            models = ["var"]

        all_results = []
        for m_type in models:
            log.info(f"Backtesting DD using {m_type} model...")
            for offset in range(n_months, 0, -1):
                self.predict_dd(horizon=1, training_offset=offset, model_type=m_type)

                # Get the date being predicted for labeling
                macro_fc = self.predict_macro_factors(horizon=1, training_offset=offset, model_type=m_type)
                target_date = macro_fc.index[-1]

                res = self.d["dd_forecast"].copy().reset_index()
                res["date"] = target_date
                all_results.append(res)

        self.d["dd_backtest"] = pd.concat(all_results, ignore_index=True)
        return self

    def optimize_portfolio(
        self,
        lambda_risk: float = 0.5,
        lgd: float = 0.4,
        use_forecast: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        strategy: str = "mean_el",
        cvar_alpha: float = 0.95,
        cutoff_date: pd.Timestamp = None,
    ) -> "Portfolio":
        """Optimizes portfolio weights using the specified strategy.

        Supported strategies:
            - ``mean_el``  (default): minimizes lambda * portfolio_volatility +
              (1 - lambda) * expected_loss.  Solved via SLSQP (QP).
            - ``cvar``: minimizes CVaR_alpha of credit-adjusted portfolio losses
              using the Rockafellar-Uryasev (2000) LP formulation.
            - ``risk_parity``: equalizes credit-adjusted risk contributions
              across all assets (Maillard, Roncalli & Teiletche, 2010).

        Args:
            lambda_risk: Risk aversion parameter (0 to 1), used by ``mean_el`` strategy.
                High lambda favors lower volatility (unexpected loss).
            lgd: Loss Given Default.
            use_forecast: If True, uses predicted PDs from self.d['pd_forecast'] or self.d['dd_forecast'].
            min_weight: Minimum weight per asset (0 to 1). Forces diversification.
                E.g. 0.02 guarantees each asset gets at least 2%.
            max_weight: Maximum weight per asset (0 to 1). Prevents concentration.
                E.g. 0.25 caps any single asset at 25%.
            strategy: Optimization strategy. One of ``"mean_el"``, ``"cvar"``,
                or ``"risk_parity"``.
            cvar_alpha: Confidence level for CVaR (default 0.95). Only used when
                ``strategy="cvar"``.
            cutoff_date: If provided, only use returns data up to (and excluding)
                this date for covariance estimation. Prevents look-ahead bias
                during backtesting.

        Returns:
            Portfolio: Self with optimized weights in self.d['optimized_weights'].
        """
        # 1. Get PDs for optimization
        if use_forecast and "dd_forecast" in self.d:
            current_pd_series = self.d["dd_forecast"]["predicted_pd"]
            tickers = current_pd_series.index.tolist()
            pds = current_pd_series.values
            log.info("Using predicted PDs (from DD model) for optimization.")
        elif use_forecast and "pd_forecast" in self.d:
            current_pd_series = self.d["pd_forecast"]["predicted_pd"]
            tickers = current_pd_series.index.tolist()
            pds = current_pd_series.values
            log.info("Using predicted PDs for optimization.")
        else:
            portfolio_df = self.d["portfolio"]
            latest_date = portfolio_df["date"].max()
            current_pd_df = portfolio_df[portfolio_df["date"] == latest_date][["ticker", "PD"]]
            tickers = current_pd_df["ticker"].tolist()
            pds = current_pd_df["PD"].values
            log.info("Using historical PDs for optimization.")

        # 2. Calculate returns matrix (daily log-returns)
        returns_df = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
        returns_df = np.log(returns_df / returns_df.shift(1)).dropna()

        # Fix look-ahead bias: use only data available at backtest date
        if cutoff_date is not None:
            returns_df = returns_df.loc[returns_df.index < cutoff_date]

        # Ensure we only use tickers present in both
        common_tickers = [t for t in tickers if t in returns_df.columns]
        if len(common_tickers) < len(tickers):
            missing = set(tickers) - set(common_tickers)
            log.warning("Some tickers missing returns data: %s. Using subset.", missing)

        tickers = common_tickers
        if use_forecast and ("dd_forecast" in self.d or "pd_forecast" in self.d):
            pds = current_pd_series.loc[tickers].values
        else:
            pds = current_pd_df.set_index("ticker").loc[tickers]["PD"].values

        if strategy == "cvar":
            weights = self._solve_cvar(returns_df[tickers].values, pds, lgd, cvar_alpha, min_weight, max_weight)
        elif strategy == "risk_parity":
            weights = self._solve_risk_parity(returns_df[tickers].values, pds, lgd, min_weight, max_weight)
        else:
            weights = self._solve_mean_el(returns_df[tickers].values, pds, lgd, lambda_risk, min_weight, max_weight)

        if weights is None:
            return self

        self.d["optimized_weights"] = pd.Series(weights, index=tickers, name="weight")
        log.info("Portfolio optimization completed successfully (strategy=%s).", strategy)

        return self

    def _solve_mean_el(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        lambda_risk: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Solves the mean-EL optimization: lambda * vol + (1-lambda) * EL -> min.

        Args:
            returns_matrix: Daily log-returns, shape (S, N).
            pds: Predicted PDs per asset, shape (N,).
            lgd: Loss Given Default.
            lambda_risk: Risk aversion coefficient.
            min_weight: Minimum weight per asset.
            max_weight: Maximum weight per asset.

        Returns:
            Optimal weight vector of shape (N,), or None on failure.
        """
        cov_matrix = np.cov(returns_matrix, rowvar=False) * self.TRADING_DAYS_PER_YEAR
        n = len(pds)

        def objective(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            expected_loss = np.sum(w * pds * lgd)
            return lambda_risk * port_vol + (1 - lambda_risk) * expected_loss

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        initial_guess = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9},
        )

        if not res.success:
            log.error("Mean-EL optimization failed: %s", res.message)
            return None
        return res.x

    def _solve_cvar(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        alpha: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Solves the CVaR optimization via Rockafellar-Uryasev LP formulation.

        Minimizes CVaR_alpha of credit-adjusted portfolio losses.
        Credit-adjusted return per scenario: r_i^(s) - PD_i * LGD.

        LP variables: x = [w_1..w_N, zeta, u_1..u_S]
        Objective:    min  zeta + 1/((1-alpha)*S) * sum(u_s)
        Constraints:  u_s >= -(w^T r_adj^(s)) - zeta,  u_s >= 0
                      sum(w) = 1,  w_min <= w_i <= w_max

        Args:
            returns_matrix: Daily log-returns, shape (S, N).
            pds: Predicted PDs per asset, shape (N,).
            lgd: Loss Given Default.
            alpha: CVaR confidence level (e.g. 0.95).
            min_weight: Minimum weight per asset.
            max_weight: Maximum weight per asset.

        Returns:
            Optimal weight vector of shape (N,), or None on failure.
        """
        s_count, n = returns_matrix.shape

        # Credit-adjusted returns: subtract daily-scaled expected loss
        daily_el = (pds * lgd) / self.TRADING_DAYS_PER_YEAR
        adjusted_returns = returns_matrix - daily_el.reshape(1, n)

        coeff = 1.0 / ((1 - alpha) * s_count)
        n_vars = n + 1 + s_count  # w, zeta, u

        # Objective: c^T x -> min
        c = np.zeros(n_vars)
        c[n] = 1.0  # zeta
        c[n + 1 :] = coeff  # u_s coefficients

        # Inequality: A_ub @ x <= b_ub
        # For each scenario s: -r_adj[s] @ w - zeta - u_s <= 0
        A_ub = np.zeros((s_count, n_vars))
        A_ub[:, :n] = -adjusted_returns  # -r_adj^(s) * w
        A_ub[:, n] = -1.0  # -zeta
        for s in range(s_count):
            A_ub[s, n + 1 + s] = -1.0  # -u_s
        b_ub = np.zeros(s_count)

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        # Bounds
        bounds_list = [(min_weight, max_weight)] * n
        bounds_list.append((None, None))  # zeta unbounded
        bounds_list.extend([(0.0, None)] * s_count)  # u_s >= 0

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
            method="highs",
        )

        if not res.success:
            log.error("CVaR optimization failed: %s", res.message)
            return None

        weights = res.x[:n]
        cvar_value = res.fun
        log.info(
            "CVaR(%.0f%%) = %.6f (daily), VaR(zeta) = %.6f",
            alpha * 100,
            cvar_value,
            res.x[n],
        )
        return weights

    def _solve_risk_parity(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Solves the Risk-Parity + Credit optimization.

        Finds weights such that each asset contributes equal credit-adjusted
        risk to the portfolio.  Uses the log-barrier formulation from
        Maillard, Roncalli & Teiletche (2010) extended with PD-based credit
        risk contributions.

        Credit Risk Contribution of asset i:
            CRC_i = w_i * (Sigma @ w)_i / sigma_p + w_i * PD_i * LGD

        Objective:
            min_w  sum_{i<j} (CRC_i - CRC_j)^2
            s.t.   sum(w) = 1,  min_weight <= w_i <= max_weight

        Args:
            returns_matrix: Daily log-returns, shape (S, N).
            pds: Predicted PDs per asset, shape (N,).
            lgd: Loss Given Default.
            min_weight: Minimum weight per asset.
            max_weight: Maximum weight per asset.

        Returns:
            Optimal weight vector of shape (N,), or None on failure.
        """
        cov_matrix = np.cov(returns_matrix, rowvar=False) * self.TRADING_DAYS_PER_YEAR
        n = len(pds)

        def credit_risk_contributions(w: np.ndarray) -> np.ndarray:
            """Computes credit-adjusted risk contribution per asset."""
            sigma_w = cov_matrix @ w
            port_vol = np.sqrt(w @ sigma_w)
            # Market risk contribution
            mrc = w * sigma_w / (port_vol + self.EPSILON)
            # Credit risk contribution
            crc = w * pds * lgd
            return mrc + crc

        def objective(w: np.ndarray) -> float:
            """Sum of squared pairwise differences in risk contributions."""
            rc = credit_risk_contributions(w)
            diff_sum = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    diff_sum += (rc[i] - rc[j]) ** 2
            return diff_sum

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        initial_guess = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not res.success:
            log.error("Risk-Parity optimization failed: %s", res.message)
            return None

        log.info(
            "Risk-Parity: objective=%.2e, max_rc_diff=%.4f",
            res.fun,
            np.max(credit_risk_contributions(res.x)) - np.min(credit_risk_contributions(res.x)),
        )
        return res.x

    def calc_portfolio_metrics(self, lgd: float = 0.4) -> "Portfolio":
        """
        Calculates key risk metrics for the optimized portfolio.

        Args:
            lgd (float): Loss Given Default.

        Returns:
            Portfolio: Self with metrics stored in self.d['portfolio_metrics'].
        """
        if "optimized_weights" not in self.d:
            log.error("No optimized weights found. Run optimize_portfolio() first.")
            return self

        weights = self.d["optimized_weights"]
        tickers = weights.index.tolist()

        # Get PDs used (from forecast or historical)
        if "dd_forecast" in self.d:
            pds = self.d["dd_forecast"]["predicted_pd"].loc[tickers]
        elif "pd_forecast" in self.d:
            pds = self.d["pd_forecast"]["predicted_pd"].loc[tickers]
        else:
            latest_date = self.d["portfolio"]["date"].max()
            pds = self.d["portfolio"][self.d["portfolio"]["date"] == latest_date].set_index("ticker").loc[tickers]["PD"]

        # 1. Expected Loss
        portfolio_el = np.sum(weights * pds * lgd)

        # 2. Portfolio Volatility (Unexpected Loss proxy)
        returns_df = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
        returns_df = np.log(returns_df / returns_df.shift(1)).dropna()
        cov_matrix = returns_df[tickers].cov().values * self.TRADING_DAYS_PER_YEAR
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # 3. Diversification index (Herfindahl-Hirschman Index for weights)
        hhi = np.sum(weights**2)
        effective_n = 1.0 / hhi if hhi > 0 else 0

        metrics = {
            "Expected Loss (%)": portfolio_el * 100,
            "Volatility (%)": portfolio_vol * 100,
            "HHI Index": hhi,
            "Effective N": effective_n,
        }

        self.d["portfolio_metrics"] = pd.Series(metrics)
        log.log_dataframe(pd.DataFrame([metrics]).T, title="Portfolio Management Metrics")

        return self

    def plot_portfolio_allocation(self, figsize: tuple = (10, 6), verbose: bool = False) -> "Portfolio":
        """
        Plots the optimized portfolio allocation.

        Args:
            figsize (tuple): Figure size.
            verbose (bool): Whether to show the plot.

        Returns:
            Portfolio: Self.
        """
        if "optimized_weights" not in self.d:
            log.error("No optimized weights found. Run optimize_portfolio() first.")
            return self

        plots.plot_portfolio_allocation(self.d["optimized_weights"], figsize=figsize, verbose=verbose)
        return self

    def plot_macro_forecast(
        self,
        horizon: int = 1,
        training_offset: int = 0,
        models: Union[str, list] = "var",
        tail: int = 12,
        target_col: str = "DD",
        verbose: bool = False,
        figsize: tuple = (12, 14),
    ):
        """
        Plots historical and forecasted values for macro parameters and Portfolio DD/PD.

        Args:
            horizon (int): Forecasting horizon.
            training_offset (int): Offset for backtesting.
            models (str|list): Model types to compare.
            tail (int): History months to show.
            target_col (str): 'DD' or 'PD'.
            verbose (bool): Whether to show the plot.
            figsize (tuple): Figure size.
        """
        if isinstance(models, str):
            models = [models]

        is_backtest = horizon < 0
        if is_backtest:
            n_months = abs(horizon)
            log.info(f"Visualizing Expanding Window backtest for last {n_months} months")
        else:
            log.info(f"Visualizing Multi-step forecast for {horizon} months")

        # Get historical data including Portfolio Target
        macro_cols = ["inflation", "interest_rate", "unemployment_rate", "rubusd_exchange_rate"]

        # Calculate daily portfolio average target
        port_target = self.d["portfolio"].groupby("date")[target_col].mean().reset_index()

        macro_df = (
            self.d["portfolio"][["date"] + macro_cols]
            .drop_duplicates("date")
            .merge(port_target, on="date", how="left")
            .set_index("date")
            .resample("ME")
            .mean()
            .dropna()
        )
        macro_df.index = macro_df.index.normalize() + pd.offsets.MonthEnd(0)

        # Collect forecasts from all specified models
        forecast_dfs = {}

        for m_type in models:
            if is_backtest:
                # WALK-FORWARD: predict each month one by one
                step_fcs = []
                for offset in range(n_months, 0, -1):
                    # Macro step
                    fc_step = self.predict_macro_factors(horizon=1, training_offset=offset, model_type=m_type)

                    # Target step
                    if target_col == "PD":
                        self.predict_pd(horizon=1, training_offset=offset, model_type=m_type)
                        fc_step["PD"] = self.d["pd_forecast"]["predicted_pd"].mean()
                    else:
                        self.predict_dd(horizon=1, training_offset=offset, model_type=m_type)
                        fc_step["DD"] = self.d["dd_forecast"]["predicted_dd"].mean()

                    step_fcs.append(fc_step)

                forecast_dfs[m_type] = pd.concat(step_fcs)

            else:
                # NORMAL: multi-step forecast
                fc_df = self.predict_macro_factors(horizon=horizon, training_offset=training_offset, model_type=m_type)

                trajectory = []
                for i in range(1, horizon + 1):
                    if target_col == "PD":
                        self.predict_pd(horizon=i, training_offset=training_offset, model_type=m_type)
                        trajectory.append(self.d["pd_forecast"]["predicted_pd"].mean())
                    else:
                        self.predict_dd(horizon=i, training_offset=training_offset, model_type=m_type)
                        trajectory.append(self.d["dd_forecast"]["predicted_dd"].mean())

                fc_df[target_col] = trajectory
                forecast_dfs[m_type] = fc_df

        plots.plot_macro_forecast(macro_df, forecast_dfs, tail=tail, figsize=figsize, verbose=verbose)
        return self

    def analyze_macro_dd_significance(self, max_lag: int = 3, verbose: bool = True) -> "Portfolio":
        """Analyzes the statistical significance of macro variables on DD.

        Performs two analyses:
        1. Pooled OLS regression: DD ~ inflation + interest_rate + unemployment + RUBUSD
           with t-tests for individual coefficients and F-test for joint significance.
        2. Granger causality tests: whether lagged macro variables improve DD prediction.

        Args:
            max_lag: Maximum number of lags for Granger causality tests.
            verbose: If True, prints detailed results.

        Returns:
            Portfolio: Self with results in self.d['macro_significance'].
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        macro_cols = ["inflation", "interest_rate", "unemployment_rate", "rubusd_exchange_rate"]

        # --- Part 1: Pooled OLS regression (monthly panel) ---
        panel = self.d["portfolio"][["date", "ticker", "DD"] + macro_cols].dropna().copy()
        panel["month"] = panel["date"].dt.to_period("M")
        monthly = (
            panel.groupby(["ticker", "month"])
            .agg(
                DD=("DD", "mean"),
                **{col: (col, "mean") for col in macro_cols},
            )
            .reset_index()
        )

        y = monthly["DD"]
        X = sm.add_constant(monthly[macro_cols])
        ols_model = sm.OLS(y, X).fit()

        ols_summary = {
            "r_squared": ols_model.rsquared,
            "adj_r_squared": ols_model.rsquared_adj,
            "f_statistic": ols_model.fvalue,
            "f_pvalue": ols_model.f_pvalue,
            "n_obs": int(ols_model.nobs),
            "coefficients": {},
        }
        for var in ["const"] + macro_cols:
            ols_summary["coefficients"][var] = {
                "coef": ols_model.params[var],
                "std_err": ols_model.bse[var],
                "t_stat": ols_model.tvalues[var],
                "p_value": ols_model.pvalues[var],
                "significant": ols_model.pvalues[var] < 0.05,
            }

        # --- Part 2: Granger causality tests (aggregate monthly DD) ---
        agg_monthly = (
            panel.groupby("month")
            .agg(DD=("DD", "mean"), **{col: (col, "mean") for col in macro_cols})
            .reset_index()
            .sort_values("month")
        )

        granger_results = {}
        for col in macro_cols:
            series = agg_monthly[["DD", col]].dropna()
            if len(series) < max_lag + 5:
                granger_results[col] = {"error": "insufficient data"}
                continue
            test_result = grangercausalitytests(series.values, maxlag=max_lag, verbose=False)
            best_lag = min(test_result, key=lambda x: test_result[x][0]["ssr_ftest"][1])
            f_stat = test_result[best_lag][0]["ssr_ftest"][0]
            p_val = test_result[best_lag][0]["ssr_ftest"][1]
            granger_results[col] = {
                "best_lag": best_lag,
                "f_statistic": f_stat,
                "p_value": p_val,
                "significant": p_val < 0.05,
            }

        self.d["macro_significance"] = {
            "ols": ols_summary,
            "granger": granger_results,
            "ols_model": ols_model,
        }

        if verbose:
            log.info("=" * 80)
            log.info("POOLED OLS REGRESSION: DD ~ macro variables")
            log.info("=" * 80)
            for line in str(ols_model.summary().tables[0]).split("\n"):
                log.info(line)

            coef_data = []
            for var in ["const"] + macro_cols:
                c = ols_summary["coefficients"][var]
                sig = (
                    "***"
                    if c["p_value"] < 0.001
                    else "**"
                    if c["p_value"] < 0.01
                    else "*"
                    if c["p_value"] < 0.05
                    else ""
                )
                coef_data.append(
                    [var, f"{c['coef']:.4f}", f"{c['std_err']:.4f}", f"{c['t_stat']:.2f}", f"{c['p_value']:.4f}", sig]
                )

            coef_df = pd.DataFrame(coef_data, columns=["Variable", "Coef", "Std Err", "t-stat", "p-value", "Sig"])
            log.info("\n" + coef_df.to_string(index=False))
            log.info(f"R-squared: {ols_summary['r_squared']:.4f}, Adj R-squared: {ols_summary['adj_r_squared']:.4f}")
            log.info(f"F-statistic: {ols_summary['f_statistic']:.2f}, p-value: {ols_summary['f_pvalue']:.2e}")
            log.info(f"N observations: {ols_summary['n_obs']}")

            log.info("=" * 80)
            log.info(f"GRANGER CAUSALITY TESTS (max_lag={max_lag}): macro -> DD")
            log.info("=" * 80)
            granger_data = []
            for col in macro_cols:
                g = granger_results[col]
                if "error" in g:
                    granger_data.append([col, "-", "-", "-", g["error"]])
                else:
                    sig = (
                        "***"
                        if g["p_value"] < 0.001
                        else "**"
                        if g["p_value"] < 0.01
                        else "*"
                        if g["p_value"] < 0.05
                        else ""
                    )
                    granger_data.append([col, g["best_lag"], f"{g['f_statistic']:.2f}", f"{g['p_value']:.4f}", sig])

            granger_df = pd.DataFrame(granger_data, columns=["Variable", "Best Lag", "F-stat", "p-value", "Sig"])
            log.info("\n" + granger_df.to_string(index=False))
            log.info("Significance: *** p<0.001, ** p<0.01, * p<0.05")

        log.info("Macro-DD significance analysis completed.")
        return self

    def compare_macro_models(
        self,
        n_months: int = 12,
        models: List[str] = None,
        target_col: str = "DD",
        verbose: bool = False,
    ) -> "Portfolio":
        """Computes quantitative accuracy metrics (MAE, RMSE, MAPE) for macro models.

        Runs walk-forward backtest: for each of the last ``n_months``, predicts
        macro variables one step ahead and compares to actuals.  Results are
        stored in ``self.d['macro_model_comparison']``.

        Args:
            n_months: Number of months in the backtest window.
            models: List of model types to evaluate. Defaults to ['var', 'sarimax', 'prophet'].
            target_col: Portfolio target metric ('DD' or 'PD').
            verbose: Whether to display the comparison table.

        Returns:
            Portfolio: Self with ``self.d['macro_model_comparison']`` populated.
        """
        if models is None:
            models = ["var", "sarimax", "prophet"]

        macro_cols = ["inflation", "interest_rate", "unemployment_rate", "rubusd_exchange_rate"]
        all_cols = macro_cols + [target_col]

        # Build monthly actuals
        port_target = self.d["portfolio"].groupby("date")[target_col].mean().reset_index()
        macro_df = (
            self.d["portfolio"][["date"] + macro_cols]
            .drop_duplicates("date")
            .merge(port_target, on="date", how="left")
            .set_index("date")
            .resample("ME")
            .mean()
            .dropna()
        )
        macro_df.index = macro_df.index.normalize() + pd.offsets.MonthEnd(0)

        rows = []
        for m_type in models:
            errors = {col: [] for col in all_cols}
            for offset in range(n_months, 0, -1):
                fc = self.predict_macro_factors(horizon=1, training_offset=offset, model_type=m_type)
                target_date = fc.index[-1]

                # Target col prediction
                if target_col == "DD":
                    self.predict_dd(horizon=1, training_offset=offset, model_type=m_type)
                    pred_target = self.d["dd_forecast"]["predicted_dd"].mean()
                else:
                    self.predict_pd(horizon=1, training_offset=offset, model_type=m_type)
                    pred_target = self.d["pd_forecast"]["predicted_pd"].mean()

                if target_date not in macro_df.index:
                    continue

                actual = macro_df.loc[target_date]
                for col in macro_cols:
                    if col in fc.columns:
                        errors[col].append(fc[col].values[0] - actual[col])
                errors[target_col].append(pred_target - actual[target_col])

            for col in all_cols:
                errs = np.array(errors[col])
                if len(errs) == 0:
                    continue
                actuals_abs = np.abs(macro_df[col].tail(n_months).values[: len(errs)])
                mape_vals = np.abs(errs) / np.where(actuals_abs > 1e-8, actuals_abs, 1e-8)
                rows.append(
                    {
                        "Model": m_type,
                        "Variable": col,
                        "MAE": round(np.mean(np.abs(errs)), 6),
                        "RMSE": round(np.sqrt(np.mean(errs**2)), 6),
                        "MAPE (%)": round(np.mean(mape_vals) * 100, 2),
                    }
                )

        comparison_df = pd.DataFrame(rows)
        self.d["macro_model_comparison"] = comparison_df

        log.log_dataframe(comparison_df, title="Macro Model Comparison (Walk-Forward)")

        if verbose:
            plots.plot_macro_model_comparison(comparison_df, verbose=verbose)

        return self

    def plot_pd_forecast(self, figsize: tuple = (12, 6), verbose: bool = False) -> "Portfolio":
        """
        Plots the comparison between predicted and reference PD for each ticker.

        Args:
            figsize (tuple): Figure size.
            verbose (bool): Whether to show the plot.

        Returns:
            Portfolio: Self.
        """
        if "pd_forecast" not in self.d:
            log.error("No PD forecast found. Run predict_pd() first.")
            return self

        plots.plot_pd_forecast(self.d["pd_forecast"], figsize=figsize, verbose=verbose)
        return self

    def plot_dd_forecast(self, figsize: tuple = (12, 6), verbose: bool = False) -> "Portfolio":
        """
        Plots the comparison between predicted and reference DD for each ticker.

        Args:
            figsize (tuple): Figure size.
            verbose (bool): Whether to show the plot.

        Returns:
            Portfolio: Self.
        """
        if "dd_forecast" not in self.d:
            log.error("No DD forecast found. Run predict_dd() first.")
            return self

        plots.plot_dd_forecast(self.d["dd_forecast"], figsize=figsize, verbose=verbose)
        return self

    def backtest_portfolio_strategies(
        self,
        n_months: int = 3,
        lambda_risk: float = 0.5,
        lgd: float = 0.4,
        model_type: str = "var",
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        strategy: Union[str, List[str]] = "mean_el",
        cvar_alpha: float = 0.95,
    ) -> "Portfolio":
        """Backtests Active (threshold-rebalanced) vs Passive (equal weights) strategy.

        Accepts one or several strategies. When a list is supplied, every
        strategy is evaluated over the **same** walk-forward window and
        a combined comparison table is logged at the end.

        Active strategy tracks actual weight drift after each period and only
        rebalances when any weight deviates from the optimized target by more
        than ``rebalance_threshold``.  Transaction costs are proportional to
        the total absolute turnover.

        Threshold rebalancing rule:
            Rebalance if  max_i |w_i^actual - w_i^target| > delta

        Weight drift formula each period:
            w_i^actual(t) = w_i(t-1) * (1 + r_i(t)) / sum_j w_j(t-1) * (1 + r_j(t))

        Transaction cost:
            TC(t) = c * sum_i |w_i^new(t) - w_i^actual(t)|

        Args:
            n_months: Number of months for the backtest window.
            lambda_risk: Risk aversion for active optimization (mean_el strategy).
            lgd: Loss Given Default.
            model_type: Macro model used for forecasts.
            rebalance_threshold: Max allowed drift before rebalancing (0-1).
                E.g. 0.05 means rebalance when any weight drifts >5 pp.
            transaction_cost: One-way proportional cost per unit of turnover.
                E.g. 0.001 = 10 basis points round-trip.
            min_weight: Minimum weight per asset (forces diversification).
            max_weight: Maximum weight per asset (prevents concentration).
            strategy: One strategy name or a list of strategy names.
                Supported values: ``"mean_el"``, ``"cvar"``, ``"risk_parity"``.
            cvar_alpha: CVaR confidence level (only used when ``strategy="cvar"``).

        Returns:
            Portfolio: Self with results stored in self.d['strategy_backtest'].
        """
        strategies: List[str] = [strategy] if isinstance(strategy, str) else list(strategy)

        log.info(
            f"Starting strategy backtest for the last {n_months} months "
            f"(strategies={strategies}, threshold={rebalance_threshold:.1%}, "
            f"tc={transaction_cost:.4f})..."
        )

        # 1. Prepare monthly data
        prices_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
        monthly_prices = prices_pivot.resample("ME").last()
        monthly_returns = monthly_prices.pct_change().dropna()

        pd_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="PD")
        monthly_pds = pd_pivot.resample("ME").last().dropna()

        # Align history
        common_index = monthly_returns.index.intersection(monthly_pds.index)
        monthly_returns = monthly_returns.loc[common_index]
        monthly_pds = monthly_pds.loc[common_index]

        # 2. Historical Baseline (BEFORE backtest) -- shared across strategies
        hist_baseline: List[dict] = []
        hist_returns = monthly_returns.iloc[:-n_months]
        for date in hist_returns.index:
            available_tickers = monthly_returns.columns[monthly_returns.loc[date].notna()]
            n = len(available_tickers)
            if n == 0:
                continue

            w_eq = 1.0 / n
            ret_eq = (monthly_returns.loc[date, available_tickers] * w_eq).sum()
            el_eq = (monthly_pds.loc[date, available_tickers] * w_eq).sum() * lgd

            hist_baseline.append(
                {
                    "Date": date,
                    "Active_Return": ret_eq,
                    "Active_EL": el_eq,
                    "Passive_Return": ret_eq,
                    "Passive_EL": el_eq,
                    "Actual_PD": monthly_pds.loc[date].mean(),
                    "Turnover": 0.0,
                    "TC": 0.0,
                    "Rebalanced": False,
                }
            )

        # 3. Walk-forward Backtest loop
        if len(monthly_returns) < n_months:
            log.error(f"Insufficient history for {n_months} months backtest.")
            return self

        # Per-strategy state
        strat_state: Dict[str, dict] = {}
        for strat in strategies:
            strat_state[strat] = {
                "w_active_actual": None,
                "w_target": None,
                "total_rebalances": 0,
                "weight_history": [],
                "results": list(hist_baseline),
            }

        for offset in range(n_months, 0, -1):
            target_date = monthly_returns.index[-offset]
            log.info(f"Backtesting month: {target_date.strftime('%Y-%m')}")

            # Predict DD once per month (shared across strategies)
            self.predict_dd(horizon=1, training_offset=offset, model_type=model_type)

            for strat in strategies:
                st = strat_state[strat]

                # --- Compute new TARGET weights from optimization ---
                # cutoff_date=target_date prevents look-ahead bias in covariance
                self.optimize_portfolio(
                    lambda_risk=lambda_risk,
                    lgd=lgd,
                    use_forecast=True,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    strategy=strat,
                    cvar_alpha=cvar_alpha,
                    cutoff_date=target_date,
                )
                w_target = self.d["optimized_weights"]

                available_tickers = w_target.index.intersection(monthly_returns.columns)
                w_target = w_target.loc[available_tickers]
                w_target = w_target / w_target.sum()

                w_active_actual = st["w_active_actual"]

                # --- ACTIVE with Threshold Rebalancing ---
                if w_active_actual is None:
                    w_active_actual = w_target.copy()
                    turnover = w_target.abs().sum()
                    rebalanced = True
                    st["total_rebalances"] += 1
                else:
                    rets = monthly_returns.loc[target_date, available_tickers].fillna(0)
                    common = w_active_actual.index.intersection(available_tickers)
                    w_drifted = w_active_actual.reindex(available_tickers, fill_value=0.0)
                    w_drifted.loc[common] = w_active_actual.loc[common] * (1 + rets.loc[common])
                    total_val = w_drifted.sum()
                    w_drifted = w_drifted / total_val if total_val > 0 else w_target.copy()

                    max_drift = (w_drifted - w_target).abs().max()
                    if max_drift > rebalance_threshold:
                        turnover = (w_target - w_drifted).abs().sum()
                        w_active_actual = w_target.copy()
                        rebalanced = True
                        st["total_rebalances"] += 1
                        log.info(
                            f"  [{strat}] Rebalancing triggered "
                            f"(max drift={max_drift:.2%}, turnover={turnover:.2%})"
                        )
                    else:
                        turnover = 0.0
                        w_active_actual = w_drifted
                        rebalanced = False

                st["w_active_actual"] = w_active_actual
                st["w_target"] = w_target

                tc = transaction_cost * turnover
                ret_active = np.sum(w_active_actual * monthly_returns.loc[target_date, available_tickers]) - tc
                el_active = np.sum(w_active_actual * monthly_pds.loc[target_date, available_tickers] * lgd)

                # --- PASSIVE (EQUAL WEIGHTS) ---
                n_tickers = len(available_tickers)
                w_passive = pd.Series(1.0 / n_tickers, index=available_tickers)
                ret_passive = np.sum(w_passive * monthly_returns.loc[target_date, available_tickers])
                el_passive = np.sum(w_passive * monthly_pds.loc[target_date, available_tickers] * lgd)

                # Record weight snapshot
                weight_record = {"date": target_date, "rebalanced": rebalanced}
                for ticker in available_tickers:
                    weight_record[ticker] = w_active_actual.get(ticker, 0.0)
                st["weight_history"].append(weight_record)

                st["results"].append(
                    {
                        "Date": target_date,
                        "Active_Return": ret_active,
                        "Active_EL": el_active,
                        "Passive_Return": ret_passive,
                        "Passive_EL": el_passive,
                        "Actual_PD": monthly_pds.loc[target_date].mean(),
                        "Turnover": turnover,
                        "TC": tc,
                        "Rebalanced": rebalanced,
                    }
                )

        # 4. Build per-strategy summaries
        all_backtests: Dict[str, pd.DataFrame] = {}
        all_weight_histories: Dict[str, pd.DataFrame] = {}
        comparison_rows: List[dict] = []

        for strat in strategies:
            st = strat_state[strat]
            backtest_df = pd.DataFrame(st["results"]).set_index("Date")
            all_backtests[strat] = backtest_df

            bt_only = backtest_df.tail(n_months)

            comparison_rows.append(
                {
                    "Strategy": strat,
                    "Total Return (%)": round(((1 + bt_only["Active_Return"]).prod() - 1) * 100, 4),
                    "Avg Realized EL (%)": round(bt_only["Active_EL"].mean() * 100, 6),
                    "Realized Vol (%)": round(bt_only["Active_Return"].std() * np.sqrt(self.MONTHS_PER_YEAR) * 100, 4),
                    "Total TC (%)": round(bt_only["TC"].sum() * 100, 4),
                    "Rebalances": int(bt_only["Rebalanced"].sum()),
                    "Avg Turnover (%)": round(bt_only["Turnover"].mean() * 100, 4),
                }
            )

            # Weight history
            wh_df = pd.DataFrame(st["weight_history"]).set_index("date")
            rebal_flags = wh_df.pop("rebalanced")
            wh_df = wh_df.fillna(0.0)
            wh_df["rebalanced"] = rebal_flags
            all_weight_histories[strat] = wh_df

            log.info(
                f"[{strat}] Backtest completed. "
                f"Rebalances: {st['total_rebalances']}/{n_months} "
                f"(threshold={rebalance_threshold:.1%})"
            )

        # Add passive baseline row (same for all strategies)
        last_bt = all_backtests[strategies[-1]]
        bt_passive = last_bt.tail(n_months)
        comparison_rows.append(
            {
                "Strategy": "passive (equal)",
                "Total Return (%)": round(((1 + bt_passive["Passive_Return"]).prod() - 1) * 100, 4),
                "Avg Realized EL (%)": round(bt_passive["Passive_EL"].mean() * 100, 6),
                "Realized Vol (%)": round(bt_passive["Passive_Return"].std() * np.sqrt(self.MONTHS_PER_YEAR) * 100, 4),
                "Total TC (%)": 0.0,
                "Rebalances": 0,
                "Avg Turnover (%)": 0.0,
            }
        )

        multi_comparison = pd.DataFrame(comparison_rows).set_index("Strategy")

        # 5. Log comparison table
        log.log_dataframe(multi_comparison.reset_index(), title="Strategy Comparison")

        # 6. Store results
        self.d["all_backtests"] = all_backtests
        self.d["all_weight_histories"] = all_weight_histories
        self.d["multi_strategy_comparison"] = multi_comparison

        # Backward-compatible keys: use the last strategy in the list
        last_strat = strategies[-1]
        self.d["strategy_backtest"] = all_backtests[last_strat]
        self.d["weight_history"] = all_weight_histories[last_strat]

        # Build backward-compatible two-row comparison (Active vs Passive)
        last_bt_only = all_backtests[last_strat].tail(n_months)
        summary_compat = {
            "Total Return (%)": [
                ((1 + last_bt_only["Active_Return"]).prod() - 1) * 100,
                ((1 + last_bt_only["Passive_Return"]).prod() - 1) * 100,
            ],
            "Avg Realized EL (%)": [
                last_bt_only["Active_EL"].mean() * 100,
                last_bt_only["Passive_EL"].mean() * 100,
            ],
            "Realized Volatility (%)": [
                last_bt_only["Active_Return"].std() * np.sqrt(self.MONTHS_PER_YEAR) * 100,
                last_bt_only["Passive_Return"].std() * np.sqrt(self.MONTHS_PER_YEAR) * 100,
            ],
            "Total Transaction Cost (%)": [
                last_bt_only["TC"].sum() * 100,
                0.0,
            ],
            "Rebalances": [
                int(last_bt_only["Rebalanced"].sum()),
                0,
            ],
            "Avg Turnover (%)": [
                last_bt_only["Turnover"].mean() * 100,
                0.0,
            ],
        }
        self.d["strategy_comparison"] = pd.DataFrame(summary_compat, index=["Active (Optimized)", "Passive (Equal)"])

        return self

    def plot_strategy_backtest(self, tail: int = 12, verbose: bool = False):
        """
        Plots the comparison of cumulative returns and EL between strategies.
        """
        if "strategy_backtest" not in self.d:
            log.error("No strategy backtest results found.")
            return self

        plots.plot_strategy_comparison(
            self.d["strategy_backtest"],
            self.d["strategy_comparison"],
            tail=tail,
            verbose=verbose,
            all_backtests=self.d.get("all_backtests"),
        )
        return self

    def log_weight_history(self) -> "Portfolio":
        """Logs the portfolio weight history as a formatted table via the logger.

        Shows how active strategy weights evolved at each backtest period,
        marking rebalanced periods with [R].

        Returns:
            Portfolio: Self for chaining.
        """
        if "weight_history" not in self.d or self.d["weight_history"] is None:
            log.error("No weight history found. Run backtest_portfolio_strategies first.")
            return self

        wh = self.d["weight_history"].copy()
        rebalanced = wh.pop("rebalanced")

        # Build a display DataFrame with date, rebalanced flag, and ticker weights
        ticker_cols = [c for c in wh.columns]
        display_rows = []
        for date, row in wh.iterrows():
            record = {
                "Date": date.strftime("%Y-%m"),
                "Reb": "[R]" if rebalanced.loc[date] else "",
            }
            for col in ticker_cols:
                record[col] = f"{row[col]:.1%}"
            display_rows.append(record)

        display_df = pd.DataFrame(display_rows)
        log.log_dataframe(display_df, title="Portfolio Weight History (Active Strategy)")
        return self

    def plot_weight_history(self, figsize: tuple = (14, 7), verbose: bool = False) -> "Portfolio":
        """Plots the evolution of portfolio weights over time during backtesting.

        Args:
            figsize: Figure size.
            verbose: Whether to display the plot interactively.

        Returns:
            Portfolio: Self for chaining.
        """
        if "weight_history" not in self.d or self.d["weight_history"] is None:
            log.error("No weight history found. Run backtest_portfolio_strategies first.")
            return self

        plots.plot_weight_history(self.d["weight_history"], figsize=figsize, verbose=verbose)
        return self

    def plot_ticker_dashboards(self, tickers: list, figsize_row: tuple = (18, 5), verbose: bool = False) -> "Portfolio":
        """
        Plots Stock, PD and Cap/Debt dashboards for each ticker in a grid.

        Args:
            tickers (list): List of tickers to plot.
            figsize_row (tuple): Size of one ticker row (3 subplots).
            verbose (bool): Whether to show the plot.
        """
        plots.plot_ticker_dashboards_grid(self.d["portfolio"], tickers, figsize_row=figsize_row, verbose=verbose)
        return self

    def plot_pd_by_tickers(self, tickers: list, figsize: tuple = (12, 5), verbose: bool = False) -> "Portfolio":
        """
        Plots the probability of default (PD) for the given tickers.

        Args:
            tickers (list): List of stock tickers (e.g., ['GAZP', 'FESH']).
            figsize (tuple): Size of the plot. Default is (12, 6).
            verbose (bool): If True, displays the plot. If False, saves the plot to a file.

        Returns:
            Portfolio: Updated portfolio with plotted probabilities of default.
        """
        plots.plot_pd_by_tickers(self.d["portfolio"], tickers, figsize, verbose)
        return self

    def calc_irf(
        self,
        impulses_responses: Dict[str, str] = None,
        figsize: Tuple[int, int] = (12, 5),
        verbose: bool = False,
    ) -> "Portfolio":
        """
        Calculates impulse response functions for the given impulses and responses.

        Args:
            impulses_responses (dict[str, str], optional): Dictionary of impulses and responses
            (e.g., {'interest_rate': 'PD', 'inflation': 'PD'}).
            figsize (tuple[int, int], optional): Size of the plot. Default is (10, 5).
            verbose (bool, optional): If True, displays the plot. If False, saves the plot to a file.

        Returns:
            Portfolio: Updated portfolio with calculated impulse response functions.
        """
        plots.calc_irf(self.d["portfolio"], impulses_responses, figsize, verbose)
        return self

    def plot_correlation_matrix(
        self,
        custom_order: list,
        save_path: str = None,
        figsize: tuple = (15, 10),
        dpi: int = 300,
        annot_size: int = 8,
        verbose: bool = False,
    ) -> "Portfolio":
        """
        Plots and saves the correlation matrix of stock closing prices.

        Args:
            custom_order (list): Order of tickers for grouping.
            save_path (str, optional): Path to save the plot (None - do not save).
            figsize (tuple[int, int], optional): Size of the plot. Default is (15, 10).
            dpi (int, optional): Quality of saving. Default is 300.
            annot_size (int, optional): Size of annotations. Default is 8.
            verbose (bool, optional): If True, displays the plot. If False, saves the plot to a file.

        Returns:
            Portfolio: Updated portfolio with plotted correlation matrix.
        """
        plots.plot_correlation_matrix(self.d["portfolio"], custom_order, save_path, figsize, dpi, annot_size, verbose)
        return self

    def plot_stocks(
        self,
        tickers: List[str],
        figsize: Tuple[int, int] = (10, 4),
        verbose: bool = False,
        fontsize: int = 16,
    ) -> "Portfolio":
        """
        Plots stock charts for the given tickers.

        Args:
            tickers (list[str]): List of stock tickers (e.g., ['FESH', 'GAZP']).
            figsize (tuple[int, int], optional): Size of the plot. Default is (10, 5).
            verbose (bool, optional): If True, displays the plot. If False, saves the plot to a file.
            fontsize (int, optional): Font size for the plot. Default is 16.

        Returns:
            Portfolio: Updated portfolio with plotted stock charts.
        """
        plots.plot_stocks(self.d["portfolio"], tickers, figsize, verbose, fontsize)
        return self

    def plot_debt_capitalization(self, verbose=False, figsize=(10, 4)):
        """
        Plots a combined chart of capitalization and debt on the same Y-axis.

        Args:
            verbose (bool, optional): If True, displays the plot. If False, saves the plot to a file.
            figsize (tuple[int, int], optional): Size of the plot. Default is (10, 5).

        Returns:
            Portfolio: Updated portfolio with plotted capitalization and debt.
        """
        plots.plot_debt_capitalization(self.d["portfolio"], verbose, figsize)
        return self

    def plot_macro_significance(
        self,
        save_path: str = "logs/graphs/macro_significance_summary.png",
        verbose: bool = False,
        figsize: tuple = (10, 6),
    ) -> "Portfolio":
        """
        Plots the significance of macroeconomic factors on Merton model parameters.

        Args:
            save_path (str): Path to save the plot. Default is "logs/graphs/macro_significance_summary.png".
            verbose (bool): If True, displays the plot. If False, saves the plot to a file. Default is False.
            figsize (tuple): Size of the plot. Default is (10, 6).

        Returns:
            Portfolio: Updated portfolio with plotted macro significance.
        """
        plots.plot_macro_significance(self.d["macro_connection_summary"], save_path, verbose, figsize)
        return self

    def log_completion(self):
        """
        Logs the completion of the analysis.

        Returns:
            Portfolio: Updated portfolio with logged completion.
        """

        self.end_time = datetime.now()

        log.info("=" * 60)
        log.info(
            "ANALYSIS COMPLETED | Duration: %.1f sec",
            (self.end_time - self.start_time).total_seconds(),
        )
        log.info("=" * 60)

        return self
