from utils.load_data import load_stock_data, load_multipliers, get_rubusd_exchange_rate
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
import pickle
from statsmodels.tsa.api import VAR
import os
from sklearn.utils import resample
from scipy.optimize import root
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import sys
import matplotlib
from datetime import datetime, timedelta

class Portfolio:
    def __init__(
        self, dt_calc: str, dt_start: str, stocks_step: int, tickers_list: list[str]
    ):
        self.dt_calc = dt_calc
        self.dt_start = dt_start
        self.stocks_step = stocks_step
        self.multipliers = None
        self.stocks = None
        self.tickers_list = tickers_list
        self.portfolio = None
        self.macro_connection_summary = None
        self.end_time = None
        self.start_time = None

    def log_system_info(self):
        """
        Logs system information.

        Returns:
            Portfolio: Updated portfolio with logged system information.
        """

        self.start_time = datetime.now()

        print("=" * 60)
        print(
            f"ANALYSIS STARTED | Python {sys.version} | Matplotlib {matplotlib.__version__}"
        )
        print("=" * 60)

        return self

    def log_completion(self):
        """
        Logs the completion of the analysis.

        Returns:
            Portfolio: Updated portfolio with logged completion.
        """

        self.end_time = datetime.now()

        print("=" * 60)
        print(
            "ANALYSIS COMPLETED | Duration: %.1f sec",
            (self.end_time - self.start_time).total_seconds(),
        )
        print("=" * 60)

        return self

    def load_stock_data(
        self,
        tickers_list: list[str] = None,
        use_backup_data: bool = False,
        create_backup: bool = False,
        backup_path: str = "data/backup/stocks.pkl",
    ) -> "Portfolio":
        """
        Loads stock data for the given tickers.

        Args:
            tickers_list (list[str], optional): List of tickers. If not specified, uses the default tickers list.
            use_backup_data (bool, optional): If True, loads stock data from backup file. Defaults to False.
            create_backup (bool, optional): If True, creates a backup file with the loaded stock data. Defaults to False.
            backup_path (str, optional): Path to the backup file. Defaults to "data/backup/stocks.pkl".

        Returns:
            Portfolio: Updated portfolio with loaded stock data.
        """

        if use_backup_data:
            if not os.path.isfile(backup_path):
                print('ERROR:', f"Backup file was not found: {backup_path}")
                raise Exception  # TODO specify exception list for this project

            with open(backup_path, "rb") as f:
                self.stocks = pickle.load(f)

            print(
                "Stocks data loaded from backup | Records: %d", self.stocks.shape[0]
            )
        else:
            self.stocks = load_stock_data(
                tickers_list=(
                    self.tickers_list if tickers_list is None else tickers_list
                ),
                start_date=self.dt_start,
                end_date=self.dt_calc,
                step=self.stocks_step,
            )

            print(f"Stocks data was loaded from finam")

        if create_backup:
            with open(backup_path, "wb") as f:
                pickle.dump(self.stocks, f)

            print(f"Backup file was saved: {backup_path}")

        self.stocks = (
            self.stocks.rename(
                columns={col: col[1:-1].lower() for col in self.stocks.columns}
            )
            .assign(date=lambda x: pd.to_datetime(x["date"]) + pd.offsets.MonthEnd(0))
            .assign(quarter=lambda x: pd.to_datetime(x["date"]).dt.quarter)
            .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
            .drop(columns=["per", "vol"])
        )

        return self

    def load_multipliers(self, tickers_list: list[str] = None):
        """
        Loads multipliers data for the given tickers.

        Args:
            tickers_list (list[str], optional): List of tickers. If not specified, uses the default tickers list.

        Returns:
            Portfolio: Updated portfolio with loaded multipliers data.
        """

        self.multipliers = load_multipliers(
            companies_list=self.tickers_list if tickers_list is None else tickers_list,
        )

        self.multipliers = (
            pd.melt(
                self.multipliers,
                id_vars=["company", "characteristic"],
                var_name="year_quarter",
                value_name="value",
            )
            .assign(year=lambda x: x["year_quarter"].str.split("_", expand=True)[0])
            .assign(quarter=lambda x: x["year_quarter"].str.split("_", expand=True)[1])
            .drop("year_quarter", axis=1)
            .astype({"year": int, "quarter": int})
            .set_index(["company", "year", "quarter", "characteristic"])["value"]
            .unstack()
            .reset_index()
            .rename(columns={"company": "ticker"})
        )

        print(
            "Multipliers data loaded | Features: %s", list(self.multipliers.columns)
        )

        return self

    def create_portfolio(self):
        """
        Creates a portfolio by merging stocks and multipliers data.

        Returns:
            Portfolio: Created portfolio.
        """

        self.portfolio = self.stocks.merge(
            self.multipliers, on=["ticker", "year", "quarter"], how="left"
        )

        print(
            "Portfolio created | Companies: %d", len(self.portfolio.ticker.unique())
        )

        return self

    def adjust_portfolio_data_types(self):
        """
        Adjusts the data types of the portfolio data.

        Returns:
            Portfolio: Updated portfolio with adjusted data types.
        """

        columns_new_names = {
            "Долг, млрд руб": "debt",
            "Капитализация, млрд руб": "capitalization",
        }

        column_to_adjust = [
            "Долг, млрд руб",
            "Капитализация, млрд руб",
            "Чистый долг, млрд руб",
            "high",
            "low",
            "close",
            "EV/EBITDA",
            "P/BV",
            "P/E",
            "P/S",
            "open",
            "Долг/EBITDA",
        ]

        for col in column_to_adjust:
            self.portfolio[col] = self.portfolio[col].str.replace(" ", "", regex=False)
            self.portfolio[col] = pd.to_numeric(self.portfolio[col], errors="coerce")
            if "млрд руб" in col:
                self.portfolio[col] *= 1e9

        self.portfolio["Долг, млрд руб"] = np.select(
            [
                (self.portfolio["Долг, млрд руб"].notna())
                & (self.portfolio["Долг, млрд руб"].ne(0)),
                (self.portfolio["Долг, млрд руб"].isna())
                & (
                    (self.portfolio["Чистый долг, млрд руб"].isna())
                    | (self.portfolio["Чистый долг, млрд руб"].eq(0))
                ),
                (self.portfolio["Долг, млрд руб"].isna())
                & (self.portfolio["Чистый долг, млрд руб"].notna())
                & (self.portfolio["Чистый долг, млрд руб"].ne(0)),
            ],
            [
                self.portfolio["Долг, млрд руб"],
                self.portfolio["Долг, млрд руб"],
                self.portfolio["Чистый долг, млрд руб"],
            ],
        )

        self.portfolio["Долг, млрд руб"] = np.where(
            self.portfolio["Долг, млрд руб"] < 0,
            np.abs(self.portfolio["Долг, млрд руб"]),
            self.portfolio["Долг, млрд руб"],
        )

        self.portfolio = self.portfolio.rename(columns=columns_new_names).drop(
            columns=["Чистый долг, млрд руб"]
        )

        print("Column types adjusted: %s", column_to_adjust)

        return self

    def add_macro_data(self):
        """
        Adds macroeconomic data to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added macroeconomic data.
        """

        # TODO create special loader for data
        unemployment = pd.read_excel("data/macro/unemployment.xlsx")
        inflation = pd.read_excel("data/macro/inflation.xlsx")
        rub_usd = get_rubusd_exchange_rate(
            dt_calc=self.dt_calc, dt_start=self.dt_start, update_backup=True
        )
        inflation["date"] = pd.to_datetime(inflation["Дата"])
        rub_usd["date"] = pd.to_datetime(rub_usd["date"])

        self.portfolio = (
            self.portfolio.merge(inflation, on="date", how="left")
            .merge(unemployment, left_on="year", right_on="Year", how="left")
            .merge(rub_usd, on="date", how="left")
            .drop(columns=["Дата", "Цель по инфляции"])
        )

        self.portfolio = self.portfolio.rename(
            columns={
                "Ключевая ставка, % годовых": "interest_rate",
                "Инфляция, % г/г": "inflation",
                "Unemployment": "unemployment_rate",
                "curs": "usd_rub",
            }
        )

        for col in ["inflation", "interest_rate", "unemployment_rate"]:
            self.portfolio[col] /= 100

        print(
            "Macro indicators added: Interest rate, Unemployment, Inflation, USD/RUB"
        )

        return self

    def fill_missing_values(self):
        """
        Fills missing values in the portfolio data.

        Returns:
            Portfolio: Updated portfolio with missing values filled.
        """

        self.portfolio = self.portfolio.sort_values(by=["ticker", "date"])

        missing = {}
        columns_to_fill = ["debt", "capitalization", "rubusd_exchange_rate"]
        for col in columns_to_fill:

            missing[col] = self.portfolio[col].isna().sum() / self.portfolio.shape[0]
            self.portfolio[col] = self.portfolio.groupby("ticker")[col].transform(
                lambda x: x.ffill().bfill()
            )

        print(
            f"Missing values share in: Debt ({np.round(missing['debt'],3)*100} %),"
            + f"Cap ({np.round(missing['capitalization'], 3)*100} %), "
            + f"USD/RUB ({np.round(missing['rubusd_exchange_rate']*100, 1)} %)"
        )

        print(f"Missing values filled in: {columns_to_fill}")

        return self

    def _solve_merton_vectorized(self, T: float = 1) -> "Portfolio":
        """
        Solves the system of equations to estimate V and sigma_V.

        Args:
            T (float): Time horizon.

        Returns:
            Portfolio: Updated portfolio with calculated capital cost and capital volatility.
        """

        E = self.portfolio["capitalization"].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)
        sigma_E = self.portfolio["quarterly_volatility"].values.astype(float)

        def equations(vars, E_i, D_i, r_i, sigma_E_i, T_i):
            V, sigma_V = vars
            d1 = np.log(V / D_i if D_i != 0 else 1e-6) + (r_i + 0.5 * sigma_V**2) * T_i
            d1 /= sigma_V * np.sqrt(T_i)
            N_d1 = norm.cdf(d1)
            eq1 = (
                V * N_d1
                - D_i * np.exp(-r_i * T_i) * norm.cdf(d1 - sigma_V * np.sqrt(T_i))
                - E_i
            )
            eq2 = N_d1 * sigma_V * V - sigma_E_i * E_i
            return [eq1, eq2]

        # Initial guesses for all elements
        initial_guess = np.vstack([E + D, sigma_E]).T

        # Solve for each element
        results = np.array(
            [
                root(
                    equations,
                    guess,
                    args=(
                        E[i],
                        D[i],
                        self.portfolio["interest_rate"][i],
                        sigma_E[i],
                        T,
                    ),
                ).x
                for i, guess in enumerate(initial_guess)
            ]
        )

        self.portfolio["V"] = np.where(results[:, 0] <= 0, 1e-6, results[:, 0])
        self.portfolio["sigma_V"] = results[:, 1]

        print(f"Capital cost and capital volatility calculated.")

        return self

    def _merton_pd(self, T: float = 1) -> "Portfolio":
        """
        Calculates the probability of default (PD) using the Merton model.

        Args:
            T (float): Time horizon for the default event (in years).

        Returns:
            Portfolio: Updated portfolio with calculated probabilities of default.
        """

        V = self.portfolio["V"].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)
        sigma_V = self.portfolio["sigma_V"]

        d2 = (
            np.log(V / np.where(D != 0, D, 1e-6))
            + (self.portfolio["interest_rate"] - 0.5 * sigma_V**2) * T
        ) / (sigma_V * np.sqrt(T))
        self.portfolio["PD"] = norm.cdf(-d2)

        print(f"Merton's probabilities of default calculated.")

        return self

    def add_merton_pd(self) -> "Portfolio":
        """
        Adds the probability of default (PD) calculated using the Merton model to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added probabilities of default.
        """
        self = self._solve_merton_vectorized()._merton_pd()

        self.portfolio = self.portfolio.drop(columns=["V", "sigma_V"])

        return self

    def plot_pd_by_tickers(
        self, tickers: list, figsize: tuple = (10, 4), verbose: bool = False
    ) -> "Portfolio":
        """
        Plots the probability of default (PD) for the given tickers.

        Args:
            tickers (list): List of stock tickers (e.g., ['GAZP', 'FESH']).
            figsize (tuple): Size of the plot. Default is (12, 6).
            verbose (bool): If True, displays the plot. If False, saves the plot to a file.

        Returns:
            Portfolio: Updated portfolio with plotted probabilities of default.
        """

        sns.set_theme(style="whitegrid")

        for ticker in tickers:

            save_path = f"logs/graphs/{ticker}_pd.png"

            data = self.portfolio.query(f"ticker == '{ticker}'")

            if data.empty:
                print(f"No data for ticker {ticker}")
                continue

            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(
                data["date"],
                data["PD"] * 100,
                marker="o",
                linestyle="--",
                color="royalblue",
                linewidth=2,
                markersize=5,
            )

            ax.set_title(f"Вероятность дефолта ({ticker})", fontsize=14, pad=20)
            ax.set_xlabel("Дата", fontsize=12)
            ax.set_ylabel("PD, %", fontsize=12)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", facecolor="white")

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if tickers:
            print(
                "PD graphs saved | "
                f"Companies: {len(tickers)} | "
                f"Path: logs/graphs/"
            )

        return self

    def calc_irf(
        self,
        impulses_responses: Dict[str, str] = None,
        figsize: Tuple[int, int] = (10, 4),
        verbose: bool = False,
    ) -> "Portfolio":
        """
        Calculates impulse response functions for the given impulses and responses.

        Args:
            impulses_responses (dict[str, str], optional): Dictionary of impulses and responses (e.g., {'interest_rate': 'PD', 'inflation': 'PD'}).
            figsize (tuple[int, int], optional): Size of the plot. Default is (10, 5).
            verbose (bool, optional): If True, displays the plot. If False, saves the plot to a file.

        Returns:
            Portfolio: Updated portfolio with calculated impulse response functions.
        """

        if impulses_responses is None:
            raise ValueError("Impulses and responses must be specified")

        columns = np.unique(
            list(impulses_responses.keys()) + list(impulses_responses.values())
        )

        data = self.portfolio.sort_values(["ticker", "date"])[columns].dropna()[columns]

        cols_before_diff = {}
        for col in data.columns:
            cols_before_diff[col] = adfuller(data[col].dropna())[1]

        if any(p > 0.05 for p in cols_before_diff.values()):

            print("p-values before differencing:\n%s", pd.Series(cols_before_diff))

            data = data.diff().dropna()
            print("Applied differencing to achieve stationarity")

            for col in data.columns:
                cols_before_diff[col] = adfuller(data[col].dropna())[1]

            print("p-values after differencing:\n%s", pd.Series(cols_before_diff))

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.RangeIndex(start=0, stop=len(data))

        model = VAR(data)
        lag_order = model.select_order(maxlags=6)
        selected_lags = lag_order.aic

        print(
            f"Optimal lag number calculated | Optimal number of lags: {selected_lags}"
        )

        results = model.fit(maxlags=selected_lags, ic="aic")

        for impulse, response in impulses_responses.items():
            irf = results.irf(periods=selected_lags)

            ax = irf.plot(
                impulse=impulse,
                response=response,
                orth=True,
                figsize=figsize,
                plot_params={"title": None, "subtitle": False},
            )

            fig = ax.get_figure()

            fig.suptitle("")
            for a in fig.axes:
                a.set_title("")

            fig.suptitle(
                f"Impulse Response Function (IRF): {impulse} -> {response}\n"
                f"Method: VAR with AIC lag selection | 95% Confidence Intervals",
                fontsize=11,
                y=1.02,
            )

            plt.xlabel("Горизонт, кварталы")
            plt.ylabel("Изменение PD, базисные пункты")

            save_path = f"logs/graphs/irf_{impulse}_{response}.png"
            plt.savefig(save_path, bbox_inches="tight", facecolor="white")

            if verbose:
                plt.show()
            else:
                plt.clf()
        plt.close()

        print(f"Impulse response functions saved | Path: logs/graphs/")

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

        if save_path is None:
            save_path = f"logs/graphs/corr_matrix.png"

        pivot_data = self.portfolio.pivot_table(
            index="date", columns="ticker", values="close"
        )

        pivot_data = pivot_data.interpolate(method="time", limit_direction="both")
        valid_tickers = [t for t in custom_order if t in pivot_data.columns]

        if not valid_tickers:
            raise ValueError("No data for plotting the matrix")

        pivot_data = pivot_data[valid_tickers]

        sector_breaks = [3, 6, 9, 12]

        corr_matrix = pivot_data.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": annot_size},
        )

        for pos in sector_breaks:
            plt.axvline(pos, color="black", linewidth=2)
            plt.axhline(pos, color="black", linewidth=2)

        plt.title("Корреляция цен закрытия акций", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Correlation matrix saved | Path: {save_path}")

        if verbose:
            plt.show()
        else:
            plt.clf()
        plt.close()

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

        for ticker in tickers:

            stock_data = self.portfolio[self.portfolio["ticker"] == ticker]

            save_path = f"logs/graphs/{ticker}_stock.png"

            if stock_data.empty:
                raise ValueError(f"Ticker {ticker} not found in portfolio")

            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(
                stock_data["date"],
                stock_data["close"],
                label="Closing price",
                color="royalblue",
                linewidth=2,
            )

            ax.set_title(f"Stock dynamics {ticker}", fontsize=fontsize, pad=20)
            ax.set_xlabel("Date")
            ax.set_ylabel(
                "Price, RUB",
            )
            ax.legend(frameon=True, facecolor="white")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
            plt.xticks(rotation=0)

            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if tickers:
            print(
                "Stock prices graphs saved | "
                f"Companies: {len(tickers)} | "
                f"Path: logs/graphs/"
            )

        return self

    def add_dynamic_features(self):
        """
        Adds dynamic features to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added dynamic features.
        """

        self.portfolio["quarterly_volatility"] = self.portfolio.groupby(
            ["ticker", pd.Grouper(key="date", freq="QE")]
        )["close"].transform(
            lambda x: np.std(np.log(x / x.shift(1)))
            * np.sqrt(63)  # 63 ≈ среднее число торговых дней в квартале
        )

        self.portfolio["quarterly_volatility"] = (
            self.portfolio["quarterly_volatility"].rolling(window=10).mean()
        )

        self.portfolio["quarterly_volatility"] = self.portfolio[
            "quarterly_volatility"
        ].bfill()

        # Adhoc values for missing quarterly volatility data
        self.portfolio["quarterly_volatility"] = 0.4

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

        save_path = f"logs/graphs/debt_catitalization.png"
        grouped = self.portfolio.groupby("ticker")

        for ticker, group in grouped:

            group = group.sort_values("date").dropna(subset=["capitalization", "debt"])
            plt.figure(figsize=figsize)

            plt.plot(
                group["date"],
                group["capitalization"],
                marker="o",
                linestyle="-",
                color="#2ecc71",
                linewidth=2,
                markersize=8,
                label="Capitalization",
            )

            plt.plot(
                group["date"],
                group["debt"],
                marker="s",
                linestyle="--",
                color="#e74c3c",
                linewidth=2,
                markersize=8,
                label="Debt",
            )

            plt.title(f"{ticker}: Capitalization vs Debt", fontsize=14, pad=20)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.legend(
                loc="upper left",
                frameon=True,
                shadow=True,
                fontsize=12,
                facecolor="white",
            )

            plt.tight_layout()

            plt.savefig(save_path, bbox_inches="tight", dpi=100)

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if save_path:
            print(
                "Capitalization-debt graphs saved | "
                f"Companies: {len(self.portfolio.ticker.unique())} | "
                f"Path: {save_path}"
            )

        return self

    def calc_macro_connections(
        self, min_samples: int = 10, n_bootstraps: int = 500, conf_level: int = 95
    ) -> "Portfolio":
        """
        Calculates macroeconomic connections for the given portfolio.

        Args:
            min_samples (int, optional): Minimum number of samples required for each ticker. Default is 10.
            n_bootstraps (int, optional): Number of bootstraps for confidence intervals. Default is 500.
            conf_level (int, optional): Confidence level for confidence intervals. Default is 95.

        Returns:
            Portfolio: Updated portfolio with calculated macroeconomic connections.
        """

        df = self.portfolio.copy()
        targets = ["debt", "capitalization"]

        results = []

        tickers = df["ticker"].unique()

        def format_ci(low, high):
            return f"[{low:.3f}, {high:.3f}]"

        for ticker in tickers:

            df_ticker = df[df["ticker"] == ticker].copy()

            if len(df_ticker) < min_samples:
                continue

            for target in targets:
                record = {
                    "ticker": ticker,
                    "target": target,
                    "best_alpha": np.nan,
                    "mse_model": np.nan,
                    "mse_baseline": np.nan,
                    "r2": np.nan,
                    "coef_inflation": np.nan,
                    "coef_inflation_ci": np.nan,
                    "coef_unemployment": np.nan,
                    "coef_unemployment_ci": np.nan,
                    "coef_usd_rub": np.nan,
                    "coef_usd_rub_ci": np.nan,
                }

                try:
                    Q1 = df_ticker[target].quantile(0.05)
                    Q3 = df_ticker[target].quantile(0.95)
                    df_target = df_ticker[
                        (df_ticker[target] >= Q1) & (df_ticker[target] <= Q3)
                    ].copy()

                    if len(df_target) < 5:
                        continue

                    y = np.log(df_target[target] + 1e-9)
                    X = df_target[
                        ["inflation", "unemployment_rate", "rubusd_exchange_rate"]
                    ]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    if len(y_test) == 0:
                        continue

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    y_pred_baseline = np.full_like(y_test, y_train.mean())
                    mse_baseline = mean_squared_error(y_test, y_pred_baseline)

                    ridge = Ridge()
                    grid = GridSearchCV(
                        ridge,
                        {"alpha": np.logspace(-3, 2, 50)},
                        cv=5,
                        scoring="neg_mean_squared_error",
                    )
                    grid.fit(X_train_scaled, y_train)
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_test_scaled)

                    coefs = []
                    for _ in range(n_bootstraps):
                        X_bs, y_bs = resample(X_train_scaled, y_train)
                        model = Ridge(alpha=grid.best_params_["alpha"])
                        model.fit(X_bs, y_bs)
                        coefs.append(model.coef_)

                    alpha = (100 - conf_level) / 2
                    ci_low, ci_high = alpha, 100 - alpha
                    coefs = np.array(coefs)

                    low_inf, high_inf = np.percentile(coefs[:, 0], [ci_low, ci_high])
                    low_unemp, high_unemp = np.percentile(
                        coefs[:, 1], [ci_low, ci_high]
                    )
                    low_usd, high_usd = np.percentile(coefs[:, 2], [ci_low, ci_high])

                    ci_inflation = format_ci(low_inf, high_inf)
                    ci_unemployment = format_ci(low_unemp, high_unemp)
                    ci_usd_rub = format_ci(low_usd, high_usd)

                    record.update(
                        {
                            "best_alpha": grid.best_params_["alpha"],
                            "mse_model": mean_squared_error(y_test, y_pred),
                            "mse_baseline": mse_baseline,
                            "r2": r2_score(y_test, y_pred),
                            "coef_inflation": best_model.coef_[0],
                            "coef_inflation_ci": ci_inflation,
                            "coef_unemployment": best_model.coef_[1],
                            "coef_unemployment_ci": ci_unemployment,
                            "coef_usd_rub": best_model.coef_[2],
                            "coef_usd_rub_ci": ci_usd_rub,
                        }
                    )

                except Exception as e:
                    print('ERROR:', f"Ошибка для {ticker}-{target}: {str(e)}")
                    continue

                results.append(record)

        result_df = pd.DataFrame(results)
        result_df = result_df.dropna(subset=["best_alpha"])

        self.macro_connection_summary = result_df
        print("Macro connection summary calculated.")

        return self

    # ========================================================================
    # МЕТОДЫ УПРАВЛЕНИЯ КРЕДИТНЫМ ПОРТФЕЛЕМ
    # ========================================================================

    def create_credit_risk_limits(
        self, 
        max_pd_threshold: float = 0.05,
        max_sector_concentration: float = 0.30,
        max_single_exposure: float = 0.10
    ) -> "Portfolio":
        """
        Создает лимиты кредитного риска для управления портфелем.
        
        Args:
            max_pd_threshold (float): Максимальная допустимая PD для выдачи кредита (5%)
            max_sector_concentration (float): Максимальная концентрация в одном секторе (30%)
            max_single_exposure (float): Максимальная доля одного заемщика (10%)
            
        Returns:
            Portfolio: Updated portfolio with credit risk limits.
        """
        
        self.credit_limits = {
            'max_pd_threshold': max_pd_threshold,
            'max_sector_concentration': max_sector_concentration,
            'max_single_exposure': max_single_exposure,
            'total_portfolio_limit': 1.0  # 100% лимит портфеля
        }
        
        print("Лимиты кредитного риска установлены:")
        print(f"   • Максимальная PD: {max_pd_threshold:.1%}")
        print(f"   • Секторная концентрация: {max_sector_concentration:.1%}")
        print(f"   • Доля одного заемщика: {max_single_exposure:.1%}")
        
        return self

    def analyze_portfolio_status(self) -> "Portfolio":
        """
        Анализирует текущее состояние портфеля и выводит статистики PD.
        
        Returns:
            Portfolio: Updated portfolio with analysis.
        """
        # Анализ текущих PD в портфеле
        current_pd_stats = self.portfolio.groupby('ticker')['PD'].last()
        
        print("Текущее состояние потенциальных заемщиков:")
        print(f"   • Средняя PD: {current_pd_stats.mean():.3f} ({current_pd_stats.mean()*100:.2f}%)")
        print(f"   • Компании с PD <= 5%: {(current_pd_stats <= 0.05).sum()}/{len(current_pd_stats)}")
        print(f"   • Компании с высоким риском (PD > 5%): {(current_pd_stats > 0.05).sum()}")

        # Показываем топ-5 самых надежных и рискованных
        print("ТОП-5 НАИБОЛЕЕ НАДЕЖНЫХ ЗАЕМЩИКОВ:")
        safest = current_pd_stats.sort_values().head(5)
        for ticker, pd_val in safest.items():
            print(f"   {ticker}: PD = {pd_val:.3f} ({pd_val*100:.2f}%)")

        print("ТОП-5 НАИБОЛЕЕ РИСКОВАННЫХ ЗАЕМЩИКОВ:")
        riskiest = current_pd_stats.sort_values(ascending=False).head(5)
        for ticker, pd_val in riskiest.items():
            print(f"   {ticker}: PD = {pd_val:.3f} ({pd_val*100:.2f}%)")
            
        return self

    def assess_credit_application(
        self, 
        borrower_ticker: str,
        loan_amount: float,
        current_portfolio_size: float = 1000000000  # 1 млрд рублей
    ) -> Dict[str, Any]:
        """
        Оценивает заявку на кредит на основе PD и лимитов риска.
        
        Args:
            borrower_ticker (str): Тикер заемщика (например, 'GAZP')
            loan_amount (float): Запрашиваемая сумма кредита в рублях
            current_portfolio_size (float): Текущий размер кредитного портфеля
            
        Returns:
            dict: Решение по кредитной заявке с обоснованием
        """
        
        if not hasattr(self, 'credit_limits'):
            self.create_credit_risk_limits()
        
        # Получаем текущую PD заемщика
        latest_data = self.portfolio[self.portfolio['ticker'] == borrower_ticker].tail(1)
        
        if latest_data.empty:
            return {
                'decision': 'ОТКЛОНИТЬ',
                'reason': f'Заемщик {borrower_ticker} отсутствует в базе данных',
                'pd': None,
                'risk_rating': 'UNKNOWN',
                'recommended_rate': None
            }
        
        borrower_pd = latest_data['PD'].iloc[0]
        
        # Определяем сектор заемщика
        sector_mapping = {
            'GAZP': 'Нефтегаз', 'LKOH': 'Нефтегаз', 'ROSN': 'Нефтегаз',
            'SBER': 'Финансы', 'VTBR': 'Финансы', 'MOEX': 'Финансы',
            'GMKN': 'Металлургия', 'NLMK': 'Металлургия', 'RUAL': 'Металлургия',
            'MTSS': 'Телеком', 'RTKM': 'Телеком', 'TTLK': 'Телеком',
            'MGNT': 'Ритейл', 'LNTA': 'Ритейл', 'FESH': 'Ритейл'
        }
        borrower_sector = sector_mapping.get(borrower_ticker, 'Прочие')
        
        # Оценка рисков
        exposure_ratio = loan_amount / current_portfolio_size
        
        # Проверяем лимиты
        checks = {
            'pd_check': borrower_pd <= self.credit_limits['max_pd_threshold'],
            'single_exposure_check': exposure_ratio <= self.credit_limits['max_single_exposure'],
            'pd_value': borrower_pd,
            'exposure_ratio': exposure_ratio,
            'sector': borrower_sector
        }
        
        # Определяем рейтинг риска
        if borrower_pd <= 0.01:
            risk_rating = 'AAA (Высший)'
            base_rate = 0.08  # 8%
        elif borrower_pd <= 0.02:
            risk_rating = 'AA (Высокий)'
            base_rate = 0.10  # 10%
        elif borrower_pd <= 0.03:
            risk_rating = 'A (Хороший)'
            base_rate = 0.12  # 12%
        elif borrower_pd <= 0.05:
            risk_rating = 'BBB (Удовлетворительный)'
            base_rate = 0.15  # 15%
        else:
            risk_rating = 'BB и ниже (Высокий риск)'
            base_rate = 0.20  # 20%
        
        # Корректировка ставки на концентрацию
        concentration_premium = max(0, (exposure_ratio - 0.05) * 2)  # +2% за каждые 5% сверх лимита
        recommended_rate = base_rate + concentration_premium
        
        # Принятие решения
        if checks['pd_check'] and checks['single_exposure_check']:
            decision = 'ОДОБРИТЬ'
            reason = f'Все лимиты соблюдены. PD: {borrower_pd:.3f} <= {self.credit_limits["max_pd_threshold"]:.3f}'
        elif not checks['pd_check']:
            decision = 'ОТКЛОНИТЬ'
            reason = f'Превышен лимит PD: {borrower_pd:.3f} > {self.credit_limits["max_pd_threshold"]:.3f}'
        elif not checks['single_exposure_check']:
            decision = 'ОТКЛОНИТЬ' if exposure_ratio > 0.15 else 'УСЛОВНО ОДОБРИТЬ'
            reason = f'Высокая концентрация: {exposure_ratio:.1%} от портфеля'
        else:
            decision = 'ОТКЛОНИТЬ'
            reason = 'Множественные нарушения лимитов'
        
        return {
            'borrower': borrower_ticker,
            'decision': decision,
            'reason': reason,
            'pd': borrower_pd,
            'risk_rating': risk_rating,
            'recommended_rate': recommended_rate,
            'loan_amount': loan_amount,
            'exposure_ratio': exposure_ratio,
            'sector': borrower_sector,
            'checks': checks
        }

    def optimize_credit_portfolio(
        self, 
        loan_applications: List[Dict],
        portfolio_budget: float = 1000000000,
        target_return: float = 0.12
    ) -> Dict[str, Any]:
        """
        Оптимизирует состав кредитного портфеля из заявок.
        
        Args:
            loan_applications (list): Список заявок [{'ticker': 'GAZP', 'amount': 100000000, 'rate': 0.10}]
            portfolio_budget (float): Общий бюджет для кредитования (1 млрд рублей)
            target_return (float): Целевая доходность портфеля (12%)
            
        Returns:
            dict: Оптимальный состав кредитного портфеля
        """
        
        if not hasattr(self, 'credit_limits'):
            self.create_credit_risk_limits()
        
        # Оцениваем каждую заявку
        evaluated_applications = []
        for app in loan_applications:
            assessment = self.assess_credit_application(
                app['ticker'], 
                app['amount'], 
                portfolio_budget
            )
            
            if assessment['decision'] in ['ОДОБРИТЬ', 'УСЛОВНО ОДОБРИТЬ']:
                app_with_assessment = app.copy()
                app_with_assessment.update(assessment)
                app_with_assessment['risk_adjusted_return'] = app.get('rate', assessment['recommended_rate']) - assessment['pd']
                evaluated_applications.append(app_with_assessment)
        
        # Сортируем по риск-скорректированной доходности
        evaluated_applications.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
        
        # Формируем оптимальный портфель
        selected_loans = []
        total_allocated = 0
        sector_allocation = {}
        
        for app in evaluated_applications:
            if total_allocated + app['amount'] <= portfolio_budget:
                # Проверяем секторные лимиты
                sector = app['sector']
                current_sector_allocation = sector_allocation.get(sector, 0)
                new_sector_allocation = (current_sector_allocation + app['amount']) / portfolio_budget
                
                if new_sector_allocation <= self.credit_limits['max_sector_concentration']:
                    selected_loans.append(app)
                    total_allocated += app['amount']
                    sector_allocation[sector] = current_sector_allocation + app['amount']
        
        # Расчет портфельных метрик
        portfolio_return = sum(loan['amount'] * loan.get('rate', loan['recommended_rate']) 
                              for loan in selected_loans) / total_allocated if total_allocated > 0 else 0
        
        portfolio_pd = sum(loan['amount'] * loan['pd'] for loan in selected_loans) / total_allocated if total_allocated > 0 else 0
        
        utilization_rate = total_allocated / portfolio_budget
        
        return {
            'selected_loans': selected_loans,
            'total_allocated': total_allocated,
            'portfolio_budget': portfolio_budget,
            'utilization_rate': utilization_rate,
            'portfolio_return': portfolio_return,
            'portfolio_pd': portfolio_pd,
            'portfolio_spread': portfolio_return - portfolio_pd,
            'sector_allocation': {k: v/portfolio_budget for k, v in sector_allocation.items()},
            'number_of_loans': len(selected_loans),
            'rejected_applications': len(loan_applications) - len(selected_loans)
        }

    def monitor_credit_portfolio_health(self) -> Dict[str, Any]:
        """
        Мониторинг состояния кредитного портфеля и раннее предупреждение рисков.
        
        Returns:
            dict: Отчет о состоянии портфеля с предупреждениями
        """
        
        if self.portfolio is None or self.portfolio.empty:
            return {'error': 'Portfolio data not available'}
        
        # Анализ текущих PD по всему портфелю
        latest_data = self.portfolio.groupby('ticker').last()
        
        # Статистики PD
        pd_stats = {
            'mean_pd': latest_data['PD'].mean(),
            'median_pd': latest_data['PD'].median(),
            'max_pd': latest_data['PD'].max(),
            'min_pd': latest_data['PD'].min(),
            'std_pd': latest_data['PD'].std(),
            'companies_count': len(latest_data)
        }
        
        # Предупреждения о рисках
        risk_warnings = []
        
        # Проверка высоких PD
        high_risk_companies = latest_data[latest_data['PD'] > 0.05]
        if not high_risk_companies.empty:
            risk_warnings.append({
                'type': 'HIGH_PD_WARNING',
                'message': f'Высокий риск дефолта у {len(high_risk_companies)} компаний',
                'companies': high_risk_companies.index.tolist(),
                'max_pd': high_risk_companies['PD'].max()
            })
        
        # Проверка роста PD (требует исторических данных)
        if len(self.portfolio) > len(self.tickers_list):  # есть история
            pd_changes = {}
            for ticker in self.tickers_list:
                ticker_data = self.portfolio[self.portfolio['ticker'] == ticker]['PD']
                if len(ticker_data) >= 2:
                    recent_change = ticker_data.iloc[-1] - ticker_data.iloc[-2]
                    if recent_change > 0.01:  # рост PD более чем на 1%
                        pd_changes[ticker] = recent_change
            
            if pd_changes:
                risk_warnings.append({
                    'type': 'PD_GROWTH_WARNING',
                    'message': f'Значительный рост PD у {len(pd_changes)} компаний',
                    'companies': pd_changes
                })
        
        # Секторный анализ рисков
        sector_mapping = {
            'GAZP': 'Нефтегаз', 'LKOH': 'Нефтегаз', 'ROSN': 'Нефтегаз',
            'SBER': 'Финансы', 'VTBR': 'Финансы', 'MOEX': 'Финансы',
            'GMKN': 'Металлургия', 'NLMK': 'Металлургия', 'RUAL': 'Металлургия',
            'MTSS': 'Телеком', 'RTKM': 'Телеком', 'TTLK': 'Телеком',
            'MGNT': 'Ритейл', 'LNTA': 'Ритейл', 'FESH': 'Ритейл'
        }
        
        latest_data['sector'] = latest_data.index.map(sector_mapping)
        sector_pd = latest_data.groupby('sector')['PD'].agg(['mean', 'max', 'count'])
        
        # Рекомендации по управлению
        recommendations = []
        
        if pd_stats['mean_pd'] > 0.03:
            recommendations.append('Средняя PD портфеля превышает 3% - рекомендуется ужесточить андеррайтинг')
        
        if pd_stats['max_pd'] > 0.08:
            recommendations.append('Есть компании с PD > 8% - требуется пересмотр лимитов или досрочное погашение')
        
        worst_sector = sector_pd['mean'].idxmax()
        if sector_pd.loc[worst_sector, 'mean'] > 0.05:
            recommendations.append(f'Сектор "{worst_sector}" показывает высокие риски - ограничить новые выдачи')
        
        return {
            'portfolio_health_score': max(0, min(100, 100 - pd_stats['mean_pd'] * 2000)),  # 0-100 шкала
            'pd_statistics': pd_stats,
            'sector_analysis': sector_pd.to_dict(),
            'risk_warnings': risk_warnings,
            'recommendations': recommendations,
            'monitoring_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def generate_credit_report(
        self, 
        loan_decisions: List[Dict] = None,
        save_path: str = None
    ) -> "Portfolio":
        """
        Генерирует отчет по управлению кредитным портфелем.
        
        Args:
            loan_decisions (list): Список решений по кредитам для включения в отчет
            save_path (str): Путь для сохранения отчета
            
        Returns:
            Portfolio: Updated portfolio with generated report
        """
        
        print("=" * 80)
        print("ОТЧЕТ ПО УПРАВЛЕНИЮ КРЕДИТНЫМ ПОРТФЕЛЕМ")
        print("=" * 80)
        
        # 1. Текущее состояние портфеля
        health_report = self.monitor_credit_portfolio_health()
        
        print(f"\n1. СОСТОЯНИЕ ПОРТФЕЛЯ (Оценка: {health_report['portfolio_health_score']:.1f}/100)")
        print("-" * 50)
        
        pd_stats = health_report['pd_statistics']
        print(f"Средняя PD портфеля: {pd_stats['mean_pd']:.3f} ({pd_stats['mean_pd']*100:.2f}%)")
        print(f"Медианная PD: {pd_stats['median_pd']:.3f}")
        print(f"Диапазон PD: {pd_stats['min_pd']:.3f} - {pd_stats['max_pd']:.3f}")
        print(f"Количество заемщиков: {pd_stats['companies_count']}")
        
        # 2. Предупреждения о рисках
        if health_report['risk_warnings']:
            print(f"\n2. ПРЕДУПРЕЖДЕНИЯ О РИСКАХ ({len(health_report['risk_warnings'])} активных)")
            print("-" * 50)
            for warning in health_report['risk_warnings']:
                print(f"⚠️  {warning['message']}")
                if 'companies' in warning and isinstance(warning['companies'], list):
                    print(f"    Компании: {', '.join(warning['companies'])}")
        else:
            print(f"\n2. ПРЕДУПРЕЖДЕНИЯ О РИСКАХ")
            print("-" * 50)
            print("✅ Критических рисков не выявлено")
        
        # 3. Секторный анализ
        print(f"\n3. АНАЛИЗ ПО СЕКТОРАМ")
        print("-" * 50)
        sector_analysis = health_report['sector_analysis']
        
        print(f"{'Сектор':<15} {'Средняя PD':<12} {'Макс PD':<10} {'Компании':<10}")
        print("-" * 50)
        for sector in sector_analysis['mean'].keys():
            mean_pd = sector_analysis['mean'][sector]
            max_pd = sector_analysis['max'][sector] 
            count = sector_analysis['count'][sector]
            print(f"{sector:<15} {mean_pd:.3f} ({mean_pd*100:>5.2f}%) {max_pd:.3f} ({max_pd*100:>5.2f}%) {count:>8}")
        
        # 4. Решения по кредитам (если предоставлены)
        if loan_decisions:
            print(f"\n4. РЕШЕНИЯ ПО КРЕДИТНЫМ ЗАЯВКАМ ({len(loan_decisions)} заявок)")
            print("-" * 50)
            
            approved = [d for d in loan_decisions if d.get('decision') == 'ОДОБРИТЬ']
            rejected = [d for d in loan_decisions if d.get('decision') == 'ОТКЛОНИТЬ']
            conditional = [d for d in loan_decisions if d.get('decision') == 'УСЛОВНО ОДОБРИТЬ']
            
            print(f"✅ Одобрено: {len(approved)} заявок")
            print(f"❌ Отклонено: {len(rejected)} заявок") 
            print(f"⚠️  Условно одобрено: {len(conditional)} заявок")
            
            if approved:
                total_approved_amount = sum(d.get('loan_amount', 0) for d in approved)
                avg_rate = sum(d.get('recommended_rate', 0) for d in approved) / len(approved)
                print(f"Общая сумма одобренных кредитов: {total_approved_amount:,.0f} руб.")
                print(f"Средняя рекомендованная ставка: {avg_rate:.2%}")
        
        # 5. Рекомендации
        print(f"\n5. РЕКОМЕНДАЦИИ ПО УПРАВЛЕНИЮ")
        print("-" * 50)
        for i, rec in enumerate(health_report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        if not health_report['recommendations']:
            print("✅ Портфель находится в хорошем состоянии. Текущая стратегия эффективна.")
        
        print(f"\n{'=' * 80}")
        print(f"Отчет сгенерирован: {health_report['monitoring_date']}")
        print(f"{'=' * 80}")
        
        # Сохранение отчета в файл (опционально)
        if save_path:
            # Здесь можно добавить сохранение в файл
            print(f"Credit portfolio report would be saved to: {save_path}")
        
        print("Credit portfolio management report generated")
        
        return self
