

from utils.load_data import load_stock_data, load_multipliers
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
from utils.logger import Logger
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

log = Logger(__name__).get_logger()


class Portfolio:
    def __init__(self, dt_calc: str, dt_start: str, stocks_step: int, tickers_list: list[str]):
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

        log.info("="*60)
        log.info(f"ANALYSIS STARTED | Python {sys.version} | Matplotlib {matplotlib.__version__}")
        log.info("="*60)

        return self

    def log_completion(self):
        """
        Logs the completion of the analysis.

        Returns:
            Portfolio: Updated portfolio with logged completion.
        """

        self.end_time = datetime.now()

        log.info("="*60)
        log.info("ANALYSIS COMPLETED | Duration: %.1f sec", (self.end_time - self.start_time).total_seconds())
        log.info("="*60)

        return self

    def load_stock_data(
            self,
            tickers_list: list[str] = None,
            use_backup_data: bool = False,
            create_backup: bool = False,
            backup_path: str = "data/backup/stocks.pkl"
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
                log.error(f"Backup file was not found: {backup_path}")
                raise Exception   # TODO specify exception list for this project

            with open(backup_path, 'rb') as f:
                self.stocks = pickle.load(f)

            log.info("Stocks data loaded from backup | Records: %d", self.stocks.shape[0])
        else:
            self.stocks = load_stock_data(
                tickers_list=self.tickers_list if tickers_list is None else tickers_list,
                start_date=self.dt_start,
                end_date=self.dt_calc,
                step=self.stocks_step
            )

            log.info(f"Stocks data was loaded from finam")

        if create_backup:
            with open(backup_path, 'wb') as f:
                pickle.dump(self.stocks, f)

            log.info(f"Backup file was saved: {backup_path}")

        self.stocks = (
            self.stocks
            .rename(columns={col: col[1:-1].lower() for col in self.stocks.columns})
            .assign(date=lambda x: pd.to_datetime(x['date']) + pd.offsets.MonthEnd(0))
            .assign(quarter=lambda x: pd.to_datetime(x['date']).dt.quarter)
            .assign(year=lambda x: pd.to_datetime(x['date']).dt.year)
            .drop(columns=['per', 'vol'])
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
                id_vars=['company', 'characteristic'],
                var_name='year_quarter',
                value_name='value'
            )
            .assign(year=lambda x: x['year_quarter'].str.split('_', expand=True)[0])
            .assign(quarter=lambda x: x['year_quarter'].str.split('_', expand=True)[1])
            .drop('year_quarter', axis=1).astype({'year': int, 'quarter': int})
            .set_index(['company', 'year', 'quarter', 'characteristic'])['value']
            .unstack().reset_index()
            .rename(columns={'company': 'ticker'})
        )

        log.info("Multipliers data loaded | Features: %s", list(self.multipliers.columns))

        return self

    def create_portfolio(self):
        """
        Creates a portfolio by merging stocks and multipliers data.

        Returns:
            Portfolio: Created portfolio.
        """

        self.portfolio = (
            self.stocks.merge(self.multipliers, on=['ticker', 'year', 'quarter'], how='left')
        )

        log.info("Portfolio created | Companies: %d", len(self.portfolio.ticker.unique()))

        return self

    def adjust_portfolio_data_types(self):
        """
        Adjusts the data types of the portfolio data.

        Returns:
            Portfolio: Updated portfolio with adjusted data types.
        """

        columns_new_names = {
            'Долг, млрд руб': 'debt',
            'Капитализация, млрд руб': 'capitalization'
        }

        column_to_adjust = [
            'Долг, млрд руб', 'Капитализация, млрд руб',
            'Чистый долг, млрд руб', 'high', 'low', 'close',
            'EV/EBITDA', 'P/BV', 'P/E', 'P/S', 'open', 'Долг/EBITDA'
        ]

        for col in column_to_adjust:
            self.portfolio[col] = self.portfolio[col].str.replace(" ", "", regex=False)
            self.portfolio[col] = pd.to_numeric(self.portfolio[col], errors="coerce")
            if 'млрд руб' in col:
                self.portfolio[col] *= 1e9

        self.portfolio['Долг, млрд руб'] = np.select(
            [
                (self.portfolio['Долг, млрд руб'].notna())
                & (self.portfolio['Долг, млрд руб'].ne(0)),

                (self.portfolio['Долг, млрд руб'].isna())
                & ((self.portfolio['Чистый долг, млрд руб'].isna())
                | (self.portfolio['Чистый долг, млрд руб'].eq(0))),

                (self.portfolio['Долг, млрд руб'].isna())
                & (self.portfolio['Чистый долг, млрд руб'].notna())
                & (self.portfolio['Чистый долг, млрд руб'].ne(0))
            ],
            [
                self.portfolio['Долг, млрд руб'],
                self.portfolio['Долг, млрд руб'],
                self.portfolio['Чистый долг, млрд руб']
            ]
        )

        self.portfolio['Долг, млрд руб'] = np.where(
            self.portfolio['Долг, млрд руб'] < 0,
                np.abs(self.portfolio['Долг, млрд руб']),
            self.portfolio['Долг, млрд руб']
        )

        self.portfolio = (
            self.portfolio
            .rename(columns=columns_new_names)
            .drop(columns=['Чистый долг, млрд руб'])
        )

        log.info("Column types adjusted: %s", column_to_adjust)

        return self

    def add_macro_data(self):
        """
        Adds macroeconomic data to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added macroeconomic data.
        """

        # TODO create special loader for data
        unemployment = pd.read_excel("data/macro/unemployment.xlsx")
        inflation = pd.read_excel('data/macro/inflation.xlsx')
        rub_usd = pd.read_excel("data/macro/rubusd.xlsx")
        inflation['date'] = pd.to_datetime(inflation["Дата"])

        self.portfolio = (
            self.portfolio
            .merge(inflation, on="date", how="left")
            .merge(unemployment, left_on="year", right_on="Year", how="left")
            .merge(rub_usd, left_on="date", right_on="data", how="left")
            .drop(columns=['Дата', 'Цель по инфляции', 'data'])
        )

        self.portfolio = self.portfolio.rename(
            columns={
                'Ключевая ставка, % годовых': 'interest_rate',
                'Инфляция, % г/г': 'inflation',
                'Unemployment': 'unemployment_rate',
                'curs': 'usd_rub'
            }
        )

        for col in ['inflation', 'interest_rate', 'unemployment_rate']:
            self.portfolio[col] /= 100

        log.info("Macro indicators added: Interest rate, Unemployment, Inflation, USD/RUB")

        return self

    def fill_missing_values(self):
        """
        Fills missing values in the portfolio data.

        Returns:
            Portfolio: Updated portfolio with missing values filled.
        """

        self.portfolio = self.portfolio.sort_values(by=['ticker', 'date'])

        missing = {}
        columns_to_fill = ['debt', 'capitalization', 'usd_rub']
        for col in columns_to_fill:

            missing[col] = self.portfolio[col].isna().sum() / self.portfolio.shape[0]
            self.portfolio[col] = self.portfolio.groupby('ticker')[col].transform(
                lambda x: x.ffill().bfill()
            )

        log.info(
            f"Missing values share in: Debt ({np.round(missing['debt'],3)*100} %),"
            + f"Cap ({np.round(missing['capitalization'], 3)*100} %), "
            + f"USD/RUB ({np.round(missing['usd_rub']*100, 1)} %)"
        )

        log.info(f"Missing values filled in: {columns_to_fill}")

        return self

    def _solve_merton_vectorized(
        self,
        T: float = 1
    ) -> "Portfolio":
        """
        Solves the system of equations to estimate V and sigma_V.

        Args:
            T (float): Time horizon.

        Returns:
            Portfolio: Updated portfolio with calculated capital cost and capital volatility.
        """

        E = self.portfolio["capitalization"].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)
        sigma_E = self.portfolio['quarterly_volatility'].values.astype(float)

        def equations(vars, E_i, D_i, r_i, sigma_E_i, T_i):
            V, sigma_V = vars
            d1 = np.log(V / D_i if D_i != 0 else 1e-6) + (r_i + 0.5 * sigma_V**2) * T_i
            d1 /= (sigma_V * np.sqrt(T_i))
            N_d1 = norm.cdf(d1)
            eq1 = V * N_d1 - D_i * np.exp(-r_i * T_i) * norm.cdf(d1 - sigma_V * np.sqrt(T_i)) - E_i
            eq2 = N_d1 * sigma_V * V - sigma_E_i * E_i
            return [eq1, eq2]

        # Initial guesses for all elements
        initial_guess = np.vstack([E + D, sigma_E]).T

        # Solve for each element
        results = np.array([
            root(equations, guess, args=(E[i], D[i], self.portfolio['interest_rate'][i], sigma_E[i], T)).x
            for i, guess in enumerate(initial_guess)
        ])

        self.portfolio['V'] = np.where(results[:, 0] <= 0, 1e-6, results[:, 0])
        self.portfolio['sigma_V'] = results[:, 1]

        log.info(f"Capital cost and capital volatility calculated.")

        return self

    def _merton_pd(
        self,
        T: float = 1
    ) -> "Portfolio":
        """
        Calculates the probability of default (PD) using the Merton model.

        Args:
            T (float): Time horizon for the default event (in years).

        Returns:
            Portfolio: Updated portfolio with calculated probabilities of default.
        """

        V = self.portfolio['V'].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)
        sigma_V = self.portfolio['sigma_V']

        d2 = (np.log(V / np.where(D != 0, D, 1e-6)) + (self.portfolio['interest_rate'] - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        self.portfolio['PD'] = norm.cdf(-d2)

        log.info(f"Merton's probabilities of default calculated.")

        return self

    def add_merton_pd(self) -> "Portfolio":
        """
        Adds the probability of default (PD) calculated using the Merton model to the portfolio data.

        Returns:
            Portfolio: Updated portfolio with added probabilities of default.
        """
        self = self._solve_merton_vectorized()._merton_pd()

        self.portfolio = self.portfolio.drop(columns=['V', 'sigma_V'])

        return self

    def plot_pd_by_tickers(
        self,
        tickers: list,
        figsize: tuple = (10, 4),
        verbose: bool = False
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

            save_path = f'logs/graphs/{ticker}_pd.png'

            data = self.portfolio.query(f"ticker == '{ticker}'")

            if data.empty:
                log.info(f"No data for ticker {ticker}")
                continue

            plt.figure(figsize=figsize)

            plt.plot(
                data['date'],
                data['PD'] * 100,
                marker='o',
                linestyle='--',
                color='royalblue',
                linewidth=2,
                markersize=5
            )

            plt.title(f'Вероятность дефолта ({ticker})', fontsize=14, pad=20)
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('PD, %', fontsize=12)
            plt.xticks(rotation=0)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if tickers:
            log.info(
                "PD graphs saved | "
                f"Companies: {len(tickers)} | "
                f"Path: logs/graphs/"
            )

        return self

    def calc_irf(
            self,
            impulses_responses: Dict[str, str] = None,
            figsize: Tuple[int, int] = (10, 4),
            verbose: bool = False
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

        columns = np.unique(list(impulses_responses.keys()) + list(impulses_responses.values()))

        data = self.portfolio.sort_values(["ticker", "date"])[columns].dropna()[columns]

        cols_before_diff = {}
        for col in data.columns:
            cols_before_diff[col] = adfuller(data[col].dropna())[1]

        if any(p > 0.05 for p in cols_before_diff.values()):

            log.info("p-values before differencing:\n%s", pd.Series(cols_before_diff))

            data = data.diff().dropna()
            log.info("Applied differencing to achieve stationarity")

            for col in data.columns:
                cols_before_diff[col] = adfuller(data[col].dropna())[1]

            log.info("p-values after differencing:\n%s", pd.Series(cols_before_diff))

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.RangeIndex(start=0, stop=len(data))

        model = VAR(data)
        lag_order = model.select_order(maxlags=6)
        selected_lags = lag_order.aic

        log.info(f'Optimal lag number calculated | Optimal number of lags: {selected_lags}')

        results = model.fit(maxlags=selected_lags, ic='aic')

        for impulse, response in impulses_responses.items():
            irf = results.irf(periods=selected_lags)

            ax = irf.plot(
                impulse=impulse,
                response=response,
                orth=True,
                figsize=figsize,
                plot_params={
                    'title': None,
                    'subtitle': False
                }
            )

            fig = ax.get_figure()

            fig.suptitle('')
            for a in fig.axes:
                a.set_title('')

            fig.suptitle(
                f"Impulse Response Function (IRF): {impulse} → {response}\n"
                f"Method: VAR with AIC lag selection | 95% Confidence Intervals",
                fontsize=11, y=1.02
            )

            plt.xlabel('Горизонт, кварталы')
            plt.ylabel('Изменение PD, базисные пункты')

            save_path = f'logs/graphs/irf_{impulse}_{response}.png'
            plt.savefig(save_path, bbox_inches='tight')

            if verbose:
                plt.show()
            else:
                plt.clf()
        plt.close()

        log.info(f"Impulse response functions saved | Path: logs/graphs/")

        return self

    def plot_correlation_matrix(
        self,
        custom_order: list,
        save_path: str = None,
        figsize: tuple = (15, 10),
        dpi: int = 300,
        annot_size: int = 8,
        verbose: bool = False
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
            save_path = f'logs/graphs/corr_matrix.png'

        pivot_data = self.portfolio.pivot_table(
            index='date',
            columns='ticker',
            values='close'
        )

        pivot_data = pivot_data.interpolate(method='time', limit_direction='both')
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
            annot_kws={"size": annot_size}
        )

        for pos in sector_breaks:
            plt.axvline(pos, color='black', linewidth=2)
            plt.axhline(pos, color='black', linewidth=2)

        plt.title("Корреляция цен закрытия акций", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            log.info(f"Correlation matrix saved | Path: {save_path}")

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

            stock_data = self.portfolio[self.portfolio['ticker'] == ticker]

            save_path = f'logs/graphs/{ticker}_stock.png'

            if stock_data.empty:
                raise ValueError(f"Ticker {ticker} not found in portfolio")

            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(
                stock_data['date'],
                stock_data['close'],
                label='Closing price',
                color='royalblue',
                linewidth=2
            )

            ax.set_title(f'Stock dynamics {ticker}', fontsize=fontsize, pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price, RUB',)
            ax.legend(frameon=True, facecolor='white')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
            plt.xticks(rotation=0)

            ax.grid(True, alpha=0.3)
            plt.tight_layout()
    
            if save_path:
                plt.savefig(
                    save_path,
                    dpi=100,
                    bbox_inches='tight',
                    facecolor='white'
                )

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if tickers:
            log.info(
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

        self.portfolio['quarterly_volatility'] = (
            self.portfolio
            .groupby(['ticker', pd.Grouper(key='date', freq='QE')])['close']
            .transform(
                lambda x: np.std(np.log(x / x.shift(1))) * np.sqrt(63)  # 63 ≈ среднее число торговых дней в квартале
            )
        )

        self.portfolio['quarterly_volatility'] = self.portfolio['quarterly_volatility'].rolling(window=10).mean()

        self.portfolio['quarterly_volatility'] = self.portfolio['quarterly_volatility'].bfill()

        # Adhoc values for missing quarterly volatility data
        self.portfolio['quarterly_volatility'] = 0.4

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

        save_path = f'logs/graphs/debt_catitalization.png'
        grouped = self.portfolio.groupby('ticker')

        for ticker, group in grouped:

            group = group.sort_values('date').dropna(subset=['capitalization', 'debt'])
            plt.figure(figsize=figsize)

            plt.plot(group['date'], group['capitalization'],
                    marker='o', linestyle='-', color='#2ecc71', linewidth=2,
                    markersize=8, label='Capitalization')

            plt.plot(group['date'], group['debt'],
                    marker='s', linestyle='--', color='#e74c3c', linewidth=2,
                    markersize=8, label='Debt')

            plt.title(f'{ticker}: Capitalization vs Debt', fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.legend(
                loc='upper left',
                frameon=True,
                shadow=True,
                fontsize=12,
                facecolor='white'
            )

            plt.tight_layout()

            plt.savefig(save_path, bbox_inches='tight', dpi=100)

            if verbose:
                plt.show()
            else:
                plt.clf()
            plt.close()

        if save_path:
            log.info(
                "Capitalization-debt graphs saved | "
                f"Companies: {len(self.portfolio.ticker.unique())} | "
                f"Path: {save_path}"
            )

        return self

    def calc_macro_connections(
        self,
        min_samples: int = 10,
        n_bootstraps: int = 500,
        conf_level: int = 95
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
        targets = ['debt', 'capitalization']

        results = []

        tickers = df['ticker'].unique()

        def format_ci(low, high):
            return f"[{low:.3f}, {high:.3f}]"

        for ticker in tickers:

            df_ticker = df[df['ticker'] == ticker].copy()

            if len(df_ticker) < min_samples:
                continue

            for target in targets:
                record = {
                    'ticker': ticker,
                    'target': target,
                    'best_alpha': np.nan,
                    'mse_model': np.nan,
                    'mse_baseline': np.nan,
                    'r2': np.nan,
                    'coef_inflation': np.nan,
                    'coef_inflation_ci': np.nan,
                    'coef_unemployment': np.nan,
                    'coef_unemployment_ci': np.nan,
                    'coef_usd_rub': np.nan,
                    'coef_usd_rub_ci': np.nan
                }

                try:
                    Q1 = df_ticker[target].quantile(0.05)
                    Q3 = df_ticker[target].quantile(0.95)
                    df_target = df_ticker[(df_ticker[target] >= Q1) & (df_ticker[target] <= Q3)].copy()

                    if len(df_target) < 5:
                        continue

                    y = np.log(df_target[target] + 1e-9)
                    X = df_target[['inflation', 'unemployment_rate', 'usd_rub']]

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
                        {'alpha': np.logspace(-3, 2, 50)},
                        cv=5,
                        scoring='neg_mean_squared_error'
                    )
                    grid.fit(X_train_scaled, y_train)
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_test_scaled)

                    coefs = []
                    for _ in range(n_bootstraps):
                        X_bs, y_bs = resample(X_train_scaled, y_train)
                        model = Ridge(alpha=grid.best_params_['alpha'])
                        model.fit(X_bs, y_bs)
                        coefs.append(model.coef_)

                    alpha = (100 - conf_level) / 2
                    ci_low, ci_high = alpha, 100 - alpha
                    coefs = np.array(coefs)

                    low_inf, high_inf = np.percentile(coefs[:, 0], [ci_low, ci_high])
                    low_unemp, high_unemp = np.percentile(coefs[:, 1], [ci_low, ci_high])
                    low_usd, high_usd = np.percentile(coefs[:, 2], [ci_low, ci_high])

                    ci_inflation = format_ci(low_inf, high_inf)
                    ci_unemployment = format_ci(low_unemp, high_unemp)
                    ci_usd_rub = format_ci(low_usd, high_usd)

                    record.update({
                        'best_alpha': grid.best_params_['alpha'],
                        'mse_model': mean_squared_error(y_test, y_pred),
                        'mse_baseline': mse_baseline,
                        'r2': r2_score(y_test, y_pred),
                        'coef_inflation': best_model.coef_[0],
                        'coef_inflation_ci': ci_inflation,
                        'coef_unemployment': best_model.coef_[1],
                        'coef_unemployment_ci': ci_unemployment,
                        'coef_usd_rub': best_model.coef_[2],
                        'coef_usd_rub_ci': ci_usd_rub
                    })

                except Exception as e:
                    log.error(f"Ошибка для {ticker}-{target}: {str(e)}")
                    continue

                results.append(record)

        result_df = pd.DataFrame(results)
        result_df = result_df.dropna(subset=['best_alpha'])

        self.macro_connection_summary = result_df
        log.info("Macro connection summary calculated.")

        return self
