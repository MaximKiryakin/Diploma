

from utils.load_data import load_stock_data, load_multipliers
from typing import List, Optional
from utils.logger import Logger
import pickle
from statsmodels.tsa.api import VAR
import os
from scipy.optimize import root
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

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

    def load_stock_data(
            self,
            tickers_list: list[str] = None,
            use_backup_data: bool = False,
            create_backup: bool = False,
            backup_path: str = "data/backup/stocks.pkl"
    ) -> "Portfolio":

        if use_backup_data:
            if not os.path.isfile(backup_path):
                log.error(f"Backup file was not found: {backup_path}")
                raise Exception   # TODO specify exception list for this project

            with open(backup_path, 'rb') as f:
                self.stocks = pickle.load(f)

            log.info(f"Stocks data was loaded from backup")
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

        log.info("Multipliers data was loaded")

        return self

    def create_portfolio(self):

        self.portfolio = (
            self.stocks.merge(self.multipliers, on=['ticker', 'year', 'quarter'], how='left')
        )

        log.info("Portfolio was created")

        return self

    def adjust_portfolio_data_types(self):

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

        #self.portfolio['Долг, млрд руб'] = np.where(
        #    (self.portfolio['Долг, млрд руб'].notna()) & (self.portfolio['Чистый долг, млрд руб'].ne(0)),
        #    self.portfolio['Долг, млрд руб'],
        #    self.portfolio['Чистый долг, млрд руб']
        #)

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

        log.info(f"The following column types were adjusted: \n{column_to_adjust}")

        return self

    def _solve_merton_vectorized(
            self,
            r: float = 0.21,
            sigma_E: float= 0.4,
            T: float = 1
    ) -> "Portfolio":
        """
        Решение системы уравнений для оценки V и sigma_V.

        Параметры:
            E (float): Рыночная капитализация (equity).
            D (float): Долг (debt).
            r (float): Безрисковая ставка.
            sigma_E (float): Волатильность акций.
            T (float): Горизонт времени.

        Возвращает:
            tuple: (V, sigma_V)
        """
        E = self.portfolio["capitalization"].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)

        def equations(vars, E_i, D_i, r_i, sigma_E_i, T_i):
            V, sigma_V = vars
            d1 = np.log(V / D_i) + (r_i + 0.5 * sigma_V**2) * T_i
            d1 /= (sigma_V * np.sqrt(T_i))
            N_d1 = norm.cdf(d1)
            eq1 = V * N_d1 - D_i * np.exp(-r_i * T_i) * norm.cdf(d1 - sigma_V * np.sqrt(T_i)) - E_i
            eq2 = N_d1 * sigma_V * V - sigma_E_i * E_i
            return [eq1, eq2]

        # Начальные приближения для всех элементов
        initial_guess = np.vstack([E + D, np.full_like(E, sigma_E)]).T

        # Решение для каждого элемента
        results = np.array([
            root(equations, guess, args=(E[i], D[i], r, sigma_E, T)).x
            for i, guess in enumerate(initial_guess)
        ])

        self.portfolio['V'] = np.where(results[:, 0] <= 0, 1e-6, results[:, 0])
        self.portfolio['sigma_V'] = results[:, 1]

        log.info(f"Capital cost and capital volatility were successfully calculated")

        return self

    def _merton_pd(
        self,
        r: float = 0.21,
        T: float = 1
    ) -> "Portfolio":
        """
        Расчет вероятности дефолта (PD) по модели Мертона.

        Параметры:
            V (float): Рыночная стоимость активов компании.
            D (float): Уровень долга (обязательства) компании.
            r (float): Безрисковая процентная ставка (десятичная дробь, например, 0.05 для 5%).
            sigma_V (float): Волатильность стоимости активов (десятичная дробь).
            T (float): Горизонт времени до погашения долга (в годах).

        Возвращает:

        """

        V = self.portfolio['V'].values.astype(float)
        D = self.portfolio["debt"].values.astype(float)
        sigma_V = self.portfolio['sigma_V']

        d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))

        self.portfolio['PD'] = norm.cdf(-d2)

        log.info(f"Merton's probabilities of default were successfully calculated")

        return self

    def add_merton_pd(self) -> "Portfolio":

        self = self._solve_merton_vectorized()._merton_pd()

        self.portfolio = self.portfolio.drop(columns=['V', 'sigma_V'])

        return self

    def fill_missing_values(self):

        self.portfolio = self.portfolio.sort_values(by=['ticker', 'date'])

        columns_to_fill = ['debt', 'capitalization']
        for col in columns_to_fill:
            self.portfolio[col] = self.portfolio.groupby('ticker')[col].transform(
                lambda x: x.ffill().bfill()
            )

        log.info(f"Missing values were filled in the following columns: {columns_to_fill}")

        return self

    def add_macro_data(self):

        df = pd.read_excel('data/interestRateInflation/inflation.xlsx')
        df['date'] = pd.to_datetime(df["Дата"])

        self.portfolio = self.portfolio.merge(df, on="date", how="left")

        self.portfolio = self.portfolio.drop(columns=['Дата', 'Цель по инфляции'])
        self.portfolio = self.portfolio.rename(
            columns={
                'Ключевая ставка, % годовых': 'interest_rate',
                'Инфляция, % г/г': 'inflation'
            }
        )

        log.info("Interest rate and inflation were added")

        return self

    def plot_pd_by_ticker(
        self,
        ticker: str,
        save_path: str = None,
        figsize: tuple = (12, 6),
        verbose: bool = False
    ) -> "Portfolio":
        """
        Строит график вероятности дефолта (PD) для указанного тикера

        Параметры:
        ticker (str): Тикер акции (например: 'GAZP')
        save_path (str, optional): Путь для сохранения графика. Пример: 'images/gazp_pd.png'
        figsize (tuple): Размер графика. По умолчанию (12, 6)
        """

        if save_path is None:
            save_path = f'logs/graphs/{ticker}_pd.png'

        data = self.portfolio.query(f"ticker == '{ticker}'")

        if data.empty:
            log.info(f"No data for ticker {ticker}")
            return self

        sns.set_theme(style="whitegrid")
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
            log.info(f"Plot was saved: {save_path}")

        if verbose:
            plt.show()
        else:
            plt.clf()

        return self

    def calc_irf(
        self,
        columns=None,
        impulse='interest_rate',
        figsize: tuple = (12, 6),
        response='PD'
    ):

        if columns is None:
            columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'interest_rate', 'inflation', 'PD']

        data = self.portfolio.sort_values(["ticker", "date"])[columns].dropna()[columns[2:]]


        cols_before_diff = {}
        for col in data.columns:
            result = adfuller(data[col].dropna())[1]
            cols_before_diff[col] = result

        log.info("p-values before differencing:\n%s",
                 pd.Series(cols_before_diff))

        if any(p > 0.05 for p in cols_before_diff.values()):
            data = data.diff().dropna()
            log.info("Applied differencing to achieve stationarity")

        for col in data.columns:
            result = adfuller(data[col].dropna())[1]
            cols_before_diff[col] = result

        log.info("p-values after differencing:\n%s", pd.Series(cols_before_diff))

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.RangeIndex(start=0, stop=len(data))

        model = VAR(data)
        lag_order = model.select_order(maxlags=6)
        selected_lags = lag_order.aic
        log.info(f'Optimal lag order: {selected_lags}')

        results = model.fit(maxlags=selected_lags, ic='aic')
        plt.figure(figsize=figsize)
        irf = results.irf(periods=selected_lags)
        irf.plot(impulse=impulse, response=response, orth=True, figsize=figsize)
        plt.title(f'Импульсный отклик: Шок {impulse} → {response}')
        plt.show()

        return self

    def plot_correlation_matrix(
        self,
        custom_order: list,
        save_path: str = None,
        figsize: tuple = (15, 10),
        dpi: int = 300,
        annot_size: int = 8
    ) -> None:
        """
        Строит и сохраняет корреляционную матрицу цен закрытия акций

        :param custom_order: Порядок тикеров для группировки
        :param save_path: Путь для сохранения графика (None - не сохранять)
        :param figsize: Размер графика
        :param dpi: Качество сохранения
        :param annot_size: Размер аннотаций
        """

        # Создаем сводную таблицу
        pivot_data = self.portfolio.pivot_table(
            index='date',
            columns='ticker',
            values='close'
        )

        # Интерполяция и фильтрация данных
        pivot_data = pivot_data.interpolate(method='time', limit_direction='both')
        valid_tickers = [t for t in custom_order if t in pivot_data.columns]

        if not valid_tickers:
            raise ValueError("Нет данных для построения матрицы")

        pivot_data = pivot_data[valid_tickers]

        # Рассчитываем позиции для разделительных линий
        sector_breaks = [3, 6, 9, 12]  # Позиции после каждой отрасли


        # Строим матрицу корреляций
        corr_matrix = pivot_data.corr()

        # Визуализация
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": annot_size}
        )

        # Добавляем разделители
        for pos in sector_breaks:
            plt.axvline(pos, color='black', linewidth=2)
            plt.axhline(pos, color='black', linewidth=2)

        plt.title("Корреляция цен закрытия акций", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Сохранение
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()
        return self








