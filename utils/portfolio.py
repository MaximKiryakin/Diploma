

from utils.load_data import load_stock_data, load_multipliers
from typing import List, Optional
from utils.logger import Logger
import pickle
import os
import pandas as pd
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
            .assign(date=lambda x: pd.to_datetime(x['date']))
            .assign(quarter=lambda x: pd.to_datetime(x['date']).dt.quarter)
            .assign(year=lambda x: pd.to_datetime(x['date']).dt.year)
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
        )

        log.info("Multipliers data was loaded")

        return self

    def create_portfolio(self):

        self.portfolio = (
            self.stocks.merge(self.multipliers, on=['year', 'quarter'], how = 'left')
        )

        log.info("Portfolio was created")

        return self







