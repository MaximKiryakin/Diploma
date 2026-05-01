import utils.data_layer as data_layer
import utils.credit_risk as credit_risk
import utils.optimization as optimization
import utils.config as cfg
from utils.logger import Logger
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from datetime import datetime
import utils.plots as plots


log = Logger(__name__)


class Portfolio:
    def __init__(self, dt_calc: str, dt_start: str, stocks_step: int, tickers_list: list[str]):
        self.dt_calc = dt_calc
        self.dt_start = dt_start
        self.stocks_step = stocks_step
        self.tickers_list = tickers_list

        # Dictionary to store all dataframes
        self.d = {"stocks": None, "multipliers": None, "portfolio": None, "macro_connection_summary": None}

        self.end_time = None
        self.start_time = None

    def log_system_info(self) -> "Portfolio":
        """Logs system information and configuration parameters.

        Returns:
            Portfolio: self with start_time set.
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
        backup_path: str = cfg.BACKUP_STOCKS_PATH,
    ) -> "Portfolio":
        """Loads stock data for the given tickers.

        Args:
            tickers_list: List of tickers. If None, uses self.tickers_list.
            use_backup_data: If True, loads from backup. Defaults to True.
            update_backup: If True, downloads and updates backup. Defaults to False.
            backup_path: Path to backup file.

        Returns:
            Portfolio: self with self.d['stocks'] populated.
        """
        return data_layer.load_stock_data_fn(
            self,
            tickers_list=tickers_list,
            use_backup_data=use_backup_data,
            update_backup=update_backup,
            backup_path=backup_path,
        )

    def load_multipliers(
        self,
        tickers_list: list[str] = None,
        use_backup: bool = True,
        update_backup: bool = False,
        backup_path: str = cfg.BACKUP_MULTIPLIERS_PATH,
    ) -> "Portfolio":
        """Loads multipliers data for the given tickers.

        Args:
            tickers_list: List of tickers. If None, uses self.tickers_list.
            use_backup: If True, loads from backup. Defaults to True.
            update_backup: If True, updates backup. Defaults to False.
            backup_path: Path to backup file.

        Returns:
            Portfolio: self with self.d['multipliers'] populated.
        """
        return data_layer.load_multipliers_fn(
            self,
            tickers_list=tickers_list,
            use_backup=use_backup,
            update_backup=update_backup,
            backup_path=backup_path,
        )

    def _log_data_period(self, df: pd.DataFrame, date_col: str, title: str) -> None:
        """Helper method to log the start and end dates of a dataframe."""
        data_layer._log_data_period(df, date_col, title)

    def load_macro_data(
        self,
        update_inflation: bool = False,
        update_rub_usd: bool = False,
        update_unemployment: bool = False,
        inflation_path: str = cfg.MACRO_INFLATION_PATH,
        rub_usd_path: str = cfg.MACRO_RUBUSD_PATH,
        unemployment_path: str = cfg.MACRO_UNEMPLOYMENT_PATH,
    ) -> "Portfolio":
        """Adds macroeconomic data to the portfolio.

        Args:
            update_inflation: If True, downloads fresh inflation data from CBR. Defaults to False.
            update_rub_usd: If True, downloads fresh USD/RUB exchange rate data. Defaults to False.
            update_unemployment: If True, downloads fresh unemployment data. Defaults to False.
            inflation_path: Path to inflation data file.
            rub_usd_path: Path to USD/RUB exchange rate file.
            unemployment_path: Path to unemployment data file.

        Returns:
            Portfolio: self with macro data populated.
        """
        return data_layer.load_macro_data_fn(
            self,
            update_inflation=update_inflation,
            update_rub_usd=update_rub_usd,
            update_unemployment=update_unemployment,
            inflation_path=inflation_path,
            rub_usd_path=rub_usd_path,
            unemployment_path=unemployment_path,
        )

    def create_portfolio(self) -> "Portfolio":
        """Merges stocks, multipliers, and macro data into self.d['portfolio'].

        Returns:
            Portfolio: self with self.d['portfolio'] populated.
        """
        return data_layer.create_portfolio_fn(self)

    def _solve_merton_vectorized(self, T: float = 1) -> "Portfolio":
        """Solves Merton equations to estimate asset value V and sigma_V.

        Args:
            T: Time horizon in years. Defaults to 1.

        Returns:
            Portfolio: self with 'V' and 'sigma_V' in self.d['portfolio'].
        """
        return data_layer._solve_merton_vectorized_fn(self, T=T)

    def _merton_pd(self, T: float = 1) -> "Portfolio":
        """Calculates PD and DD from solved Merton V and sigma_V.

        Args:
            T: Time horizon in years. Defaults to 1.

        Returns:
            Portfolio: self with 'PD' and 'DD' in self.d['portfolio'].
        """
        return data_layer._merton_pd_fn(self, T=T)

    def add_merton_pd(self) -> "Portfolio":
        """Computes PD and DD via Merton model and adds them to the portfolio.

        Returns:
            Portfolio: self with 'PD' and 'DD' added, V/sigma_V removed.
        """
        return data_layer.add_merton_pd_fn(self)

    def add_dynamic_features(self) -> "Portfolio":
        """Adds EWMA annualized volatility to self.d['portfolio'].

        Returns:
            Portfolio: self with 'volatility' column added.
        """
        return data_layer.add_dynamic_features_fn(self)

    def predict_macro_factors(
        self, horizon: int = 1, training_offset: int = 0, model_type: str = "var"
    ) -> pd.DataFrame:
        """Predicts macroeconomic factors using specified model type."""
        return credit_risk.predict_macro_factors_fn(
            self, horizon=horizon, training_offset=training_offset, model_type=model_type
        )

    def predict_pd(self, horizon: int = 1, training_offset: int = 0, model_type: str = "var") -> "Portfolio":
        """Predicts PD for portfolio assets based on macro OLS model."""
        return credit_risk.predict_pd_fn(self, horizon=horizon, training_offset=training_offset, model_type=model_type)

    def predict_dd(self, horizon: int = 1, training_offset: int = 0, model_type: str = "var") -> "Portfolio":
        """Predicts DD and derived PD for portfolio assets based on macro OLS model."""
        return credit_risk.predict_dd_fn(self, horizon=horizon, training_offset=training_offset, model_type=model_type)

    def backtest_pd(self, n_months: int = 12, models: List[str] = None) -> "Portfolio":
        """Walk-forward backtest of PD predictions across macro models."""
        return credit_risk.backtest_pd_fn(self, n_months=n_months, models=models)

    def backtest_dd(self, n_months: int = 12, models: List[str] = None) -> "Portfolio":
        """Walk-forward backtest of DD predictions across macro models."""
        return credit_risk.backtest_dd_fn(self, n_months=n_months, models=models)

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
        """Optimizes portfolio weights using the specified strategy."""
        return optimization.optimize_portfolio_fn(
            self,
            lambda_risk=lambda_risk,
            lgd=lgd,
            use_forecast=use_forecast,
            min_weight=min_weight,
            max_weight=max_weight,
            strategy=strategy,
            cvar_alpha=cvar_alpha,
            cutoff_date=cutoff_date,
        )

    def _solve_mean_el(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        lambda_risk: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Delegates to optimization._solve_mean_el_fn."""
        return optimization._solve_mean_el_fn(returns_matrix, pds, lgd, lambda_risk, min_weight, max_weight)

    def _solve_cvar(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        alpha: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Delegates to optimization._solve_cvar_fn."""
        return optimization._solve_cvar_fn(returns_matrix, pds, lgd, alpha, min_weight, max_weight)

    def _solve_risk_parity(
        self,
        returns_matrix: np.ndarray,
        pds: np.ndarray,
        lgd: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Delegates to optimization._solve_risk_parity_fn."""
        return optimization._solve_risk_parity_fn(returns_matrix, pds, lgd, min_weight, max_weight)

    def calc_portfolio_metrics(self, lgd: float = 0.4) -> "Portfolio":
        """Calculates key risk metrics for the optimized portfolio."""
        return optimization.calc_portfolio_metrics_fn(self, lgd=lgd)

    def plot_portfolio_allocation(self, figsize: tuple = (10, 6), verbose: bool = False) -> "Portfolio":
        """Plots the optimized portfolio allocation.

        Args:
            figsize: Figure size.
            verbose: Whether to show the plot.

        Returns:
            Portfolio: self.
        """
        if "optimized_weights" not in self.d:
            log.error("No optimized weights found. Run optimize_portfolio() first.")
            return self

        plots.plot_portfolio_allocation(self.d["optimized_weights"], figsize=figsize, verbose=verbose)
        return self

    def forecast_macro(
        self,
        horizon: int = 1,
        training_offset: int = 0,
        models: Union[str, list] = "var",
        target_col: str = "DD",
    ) -> "Portfolio":
        """Computes macro and DD/PD forecasts, stores in self.d['macro_forecast_data'].

        Args:
            horizon: Forecasting horizon (negative = walk-forward backtest).
            training_offset: Offset for backtesting.
            models: Model type(s) to compare.
            target_col: 'DD' or 'PD'.

        Returns:
            Portfolio: self.
        """
        return credit_risk.forecast_macro_fn(
            self,
            horizon=horizon,
            training_offset=training_offset,
            models=models,
            target_col=target_col,
        )

    def plot_macro_forecast(
        self,
        horizon: int = 1,
        training_offset: int = 0,
        models: Union[str, list] = "var",
        tail: int = 12,
        target_col: str = "DD",
        verbose: bool = False,
        figsize: tuple = (12, 14),
    ) -> "Portfolio":
        """Plots historical and forecasted macro and portfolio DD/PD."""
        return credit_risk.plot_macro_forecast_fn(
            self,
            horizon=horizon,
            training_offset=training_offset,
            models=models,
            tail=tail,
            target_col=target_col,
            verbose=verbose,
            figsize=figsize,
        )

    def analyze_macro_dd_significance(self, verbose: bool = True) -> "Portfolio":
        """Analyzes the statistical significance of macro variables on DD.

        Delegates to credit_risk.analyze_macro_dd_significance_fn.

        Args:
            verbose: If True, prints detailed results.

        Returns:
            Portfolio: Self with results in self.d['macro_significance'].
        """
        return credit_risk.analyze_macro_dd_significance_fn(self, verbose=verbose)

    def compare_macro_models(
        self,
        n_months: int = 12,
        models: List[str] = None,
        target_col: str = "DD",
        verbose: bool = False,
    ) -> "Portfolio":
        """Computes MAE/RMSE/MAPE for macro models via walk-forward backtest."""
        return credit_risk.compare_macro_models_fn(
            self, n_months=n_months, models=models, target_col=target_col, verbose=verbose
        )

    def plot_pd_forecast(self, figsize: tuple = (12, 6), verbose: bool = False) -> "Portfolio":
        """Plots predicted vs reference PD for each ticker."""
        return credit_risk.plot_pd_forecast_fn(self, figsize=figsize, verbose=verbose)

    def plot_dd_forecast(self, figsize: tuple = (12, 6), verbose: bool = False) -> "Portfolio":
        """Plots predicted vs reference DD for each ticker."""
        return credit_risk.plot_dd_forecast_fn(self, figsize=figsize, verbose=verbose)

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
        """Backtests Active vs Passive strategy."""
        return optimization.backtest_portfolio_strategies_fn(
            self,
            n_months=n_months,
            lambda_risk=lambda_risk,
            lgd=lgd,
            model_type=model_type,
            rebalance_threshold=rebalance_threshold,
            transaction_cost=transaction_cost,
            min_weight=min_weight,
            max_weight=max_weight,
            strategy=strategy,
            cvar_alpha=cvar_alpha,
        )

    def plot_strategy_backtest(self, tail: int = 12, verbose: bool = False) -> "Portfolio":
        """Plots the comparison of cumulative returns and EL between strategies.

        Args:
            tail: Number of recent periods to display.
            verbose: Whether to show the plot interactively.

        Returns:
            Portfolio: self.
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
        """Plots Stock, PD and Cap/Debt dashboards for each ticker in a grid.

        Args:
            tickers: List of tickers to plot.
            figsize_row: Size of one ticker row (3 subplots).
            verbose: Whether to show the plot interactively.

        Returns:
            Portfolio: self.
        """
        plots.plot_ticker_dashboards_grid(self.d["portfolio"], tickers, figsize_row=figsize_row, verbose=verbose)
        return self

    def plot_pd_by_tickers(self, tickers: list, figsize: tuple = (12, 5), verbose: bool = False) -> "Portfolio":
        """Plots the probability of default (PD) for the given tickers.

        Args:
            tickers: List of stock tickers (e.g., ['GAZP', 'FESH']).
            figsize: Figure size.
            verbose: If True, displays the plot. If False, saves to file.

        Returns:
            Portfolio: self.
        """
        plots.plot_pd_by_tickers(self.d["portfolio"], tickers, figsize, verbose)
        return self

    def calc_irf(
        self,
        impulses_responses: Dict[str, str] = None,
        figsize: Tuple[int, int] = (12, 5),
        verbose: bool = False,
    ) -> "Portfolio":
        """Calculates impulse response functions for the given impulses and responses.

        Args:
            impulses_responses: Dict mapping impulse columns to response columns
                (e.g., {'interest_rate': 'PD', 'inflation': 'PD'}).
            figsize: Figure size.
            verbose: If True, displays the plot. If False, saves to file.

        Returns:
            Portfolio: self.
        """
        return credit_risk.calc_irf_fn(self, impulses_responses, figsize, verbose)

    def plot_correlation_matrix(
        self,
        custom_order: list,
        save_path: str = None,
        figsize: tuple = (15, 10),
        dpi: int = 300,
        annot_size: int = 8,
        verbose: bool = False,
    ) -> "Portfolio":
        """Plots and saves the correlation matrix of stock closing prices.

        Args:
            custom_order: Order of tickers for grouping.
            save_path: Path to save the plot (None uses default path).
            figsize: Figure size.
            dpi: Image resolution.
            annot_size: Font size for correlation annotations.
            verbose: If True, displays the plot. If False, saves to file.

        Returns:
            Portfolio: self.
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
        """Plots stock charts for the given tickers.

        Args:
            tickers: List of stock tickers (e.g., ['FESH', 'GAZP']).
            figsize: Figure size.
            verbose: If True, displays the plot. If False, saves to file.
            fontsize: Title font size.

        Returns:
            Portfolio: self.
        """
        plots.plot_stocks(self.d["portfolio"], tickers, figsize, verbose, fontsize)
        return self

    def plot_debt_capitalization(self, verbose: bool = False, figsize: tuple = (10, 4)) -> "Portfolio":
        """Plots a combined chart of capitalization and debt on the same Y-axis.

        Args:
            verbose: If True, displays the plot. If False, saves to file.
            figsize: Figure size.

        Returns:
            Portfolio: self.
        """
        plots.plot_debt_capitalization(self.d["portfolio"], verbose, figsize)
        return self

    def plot_macro_significance(
        self,
        save_path: str = cfg.MACRO_SIGNIFICANCE_PLOT_PATH,
        verbose: bool = False,
        figsize: tuple = (10, 6),
    ) -> "Portfolio":
        """Plots the significance of macroeconomic factors on Merton model parameters.

        Args:
            save_path: Path to save the plot.
            verbose: If True, displays the plot. If False, saves to file.
            figsize: Figure size.

        Returns:
            Portfolio: self.
        """
        plots.plot_macro_significance(self.d["macro_connection_summary"], save_path, verbose, figsize)
        return self

    def optimize_portfolio_with_amounts(
        self,
        loan_applications: pd.DataFrame,
        budget: float,
        lambda_return: float = 0.5,
        lambda_vol: float = 0.25,
        lambda_el: float = 0.25,
        max_sector_share: float = 0.3,
        sector_map: Dict[str, str] = None,
        cutoff_date: pd.Timestamp = None,
        strategy: str = "mean_el",
        cvar_alpha: float = 0.95,
    ) -> "Portfolio":
        """Optimizes portfolio allocations using real loan amounts."""
        return optimization.optimize_portfolio_with_amounts_fn(
            self,
            loan_applications=loan_applications,
            budget=budget,
            lambda_return=lambda_return,
            lambda_vol=lambda_vol,
            lambda_el=lambda_el,
            max_sector_share=max_sector_share,
            sector_map=sector_map,
            cutoff_date=cutoff_date,
            strategy=strategy,
            cvar_alpha=cvar_alpha,
        )

    def backtest_portfolio_with_amounts(
        self,
        loan_applications: pd.DataFrame,
        budget: float,
        n_months: int = 12,
        lambda_return: float = 0.5,
        lambda_vol: float = 0.25,
        lambda_el: float = 0.25,
        model_type: str = "var",
        max_sector_share: float = 0.3,
        sector_map: Dict[str, str] = None,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
        strategy: str = "mean_el",
        cvar_alpha: float = 0.95,
    ) -> "Portfolio":
        """Backtests the amount-based optimization strategy vs equal allocation."""
        return optimization.backtest_portfolio_with_amounts_fn(
            self,
            loan_applications=loan_applications,
            budget=budget,
            n_months=n_months,
            lambda_return=lambda_return,
            lambda_vol=lambda_vol,
            lambda_el=lambda_el,
            model_type=model_type,
            max_sector_share=max_sector_share,
            sector_map=sector_map,
            rebalance_threshold=rebalance_threshold,
            transaction_cost=transaction_cost,
            strategy=strategy,
            cvar_alpha=cvar_alpha,
        )

    def compare_amount_strategies(
        self,
        loan_applications: pd.DataFrame,
        budget: float,
        strategies: List[str] = None,
        n_months: int = 12,
        lambda_return: float = 0.5,
        lambda_vol: float = 0.25,
        lambda_el: float = 0.25,
        model_type: str = "var",
        max_sector_share: float = 0.3,
        sector_map: Dict[str, str] = None,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
        cvar_alpha: float = 0.95,
    ) -> "Portfolio":
        """Runs amount-based backtest for multiple strategies and builds comparison."""
        return optimization.compare_amount_strategies_fn(
            self,
            loan_applications=loan_applications,
            budget=budget,
            strategies=strategies,
            n_months=n_months,
            lambda_return=lambda_return,
            lambda_vol=lambda_vol,
            lambda_el=lambda_el,
            model_type=model_type,
            max_sector_share=max_sector_share,
            sector_map=sector_map,
            rebalance_threshold=rebalance_threshold,
            transaction_cost=transaction_cost,
            cvar_alpha=cvar_alpha,
        )

    def plot_amount_backtest(self, tail: int = 12, verbose: bool = False) -> "Portfolio":
        """Plots multi-strategy comparison for amount-based backtests.

        Args:
            tail: Number of last periods to show (0 = all).
            verbose: Whether to display the plot interactively.

        Returns:
            Portfolio: Self for method chaining.
        """
        if "all_amount_backtests" not in self.d:
            log.error("No amount-based multi-strategy results. Run compare_amount_strategies first.")
            return self

        plots.plot_amount_strategy_comparison(
            all_backtests=self.d["all_amount_backtests"],
            comparison_df=self.d["amount_multi_strategy_comparison"],
            tail=tail,
            verbose=verbose,
        )
        return self

    def log_completion(self) -> "Portfolio":
        """Logs the duration and completion of the analysis.

        Returns:
            Portfolio: self.
        """

        self.end_time = datetime.now()

        log.info("=" * 60)
        log.info(
            "ANALYSIS COMPLETED | Duration: %.1f sec",
            (self.end_time - self.start_time).total_seconds(),
        )
        log.info("=" * 60)

        return self
