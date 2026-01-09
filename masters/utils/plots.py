import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import re
import numpy as np
import pandas as pd
from utils.logger import Logger

log = Logger(__name__)

def plot_pd_by_tickers(
    portfolio_df: pd.DataFrame, tickers: list, figsize: tuple = (12, 5), verbose: bool = False
):
    """
    Plots the probability of default (PD) for the given tickers.
    """

    for ticker in tickers:
        save_path = f"logs/graphs/{ticker}_pd.png"
        data = portfolio_df.query(f"ticker == '{ticker}'")

        if data.empty:
            log.warning(f"No data for ticker {ticker}")
            continue

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        ax.plot(
            data["date"], data["PD"] * 100, color="royalblue",
            linewidth=2, markersize=5,
        )

        ax.set_title(f"Вероятность дефолта ({ticker})", fontsize=14, pad=10)
        ax.set_xlabel("Дата", fontsize=12), ax.set_ylabel("PD, %", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90), plt.grid(True, alpha=0.3), plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", facecolor="white")

        plt.show() if verbose else plt.clf()
        plt.close()

    if tickers:
        log.info("PD graphs saved: logs/graphs/")


def calc_irf(
    portfolio_df: pd.DataFrame,
    impulses_responses: dict,
    figsize: tuple = (10, 4),
    verbose: bool = False,
):
    """
    Calculates impulse response functions for the given impulses and responses.
    """
    if impulses_responses is None:
        raise ValueError("Impulses and responses must be specified")

    columns = np.unique(
        list(impulses_responses.keys()) + list(impulses_responses.values())
    )

    # Group by date to aggregate data across tickers (e.g., mean PD)
    # This creates a single time series for the VAR model
    if 'date' in portfolio_df.columns:
        data = portfolio_df.groupby("date")[columns].mean().sort_index().dropna()
    else:
        log.warning("'date' column not found. Using raw data (stacked tickers?) for VAR.")
        data = portfolio_df.sort_values(["ticker", "date"])[columns].dropna()[columns]

    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        log.error(f"Constant columns detected: {constant_cols}. VAR model requires time-varying data.")
        # If critical columns are constant, we can't proceed.
        # We could try to proceed without them, but if the user asked for them, it's better to stop or warn.
        # For now, let's drop them and see if anything remains.
        data = data.drop(columns=constant_cols)
        if data.empty or len(data.columns) < 2:
             log.error("Not enough variables left for VAR analysis.")
             return

    cols_before_diff = {}
    for col in data.columns:
        cols_before_diff[col] = adfuller(data[col].dropna())[1]

    if any(p > 0.05 for p in cols_before_diff.values()):
        log.info("p-values before differencing:\n%s", pd.Series(cols_before_diff))
        data = data.diff().dropna()
        log.info("Applied differencing to achieve stationarity")

        # Re-check for constant columns after differencing
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
             log.warning(f"Constant columns after differencing: {constant_cols}. Dropping them.")
             data = data.drop(columns=constant_cols)

        if data.empty or len(data.columns) < 2:
             log.error("Not enough variables left for VAR analysis after differencing.")
             return

        for col in data.columns:
            cols_before_diff[col] = adfuller(data[col].dropna())[1]
        log.info("p-values after differencing:\n%s", pd.Series(cols_before_diff))

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.RangeIndex(start=0, stop=len(data))

    try:
        model = VAR(data)
        lag_order = model.select_order(maxlags=6)
        selected_lags = lag_order.aic

        log.info(
            f"Optimal lag number calculated | Optimal number of lags: {selected_lags}"
        )

        results = model.fit(maxlags=selected_lags, ic="aic")

        for impulse, response in impulses_responses.items():
            if impulse not in data.columns or response not in data.columns:
                log.warning(f"Skipping IRF for {impulse} -> {response}: variable missing in data.")
                continue

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

            # ... existing code ...

            # Since we don't have the rest of the loop body here, I will just close the try block
            # and let the user re-run. Wait, I need to be careful with replace_string.
            # The original code had the loop. I should include it.

            plt.title(f"Impulse Response: {impulse} -> {response}", fontsize=14)
            plt.xlabel("Periods", fontsize=12)
            plt.ylabel("Response", fontsize=12)
            plt.grid()
            plt.tight_layout()

            if verbose:
                plt.show()
            else:
                plt.savefig(f"logs/graphs/irf_{impulse}_{response}.png")
                plt.clf()
            plt.close()

    except Exception as e:
        log.error(f"VAR model failed: {str(e)}")
        # Optional: try with a simpler model or just return
        return

        fig.suptitle(
            f"Impulse Response Function (IRF): {impulse} -> {response}\n"
            f"Method: VAR with AIC lag selection | 95% Confidence Intervals",
            fontsize=11,
            y=1.02,
        )

        plt.xlabel("Горизонт, кварталы")
        plt.ylabel("Изменение PD, базисные пункты")

        save_path = f"logs/graphs/irf_{impulse}_{response}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")

        if verbose:
            plt.show()
        else:
            plt.clf()
    plt.close()

    log.info(f"Impulse response functions saved | Path: logs/graphs/")

def plot_correlation_matrix(
    portfolio_df: pd.DataFrame,
    custom_order: list,
    save_path: str = None,
    figsize: tuple = (15, 10),
    dpi: int = 300,
    annot_size: int = 8,
    verbose: bool = False,
):
    """
    Plots and saves the correlation matrix of stock closing prices.
    """
    if save_path is None:
        save_path = f"logs/graphs/corr_matrix.png"

    pivot_data = portfolio_df.pivot_table(
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
        log.info(f"Correlation matrix saved | Path: {save_path}")

    if verbose:
        plt.show()
    else:
        plt.clf()
    plt.close()

def plot_stocks(
    portfolio_df: pd.DataFrame,
    tickers: list,
    figsize: tuple = (12, 5),
    verbose: bool = False,
    fontsize: int = 16,
):
    """
    Plots stock charts for the given tickers.
    """
    for ticker in tickers:
        stock_data = portfolio_df[portfolio_df["ticker"] == ticker]
        save_path = f"logs/graphs/{ticker}_stock.png"

        if stock_data.empty:
            raise ValueError(f"Ticker {ticker} not found in portfolio")

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        ax.plot(
            stock_data["date"], stock_data["close"],
            label="Closing price", color="royalblue",
            linewidth=2,
        )

        ax.set_title(f"Stock dynamics {ticker}", fontsize=fontsize, pad=10)
        ax.set_xlabel("Date", fontsize=12), ax.set_ylabel("Price, RUB", fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90), ax.grid(True, alpha=0.3), plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")

        plt.show() if verbose else plt.clf()
        plt.close()

    if tickers:
        log.info("Stock prices graphs saved: logs/graphs/")


def plot_debt_capitalization(
    portfolio_df: pd.DataFrame, verbose: bool = False, figsize: tuple = (12, 5)
):
    """
    Plots a combined chart of capitalization and debt on the same Y-axis.
    """
    save_path = f"logs/graphs/debt_catitalization.png"
    grouped = portfolio_df.groupby("ticker")

    for ticker, group in grouped:
        group = group.sort_values("date").dropna(subset=["capitalization", "debt"])

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        ax.plot(
            group["date"], group["capitalization"],
            color="#2ecc71", label="Capitalization",
            linewidth=2
        )

        ax.plot(
            group["date"], group["debt"],
            color="#e74c3c",
            linewidth=2, label="Debt"
        )

        ax.set_title(f"{ticker}: Capitalization vs Debt", fontsize=14, pad=10)
        ax.set_xlabel("Date", fontsize=12), ax.set_ylabel("Value", fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90), ax.grid(True, alpha=0.3)

        plt.legend(), plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=100)

        plt.show() if verbose else plt.clf()
        plt.close()

    if save_path:
        log.info(f"Capitalization-debt graphs saved: {save_path}")

def plot_macro_significance(
    macro_connection_summary: pd.DataFrame,
    save_path: str = "logs/graphs/macro_significance_summary.png",
    verbose: bool = False,
    figsize: tuple = (10, 6)
):
    """
    Plots the significance of macroeconomic factors on Merton model parameters.
    """
    if macro_connection_summary is None:
        raise ValueError("Macro connection summary not calculated.")

    factors = ['inflation', 'unemployment', 'usd_rub']
    factor_labels = ['Инфляция', 'Безработица', 'USD/RUB']
    significance_data = {'capitalization': [], 'debt': []}

    for target in ['capitalization', 'debt']:
        target_data = macro_connection_summary[macro_connection_summary['target'] == target]
        for factor in factors:
            ci_col = f'coef_{factor}_ci'
            significant = sum(
                1 for _, row in target_data.iterrows()
                if pd.notna(row[ci_col]) and (
                    lambda nums: len(nums) == 2 and (
                        (float(nums[0]) > 0 and float(nums[1]) > 0) or
                        (float(nums[0]) < 0 and float(nums[1]) < 0)
                    )
                )(re.findall(r'-?\d+\.\d+', str(row[ci_col])))
            )
            total = target_data[ci_col].notna().sum()
            significance_data[target].append(significant / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(factor_labels))
    width = 0.35

    cap_bars = ax.bar(x - width/2, significance_data['capitalization'], width,
                     label='Капитализация', color='steelblue', alpha=0.8)
    debt_bars = ax.bar(x + width/2, significance_data['debt'], width,
                      label='Долг', color='darkred', alpha=0.8)

    ax.set_xlabel('Макроэкономические факторы', fontsize=12, fontweight='bold')
    ax.set_ylabel('Доля значимых связей, %', fontsize=12, fontweight='bold')
    ax.set_title('Влияние макрофакторов на параметры модели Мертона',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(factor_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [cap_bars, debt_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        log.info(f"Macro significance plot saved | Path: {save_path}")

    if verbose:
        plt.show()
    else:
        plt.clf()
    plt.close()

def plot_macro_forecast(
    macro_df: pd.DataFrame,
    forecast_dfs: dict,
    tail: int = 0,
    figsize: tuple = (12, 10),
    verbose: bool = False,
):
    """
    Plots historical and forecasted values for macro parameters from multiple models.

    Args:
        macro_df (pd.DataFrame): Historical macro data.
        forecast_dfs (dict): Dictionary mapping model labels to forecast DataFrames.
        tail (int): Number of last months to show.
        figsize (tuple): Figure size.
        verbose (bool): Whether to show the plot.
    """
    factors = list(macro_df.columns)
    n_factors = len(factors)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
    if n_factors == 1:
        axes = [axes]

    # Define colors for different models
    colors = ['darkred', 'darkgreen', 'orange', 'purple', 'brown']

    for i, factor in enumerate(factors):
        ax = axes[i]

        # Plot Macro factor
        plot_df = macro_df.tail(tail) if tail > 0 else macro_df
        ax.plot(plot_df.index, plot_df[factor], label="Fact", color="royalblue", linewidth=2.5)

        for idx, (label, forecast_df) in enumerate(forecast_dfs.items()):
            color = colors[idx % len(colors)]
            first_fc_date = forecast_df.index[0]
            history_before_fc = macro_df[macro_df.index < first_fc_date]

            if not history_before_fc.empty:
                fc_dates = [history_before_fc.index[-1]] + list(forecast_df.index)
                fc_vals = [history_before_fc[factor].iloc[-1]] + list(forecast_df[factor])
            else:
                fc_dates, fc_vals = list(forecast_df.index), list(forecast_df[factor])

            ax.plot(fc_dates, fc_vals, label=f"FC: {label}", color=color, linestyle="--", linewidth=1.5)
            ax.scatter(forecast_df.index, forecast_df[factor], color=color, s=20)

        ax.set_title(f"Macro Factor: {factor}", fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xlabel("Date", fontsize=10)
    plt.xticks(rotation=90)
    plt.tight_layout()

    save_path = "logs/graphs/macro_comparison.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if verbose:
        plt.show()
    else:
        plt.clf()
    plt.close()

    log.info(f"Macro comparison plot saved: {save_path}")


def plot_pd_forecast(forecast_df: pd.DataFrame, figsize: tuple = (12, 6), verbose: bool = False):
    """
    Plots a bar chart comparing predicted PD vs reference PD for tickers.

    Args:
        forecast_df (pd.DataFrame): DataFrame with columns 'predicted_pd' and 'reference_pd'.
        figsize (tuple): Figure size.
        verbose (bool): Whether to show the plot.
    """
    if forecast_df.empty:
        log.warning("Forecast DataFrame is empty. Cannot plot PD forecast.")
        return

    df = forecast_df.sort_values("predicted_pd", ascending=False)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width / 2,
        df["reference_pd"] * 100,
        width,
        label="Fact (Reference)",
        color="royalblue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        df["predicted_pd"] * 100,
        width,
        label="Forecast",
        color="darkred",
        alpha=0.8,
    )

    ax.set_ylabel("PD, %", fontsize=12)
    ax.set_title("Comparison: Predicted PD vs Ground Truth", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = "logs/graphs/pd_forecast_comparison.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")

    if verbose:
        plt.show()
    else:
        plt.clf()
    plt.close()
    log.info(f"PD forecast comparison plot saved: {save_path}")

