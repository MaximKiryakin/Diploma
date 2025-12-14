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

    data = portfolio_df.sort_values(["ticker", "date"])[columns].dropna()[columns]

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

    log.info(
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
