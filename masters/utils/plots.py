import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path
import re
import numpy as np
import pandas as pd
import utils.config as cfg
from utils.logger import Logger

log = Logger(__name__)


def _finalize_plot(save_path: str, verbose: bool, dpi: int = 150) -> None:
    """Saves the current figure, optionally shows it, and closes it."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if verbose:
        plt.show()
    else:
        plt.clf()
    plt.close()


def _is_ci_significant(ci_str: str) -> bool:
    """Checks whether a confidence interval string indicates statistical significance.

    A CI is significant when both bounds have the same sign (both positive or both negative),
    meaning the interval does not cross zero.

    Args:
        ci_str: String representation of CI, e.g. "(0.12, 0.45)" or "(-0.3, -0.1)".

    Returns:
        True if CI is significant, False otherwise.
    """
    nums = re.findall(r"-?\d+\.\d+", ci_str)
    if len(nums) != 2:
        return False
    lower, upper = float(nums[0]), float(nums[1])
    return (lower > 0 and upper > 0) or (lower < 0 and upper < 0)


def plot_pd_by_tickers(portfolio_df: pd.DataFrame, tickers: list, figsize: tuple = (12, 5), verbose: bool = False):
    """Plots the probability of default (PD) for the given tickers.

    Uses symmetric log scale to handle large PD spikes alongside near-zero values.
    """
    for ticker in tickers:
        save_path = str(cfg.GRAPHS_DIR / f"{ticker}_pd.png")
        data = portfolio_df.query(f"ticker == '{ticker}'")

        if data.empty:
            log.warning("No data for ticker %s", ticker)
            continue

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        pd_pct = data["PD"] * 100

        ax.plot(
            data["date"],
            pd_pct,
            color="royalblue",
            linewidth=2,
            markersize=5,
        )

        # Symmetric log scale: linear near zero (linthresh), log for large values
        ax.set_yscale("symlog", linthresh=0.01)

        ax.set_title(f"Probability of Default ({ticker})", fontsize=14, pad=10)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("PD, % (log scale)", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        _finalize_plot(save_path, verbose)

    if tickers:
        log.info("PD graphs saved: %s", cfg.GRAPHS_DIR)


def plot_irf(
    results,
    impulses_responses: dict,
    selected_lags: int,
    figsize: tuple = (10, 4),
    verbose: bool = False,
) -> None:
    """Plots impulse response functions from a fitted VAR model.

    Args:
        results: Fitted VAR model results object (statsmodels VARResults).
        impulses_responses: Dict mapping impulse column names to response column names.
        selected_lags: Number of periods to compute the IRF over.
        figsize: Figure size for each IRF plot.
        verbose: Whether to display the plot interactively.
    """
    for impulse, response in impulses_responses.items():
        if impulse not in results.names or response not in results.names:
            log.warning("Skipping IRF for %s -> %s: variable missing in data.", impulse, response)
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

        plt.title(
            f"Impulse Response Function (IRF): {impulse} -> {response}\n"
            f"Method: VAR with AIC lag selection | 95% Confidence Intervals",
            fontsize=11,
        )
        plt.xlabel("Periods (quarters)", fontsize=12)
        plt.ylabel("Response (basis points)", fontsize=12)
        plt.grid()
        plt.tight_layout()

        save_path = str(cfg.GRAPHS_DIR / f"irf_{impulse}_{response}.png")
        _finalize_plot(save_path, verbose)

    log.info("Impulse response functions saved | Path: %s", cfg.GRAPHS_DIR)


def calc_irf(
    portfolio_df: pd.DataFrame,
    impulses_responses: dict,
    figsize: tuple = (10, 4),
    verbose: bool = False,
) -> None:
    """Calculates and plots impulse response functions for the given impulses and responses.

    .. deprecated::
        VAR analysis logic has been moved to ``credit_risk.calc_irf_fn``.
        Call ``Portfolio.calc_irf()`` instead of this function directly.

    Args:
        portfolio_df: Portfolio DataFrame with date column and numeric columns for VAR.
        impulses_responses: Dict mapping impulse column names to response column names.
        figsize: Figure size for each IRF plot.
        verbose: Whether to display the plot interactively.
    """
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller

    if impulses_responses is None:
        raise ValueError("Impulses and responses must be specified")

    columns = np.unique(list(impulses_responses.keys()) + list(impulses_responses.values()))

    if "date" in portfolio_df.columns:
        data = portfolio_df.groupby("date")[columns].mean().sort_index().dropna()
    else:
        log.warning("'date' column not found. Using raw data (stacked tickers?) for VAR.")
        data = portfolio_df.sort_values(["ticker", "date"])[columns].dropna()[columns]

    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        log.error("Constant columns detected: %s. VAR model requires time-varying data.", constant_cols)
        data = data.drop(columns=constant_cols)
        if data.empty or len(data.columns) < 2:
            log.error("Not enough variables left for VAR analysis.")
            return

    pvalues = {col: adfuller(data[col].dropna())[1] for col in data.columns}

    if any(p > 0.05 for p in pvalues.values()):
        log.info("p-values before differencing:\n%s", pd.Series(pvalues))
        data = data.diff().dropna()
        log.info("Applied differencing to achieve stationarity")

        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            log.warning("Constant columns after differencing: %s. Dropping them.", constant_cols)
            data = data.drop(columns=constant_cols)

        if data.empty or len(data.columns) < 2:
            log.error("Not enough variables left for VAR analysis after differencing.")
            return

        pvalues = {col: adfuller(data[col].dropna())[1] for col in data.columns}
        log.info("p-values after differencing:\n%s", pd.Series(pvalues))

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.RangeIndex(start=0, stop=len(data))

    model = VAR(data)
    lag_order = model.select_order(maxlags=6)
    selected_lags = lag_order.aic
    log.info("Optimal lag number calculated | Optimal number of lags: %d", selected_lags)
    results = model.fit(maxlags=selected_lags, ic="aic")

    plot_irf(results, impulses_responses, selected_lags, figsize, verbose)


def plot_correlation_matrix(
    portfolio_df: pd.DataFrame,
    custom_order: list,
    save_path: str = None,
    figsize: tuple = (15, 10),
    dpi: int = 300,
    annot_size: int = 8,
    verbose: bool = False,
    sector_breaks: list = None,
) -> None:
    """Plots and saves the correlation matrix of stock closing prices.

    Args:
        portfolio_df: Portfolio DataFrame with date, ticker, close columns.
        custom_order: Desired ticker ordering for the matrix.
        save_path: File path to save the plot.
        figsize: Figure size.
        dpi: Resolution for the saved image.
        annot_size: Font size for correlation annotations.
        sector_breaks: List of positions for sector separator lines.
        verbose: Whether to display the plot interactively.
    """
    if save_path is None:
        save_path = str(cfg.GRAPHS_DIR / "corr_matrix.png")

    pivot_data = portfolio_df.pivot_table(index="date", columns="ticker", values="close")
    pivot_data = pivot_data.interpolate(method="time", limit_direction="both")
    valid_tickers = [t for t in custom_order if t in pivot_data.columns]

    if not valid_tickers:
        raise ValueError("No data for plotting the matrix")

    pivot_data = pivot_data[valid_tickers]
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

    if sector_breaks:
        for pos in sector_breaks:
            plt.axvline(pos, color="black", linewidth=2)
            plt.axhline(pos, color="black", linewidth=2)

    plt.title("Stock Closing Price Correlation", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    _finalize_plot(save_path, verbose, dpi=dpi)
    log.info("Correlation matrix saved | Path: %s", save_path)


def plot_stocks(
    portfolio_df: pd.DataFrame,
    tickers: list,
    figsize: tuple = (12, 5),
    verbose: bool = False,
    fontsize: int = 16,
) -> None:
    """Plots stock price charts for the given tickers.

    Args:
        portfolio_df: Portfolio DataFrame with date, ticker, close columns.
        tickers: List of ticker symbols to plot.
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
        fontsize: Title font size.
    """
    for ticker in tickers:
        stock_data = portfolio_df[portfolio_df["ticker"] == ticker]
        save_path = str(cfg.GRAPHS_DIR / f"{ticker}_stock.png")

        if stock_data.empty:
            raise ValueError(f"Ticker {ticker} not found in portfolio")

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        ax.plot(
            stock_data["date"],
            stock_data["close"],
            label="Closing price",
            color="royalblue",
            linewidth=2,
        )

        ax.set_title(f"Stock Dynamics: {ticker}", fontsize=fontsize, pad=10)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price, RUB", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        _finalize_plot(save_path, verbose, dpi=100)

    if tickers:
        log.info("Stock prices graphs saved: %s", cfg.GRAPHS_DIR)


def plot_debt_capitalization(portfolio_df: pd.DataFrame, verbose: bool = False, figsize: tuple = (12, 5)) -> None:
    """Plots a combined chart of capitalization and debt on the same Y-axis.

    Args:
        portfolio_df: Portfolio DataFrame with date, ticker, capitalization, debt columns.
        verbose: Whether to display the plot interactively.
        figsize: Figure size.
    """
    grouped = portfolio_df.groupby("ticker")
    summary_rows = []

    for ticker, group in grouped:
        group = group.sort_values("date").dropna(subset=["capitalization", "debt"])

        summary_rows.append(
            {
                "ticker": ticker,
                "cap_mean": group["capitalization"].mean(),
                "cap_min": group["capitalization"].min(),
                "cap_max": group["capitalization"].max(),
                "debt_mean": group["debt"].mean(),
                "debt_min": group["debt"].min(),
                "debt_max": group["debt"].max(),
                "cap_debt_ratio": (
                    group["capitalization"].mean() / group["debt"].mean() if group["debt"].mean() != 0 else np.inf
                ),
                "rows": len(group),
            }
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        ax.plot(group["date"], group["capitalization"], color="#2ecc71", label="Capitalization", linewidth=2)
        ax.plot(group["date"], group["debt"], color="#e74c3c", linewidth=2, label="Debt")

        ax.set_title(f"{ticker}: Capitalization vs Debt", fontsize=14, pad=10)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=90)
        ax.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = str(cfg.GRAPHS_DIR / f"{ticker}_debt_capitalization.png")
        _finalize_plot(save_path, verbose, dpi=100)

    log.info("Capitalization-debt graphs saved: %s", cfg.GRAPHS_DIR)

    summary_df = pd.DataFrame(summary_rows)
    for col in ["cap_mean", "cap_min", "cap_max", "debt_mean", "debt_min", "debt_max"]:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:,.0f}")
    summary_df["cap_debt_ratio"] = summary_df["cap_debt_ratio"].apply(lambda x: f"{x:.2f}" if x != np.inf else "inf")
    log.log_dataframe(summary_df, title="Debt & Capitalization Summary")


def plot_ticker_dashboards_grid(
    portfolio_df: pd.DataFrame, tickers: list, figsize_row: tuple = (26, 5), verbose: bool = False
) -> None:
    """Plots a multi-panel dashboard for each ticker.

    One row per ticker: [DD] | [PD] | [Stock Price] | [Debt vs Assets]

    Args:
        portfolio_df: Portfolio DataFrame.
        tickers: List of ticker symbols.
        figsize_row: Size of one ticker row (width, height).
        verbose: Whether to display the plot interactively.
    """
    n_tickers = len(tickers)
    n_cols = 4
    fig, axes = plt.subplots(n_tickers, n_cols, figsize=(figsize_row[0], figsize_row[1] * n_tickers))

    # Handle single ticker case (axes becomes 1D)
    if n_tickers == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, ticker in enumerate(tickers):
        data = portfolio_df.query(f"ticker == '{ticker}'").sort_values("date")
        if data.empty:
            continue

        # 1. Distance to Default DD (Column 0)
        ax_dd = axes[i, 0]
        ax_dd.plot(data["date"], data["DD"], color="crimson", linewidth=2)
        ax_dd.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_dd.set_title(f"{ticker}: DD", fontsize=12, pad=10)
        ax_dd.set_ylabel("DD (std. dev.)")
        ax_dd.grid(True, alpha=0.3)

        # 2. Probability of Default PD (Column 1)
        ax_pd = axes[i, 1]
        ax_pd.plot(data["date"], data["PD"] * 100, color="darkorange", linewidth=2)
        ax_pd.set_yscale("symlog", linthresh=0.1)
        ax_pd.set_title(f"{ticker}: PD (%)", fontsize=12, pad=10)
        ax_pd.set_ylabel("PD, %")
        ax_pd.grid(True, alpha=0.3, which="both")

        # 3. Stock Price (Column 2)
        ax_stock = axes[i, 2]
        ax_stock.plot(data["date"], data["close"], color="royalblue", linewidth=2)
        ax_stock.set_title(f"{ticker}: Stock Price", fontsize=12, pad=10)
        ax_stock.set_ylabel("Price, RUB")
        ax_stock.grid(True, alpha=0.3)

        # 4. Debt vs Assets (Column 3)
        ax_debt = axes[i, 3]
        ax_debt.plot(data["date"], data["capitalization"], color="#2ecc71", label="Cap", linewidth=2)
        ax_debt.plot(data["date"], data["debt"], color="#e74c3c", label="Debt", linewidth=2)
        ax_debt.set_title(f"{ticker}: Cap vs Debt", fontsize=12, pad=10)
        ax_debt.set_ylabel("Value, RUB")
        ax_debt.legend(fontsize=8)
        ax_debt.grid(True, alpha=0.3)

        # Apply common date formatting to all axes in the row
        for ax in axes[i]:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=90, labelsize=8)

    plt.tight_layout()
    save_path = str(cfg.GRAPHS_DIR / "ticker_dashboards.png")
    _finalize_plot(save_path, verbose)


def plot_macro_significance(
    macro_connection_summary: pd.DataFrame,
    save_path: str = None,
    verbose: bool = False,
    figsize: tuple = (10, 6),
) -> None:
    """Plots the significance of macroeconomic factors on Merton model parameters.

    Args:
        macro_connection_summary: Summary DataFrame with CI columns per factor.
        save_path: File path for saving the plot.
        verbose: Whether to display the plot interactively.
        figsize: Figure size.
    """
    if macro_connection_summary is None:
        raise ValueError("Macro connection summary not calculated.")

    if save_path is None:
        save_path = str(cfg.GRAPHS_DIR / "macro_significance_summary.png")

    factors = ["inflation", "unemployment", "usd_rub"]
    factor_labels = ["Inflation", "Unemployment", "USD/RUB"]
    significance_data: dict[str, list[float]] = {"capitalization": [], "debt": []}

    for target in ["capitalization", "debt"]:
        target_data = macro_connection_summary[macro_connection_summary["target"] == target]
        for factor in factors:
            ci_col = f"coef_{factor}_ci"
            significant = sum(
                1 for _, row in target_data.iterrows() if pd.notna(row[ci_col]) and _is_ci_significant(str(row[ci_col]))
            )
            total = target_data[ci_col].notna().sum()
            significance_data[target].append(significant / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(factor_labels))
    width = 0.35

    cap_bars = ax.bar(
        x - width / 2, significance_data["capitalization"], width, label="Capitalization", color="steelblue", alpha=0.8
    )
    debt_bars = ax.bar(x + width / 2, significance_data["debt"], width, label="Debt", color="darkred", alpha=0.8)

    ax.set_xlabel("Macroeconomic Factors", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share of Significant Links, %", fontsize=12, fontweight="bold")
    ax.set_title("Macro Factor Impact on Merton Model Parameters", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(factor_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bars in [cap_bars, debt_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    plt.tight_layout()

    _finalize_plot(save_path, verbose, dpi=300)
    log.info("Macro significance plot saved | Path: %s", save_path)


def plot_macro_forecast(
    macro_df: pd.DataFrame,
    forecast_dfs: dict,
    tail: int = 0,
    figsize: tuple = (10, 5),
    verbose: bool = False,
) -> None:
    """Plots historical and forecasted values for each macro factor.

    Each factor is rendered as a separate figure and saved to
    ``logs/graphs/macro_forecast_<factor>.png``.

    Args:
        macro_df: Historical macro data (including DD/PD).
        forecast_dfs: Dictionary mapping model labels to forecast DataFrames.
        tail: Number of last periods to show (0 = all).
        figsize: Per-figure size.
        verbose: Whether to display the figures interactively.
    """
    factor_titles = {
        "inflation": "Инфляция",
        "interest_rate": "Ключевая ставка",
        "unemployment_rate": "Безработица",
        "rubusd_exchange_rate": "Курс RUB/USD",
        "DD": "Дистанция до дефолта (DD) портфеля",
        "PD": "Вероятность дефолта (PD) портфеля, %",
    }
    model_colors = {
        "var": "darkred",
        "sarimax": "darkgreen",
        "prophet": "orange",
    }
    fallback_colors = ["purple", "brown", "teal", "magenta"]

    for factor in macro_df.columns:
        is_pd = factor == "PD"

        fig, ax = plt.subplots(figsize=figsize)

        plot_df = macro_df.tail(tail) if tail > 0 else macro_df
        fact_vals = plot_df[factor] * 100 if is_pd else plot_df[factor]
        ax.plot(plot_df.index, fact_vals, label="Факт", color="royalblue", linewidth=2.5)

        for idx, (label, forecast_df) in enumerate(forecast_dfs.items()):
            if factor not in forecast_df.columns:
                continue

            color = model_colors.get(label.lower(), fallback_colors[idx % len(fallback_colors)])
            first_fc_date = forecast_df.index[0]
            history_before_fc = macro_df[macro_df.index < first_fc_date]

            fc_raw = list(forecast_df[factor])
            fc_scaled = [v * 100 for v in fc_raw] if is_pd else fc_raw

            if not history_before_fc.empty:
                fc_dates = [history_before_fc.index[-1]] + list(forecast_df.index)
                hist_val = history_before_fc[factor].iloc[-1]
                fc_vals = [hist_val * 100 if is_pd else hist_val] + fc_scaled
            else:
                fc_dates = list(forecast_df.index)
                fc_vals = fc_scaled

            ax.plot(
                fc_dates,
                fc_vals,
                label=f"Прогноз: {label.upper()}",
                color=color,
                linestyle="--",
                linewidth=1.8,
            )
            ax.scatter(forecast_df.index, fc_scaled, color=color, s=30, zorder=3)

        ax.set_title(factor_titles.get(factor, factor), fontsize=14)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Дата", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

        plt.tight_layout()

        save_path = str(cfg.GRAPHS_DIR / f"macro_forecast_{factor}.png")
        _finalize_plot(save_path, verbose)
        log.info("Macro forecast plot saved | factor=%s | path=%s", factor, save_path)


def _plot_metric_forecast(
    forecast_df: pd.DataFrame,
    metric: str,
    figsize: tuple = (12, 6),
    verbose: bool = False,
) -> None:
    """Plots a bar chart comparing predicted vs reference values for a given metric.

    Args:
        forecast_df: DataFrame with columns 'predicted_{metric}' and 'reference_{metric}'.
        metric: Metric name, either 'pd' or 'dd'.
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
    """
    predicted_col = f"predicted_{metric}"
    reference_col = f"reference_{metric}"

    if forecast_df.empty:
        log.warning("Forecast DataFrame is empty. Cannot plot %s forecast.", metric.upper())
        return

    is_pd = metric == "pd"
    scale = 100 if is_pd else 1
    ylabel = "PD, %" if is_pd else "Distance to Default (DD)"
    title = f"Comparison: Predicted {metric.upper()} vs Ground Truth"
    ascending = False if is_pd else True

    df = forecast_df.sort_values(predicted_col, ascending=ascending)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width / 2,
        df[reference_col] * scale,
        width,
        label="Fact (Reference)",
        color="royalblue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        df[predicted_col] * scale,
        width,
        label="Forecast",
        color="darkred",
        alpha=0.8,
    )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / f"{metric}_forecast_comparison.png")
    _finalize_plot(save_path, verbose)
    log.info("%s forecast comparison plot saved: %s", metric.upper(), save_path)


def plot_pd_forecast(forecast_df: pd.DataFrame, figsize: tuple = (12, 6), verbose: bool = False) -> None:
    """Plots a bar chart comparing predicted PD vs reference PD for tickers.

    Args:
        forecast_df: DataFrame with columns 'predicted_pd' and 'reference_pd'.
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
    """
    _plot_metric_forecast(forecast_df, metric="pd", figsize=figsize, verbose=verbose)


def plot_dd_forecast(forecast_df: pd.DataFrame, figsize: tuple = (12, 6), verbose: bool = False) -> None:
    """Plots a bar chart comparing predicted DD vs reference DD for tickers.

    Args:
        forecast_df: DataFrame with columns 'predicted_dd' and 'reference_dd'.
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
    """
    _plot_metric_forecast(forecast_df, metric="dd", figsize=figsize, verbose=verbose)


def plot_portfolio_allocation(weights: pd.Series, figsize: tuple = (10, 6), verbose: bool = False) -> None:
    """Plots a bar chart of portfolio weights.

    Args:
        weights: Optimized weights by ticker.
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
    """
    if weights.empty:
        log.warning("Weights series is empty. Cannot plot allocation.")
        return

    df = weights[weights > 0.001].sort_values(ascending=False).reset_index()
    df.columns = ["ticker", "weight"]

    plt.figure(figsize=figsize, dpi=150)
    sns.barplot(data=df, x="ticker", y="weight", palette="viridis")

    plt.title("Optimized Portfolio Allocation", fontsize=14, pad=15)
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Ticker", fontsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "portfolio_allocation.png")
    _finalize_plot(save_path, verbose)
    log.info("Portfolio allocation plot saved: %s", save_path)


def plot_strategy_comparison(
    backtest_df: pd.DataFrame,
    comparison_df: pd.DataFrame = None,
    tail: int = 12,
    verbose: bool = False,
    all_backtests: dict = None,
) -> None:
    """Plots strategy backtest results as two separate figures.

    Figure 1: cumulative returns per active strategy plus passive buy-and-hold.
    Figure 2: portfolio average PD (bars, left axis) combined with realized EL
              per strategy (lines, right axis), annotated with curated
              economic events for August 2023, January 2025 and March 2025.

    Args:
        backtest_df: Backtest results with Active/Passive returns and EL.
        comparison_df: Legacy parameter, kept for backward compatibility.
        tail: Number of last periods to show (0 = all).
        verbose: Whether to display plots interactively.
        all_backtests: Dict mapping strategy name -> backtest DataFrame.
    """
    if backtest_df.empty:
        log.warning("Strategy backtest DataFrame is empty.")
        return
    if not all_backtests:
        log.warning("plot_strategy_comparison: all_backtests is empty.")
        return

    strategy_colors = {
        "mean_el": "#1f77b4",
        "cvar": "#ff7f0e",
        "risk_parity": "#2ca02c",
    }
    strategy_labels = {
        "mean_el": "Активная: Mean-EL",
        "cvar": "Активная: CVaR",
        "risk_parity": "Активная: Risk Parity",
    }

    last_bt = list(all_backtests.values())[-1]
    plot_passive = last_bt.tail(tail) if tail > 0 else last_bt

    diff_mask = backtest_df["Active_Return"] != backtest_df["Passive_Return"]
    bt_start = backtest_df.index[diff_mask.argmax()] if diff_mask.any() else None

    def _draw_bt_line(ax) -> None:
        if bt_start is not None and bt_start in plot_passive.index:
            ax.axvline(x=bt_start, color="black", linestyle=":", alpha=0.5)

    def _rotate_xticks(ax) -> None:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    def _local_peak(series: pd.Series, target_ts: pd.Timestamp, window_months: int = 1):
        lo = target_ts - pd.DateOffset(months=window_months)
        hi = target_ts + pd.DateOffset(months=window_months)
        sub = series[(series.index >= lo) & (series.index <= hi)]
        if sub.empty:
            pos = series.index.get_indexer([target_ts], method="nearest")[0]
            idx = series.index[pos]
            return idx, series.loc[idx]
        peak_idx = sub.idxmax()
        return peak_idx, sub.loc[peak_idx]

    def _annotate_event(ax, x, y, text: str, color: str, xytext, ha: str = "left") -> None:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            fontsize=8,
            color="black",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec=color, alpha=0.95),
        )

    # --- Figure 1: cumulative return ---
    fig, ax = plt.subplots(figsize=(11, 5))
    for strat_name, bt_df in all_backtests.items():
        plot_df = bt_df.tail(tail) if tail > 0 else bt_df
        cum_ret = ((1 + plot_df["Active_Return"]).cumprod() - 1) * 100
        ax.plot(
            plot_df.index,
            cum_ret,
            label=strategy_labels.get(strat_name, strat_name),
            color=strategy_colors.get(strat_name),
            linewidth=2.5,
            marker="o",
            markersize=3,
        )
    cum_ret_p = ((1 + plot_passive["Passive_Return"]).cumprod() - 1) * 100
    ax.plot(
        plot_passive.index,
        cum_ret_p,
        label="Пассивная (равные веса, buy-and-hold)",
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
    )
    _draw_bt_line(ax)
    ax.set_title("Накопленная доходность стратегий, %", fontsize=14)
    ax.set_ylabel("Доходность, %", fontsize=12)
    ax.set_xlabel("Дата", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    _rotate_xticks(ax)
    plt.tight_layout()
    save_path_1 = str(cfg.GRAPHS_DIR / "strategy_backtest_cumret.png")
    _finalize_plot(save_path_1, verbose)
    log.info("Strategy cumret plot saved: %s", save_path_1)

    # --- Figure 2: realized EL (lines) + portfolio PD (bars) on twin axes ---
    fig, ax = plt.subplots(figsize=(11, 5))
    pd_pct = None
    if "Actual_PD" in plot_passive.columns:
        pd_pct = plot_passive["Actual_PD"] * 100
        ax.bar(
            plot_passive.index,
            pd_pct,
            color="steelblue",
            alpha=0.25,
            width=20,
            label="Средняя PD портфеля, %",
        )
    ax.set_ylabel("PD, %", fontsize=12, color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax.twinx()
    ref_el = None
    for strat_name, bt_df in all_backtests.items():
        plot_df = bt_df.tail(tail) if tail > 0 else bt_df
        el_pct = plot_df["Active_EL"] * 100
        ax2.plot(
            plot_df.index,
            el_pct,
            label=strategy_labels.get(strat_name, strat_name),
            color=strategy_colors.get(strat_name),
            linewidth=2,
        )
        if strat_name == "mean_el":
            ref_el = el_pct
    ax2.plot(
        plot_passive.index,
        plot_passive["Passive_EL"] * 100,
        label="Пассивная",
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
    )
    ax2.set_ylabel("EL, %", fontsize=12)

    rebal_df = list(all_backtests.values())[0].tail(tail) if tail > 0 else list(all_backtests.values())[0]
    rebal_dates = rebal_df.index[rebal_df["Rebalanced"].fillna(False)]
    for d in rebal_dates:
        ax.axvline(x=d, color="crimson", linestyle="-", alpha=0.15, linewidth=1)

    events = [
        {
            "target": pd.Timestamp("2023-08-31"),
            "text": (
                "Авг 2023: экстренное повышение ключевой\n"
                "ставки ЦБ с 8.5% до 12% (15 авг) на фоне\n"
                "падения рубля до 100/USD. Equity emitents\n"
                "дешевеют - DD падает, PD растёт."
            ),
            "color": "#1f77b4",
            "xytext": (-170, 40),
            "ha": "left",
        },
        {
            "target": pd.Timestamp("2024-12-31"),
            "text": (
                "Янв 2025: ключевая ставка ЦБ 21% (пик цикла).\n"
                "Ребалансировка после ралли IMOEX в дек 2024:\n"
                "веса концентрируются в высоковолатильных\n"
                "тикерах -> резкий скачок EL портфеля."
            ),
            "color": "#1f77b4",
            "xytext": (-220, -10),
            "ha": "left",
        },
        {
            "target": pd.Timestamp("2025-03-31"),
            "text": (
                "Мар 2025: коррекция IMOEX (-10% от фев пика)\n"
                "на срыве ожиданий мирных переговоров и\n"
                "падении нефти к $70/bbl. ЦБ удерживает 21%.\n"
                "Цикличные эмитенты (нефть, металлы) теряют\n"
                "в стоимости -> DD сокращается, PD/EL растут."
            ),
            "color": "#1f77b4",
            "xytext": (-255, 72),
            "ha": "left",
        },
    ]

    if ref_el is not None:
        for ev in events:
            x, y = _local_peak(ref_el, ev["target"], ev.get("peak_window", 1))
            _annotate_event(ax2, x, y, ev["text"], ev["color"], ev["xytext"], ha=ev["ha"])

    _draw_bt_line(ax)
    ax.set_title("Реализованные EL и средневзвешенная PD портфеля, %", fontsize=14)
    ax.set_xlabel("Дата", fontsize=12)
    ax.grid(True, alpha=0.3)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)
    _rotate_xticks(ax)
    plt.tight_layout()
    save_path_2 = str(cfg.GRAPHS_DIR / "strategy_backtest_el_pd.png")
    _finalize_plot(save_path_2, verbose)
    log.info("Strategy EL/PD plot saved: %s", save_path_2)


def plot_amount_strategy_comparison(
    all_backtests: dict,
    comparison_df: pd.DataFrame,
    tail: int = 12,
    verbose: bool = False,
) -> None:
    """Plots multi-strategy comparison for amount-based backtests.

    Produces two figures:
    1. Time-series: cumulative net income, realized EL, credit income per strategy.
    2. Bar chart: key summary metrics (RAROC, EL, Vol, Net Income) across strategies.

    Args:
        all_backtests: Dict mapping strategy name -> backtest DataFrame.
        comparison_df: Summary DataFrame indexed by strategy name.
        tail: Number of last periods to show (0 = all).
        verbose: Whether to display plots interactively.
    """
    if not all_backtests:
        log.warning("No amount-based backtest results to plot.")
        return

    strategy_colors = {
        "mean_el": "#1f77b4",
        "cvar": "#ff7f0e",
        "risk_parity": "#2ca02c",
    }
    strategy_labels = {
        "mean_el": "Mean-EL",
        "cvar": "CVaR",
        "risk_parity": "Risk Parity",
    }

    # --- Figure 1: Time-series (3 subplots) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 14), sharex=True)

    for strat_name, bt_df in all_backtests.items():
        plot_df = bt_df.tail(tail) if tail > 0 else bt_df
        plot_df = plot_df.copy()
        plot_df["Active_CumNet"] = (1 + plot_df["Active_Net"]).cumprod() - 1
        color = strategy_colors.get(strat_name, None)
        label = strategy_labels.get(strat_name, strat_name)

        ax1.plot(
            plot_df.index,
            plot_df["Active_CumNet"] * 100,
            label=label,
            linewidth=2.5,
            marker="o",
            markersize=3,
            color=color,
        )
        ax2.plot(
            plot_df.index,
            plot_df["Active_EL"] * 100,
            label=f"{label}",
            linewidth=2,
            color=color,
        )
        ax3.plot(
            plot_df.index,
            plot_df["Active_Credit_Income"] * 100,
            label=f"{label}",
            linewidth=2,
            color=color,
        )

    # Passive baseline from last strategy backtest
    last_bt = list(all_backtests.values())[-1]
    plot_passive = last_bt.tail(tail) if tail > 0 else last_bt
    plot_passive = plot_passive.copy()
    plot_passive["Passive_CumNet"] = (1 + plot_passive["Passive_Net"]).cumprod() - 1

    ax1.plot(
        plot_passive.index,
        plot_passive["Passive_CumNet"] * 100,
        label="Passive (Equal)",
        linewidth=2,
        linestyle="--",
        color="gray",
        alpha=0.8,
    )
    ax2.plot(
        plot_passive.index,
        plot_passive["Passive_EL"] * 100,
        label="Passive",
        color="gray",
        linestyle="--",
    )
    ax3.plot(
        plot_passive.index,
        plot_passive["Passive_Credit_Income"] * 100,
        label="Passive",
        color="gray",
        linestyle="--",
    )

    ax1.set_title("Cumulative Net Income (Market Return + Credit Income - EL)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Net Income (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.set_title("Monthly Realized Expected Loss (EL)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("EL (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax3.set_title("Monthly Credit Income (Interest Rate Revenue)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Credit Income (%)", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.xlabel("Date", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "amount_strategy_timeseries.png")
    _finalize_plot(save_path, verbose)
    log.info("Amount strategy time-series plot saved: %s", save_path)

    # --- Figure 2: Bar chart of summary metrics ---
    if comparison_df is None or comparison_df.empty:
        return

    metrics_to_plot = ["Avg Net Income (%)", "Avg Realized EL (%)", "Realized Vol (%)", "RAROC (%)"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    bar_colors = []
    for strat in comparison_df.index:
        bar_colors.append(strategy_colors.get(strat, "gray"))

    for ax, metric in zip(axes, available_metrics):
        values = comparison_df[metric]
        bars = ax.bar(range(len(values)), values, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(values)))
        labels = [strategy_labels.get(s, s) for s in comparison_df.index]
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Amount-Based Strategy Comparison: Key Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "amount_strategy_metrics.png")
    _finalize_plot(save_path, verbose)
    log.info("Amount strategy metrics plot saved: %s", save_path)


def plot_macro_model_comparison(
    comparison_df: pd.DataFrame,
    verbose: bool = False,
    figsize: tuple | None = None,
    value_fmt: str = "%.3f",
) -> None:
    """Plots a grid of horizontal bar charts comparing macro model metrics.

    Layout: rows = variables, columns = metrics. Each cell has its own
    independent X-axis so small values are not dwarfed by large ones.
    Each bar is labelled in Russian inside the bar (model name, left-
    aligned, bold) and on the right edge with the numeric value.

    Args:
        comparison_df: DataFrame with columns
            [Model, Variable, MAE, RMSE, MAPE (%), ...].
        verbose: Whether to display the plot interactively.
        figsize: Figure size; if None, sized as (22, 4 * n_vars).
        value_fmt: printf-style format for numeric value labels.
    """
    if comparison_df.empty:
        log.warning("Macro model comparison DataFrame is empty.")
        return

    var_ru = {
        "inflation": "Инфляция",
        "interest_rate": "Ключевая ставка",
        "unemployment_rate": "Безработица",
        "rubusd_exchange_rate": "Курс RUB/USD",
        "DD": "Дистанция до дефолта",
        "PD": "Вероятность дефолта",
    }
    metric_titles = {
        "MAE": "Средняя абсолютная ошибка (MAE)",
        "RMSE": "Корень из среднеквадратичной ошибки (RMSE)",
        "MAPE (%)": "Средняя абсолютная процентная ошибка (MAPE, %)",
    }
    metrics = [m for m in ["MAE", "RMSE", "MAPE (%)"] if m in comparison_df.columns]

    models = list(comparison_df["Model"].unique())
    variables = list(comparison_df["Variable"].unique())
    n_models = len(models)
    n_vars = len(variables)
    n_metrics = len(metrics)
    color_map = {m: c for m, c in zip(models, plt.cm.tab10.colors[:n_models])}

    if figsize is None:
        figsize = (22, 4.0 * n_vars)

    fig, axes = plt.subplots(n_vars, n_metrics, figsize=figsize, squeeze=False)

    for r, var in enumerate(variables):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            sub = comparison_df[comparison_df["Variable"] == var].set_index("Model").reindex(models)
            values = sub[metric].values.astype(float)
            ypos = np.arange(n_models)
            bars = ax.barh(
                ypos,
                values,
                color=[color_map[m] for m in models],
                edgecolor="white",
                height=0.7,
            )
            xmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
            inset = xmax * 0.02 if xmax > 0 else 0.0
            for rect, val, model in zip(bars, values, models):
                w = rect.get_width()
                if np.isfinite(val) and w > 0:
                    ax.text(
                        inset,
                        rect.get_y() + rect.get_height() / 2,
                        model.upper(),
                        ha="left",
                        va="center",
                        fontsize=18,
                        color="white",
                        fontweight="bold",
                    )
            ax.bar_label(bars, fmt=value_fmt, padding=4, fontsize=17)

            ax.set_yticks(ypos)
            ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis="x")
            ax.tick_params(axis="x", labelsize=12)
            if r == 0:
                ax.set_title(metric_titles.get(metric, metric), fontsize=17)
            if c == 0:
                ax.set_ylabel(
                    var_ru.get(var, var),
                    fontsize=20,
                    rotation=90,
                    ha="center",
                    va="center",
                    labelpad=16,
                )
            if np.isfinite(values).any() and xmax > 0:
                ax.set_xlim(0, xmax * 1.22)

    plt.suptitle(
        "Сравнение макропрогнозных моделей (walk-forward)",
        fontsize=26,
        y=1.005,
    )
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "macro_model_comparison.png")
    _finalize_plot(save_path, verbose)
    log.info("Macro model comparison plot saved: %s", save_path)
