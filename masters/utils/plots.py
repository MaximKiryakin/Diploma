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
            log.warning(f"No data for ticker {ticker}")
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


def calc_irf(
    portfolio_df: pd.DataFrame,
    impulses_responses: dict,
    figsize: tuple = (10, 4),
    verbose: bool = False,
) -> None:
    """Calculates and plots impulse response functions for the given impulses and responses.

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

    # Group by date to aggregate data across tickers (e.g., mean PD)
    if "date" in portfolio_df.columns:
        data = portfolio_df.groupby("date")[columns].mean().sort_index().dropna()
    else:
        log.warning("'date' column not found. Using raw data (stacked tickers?) for VAR.")
        data = portfolio_df.sort_values(["ticker", "date"])[columns].dropna()[columns]

    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        log.error(f"Constant columns detected: {constant_cols}. VAR model requires time-varying data.")
        data = data.drop(columns=constant_cols)
        if data.empty or len(data.columns) < 2:
            log.error("Not enough variables left for VAR analysis.")
            return

    pvalues = {col: adfuller(data[col].dropna())[1] for col in data.columns}

    if any(p > 0.05 for p in pvalues.values()):
        log.info("p-values before differencing:\n%s", pd.Series(pvalues))
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

        pvalues = {col: adfuller(data[col].dropna())[1] for col in data.columns}
        log.info("p-values after differencing:\n%s", pd.Series(pvalues))

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.RangeIndex(start=0, stop=len(data))

    model = VAR(data)
    lag_order = model.select_order(maxlags=6)
    selected_lags = lag_order.aic

    log.info(f"Optimal lag number calculated | Optimal number of lags: {selected_lags}")

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
    log.info(f"Correlation matrix saved | Path: {save_path}")


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

    for ticker, group in grouped:
        group = group.sort_values("date").dropna(subset=["capitalization", "debt"])

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
    log.info(f"Macro significance plot saved | Path: {save_path}")


def plot_macro_forecast(
    macro_df: pd.DataFrame,
    forecast_dfs: dict,
    tail: int = 0,
    figsize: tuple = (12, 10),
    verbose: bool = False,
) -> None:
    """Plots historical and forecasted values for macro parameters and Portfolio PD.

    Args:
        macro_df: Historical macro data (including PD).
        forecast_dfs: Dictionary mapping model labels to forecast DataFrames.
        tail: Number of last periods to show (0 = all).
        figsize: Figure size.
        verbose: Whether to display the plot interactively.
    """
    factors = list(macro_df.columns)
    n_factors = len(factors)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
    if n_factors == 1:
        axes = [axes]

    colors = ["darkred", "darkgreen", "orange", "purple", "brown"]

    for i, factor in enumerate(factors):
        ax = axes[i]
        is_pd = factor == "PD"
        is_dd = factor == "DD"

        plot_df = macro_df.tail(tail) if tail > 0 else macro_df

        fact_vals = plot_df[factor] * 100 if is_pd else plot_df[factor]
        ax.plot(plot_df.index, fact_vals, label="Fact", color="royalblue", linewidth=2.5)

        for idx, (label, forecast_df) in enumerate(forecast_dfs.items()):
            if factor not in forecast_df.columns:
                continue

            color = colors[idx % len(colors)]
            first_fc_date = forecast_df.index[0]
            history_before_fc = macro_df[macro_df.index < first_fc_date]

            fc_vals_raw = list(forecast_df[factor])
            fc_vals_scaled = [v * 100 for v in fc_vals_raw] if is_pd else fc_vals_raw

            if not history_before_fc.empty:
                fc_dates = [history_before_fc.index[-1]] + list(forecast_df.index)
                hist_val = history_before_fc[factor].iloc[-1]
                fc_vals = [hist_val * 100 if is_pd else hist_val] + fc_vals_scaled
            else:
                fc_dates = list(forecast_df.index)
                fc_vals = fc_vals_scaled

            ax.plot(fc_dates, fc_vals, label=f"FC: {label}", color=color, linestyle="--", linewidth=1.5)
            ax.scatter(forecast_df.index, fc_vals_scaled, color=color, s=20)

        if is_pd:
            title = "Portfolio Average PD (%)"
        elif is_dd:
            title = "Portfolio Average Distance to Default (DD)"
        else:
            title = f"Macro Factor: {factor}"

        ax.set_title(title, fontsize=12, fontweight="bold" if (is_pd or is_dd) else "normal")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xlabel("Date", fontsize=10)
    plt.xticks(rotation=90)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "macro_pd_comparison.png")
    _finalize_plot(save_path, verbose)
    log.info(f"Macro comparison plot saved: {save_path}")


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
        log.warning(f"Forecast DataFrame is empty. Cannot plot {metric.upper()} forecast.")
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
    log.info(f"{metric.upper()} forecast comparison plot saved: {save_path}")


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
    log.info(f"Portfolio allocation plot saved: {save_path}")


def plot_strategy_comparison(
    backtest_df: pd.DataFrame,
    comparison_df: pd.DataFrame = None,
    tail: int = 12,
    verbose: bool = False,
    all_backtests: dict = None,
) -> None:
    """Plots the cumulative returns and realized EL comparison between strategies.

    When ``all_backtests`` is provided (dict[str, DataFrame]), plots one line
    per strategy.  Falls back to the legacy two-line (Active vs Passive) view
    when only ``backtest_df`` is given.

    Args:
        backtest_df: Backtest results with Active/Passive returns and EL (legacy, used for Passive baseline).
        comparison_df: Legacy parameter, kept for backward compatibility.
        tail: Number of last periods to show (0 = all).
        verbose: Whether to display the plot interactively.
        all_backtests: Dict mapping strategy name -> backtest DataFrame.
    """
    if backtest_df.empty:
        log.warning("Strategy backtest DataFrame is empty.")
        return

    # Color palette for strategies
    strategy_colors = {
        "mean_el": "#1f77b4",
        "cvar": "#ff7f0e",
        "risk_parity": "#2ca02c",
    }
    strategy_labels = {
        "mean_el": "Active: Mean-EL (SLSQP)",
        "cvar": "Active: CVaR (LP)",
        "risk_parity": "Active: Risk Parity",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    if all_backtests and len(all_backtests) > 0:
        # Multi-strategy mode
        for strat_name, bt_df in all_backtests.items():
            plot_df = bt_df.tail(tail) if tail > 0 else bt_df
            plot_df = plot_df.copy()
            plot_df["Active_CumRet"] = (1 + plot_df["Active_Return"]).cumprod() - 1
            color = strategy_colors.get(strat_name, None)
            label = strategy_labels.get(strat_name, strat_name)

            ax1.plot(
                plot_df.index,
                plot_df["Active_CumRet"] * 100,
                label=label,
                linewidth=2.5,
                marker="o",
                markersize=3,
                color=color,
            )
            ax2.plot(
                plot_df.index,
                plot_df["Active_EL"] * 100,
                label=f"{label} EL",
                linewidth=2,
                color=color,
            )

        # Passive baseline (same for all strategies -- take from last backtest)
        last_bt = list(all_backtests.values())[-1]
        plot_passive = last_bt.tail(tail) if tail > 0 else last_bt
        plot_passive = plot_passive.copy()
        plot_passive["Passive_CumRet"] = (1 + plot_passive["Passive_Return"]).cumprod() - 1

        ax1.plot(
            plot_passive.index,
            plot_passive["Passive_CumRet"] * 100,
            label="Passive (Equal-Weight)",
            linewidth=2,
            linestyle="--",
            color="gray",
            alpha=0.8,
        )
        ax2.plot(
            plot_passive.index,
            plot_passive["Passive_EL"] * 100,
            label="Passive EL",
            color="gray",
            linestyle="--",
        )

        if "Actual_PD" in plot_passive.columns:
            ax2.bar(
                plot_passive.index,
                plot_passive["Actual_PD"] * 100,
                label="Portfolio Avg PD (%)",
                color="blue",
                alpha=0.12,
                width=20,
            )
    else:
        # Legacy two-line mode
        plot_df = backtest_df.tail(tail) if tail > 0 else backtest_df
        plot_df = plot_df.copy()
        plot_df["Active_CumRet"] = (1 + plot_df["Active_Return"]).cumprod() - 1
        plot_df["Passive_CumRet"] = (1 + plot_df["Passive_Return"]).cumprod() - 1

        ax1.plot(
            plot_df.index,
            plot_df["Active_CumRet"] * 100,
            label="Active (Optimized)",
            linewidth=3,
            marker="o",
            markersize=4,
        )
        ax1.plot(
            plot_df.index,
            plot_df["Passive_CumRet"] * 100,
            label="Passive (Equal)",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )
        ax2.plot(plot_df.index, plot_df["Active_EL"] * 100, label="Active Realized EL (%)", color="red", linewidth=2)
        ax2.plot(
            plot_df.index, plot_df["Passive_EL"] * 100, label="Passive Realized EL (%)", color="gray", linestyle="--"
        )
        if "Actual_PD" in plot_df.columns:
            ax2.bar(
                plot_df.index,
                plot_df["Actual_PD"] * 100,
                label="Portfolio Avg PD (%)",
                color="blue",
                alpha=0.15,
                width=20,
            )

    ax1.set_title("Strategy Backtest: Cumulative Returns (%)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Return (%)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.set_title("Portfolio Credit Risk: Realized EL and Average PD", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Loss / Probability (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Add vertical line for backtest start
    diff = backtest_df["Active_Return"] != backtest_df["Passive_Return"]
    backtest_start_date = backtest_df.index[diff.argmax()] if diff.any() else None

    if backtest_start_date:
        ref_df = backtest_df.tail(tail) if tail > 0 else backtest_df
        if backtest_start_date in ref_df.index:
            for ax in [ax1, ax2]:
                ax.axvline(x=backtest_start_date, color="black", linestyle=":", alpha=0.5)
                ax.text(backtest_start_date, ax.get_ylim()[1], " Backtest Start", verticalalignment="top")

    plt.xlabel("Date", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "strategy_backtest_comparison.png")
    _finalize_plot(save_path, verbose)
    log.info(f"Strategy comparison plot saved: {save_path}")


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
    log.info(f"Amount strategy time-series plot saved: {save_path}")

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
    log.info(f"Amount strategy metrics plot saved: {save_path}")


def plot_macro_model_comparison(
    comparison_df: pd.DataFrame,
    verbose: bool = False,
    figsize: tuple = (14, 6),
) -> None:
    """Plots a grouped bar chart comparing macro model accuracy metrics.

    Args:
        comparison_df: DataFrame with columns [Model, Variable, MAE, RMSE, MAPE (%)].
        verbose: Whether to display the plot interactively.
        figsize: Figure size.
    """
    if comparison_df.empty:
        log.warning("Macro model comparison DataFrame is empty.")
        return

    models = comparison_df["Model"].unique()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metrics = ["MAE", "RMSE", "MAPE (%)"]
    titles = ["Mean Absolute Error", "Root Mean Squared Error", "Mean Absolute Percentage Error (%)"]

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        pivot = comparison_df.pivot(index="Variable", columns="Model", values=metric)
        pivot = pivot.reindex(columns=models)
        pivot.plot(kind="bar", ax=ax, rot=30, width=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.legend(title="Model", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Macro Forecast Model Comparison (Walk-Forward)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = str(cfg.GRAPHS_DIR / "macro_model_comparison.png")
    _finalize_plot(save_path, verbose)
    log.info(f"Macro model comparison plot saved: {save_path}")
