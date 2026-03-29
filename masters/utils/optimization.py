"""Portfolio optimization and backtesting functions for the Portfolio class.

Covers weight-based optimization (mean_el / CVaR / risk-parity), portfolio
metrics, walk-forward backtests (strategy and amount-based) and multi-strategy
comparison.

All public functions accept a ``Portfolio`` instance as the first argument and
mutate ``self.d`` in place, returning ``self`` for method chaining.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

import utils.config as cfg
from utils.logger import Logger

if TYPE_CHECKING:
    from utils.portfolio import Portfolio

log = Logger(__name__)

# Rounding specification for Loan Allocations display (column -> decimal places).
_ALLOC_ROUNDING: dict = {
    "requested": 0,
    "approved": 2,
    "approval_rate": 2,
    "weight": 2,
    "rate": 2,
    "term": 2,
    "pd": 2,
    "pd_cum": 2,
    "lgd": 2,
    "risk_adj_return": 2,
    "expected_income": 0,
    "expected_loss": 0,
}


# ---------------------------------------------------------------------------
# Internal solvers
# ---------------------------------------------------------------------------


def _solve_mean_el_fn(
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
    cov_matrix = np.cov(returns_matrix, rowvar=False) * cfg.TRADING_DAYS_PER_YEAR
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


def _solve_cvar_fn(
    returns_matrix: np.ndarray,
    pds: np.ndarray,
    lgd: float,
    alpha: float,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Solves CVaR optimization via Rockafellar-Uryasev LP formulation.

    Minimizes CVaR_alpha of credit-adjusted portfolio losses.

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
    daily_el = (pds * lgd) / cfg.TRADING_DAYS_PER_YEAR
    adjusted_returns = returns_matrix - daily_el.reshape(1, n)

    coeff = 1.0 / ((1 - alpha) * s_count)
    n_vars = n + 1 + s_count

    c = np.zeros(n_vars)
    c[n] = 1.0
    c[n + 1 :] = coeff

    A_ub = np.zeros((s_count, n_vars))
    A_ub[:, :n] = -adjusted_returns
    A_ub[:, n] = -1.0
    for s in range(s_count):
        A_ub[s, n + 1 + s] = -1.0
    b_ub = np.zeros(s_count)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds_list = [(min_weight, max_weight)] * n
    bounds_list.append((None, None))
    bounds_list.extend([(0.0, None)] * s_count)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_list, method="highs")

    if not res.success:
        log.error("CVaR optimization failed: %s", res.message)
        return None

    log.info("CVaR(%.0f%%) = %.6f (daily), VaR(zeta) = %.6f", alpha * 100, res.fun, res.x[n])
    return res.x[:n]


def _solve_risk_parity_fn(
    returns_matrix: np.ndarray,
    pds: np.ndarray,
    lgd: float,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Solves Risk-Parity + Credit optimization.

    Finds weights such that each asset contributes equal credit-adjusted risk.
    Uses the log-barrier formulation from Maillard, Roncalli & Teiletche (2010).

    Args:
        returns_matrix: Daily log-returns, shape (S, N).
        pds: Predicted PDs per asset, shape (N,).
        lgd: Loss Given Default.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        Optimal weight vector of shape (N,), or None on failure.
    """
    cov_matrix = np.cov(returns_matrix, rowvar=False) * cfg.TRADING_DAYS_PER_YEAR
    n = len(pds)

    def credit_risk_contributions(w: np.ndarray) -> np.ndarray:
        sigma_w = cov_matrix @ w
        port_vol = np.sqrt(w @ sigma_w)
        mrc = w * sigma_w / (port_vol + cfg.EPSILON)
        crc = w * pds * lgd
        return mrc + crc

    def objective(w: np.ndarray) -> float:
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


# ---------------------------------------------------------------------------
# Portfolio weight optimization
# ---------------------------------------------------------------------------


def optimize_portfolio_fn(
    self: "Portfolio",
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

    Args:
        self: Portfolio instance.
        lambda_risk: Risk aversion parameter (0-1), used by mean_el strategy.
        lgd: Loss Given Default.
        use_forecast: If True, uses predicted PDs from self.d['pd_forecast'] or
            self.d['dd_forecast'].
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.
        strategy: One of 'mean_el', 'cvar', 'risk_parity'.
        cvar_alpha: CVaR confidence level (only used when strategy='cvar').
        cutoff_date: If provided, limits return history to avoid look-ahead bias.

    Returns:
        Portfolio: self with optimized weights in self.d['optimized_weights'].
    """
    if use_forecast and "dd_forecast" in self.d:
        current_pd_series = self.d["dd_forecast"]["predicted_pd"]
        tickers = current_pd_series.index.tolist()
        pds = current_pd_series.values
        log.debug("Using predicted PDs (from DD model) for optimization.")
    elif use_forecast and "pd_forecast" in self.d:
        current_pd_series = self.d["pd_forecast"]["predicted_pd"]
        tickers = current_pd_series.index.tolist()
        pds = current_pd_series.values
        log.debug("Using predicted PDs for optimization.")
    else:
        portfolio_df = self.d["portfolio"]
        latest_date = portfolio_df["date"].max()
        current_pd_df = portfolio_df[portfolio_df["date"] == latest_date][["ticker", "PD"]]
        tickers = current_pd_df["ticker"].tolist()
        pds = current_pd_df["PD"].values
        log.debug("Using historical PDs for optimization.")

    returns_df = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
    returns_df = np.log(returns_df / returns_df.shift(1)).dropna()

    if cutoff_date is not None:
        returns_df = returns_df.loc[returns_df.index < cutoff_date]

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
        weights = _solve_cvar_fn(returns_df[tickers].values, pds, lgd, cvar_alpha, min_weight, max_weight)
    elif strategy == "risk_parity":
        weights = _solve_risk_parity_fn(returns_df[tickers].values, pds, lgd, min_weight, max_weight)
    else:
        weights = _solve_mean_el_fn(returns_df[tickers].values, pds, lgd, lambda_risk, min_weight, max_weight)

    if weights is None:
        return self

    self.d["optimized_weights"] = pd.Series(weights, index=tickers, name="weight")
    log.debug("Portfolio optimization completed successfully (strategy=%s).", strategy)
    return self


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------


def calc_portfolio_metrics_fn(self: "Portfolio", lgd: float = 0.4) -> "Portfolio":
    """Calculates key risk metrics for the optimized portfolio.

    Args:
        self: Portfolio instance.
        lgd: Loss Given Default.

    Returns:
        Portfolio: self with metrics in self.d['portfolio_metrics'].
    """
    if "optimized_weights" not in self.d:
        log.error("No optimized weights found. Run optimize_portfolio() first.")
        return self

    weights = self.d["optimized_weights"]
    tickers = weights.index.tolist()

    if "dd_forecast" in self.d:
        pds = self.d["dd_forecast"]["predicted_pd"].loc[tickers]
    elif "pd_forecast" in self.d:
        pds = self.d["pd_forecast"]["predicted_pd"].loc[tickers]
    else:
        latest_date = self.d["portfolio"]["date"].max()
        pds = self.d["portfolio"][self.d["portfolio"]["date"] == latest_date].set_index("ticker").loc[tickers]["PD"]

    portfolio_el = np.sum(weights * pds * lgd)

    returns_df = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
    returns_df = np.log(returns_df / returns_df.shift(1)).dropna()
    cov_matrix = returns_df[tickers].cov().values * cfg.TRADING_DAYS_PER_YEAR
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

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


# ---------------------------------------------------------------------------
# Strategy backtest (weight-based)
# ---------------------------------------------------------------------------


def backtest_portfolio_strategies_fn(
    self: "Portfolio",
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

    Accepts one or several strategies evaluated over the same walk-forward window.

    Args:
        self: Portfolio instance.
        n_months: Number of months for the backtest window.
        lambda_risk: Risk aversion for mean_el strategy.
        lgd: Loss Given Default.
        model_type: Macro model used for DD forecasts.
        rebalance_threshold: Max allowed drift before rebalancing (0-1).
        transaction_cost: One-way proportional cost per unit of turnover.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.
        strategy: One strategy name or a list of strategy names.
        cvar_alpha: CVaR confidence level (only used when strategy='cvar').

    Returns:
        Portfolio: self with results in self.d['strategy_backtest'].
    """
    strategies: List[str] = [strategy] if isinstance(strategy, str) else list(strategy)

    prices_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
    monthly_prices = prices_pivot.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    pd_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="PD")
    monthly_pds = pd_pivot.resample("ME").last().dropna()

    common_index = monthly_returns.index.intersection(monthly_pds.index)
    monthly_returns = monthly_returns.loc[common_index]
    monthly_pds = monthly_pds.loc[common_index]

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

    if len(monthly_returns) < n_months:
        log.error(f"Insufficient history for {n_months} months backtest.")
        return self

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

        self.predict_dd(horizon=1, training_offset=offset, model_type=model_type)

        for strat in strategies:
            st = strat_state[strat]

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
                        f"  [{strat}] Rebalancing triggered " f"(max drift={max_drift:.2%}, turnover={turnover:.2%})"
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

            n_tickers = len(available_tickers)
            w_passive = pd.Series(1.0 / n_tickers, index=available_tickers)
            ret_passive = np.sum(w_passive * monthly_returns.loc[target_date, available_tickers])
            el_passive = np.sum(w_passive * monthly_pds.loc[target_date, available_tickers] * lgd)

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
                "Realized Vol (%)": round(bt_only["Active_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100, 4),
                "Total TC (%)": round(bt_only["TC"].sum() * 100, 4),
                "Rebalances": int(bt_only["Rebalanced"].sum()),
                "Avg Turnover (%)": round(bt_only["Turnover"].mean() * 100, 4),
            }
        )

        wh_df = pd.DataFrame(st["weight_history"]).set_index("date")
        rebal_flags = wh_df.pop("rebalanced")
        wh_df = wh_df.fillna(0.0)
        wh_df["rebalanced"] = rebal_flags
        all_weight_histories[strat] = wh_df

    last_bt = all_backtests[strategies[-1]]
    bt_passive = last_bt.tail(n_months)
    comparison_rows.append(
        {
            "Strategy": "passive (equal)",
            "Total Return (%)": round(((1 + bt_passive["Passive_Return"]).prod() - 1) * 100, 4),
            "Avg Realized EL (%)": round(bt_passive["Passive_EL"].mean() * 100, 6),
            "Realized Vol (%)": round(bt_passive["Passive_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100, 4),
            "Total TC (%)": 0.0,
            "Rebalances": 0,
            "Avg Turnover (%)": 0.0,
        }
    )

    multi_comparison = pd.DataFrame(comparison_rows).set_index("Strategy")
    log.log_dataframe(multi_comparison.reset_index(), title="Strategy Comparison")

    self.d["all_backtests"] = all_backtests
    self.d["all_weight_histories"] = all_weight_histories
    self.d["multi_strategy_comparison"] = multi_comparison

    last_strat = strategies[-1]
    self.d["strategy_backtest"] = all_backtests[last_strat]
    self.d["weight_history"] = all_weight_histories[last_strat]

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
            last_bt_only["Active_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100,
            last_bt_only["Passive_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100,
        ],
        "Total Transaction Cost (%)": [last_bt_only["TC"].sum() * 100, 0.0],
        "Rebalances": [int(last_bt_only["Rebalanced"].sum()), 0],
        "Avg Turnover (%)": [last_bt_only["Turnover"].mean() * 100, 0.0],
    }
    self.d["strategy_comparison"] = pd.DataFrame(summary_compat, index=["Active (Optimized)", "Passive (Equal)"])
    return self


# ---------------------------------------------------------------------------
# Amount-based optimization
# ---------------------------------------------------------------------------


def optimize_portfolio_with_amounts_fn(
    self: "Portfolio",
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
    """Optimizes portfolio allocations using real loan amounts and return maximization.

    Args:
        self: Portfolio instance.
        loan_applications: DataFrame with columns [ticker, amount, rate, lgd]
            and optional ``term`` (loan maturity in months, default 12).
        budget: Total lending budget.
        lambda_return: Weight for return maximization component (0-1).
        lambda_vol: Weight for risk measure component (0-1).
        lambda_el: Weight for expected loss minimization component (0-1).
        max_sector_share: Maximum share of budget per sector.
        sector_map: Dict mapping ticker -> sector name.
        cutoff_date: If provided, limits return history to avoid look-ahead bias.
        strategy: Risk measure to use: 'mean_el', 'cvar', 'risk_parity'.
        cvar_alpha: CVaR confidence level (only for strategy='cvar').

    Returns:
        Portfolio: self with results in self.d['loan_allocations'] and
            self.d['optimized_weights'].
    """
    tickers = loan_applications["ticker"].tolist()
    amounts = loan_applications["amount"].values.astype(float)
    rates = loan_applications["rate"].values.astype(float)
    lgds = loan_applications["lgd"].values.astype(float)
    terms = (
        loan_applications["term"].values.astype(float)
        if "term" in loan_applications.columns
        else np.full(len(tickers), 12.0)
    )
    n = len(tickers)

    if "dd_forecast" in self.d:
        pds = self.d["dd_forecast"]["predicted_pd"].loc[tickers].values.astype(float)
        log.debug("Using predicted PDs (from DD model) for amount-based optimization.")
    elif "pd_forecast" in self.d:
        pds = self.d["pd_forecast"]["predicted_pd"].loc[tickers].values.astype(float)
        log.debug("Using predicted PDs for amount-based optimization.")
    else:
        latest_date = self.d["portfolio"]["date"].max()
        pds = (
            self.d["portfolio"][self.d["portfolio"]["date"] == latest_date]
            .set_index("ticker")
            .loc[tickers]["PD"]
            .values.astype(float)
        )
        log.debug("Using historical PDs for amount-based optimization.")

    pd_cum = 1.0 - (1.0 - pds) ** (terms / 12.0)
    risk_adj_return = rates * (1.0 - pd_cum) - pd_cum * lgds * 12.0 / terms

    returns_df = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
    returns_df = np.log(returns_df / returns_df.shift(1)).dropna()
    if cutoff_date is not None:
        returns_df = returns_df.loc[returns_df.index < cutoff_date]
    returns_matrix = returns_df[tickers].values
    cov_matrix = returns_df[tickers].cov().values * cfg.TRADING_DAYS_PER_YEAR

    el_annual = pd_cum * lgds * 12.0 / terms
    daily_el = (pd_cum * lgds) / cfg.TRADING_DAYS_PER_YEAR
    adjusted_returns = returns_matrix - daily_el.reshape(1, n)

    # Decision variable z = x/budget: rescales gradients from O(rate/budget)~5e-12
    # to O(rate)~0.1, which is required for SLSQP to differentiate strategies.
    if strategy == "mean_el":

        def objective(z: np.ndarray) -> float:
            total = z.sum() + cfg.EPSILON
            w = z / total
            income = np.sum(z * risk_adj_return)
            port_vol = np.sqrt(w @ cov_matrix @ w)
            el = np.sum(w * el_annual)
            return -lambda_return * income + lambda_vol * port_vol + lambda_el * el

    elif strategy == "cvar":

        def objective(z: np.ndarray) -> float:
            total = z.sum() + cfg.EPSILON
            w = z / total
            income = np.sum(z * risk_adj_return)
            port_losses = -(adjusted_returns @ w)
            sorted_losses = np.sort(port_losses)[::-1]
            cutoff = max(int(np.ceil((1 - cvar_alpha) * len(sorted_losses))), 1)
            cvar_val = np.mean(sorted_losses[:cutoff]) * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)
            el = np.sum(w * el_annual)
            return -lambda_return * income + lambda_vol * cvar_val + lambda_el * el

    elif strategy == "risk_parity":

        def objective(z: np.ndarray) -> float:
            total = z.sum() + cfg.EPSILON
            w = z / total
            income = np.sum(z * risk_adj_return)
            sigma_w = cov_matrix @ w
            port_vol = np.sqrt(w @ sigma_w + cfg.EPSILON)
            mrc = w * sigma_w / (port_vol + cfg.EPSILON)
            crc = mrc + w * el_annual
            total_crc = crc.sum() + cfg.EPSILON
            crc_norm = crc / total_crc
            target = 1.0 / n
            diff_sum = float(np.sum((crc_norm - target) ** 2))
            return -lambda_return * income + lambda_vol * diff_sum

    else:
        log.error("Unknown strategy '%s'. Use 'mean_el', 'cvar', or 'risk_parity'.", strategy)
        return self

    log.debug("Amount-based optimization: strategy=%s", strategy)

    constraints = [{"type": "ineq", "fun": lambda z: 1.0 - np.sum(z)}]
    if sector_map is not None:
        sectors = set(sector_map.values())
        for sector in sectors:
            idx = [i for i, t in enumerate(tickers) if sector_map.get(t) == sector]
            if idx:
                constraints.append(
                    {"type": "ineq", "fun": lambda z, _idx=idx: max_sector_share - sum(z[i] for i in _idx)}
                )

    bounds = tuple((0.0, a / budget) for a in amounts)
    x0 = np.minimum(amounts, budget / n) / budget

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not res.success:
        log.error("Amount-based optimization failed: %s", res.message)
        return self

    approved = res.x * budget
    total_approved = approved.sum()

    self.d["loan_allocations"] = pd.DataFrame(
        {
            "ticker": tickers,
            "requested": amounts,
            "approved": approved,
            "approval_rate": approved / np.where(amounts > 0, amounts, cfg.EPSILON),
            "weight": approved / (total_approved + cfg.EPSILON),
            "rate": rates,
            "term": terms,
            "pd": pds,
            "pd_cum": pd_cum,
            "lgd": lgds,
            "risk_adj_return": risk_adj_return,
            "expected_income": approved * risk_adj_return,
            "expected_loss": approved * pd_cum * lgds,
        }
    )

    weights = approved / (total_approved + cfg.EPSILON)
    self.d["optimized_weights"] = pd.Series(weights, index=tickers, name="weight")

    total_income = np.sum(approved * risk_adj_return)
    total_el = np.sum(approved * pd_cum * lgds * 12.0 / terms)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)

    metrics = {
        "Budget": budget,
        "Total Approved": total_approved,
        "Budget Utilization (%)": total_approved / budget * 100,
        "Expected Income": total_income,
        "Expected Loss": total_el,
        "Net Expected Income": total_income - total_el,
        "RAROC (%)": (total_income / (total_approved + cfg.EPSILON)) * 100,
        "Portfolio Volatility (%)": port_vol * 100,
        "Applications": n,
        "Fully Approved": int(np.sum(np.isclose(approved, amounts, rtol=0.01))),
        "Partially Approved": int(np.sum((approved > 0) & (~np.isclose(approved, amounts, rtol=0.01)))),
        "Rejected": int(np.sum(approved < cfg.EPSILON)),
    }
    self.d["loan_metrics"] = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    alloc_display = self.d["loan_allocations"].copy()
    for _col, _dec in _ALLOC_ROUNDING.items():
        if _col in alloc_display.columns:
            alloc_display[_col] = alloc_display[_col].round(_dec)
    log.log_dataframe(alloc_display, title="Loan Allocations", level=logging.DEBUG)
    log.log_dataframe(self.d["loan_metrics"], title="Loan Portfolio Metrics", level=logging.DEBUG)
    log.debug("Amount-based portfolio optimization completed successfully.")
    return self


# ---------------------------------------------------------------------------
# Amount-based backtest
# ---------------------------------------------------------------------------


def backtest_portfolio_with_amounts_fn(
    self: "Portfolio",
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
    """Backtests the amount-based optimization strategy vs equal allocation.

    Args:
        self: Portfolio instance.
        loan_applications: DataFrame with columns [ticker, amount, rate, lgd, term].
        budget: Total lending budget.
        n_months: Number of months for the backtest window.
        lambda_return: Weight for return component.
        lambda_vol: Weight for risk measure component.
        lambda_el: Weight for expected loss component.
        model_type: Macro model for DD forecasts.
        max_sector_share: Maximum share of budget per sector.
        sector_map: Dict mapping ticker -> sector.
        rebalance_threshold: Max weight drift before rebalancing.
        transaction_cost: One-way proportional cost per unit of turnover.
        strategy: Risk measure: 'mean_el', 'cvar', 'risk_parity'.
        cvar_alpha: CVaR confidence level (only for strategy='cvar').

    Returns:
        Portfolio: self with results in self.d['amount_backtest'].
    """
    tickers = loan_applications["ticker"].tolist()
    rates = loan_applications.set_index("ticker")["rate"]
    lgds_map = loan_applications.set_index("ticker")["lgd"]

    prices_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="close")
    monthly_prices = prices_pivot.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    pd_pivot = self.d["portfolio"].pivot(index="date", columns="ticker", values="PD")
    monthly_pds = pd_pivot.resample("ME").last().dropna()

    common_index = monthly_returns.index.intersection(monthly_pds.index)
    monthly_returns = monthly_returns.loc[common_index]
    monthly_pds = monthly_pds.loc[common_index]

    available_tickers = [t for t in tickers if t in monthly_returns.columns]
    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
        log.warning("Tickers missing from returns data: %s", missing)

    if len(monthly_returns) < n_months:
        log.error(f"Insufficient history for {n_months} months backtest.")
        return self

    if "term" in loan_applications.columns:
        terms_map = loan_applications.set_index("ticker")["term"].astype(float)
    else:
        terms_map = pd.Series(12.0, index=loan_applications["ticker"].values)
    remaining_terms = terms_map.reindex(available_tickers, fill_value=12.0).copy()

    results = []
    w_active_actual = None
    total_rebalances = 0
    weight_history = []

    log.info(f"Starting amount-based backtest for {n_months} months " f"(budget={budget:,.0f}, model={model_type})...")

    for offset in range(n_months, 0, -1):
        target_date = monthly_returns.index[-offset]
        log.info(f"Amount backtest month: {target_date.strftime('%Y-%m')}")

        active_tickers_month = [t for t in available_tickers if remaining_terms[t] > 0]
        n_active = len(active_tickers_month)

        if n_active == 0:
            log.info("All loans expired at %s.", target_date.strftime("%Y-%m"))
            results.append(
                {
                    "Date": target_date,
                    "Active_Return": 0.0,
                    "Active_EL": 0.0,
                    "Active_Credit_Income": 0.0,
                    "Active_Net": 0.0,
                    "Passive_Return": 0.0,
                    "Passive_EL": 0.0,
                    "Passive_Credit_Income": 0.0,
                    "Passive_Net": 0.0,
                    "Turnover": 0.0,
                    "TC": 0.0,
                    "Rebalanced": False,
                    "Active_Loans": 0,
                }
            )
            weight_record = {"date": target_date, "rebalanced": False}
            for ticker in available_tickers:
                weight_record[ticker] = 0.0
            weight_history.append(weight_record)
            continue

        self.predict_dd(horizon=1, training_offset=offset, model_type=model_type)

        active_apps = loan_applications[loan_applications["ticker"].isin(active_tickers_month)].copy()
        active_apps["term"] = active_apps["ticker"].map(remaining_terms)

        self.optimize_portfolio_with_amounts(
            loan_applications=active_apps,
            budget=budget,
            lambda_return=lambda_return,
            lambda_vol=lambda_vol,
            lambda_el=lambda_el,
            max_sector_share=max_sector_share,
            sector_map=sector_map,
            cutoff_date=target_date,
            strategy=strategy,
            cvar_alpha=cvar_alpha,
        )

        alloc = self.d["loan_allocations"].set_index("ticker")
        w_target = alloc["weight"].reindex(available_tickers, fill_value=0.0)
        w_target = w_target / (w_target.sum() + cfg.EPSILON)

        if w_active_actual is None:
            w_active_actual = w_target.copy()
            turnover = w_target.abs().sum()
            rebalanced = True
            total_rebalances += 1
        else:
            rets = monthly_returns.loc[target_date, available_tickers].fillna(0)
            w_drifted = w_active_actual * (1 + rets)
            total_val = w_drifted.sum()
            w_drifted = w_drifted / total_val if total_val > 0 else w_target.copy()

            max_drift = (w_drifted - w_target).abs().max()
            if max_drift > rebalance_threshold:
                turnover = (w_target - w_drifted).abs().sum()
                w_active_actual = w_target.copy()
                rebalanced = True
                total_rebalances += 1
            else:
                turnover = 0.0
                w_active_actual = w_drifted
                rebalanced = False

        tc = transaction_cost * turnover
        pds_t = monthly_pds.loc[target_date, available_tickers].fillna(0)

        active_return = np.sum(w_active_actual * monthly_returns.loc[target_date, available_tickers]) - tc
        active_el = np.sum(w_active_actual * pds_t * lgds_map.reindex(available_tickers, fill_value=0.4))
        active_credit_income = np.sum(
            w_active_actual.reindex(active_tickers_month, fill_value=0)
            * rates.reindex(active_tickers_month, fill_value=0)
            / cfg.MONTHS_PER_YEAR
        )

        w_passive = pd.Series(0.0, index=available_tickers)
        w_passive[active_tickers_month] = 1.0 / n_active
        passive_return = np.sum(w_passive * monthly_returns.loc[target_date, available_tickers])
        passive_el = np.sum(w_passive * pds_t * lgds_map.reindex(available_tickers, fill_value=0.4))
        passive_credit_income = np.sum(
            w_passive.reindex(active_tickers_month, fill_value=0)
            * rates.reindex(active_tickers_month, fill_value=0)
            / cfg.MONTHS_PER_YEAR
        )

        weight_record = {"date": target_date, "rebalanced": rebalanced}
        for ticker in available_tickers:
            weight_record[ticker] = w_active_actual.get(ticker, 0.0)
        weight_history.append(weight_record)

        results.append(
            {
                "Date": target_date,
                "Active_Return": active_return,
                "Active_EL": active_el,
                "Active_Credit_Income": active_credit_income,
                "Active_Net": active_return + active_credit_income - active_el,
                "Passive_Return": passive_return,
                "Passive_EL": passive_el,
                "Passive_Credit_Income": passive_credit_income,
                "Passive_Net": passive_return + passive_credit_income - passive_el,
                "Turnover": turnover,
                "TC": tc,
                "Rebalanced": rebalanced,
                "Active_Loans": n_active,
            }
        )

        remaining_terms -= 1

    backtest_df = pd.DataFrame(results).set_index("Date")

    summary = pd.DataFrame(
        {
            "Total Market Return (%)": [
                ((1 + backtest_df["Active_Return"]).prod() - 1) * 100,
                ((1 + backtest_df["Passive_Return"]).prod() - 1) * 100,
            ],
            "Avg Credit Income (%)": [
                backtest_df["Active_Credit_Income"].mean() * 100,
                backtest_df["Passive_Credit_Income"].mean() * 100,
            ],
            "Avg Realized EL (%)": [
                backtest_df["Active_EL"].mean() * 100,
                backtest_df["Passive_EL"].mean() * 100,
            ],
            "Avg Net Income (%)": [
                backtest_df["Active_Net"].mean() * 100,
                backtest_df["Passive_Net"].mean() * 100,
            ],
            "Realized Vol (%)": [
                backtest_df["Active_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100,
                backtest_df["Passive_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100,
            ],
            "Total TC (%)": [backtest_df["TC"].sum() * 100, 0.0],
            "Rebalances": [total_rebalances, 0],
        },
        index=["Active (Optimized)", "Passive (Equal)"],
    )

    self.d["amount_backtest"] = backtest_df
    self.d["amount_backtest_summary"] = summary
    self.d["amount_weight_history"] = pd.DataFrame(weight_history).set_index("date")

    log.log_dataframe(summary.reset_index(), title="Amount-Based Strategy Comparison")
    log.info(f"Amount-based backtest completed. Rebalances: {total_rebalances}/{n_months}")
    return self


# ---------------------------------------------------------------------------
# Multi-strategy amount comparison
# ---------------------------------------------------------------------------


def compare_amount_strategies_fn(
    self: "Portfolio",
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
    """Runs amount-based backtest for multiple strategies and builds comparison.

    Args:
        self: Portfolio instance.
        loan_applications: DataFrame with columns [ticker, amount, rate, lgd, term].
        budget: Total lending budget.
        strategies: List of strategy names (default: all three).
        n_months: Backtest window length in months.
        lambda_return: Weight for return component.
        lambda_vol: Weight for risk measure component.
        lambda_el: Weight for expected loss component.
        model_type: Macro model for DD forecasts.
        max_sector_share: Maximum share of budget per sector.
        sector_map: Dict mapping ticker -> sector.
        rebalance_threshold: Max weight drift before rebalancing.
        transaction_cost: One-way proportional cost per unit of turnover.
        cvar_alpha: CVaR confidence level (only for strategy='cvar').

    Returns:
        Portfolio: self with self.d['all_amount_backtests'] and
            self.d['amount_multi_strategy_comparison'] populated.
    """
    if strategies is None:
        strategies = ["mean_el", "cvar", "risk_parity"]

    all_backtests: Dict[str, pd.DataFrame] = {}
    comparison_rows: List[dict] = []

    for strat in strategies:
        log.info("=" * 60)
        log.info("Amount-based backtest: strategy=%s", strat)
        log.info("=" * 60)

        self.backtest_portfolio_with_amounts(
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
            strategy=strat,
            cvar_alpha=cvar_alpha,
        )
        bt = self.d["amount_backtest"].copy()
        all_backtests[strat] = bt

        alloc = self.d.get("loan_allocations")
        if alloc is not None:
            alloc_display = alloc.copy()
            for _col, _dec in _ALLOC_ROUNDING.items():
                if _col in alloc_display.columns:
                    alloc_display[_col] = alloc_display[_col].round(_dec)
            log.log_dataframe(alloc_display, title=f"[{strat}] Loan Allocations (last period)")

        net_vol = bt["Active_Net"].std() * np.sqrt(cfg.MONTHS_PER_YEAR)
        comparison_rows.append(
            {
                "Strategy": strat,
                "Total Market Return (%)": round(((1 + bt["Active_Return"]).prod() - 1) * 100, 4),
                "Avg Credit Income (%)": round(bt["Active_Credit_Income"].mean() * 100, 4),
                "Avg Realized EL (%)": round(bt["Active_EL"].mean() * 100, 6),
                "Avg Net Income (%)": round(bt["Active_Net"].mean() * 100, 4),
                "Realized Vol (%)": round(bt["Active_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100, 4),
                "Total TC (%)": round(bt["TC"].sum() * 100, 4),
                "Rebalances": int(bt["Rebalanced"].sum()),
                "RAROC (%)": round(bt["Active_Net"].mean() / (net_vol + cfg.EPSILON) * 100, 4),
            }
        )

    last_bt = all_backtests[strategies[-1]]
    passive_net_vol = last_bt["Passive_Net"].std() * np.sqrt(cfg.MONTHS_PER_YEAR)
    comparison_rows.append(
        {
            "Strategy": "passive (equal)",
            "Total Market Return (%)": round(((1 + last_bt["Passive_Return"]).prod() - 1) * 100, 4),
            "Avg Credit Income (%)": round(last_bt["Passive_Credit_Income"].mean() * 100, 4),
            "Avg Realized EL (%)": round(last_bt["Passive_EL"].mean() * 100, 6),
            "Avg Net Income (%)": round(last_bt["Passive_Net"].mean() * 100, 4),
            "Realized Vol (%)": round(last_bt["Passive_Return"].std() * np.sqrt(cfg.MONTHS_PER_YEAR) * 100, 4),
            "Total TC (%)": 0.0,
            "Rebalances": 0,
            "RAROC (%)": round(last_bt["Passive_Net"].mean() / (passive_net_vol + cfg.EPSILON) * 100, 4),
        }
    )

    multi_comparison = pd.DataFrame(comparison_rows).set_index("Strategy")
    self.d["all_amount_backtests"] = all_backtests
    self.d["amount_multi_strategy_comparison"] = multi_comparison

    log.log_dataframe(multi_comparison.reset_index(), title="Amount-Based Multi-Strategy Comparison")
    return self
