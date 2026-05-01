"""Credit risk forecasting functions for the Portfolio class.

Covers macro factor prediction, PD/DD forecasting, walk-forward backtests,
OLS macro significance analysis, and model comparison.

All public functions accept a ``Portfolio`` instance as the first argument and
mutate ``self.d`` in place, returning ``self`` for method chaining.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

import utils.plots as plots
import utils.config as cfg
from utils.logger import Logger

if TYPE_CHECKING:
    from utils.portfolio import Portfolio

log = Logger(__name__)


# ---------------------------------------------------------------------------
# Macro factor prediction
# ---------------------------------------------------------------------------


def predict_macro_factors_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
) -> pd.DataFrame:
    """Predicts macroeconomic factors using the specified model.

    Args:
        self: Portfolio instance.
        horizon: Number of months to predict.
        training_offset: Months of history to hide (for walk-forward backtest).
        model_type: 'var', 'sarimax', or 'prophet'.

    Returns:
        pd.DataFrame: Forecasted macro variables indexed by future dates.
    """
    if horizon < 0:
        training_offset = abs(horizon)
        horizon = abs(horizon)

    macro_df = (
        self.d["portfolio"][["date"] + cfg.MACRO_COLS]
        .drop_duplicates("date")
        .set_index("date")
        .resample("ME")
        .mean()
        .dropna()
    )

    if training_offset > 0:
        macro_df = macro_df.iloc[:-training_offset]

    future_dates = pd.date_range(start=macro_df.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")

    if model_type.lower() == "var":
        results = VAR(macro_df).fit(maxlags=max(1, min(3, len(macro_df) // 2 - 1)))
        fc = results.forecast(y=macro_df.values[-results.k_ar :], steps=horizon)
        return pd.DataFrame(fc, index=future_dates, columns=macro_df.columns)

    if model_type.lower() == "sarimax":
        forecast_results = {}
        for col in macro_df.columns:
            d = 1 if adfuller(macro_df[col])[1] > 0.05 else 0
            exog_train = macro_df.drop(columns=[col]).shift(1).bfill()

            # Scaling is crucial: macro variables have very different magnitudes.
            sy, sx = StandardScaler(), StandardScaler()
            y_scaled = sy.fit_transform(macro_df[[col]])
            x_scaled = sx.fit_transform(exog_train)

            results = SARIMAX(
                y_scaled,
                exog=x_scaled,
                order=(1, d, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=500)

            exog_fc = pd.DataFrame(
                [macro_df.drop(columns=[col]).iloc[-1]] * horizon,
                columns=exog_train.columns,
            )
            fc_scaled = results.forecast(steps=horizon, exog=sx.transform(exog_fc))
            forecast_results[col] = sy.inverse_transform(fc_scaled.reshape(-1, 1)).flatten()
        return pd.DataFrame(forecast_results, index=future_dates)

    if model_type.lower() == "prophet":
        forecast_results = {}
        for col in macro_df.columns:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            m.fit(macro_df[[col]].reset_index().rename(columns={"date": "ds", col: "y"}))
            future = m.make_future_dataframe(periods=horizon, freq="ME")
            forecast_results[col] = m.predict(future)["yhat"].tail(horizon).values
        return pd.DataFrame(forecast_results, index=future_dates)

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# PD / DD prediction
# ---------------------------------------------------------------------------


def predict_pd_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
) -> "Portfolio":
    """Predicts PD for portfolio assets based on macro OLS model.

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon in months.
        training_offset: Months of history to hide for backtesting.
        model_type: Macro model type ('var', 'sarimax', 'prophet').

    Returns:
        Portfolio: self with results in self.d['pd_forecast'].
    """
    if horizon < 0:
        training_offset = abs(horizon)
        horizon = abs(horizon)

    macro_forecast = predict_macro_factors_fn(
        self, horizon=horizon, training_offset=training_offset, model_type=model_type
    )

    pd_pivot = self.d["portfolio"][["date", "ticker", "PD"]].pivot(index="date", columns="ticker", values="PD")
    pd_monthly = pd_pivot.resample("ME").last()
    pd_monthly.index = pd_monthly.index.normalize() + pd.offsets.MonthEnd(0)

    macro_cols = macro_forecast.columns.tolist()
    macro_hist = (
        self.d["portfolio"][["date"] + macro_cols]
        .drop_duplicates("date")
        .set_index("date")
        .resample("ME")
        .mean()
        .dropna()
    )
    macro_hist.index = macro_hist.index.normalize() + pd.offsets.MonthEnd(0)

    combined = pd.concat([pd_monthly, macro_hist], axis=1).dropna()
    train = combined.iloc[:-training_offset] if training_offset > 0 else combined

    predictions = {}
    for ticker in pd_monthly.columns:
        if ticker not in train.columns:
            continue
        y = train[ticker]
        x = sm.add_constant(train[macro_cols])
        model = sm.OLS(y, x).fit()
        x_pred = sm.add_constant(macro_forecast, has_constant="add")
        if "const" not in x_pred.columns:
            x_pred["const"] = 1.0
        predictions[ticker] = model.predict(x_pred).iloc[-1]

    result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["predicted_pd"])
    result_df.index.name = "ticker"
    target_date = macro_forecast.index[-1]
    comparison_pd = pd_monthly.loc[target_date] if target_date in pd_monthly.index else pd_pivot.ffill().iloc[-1]
    result_df["reference_pd"] = comparison_pd
    result_df["delta"] = result_df["predicted_pd"] - result_df["reference_pd"]
    result_df["predicted_pd"] = result_df["predicted_pd"].clip(0, 1)
    result_df["model"] = model_type

    self.d["pd_forecast"] = result_df
    return self


def predict_dd_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
) -> "Portfolio":
    """Predicts Distance to Default (DD) based on macro OLS model.

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon in months.
        training_offset: Months of history to hide for backtesting.
        model_type: Macro model type ('var', 'sarimax', 'prophet').

    Returns:
        Portfolio: self with results in self.d['dd_forecast'].
    """
    if horizon < 0:
        training_offset = abs(horizon)
        horizon = abs(horizon)

    macro_forecast = predict_macro_factors_fn(
        self, horizon=horizon, training_offset=training_offset, model_type=model_type
    )

    dd_pivot = self.d["portfolio"][["date", "ticker", "DD"]].pivot(index="date", columns="ticker", values="DD")
    dd_monthly = dd_pivot.resample("ME").last()
    dd_monthly.index = dd_monthly.index.normalize() + pd.offsets.MonthEnd(0)

    macro_cols = macro_forecast.columns.tolist()
    macro_hist = (
        self.d["portfolio"][["date"] + macro_cols]
        .drop_duplicates("date")
        .set_index("date")
        .resample("ME")
        .mean()
        .dropna()
    )
    macro_hist.index = macro_hist.index.normalize() + pd.offsets.MonthEnd(0)

    combined = pd.concat([dd_monthly, macro_hist], axis=1).dropna()
    train = combined.iloc[:-training_offset] if training_offset > 0 else combined

    predictions = {}
    for ticker in dd_monthly.columns:
        if ticker not in train.columns:
            continue
        y = train[ticker]
        x = sm.add_constant(train[macro_cols])
        model = sm.OLS(y, x).fit()
        x_pred = sm.add_constant(macro_forecast, has_constant="add")
        if "const" not in x_pred.columns:
            x_pred["const"] = 1.0
        predictions[ticker] = model.predict(x_pred).iloc[-1]

    result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["predicted_dd"])
    result_df.index.name = "ticker"
    target_date = macro_forecast.index[-1]
    comparison_dd = dd_monthly.loc[target_date] if target_date in dd_monthly.index else dd_pivot.ffill().iloc[-1]
    result_df["reference_dd"] = comparison_dd
    result_df["delta"] = result_df["predicted_dd"] - result_df["reference_dd"]
    result_df["model"] = model_type
    result_df["predicted_pd"] = norm.cdf(-result_df["predicted_dd"].astype(float))
    result_df["reference_pd"] = norm.cdf(-result_df["reference_dd"].astype(float))

    self.d["dd_forecast"] = result_df
    return self


# ---------------------------------------------------------------------------
# Walk-forward backtests for PD / DD
# ---------------------------------------------------------------------------


def backtest_pd_fn(
    self: "Portfolio",
    n_months: int = 12,
    models: List[str] = None,
) -> "Portfolio":
    """Walk-forward backtest of PD predictions across macro models.

    Args:
        self: Portfolio instance.
        n_months: Number of months in the backtest window.
        models: List of model types. Defaults to ['var'].

    Returns:
        Portfolio: self with results in self.d['pd_backtest'].
    """
    if models is None:
        models = ["var"]

    all_results = []
    for m_type in models:
        log.info("Backtesting PD using %s model...", m_type)
        for offset in range(n_months, 0, -1):
            predict_pd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
            macro_fc = predict_macro_factors_fn(self, horizon=1, training_offset=offset, model_type=m_type)
            target_date = macro_fc.index[-1]
            res = self.d["pd_forecast"].copy().reset_index()
            res["date"] = target_date
            all_results.append(res)

    self.d["pd_backtest"] = pd.concat(all_results, ignore_index=True)
    return self


def backtest_dd_fn(
    self: "Portfolio",
    n_months: int = 12,
    models: List[str] = None,
) -> "Portfolio":
    """Walk-forward backtest of DD predictions across macro models.

    Args:
        self: Portfolio instance.
        n_months: Number of months in the backtest window.
        models: List of model types. Defaults to ['var'].

    Returns:
        Portfolio: self with results in self.d['dd_backtest'].
    """
    if models is None:
        models = ["var"]

    all_results = []
    for m_type in models:
        log.info("Backtesting DD using %s model...", m_type)
        for offset in range(n_months, 0, -1):
            predict_dd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
            macro_fc = predict_macro_factors_fn(self, horizon=1, training_offset=offset, model_type=m_type)
            target_date = macro_fc.index[-1]
            res = self.d["dd_forecast"].copy().reset_index()
            res["date"] = target_date
            all_results.append(res)

    self.d["dd_backtest"] = pd.concat(all_results, ignore_index=True)
    return self


# ---------------------------------------------------------------------------
# Macro forecast visualization
# ---------------------------------------------------------------------------


def forecast_macro_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    models: Union[str, list] = "var",
    target_col: str = "DD",
) -> "Portfolio":
    """Computes macro and DD/PD forecasts, stores results in self.d['macro_forecast_data'].

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon (negative = walk-forward backtest).
        training_offset: Offset for backtesting.
        models: Model type(s) to compare.
        target_col: 'DD' or 'PD'.

    Returns:
        Portfolio: self with self.d['macro_forecast_data'] populated.
    """
    if isinstance(models, str):
        models = [models]

    is_backtest = horizon < 0
    if is_backtest:
        n_months = abs(horizon)
        log.info("Computing Expanding Window backtest for last %d months", n_months)
    else:
        log.info("Computing Multi-step forecast for %d months", horizon)

    port_target = self.d["portfolio"].groupby("date")[target_col].mean().reset_index()
    macro_df = (
        self.d["portfolio"][["date"] + cfg.MACRO_COLS]
        .drop_duplicates("date")
        .merge(port_target, on="date", how="left")
        .set_index("date")
        .resample("ME")
        .mean()
        .dropna()
    )
    macro_df.index = macro_df.index.normalize() + pd.offsets.MonthEnd(0)

    forecast_dfs = {}
    for m_type in models:
        if is_backtest:
            step_fcs = []
            for offset in range(n_months, 0, -1):
                fc_step = predict_macro_factors_fn(self, horizon=1, training_offset=offset, model_type=m_type)
                if target_col == "PD":
                    predict_pd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
                    fc_step["PD"] = self.d["pd_forecast"]["predicted_pd"].mean()
                else:
                    predict_dd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
                    fc_step["DD"] = self.d["dd_forecast"]["predicted_dd"].mean()
                step_fcs.append(fc_step)
            forecast_dfs[m_type] = pd.concat(step_fcs)
        else:
            fc_df = predict_macro_factors_fn(self, horizon=horizon, training_offset=training_offset, model_type=m_type)
            trajectory = []
            for i in range(1, horizon + 1):
                if target_col == "PD":
                    predict_pd_fn(self, horizon=i, training_offset=training_offset, model_type=m_type)
                    trajectory.append(self.d["pd_forecast"]["predicted_pd"].mean())
                else:
                    predict_dd_fn(self, horizon=i, training_offset=training_offset, model_type=m_type)
                    trajectory.append(self.d["dd_forecast"]["predicted_dd"].mean())
            fc_df[target_col] = trajectory
            forecast_dfs[m_type] = fc_df

    self.d["macro_forecast_data"] = {"macro_df": macro_df, "forecast_dfs": forecast_dfs}
    log.info("Macro forecast data computed and stored.")
    return self


def plot_macro_forecast_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    models: Union[str, list] = "var",
    tail: int = 12,
    target_col: str = "DD",
    verbose: bool = False,
    figsize: tuple = (12, 14),
) -> "Portfolio":
    """Plots historical and forecasted macro and portfolio DD/PD.

    If self.d['macro_forecast_data'] is not populated, computes forecasts first.

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon (negative = walk-forward backtest).
        training_offset: Offset for backtesting.
        models: Model type(s) to compare.
        tail: History months to show in plots.
        target_col: 'DD' or 'PD'.
        verbose: Whether to display the plot.
        figsize: Figure size.

    Returns:
        Portfolio: self.
    """
    if "macro_forecast_data" not in self.d:
        forecast_macro_fn(self, horizon=horizon, training_offset=training_offset, models=models, target_col=target_col)

    macro_df = self.d["macro_forecast_data"]["macro_df"]
    forecast_dfs = self.d["macro_forecast_data"]["forecast_dfs"]

    plots.plot_macro_forecast(macro_df, forecast_dfs, tail=tail, figsize=figsize, verbose=verbose)
    return self


# ---------------------------------------------------------------------------
# OLS macro significance analysis
# ---------------------------------------------------------------------------


def analyze_macro_dd_significance_fn(
    self: "Portfolio",
    verbose: bool = True,
) -> "Portfolio":
    """Analyzes statistical significance of macro variables on DD.

    Runs Panel OLS with entity Fixed Effects and clustered standard errors.

    Args:
        self: Portfolio instance.
        verbose: If True, prints detailed results.

    Returns:
        Portfolio: self with results in self.d['macro_significance'].
    """
    from linearmodels.panel import PanelOLS

    panel = self.d["portfolio"][["date", "ticker", "DD"] + cfg.MACRO_COLS].dropna().copy()
    panel["month"] = panel["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        panel.groupby(["ticker", "month"])
        .agg(DD=("DD", "mean"), **{col: (col, "mean") for col in cfg.MACRO_COLS})
        .reset_index()
    )

    monthly = monthly.set_index(["ticker", "month"])
    panel_model = PanelOLS(
        monthly["DD"],
        sm.add_constant(monthly[cfg.MACRO_COLS]),
        entity_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    ols_summary = {
        "r_squared": panel_model.rsquared,
        "r_squared_within": panel_model.rsquared_within,
        "f_statistic": panel_model.f_statistic.stat,
        "f_pvalue": panel_model.f_statistic.pval,
        "n_obs": int(panel_model.nobs),
        "n_entities": int(panel_model.entity_info.total),
        "coefficients": {},
    }
    for var in ["const"] + cfg.MACRO_COLS:
        ols_summary["coefficients"][var] = {
            "coef": panel_model.params[var],
            "std_err": panel_model.std_errors[var],
            "t_stat": panel_model.tstats[var],
            "p_value": panel_model.pvalues[var],
            "significant": panel_model.pvalues[var] < 0.05,
        }

    self.d["macro_significance"] = {
        "ols": ols_summary,
        "panel_model": panel_model,
    }

    if verbose:
        log.info("=" * 80)
        log.info("PANEL OLS (Entity FE, Clustered SE): DD ~ macro variables")
        log.info("=" * 80)
        log.info(str(panel_model.summary.tables[0]))

        coef_data = []
        for var in ["const"] + cfg.MACRO_COLS:
            c = ols_summary["coefficients"][var]
            sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else ""
            coef_data.append(
                [var, f"{c['coef']:.4f}", f"{c['std_err']:.4f}", f"{c['t_stat']:.2f}", f"{c['p_value']:.4f}", sig]
            )
        coef_df = pd.DataFrame(coef_data, columns=["Variable", "Coef", "Std Err", "t-stat", "p-value", "Sig"])
        log.info("\n" + coef_df.to_string(index=False))
        log.info(
            "R-squared: %.4f, R-squared (within): %.4f",
            ols_summary["r_squared"],
            ols_summary["r_squared_within"],
        )
        log.info("F-statistic: %.2f, p-value: %.2e", ols_summary["f_statistic"], ols_summary["f_pvalue"])
        log.info("N observations: %d, N entities: %d", ols_summary["n_obs"], ols_summary["n_entities"])

    log.info("Macro-DD significance analysis completed.")
    return self


# ---------------------------------------------------------------------------
# Model quality comparison
# ---------------------------------------------------------------------------


def compare_macro_models_fn(
    self: "Portfolio",
    n_months: int = 12,
    models: List[str] = None,
    target_col: str = "DD",
    verbose: bool = False,
) -> "Portfolio":
    """Computes MAE/RMSE/MAPE for macro models via walk-forward backtest.

    Args:
        self: Portfolio instance.
        n_months: Number of months in the backtest window.
        models: List of model types. Defaults to ['var', 'sarimax', 'prophet'].
        target_col: Portfolio target metric ('DD' or 'PD').
        verbose: Whether to display the comparison table.

    Returns:
        Portfolio: self with self.d['macro_model_comparison'] populated.
    """
    if models is None:
        models = ["var", "sarimax", "prophet"]

    all_cols = cfg.MACRO_COLS + [target_col]

    port_target = self.d["portfolio"].groupby("date")[target_col].mean().reset_index()
    macro_df = (
        self.d["portfolio"][["date"] + cfg.MACRO_COLS]
        .drop_duplicates("date")
        .merge(port_target, on="date", how="left")
        .set_index("date")
        .resample("ME")
        .mean()
        .dropna()
    )
    macro_df.index = macro_df.index.normalize() + pd.offsets.MonthEnd(0)

    rows = []
    for m_type in models:
        errors = {col: [] for col in all_cols}
        for offset in range(n_months, 0, -1):
            fc = predict_macro_factors_fn(self, horizon=1, training_offset=offset, model_type=m_type)
            target_date = fc.index[-1]

            if target_col == "DD":
                predict_dd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
                pred_target = self.d["dd_forecast"]["predicted_dd"].mean()
            else:
                predict_pd_fn(self, horizon=1, training_offset=offset, model_type=m_type)
                pred_target = self.d["pd_forecast"]["predicted_pd"].mean()

            if target_date not in macro_df.index:
                continue

            actual = macro_df.loc[target_date]
            for col in cfg.MACRO_COLS:
                if col in fc.columns:
                    errors[col].append(fc[col].values[0] - actual[col])
            errors[target_col].append(pred_target - actual[target_col])

        for col in all_cols:
            errs = np.array(errors[col])
            if len(errs) == 0:
                continue
            actuals_abs = np.abs(macro_df[col].tail(n_months).values[: len(errs)])
            mape_vals = np.abs(errs) / np.where(actuals_abs > 1e-8, actuals_abs, 1e-8)
            rows.append(
                {
                    "Model": m_type,
                    "Variable": col,
                    "MAE": round(np.mean(np.abs(errs)), 6),
                    "RMSE": round(np.sqrt(np.mean(errs**2)), 6),
                    "MAPE (%)": round(np.mean(mape_vals) * 100, 2),
                }
            )

    comparison_df = pd.DataFrame(rows)
    self.d["macro_model_comparison"] = comparison_df
    log.log_dataframe(comparison_df, title="Macro Model Comparison (Walk-Forward)")

    if verbose:
        plots.plot_macro_model_comparison(comparison_df, verbose=verbose)

    return self


# ---------------------------------------------------------------------------
# Thin plot wrappers
# ---------------------------------------------------------------------------


def plot_pd_forecast_fn(
    self: "Portfolio",
    figsize: tuple = (12, 6),
    verbose: bool = False,
) -> "Portfolio":
    """Plots predicted vs reference PD for each ticker.

    Args:
        self: Portfolio instance.
        figsize: Figure size.
        verbose: Whether to show the plot.

    Returns:
        Portfolio: self.
    """
    if "pd_forecast" not in self.d:
        log.error("No PD forecast found. Run predict_pd() first.")
        return self
    plots.plot_pd_forecast(self.d["pd_forecast"], figsize=figsize, verbose=verbose)
    return self


def plot_dd_forecast_fn(
    self: "Portfolio",
    figsize: tuple = (12, 6),
    verbose: bool = False,
) -> "Portfolio":
    """Plots predicted vs reference DD for each ticker.

    Args:
        self: Portfolio instance.
        figsize: Figure size.
        verbose: Whether to show the plot.

    Returns:
        Portfolio: self.
    """
    if "dd_forecast" not in self.d:
        log.error("No DD forecast found. Run predict_dd() first.")
        return self
    plots.plot_dd_forecast(self.d["dd_forecast"], figsize=figsize, verbose=verbose)
    return self


# ---------------------------------------------------------------------------
# Impulse response function (VAR analysis)
# ---------------------------------------------------------------------------


def calc_irf_fn(
    self: "Portfolio",
    impulses_responses: dict,
    figsize: tuple = (10, 4),
    verbose: bool = False,
) -> "Portfolio":
    """Fits a VAR model on portfolio data and plots impulse response functions.

    Performs stationarity checks (ADF test), applies first-differencing if
    needed, selects the optimal lag order via AIC, and delegates plotting
    to ``plots.plot_irf``.

    Args:
        self: Portfolio instance.
        impulses_responses: Dict mapping impulse column names to response column names
            (e.g., ``{'inflation': 'DD', 'rubusd_exchange_rate': 'DD'}``).
        figsize: Figure size for each IRF plot.
        verbose: Whether to display the plot interactively.

    Returns:
        Portfolio: self.
    """
    if impulses_responses is None:
        raise ValueError("impulses_responses must be specified")

    portfolio_df = self.d["portfolio"]
    columns = np.unique(list(impulses_responses.keys()) + list(impulses_responses.values()))

    if "date" in portfolio_df.columns:
        data = portfolio_df.groupby("date")[columns].mean().sort_index().dropna()
    else:
        log.warning("'date' column not found. Using raw data for VAR.")
        data = portfolio_df.sort_values(["ticker", "date"])[columns].dropna()[columns]

    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        log.error("Constant columns detected: %s. VAR requires time-varying data.", constant_cols)
        data = data.drop(columns=constant_cols)
        if data.empty or len(data.columns) < 2:
            log.error("Not enough variables left for VAR analysis.")
            return self

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
            return self

        pvalues = {col: adfuller(data[col].dropna())[1] for col in data.columns}
        log.info("p-values after differencing:\n%s", pd.Series(pvalues))

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.RangeIndex(start=0, stop=len(data))

    model = VAR(data)
    lag_order = model.select_order(maxlags=6)
    selected_lags = lag_order.aic
    log.info("Optimal lag number | AIC lags: %d", selected_lags)
    results = model.fit(maxlags=selected_lags, ic="aic")

    plots.plot_irf(results, impulses_responses, selected_lags, figsize, verbose)
    return self
