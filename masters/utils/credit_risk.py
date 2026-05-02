"""Credit risk forecasting functions for the Portfolio class.

Covers macro factor prediction, PD/DD forecasting, walk-forward backtests,
OLS macro significance analysis, and model comparison.

All public functions accept a ``Portfolio`` instance as the first argument and
mutate ``self.d`` in place, returning ``self`` for method chaining.
"""

from __future__ import annotations

import time
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

    # Instrumentation: count macro-model fits during a forecast_macro_fn run.
    # The counter dict is created/cleared by forecast_macro_fn; outside of
    # that scope this is a no-op.
    fit_counter = self.d.get("_macro_fit_counter")
    if fit_counter is not None:
        key = model_type.lower()
        fit_counter[key] = fit_counter.get(key, 0) + 1
    fit_t0 = time.perf_counter()

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

    result = pd.DataFrame()
    if model_type.lower() == "var":
        # Lag order is selected automatically by minimizing BIC over a
        # candidate range. The upper bound is bounded by sample length to
        # keep the VAR identifiable. Falling back to lag=1 if BIC fails to
        # pick a positive order (degenerate short series).
        max_lag_cap = max(1, min(12, len(macro_df) // 4))
        var_model = VAR(macro_df)
        order_sel = var_model.select_order(maxlags=max_lag_cap)
        chosen_lag = max(1, int(order_sel.bic))
        results = var_model.fit(chosen_lag)
        fc = results.forecast(y=macro_df.values[-results.k_ar :], steps=horizon)
        result = pd.DataFrame(fc, index=future_dates, columns=macro_df.columns)
        log.info(
            "VAR fit | chosen_lag=%d (BIC) | maxlags_cap=%d | n_obs=%d",
            chosen_lag,
            max_lag_cap,
            len(macro_df),
        )

    elif model_type.lower() in ("sarimax", "prophet"):
        # Dynamic exogenous projection: instead of repeating the last
        # observation horizon times (which contradicts the multi-factor
        # forecast idea), forecast the entire macro panel via a small
        # auxiliary VAR(BIC) once, and slice the appropriate columns as
        # exog for each per-target univariate fit. Used by both SARIMAX
        # (.forecast(exog=...)) and Prophet (.add_regressor(...)).
        aux_max_lag = max(1, min(12, len(macro_df) // 4))
        aux_var = VAR(macro_df)
        aux_lag = max(1, int(aux_var.select_order(maxlags=aux_max_lag).bic))
        aux_results = aux_var.fit(aux_lag)
        aux_fc = aux_results.forecast(y=macro_df.values[-aux_results.k_ar :], steps=horizon)
        exog_forecast_panel = pd.DataFrame(aux_fc, index=future_dates, columns=macro_df.columns)
        log.info(
            "%s exog projection | aux VAR lag=%d (BIC) | maxlags_cap=%d",
            model_type.upper(),
            aux_lag,
            aux_max_lag,
        )

        if model_type.lower() == "sarimax":
            forecast_results = {}
            # Small AIC grid for non-seasonal ARIMA(p, d, q) with exogenous
            # regressors. p,q in {0,1,2} excluding (0,0). Sample size (~75
            # months) is too short to reliably identify a seasonal
            # component, so seasonal_order is left at default.
            order_grid = [(p, q) for p in range(3) for q in range(3) if (p, q) != (0, 0)]
            chosen_orders = {}

            for col in macro_df.columns:
                d = 1 if adfuller(macro_df[col])[1] > 0.05 else 0
                exog_train = macro_df.drop(columns=[col]).shift(1).bfill()

                # Scaling is crucial: macro variables have different magnitudes.
                sy, sx = StandardScaler(), StandardScaler()
                y_scaled = sy.fit_transform(macro_df[[col]])
                x_scaled = sx.fit_transform(exog_train)

                best_aic = np.inf
                best_results = None
                best_pq = (1, 0)
                for p, q in order_grid:
                    candidate = SARIMAX(
                        y_scaled,
                        exog=x_scaled,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=500)
                    if np.isfinite(candidate.aic) and candidate.aic < best_aic:
                        best_aic = candidate.aic
                        best_results = candidate
                        best_pq = (p, q)
                chosen_orders[col] = (best_pq[0], d, best_pq[1])

                exog_fc = exog_forecast_panel.drop(columns=[col])[exog_train.columns]
                fc_scaled = best_results.forecast(steps=horizon, exog=sx.transform(exog_fc))
                forecast_results[col] = sy.inverse_transform(fc_scaled.reshape(-1, 1)).flatten()
            log.info("SARIMAX orders chosen by AIC: %s", chosen_orders)
            result = pd.DataFrame(forecast_results, index=future_dates)
        else:  # prophet
            # For each target macrofactor, the remaining factors are used
            # as Prophet additional regressors. Future regressor values
            # come from the auxiliary VAR forecast above (instead of
            # being naively held constant).
            forecast_results = {}
            for col in macro_df.columns:
                regressor_cols = [c for c in macro_df.columns if c != col]
                train_df = macro_df.reset_index().rename(columns={"date": "ds", col: "y"})
                train_df = train_df[["ds", "y"] + regressor_cols]

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                )
                for reg in regressor_cols:
                    m.add_regressor(reg)
                m.fit(train_df)

                future = m.make_future_dataframe(periods=horizon, freq="ME")
                # Fill historical regressor values from training data and
                # forecasted values from the aux VAR for the horizon tail.
                hist_regs = macro_df[regressor_cols].reset_index(drop=True)
                fut_regs = exog_forecast_panel[regressor_cols].reset_index(drop=True)
                future = future.reset_index(drop=True)
                combined_regs = pd.concat([hist_regs, fut_regs], ignore_index=True)
                for reg in regressor_cols:
                    future[reg] = combined_regs[reg].values
                forecast_results[col] = m.predict(future)["yhat"].tail(horizon).values
            result = pd.DataFrame(forecast_results, index=future_dates)

    # Instrumentation: accumulate fit time per model when invoked from
    # forecast_macro_fn (which sets up self.d['_macro_fit_time']).
    fit_time_acc = self.d.get("_macro_fit_time")
    if fit_time_acc is not None:
        key = model_type.lower()
        fit_time_acc[key] = fit_time_acc.get(key, 0.0) + (time.perf_counter() - fit_t0)
    return result


# ---------------------------------------------------------------------------
# PD / DD prediction
# ---------------------------------------------------------------------------


def _predict_target_fn(
    self: "Portfolio",
    target: str,
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
    macro_forecast: pd.DataFrame = None,
) -> "Portfolio":
    """Internal OLS-based forecaster for PD or DD.

    Args:
        self: Portfolio instance.
        target: 'PD' or 'DD'.
        horizon: Forecasting horizon in months.
        training_offset: Months of history to hide for backtesting.
        model_type: Macro model type ('var', 'sarimax', 'prophet').
        macro_forecast: Optional pre-computed macro factor forecast.

    Returns:
        Portfolio: self with results in self.d['<target>_forecast'] and
            full per-ticker trajectory in self.d['<target>_forecast_path'].
    """
    if target not in {"PD", "DD"}:
        raise ValueError(f"target must be 'PD' or 'DD', got {target!r}")
    is_pd = target == "PD"
    key_lc = target.lower()
    pred_col = f"predicted_{key_lc}"
    ref_col = f"reference_{key_lc}"

    if horizon < 0:
        training_offset = abs(horizon)
        horizon = abs(horizon)

    if macro_forecast is None:
        macro_forecast = predict_macro_factors_fn(
            self, horizon=horizon, training_offset=training_offset, model_type=model_type
        )

    target_pivot = self.d["portfolio"][["date", "ticker", target]].pivot(index="date", columns="ticker", values=target)
    target_monthly = target_pivot.resample("ME").last()
    target_monthly.index = target_monthly.index.normalize() + pd.offsets.MonthEnd(0)

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

    combined = pd.concat([target_monthly, macro_hist], axis=1).dropna()
    train = combined.iloc[:-training_offset] if training_offset > 0 else combined

    predictions = {}
    paths = {}
    for ticker in target_monthly.columns:
        if ticker not in train.columns:
            continue
        y = train[ticker]
        x = sm.add_constant(train[macro_cols])
        model = sm.OLS(y, x).fit()
        x_pred = sm.add_constant(macro_forecast, has_constant="add")
        if "const" not in x_pred.columns:
            x_pred["const"] = 1.0
        full_path = model.predict(x_pred)
        paths[ticker] = full_path.values
        predictions[ticker] = full_path.iloc[-1]

    result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=[pred_col])
    result_df.index.name = "ticker"
    target_date = macro_forecast.index[-1]
    comparison = (
        target_monthly.loc[target_date] if target_date in target_monthly.index else target_pivot.ffill().iloc[-1]
    )
    result_df[ref_col] = comparison
    result_df["delta"] = result_df[pred_col] - result_df[ref_col]
    result_df["model"] = model_type

    path_df = pd.DataFrame(paths, index=macro_forecast.index)
    if is_pd:
        # PD must lie in [0, 1] — clip both scalar prediction and full
        # trajectory consistently.
        result_df[pred_col] = result_df[pred_col].clip(0.0, 1.0)
        path_df = path_df.clip(0.0, 1.0)
    else:
        # Derive PD from DD via the standard normal CDF; norm.cdf already
        # returns values in [0, 1] so no clip is needed.
        result_df["predicted_pd"] = norm.cdf(-result_df[pred_col].astype(float))
        result_df["reference_pd"] = norm.cdf(-result_df[ref_col].astype(float))

    self.d[f"{key_lc}_forecast"] = result_df
    self.d[f"{key_lc}_forecast_path"] = path_df
    return self


def predict_pd_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
    macro_forecast: pd.DataFrame = None,
) -> "Portfolio":
    """Predicts PD for portfolio assets based on macro OLS model.

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon in months.
        training_offset: Months of history to hide for backtesting.
        model_type: Macro model type ('var', 'sarimax', 'prophet').
        macro_forecast: Optional pre-computed macro factor forecast.

    Returns:
        Portfolio: self with results in self.d['pd_forecast'] and full
            trajectory in self.d['pd_forecast_path'].
    """
    return _predict_target_fn(
        self,
        target="PD",
        horizon=horizon,
        training_offset=training_offset,
        model_type=model_type,
        macro_forecast=macro_forecast,
    )


def predict_dd_fn(
    self: "Portfolio",
    horizon: int = 1,
    training_offset: int = 0,
    model_type: str = "var",
    macro_forecast: pd.DataFrame = None,
) -> "Portfolio":
    """Predicts DD for portfolio assets based on macro OLS model.

    Args:
        self: Portfolio instance.
        horizon: Forecasting horizon in months.
        training_offset: Months of history to hide for backtesting.
        model_type: Macro model type ('var', 'sarimax', 'prophet').
        macro_forecast: Optional pre-computed macro factor forecast.

    Returns:
        Portfolio: self with results in self.d['dd_forecast'] and full
            trajectory in self.d['dd_forecast_path'].
    """
    return _predict_target_fn(
        self,
        target="DD",
        horizon=horizon,
        training_offset=training_offset,
        model_type=model_type,
        macro_forecast=macro_forecast,
    )


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

    # Instrumentation: reset fit counters/timers so that nested
    # predict_macro_factors_fn calls feed into them. Aggregated and logged
    # at the end of this function for before/after benchmarking.
    self.d["_macro_fit_counter"] = {}
    self.d["_macro_fit_time"] = {}
    overall_t0 = time.perf_counter()

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
                    predict_pd_fn(self, horizon=1, training_offset=offset, model_type=m_type, macro_forecast=fc_step)
                    fc_step["PD"] = self.d["pd_forecast"]["predicted_pd"].mean()
                else:
                    predict_dd_fn(self, horizon=1, training_offset=offset, model_type=m_type, macro_forecast=fc_step)
                    fc_step["DD"] = self.d["dd_forecast"]["predicted_dd"].mean()
                step_fcs.append(fc_step)
            forecast_dfs[m_type] = pd.concat(step_fcs)
        else:
            fc_df = predict_macro_factors_fn(self, horizon=horizon, training_offset=training_offset, model_type=m_type)
            # Single OLS pass over the full H-step macro forecast yields
            # the entire trajectory at once via predict_*_fn's *_path
            # output. Avoids the H-times redundant OLS refit of the
            # previous design.
            if target_col == "PD":
                predict_pd_fn(
                    self,
                    horizon=horizon,
                    training_offset=training_offset,
                    model_type=m_type,
                    macro_forecast=fc_df,
                )
                trajectory = self.d["pd_forecast_path"].mean(axis=1).values
            else:
                predict_dd_fn(
                    self,
                    horizon=horizon,
                    training_offset=training_offset,
                    model_type=m_type,
                    macro_forecast=fc_df,
                )
                trajectory = self.d["dd_forecast_path"].mean(axis=1).values
            fc_df[target_col] = trajectory
            forecast_dfs[m_type] = fc_df

    self.d["macro_forecast_data"] = {"macro_df": macro_df, "forecast_dfs": forecast_dfs}

    # Instrumentation summary: total wall time, number of macro fits per
    # model, cumulative fit time per model, and per-model trajectory stats
    # for the forecast target (DD / PD). Logged for before/after comparison.
    elapsed = time.perf_counter() - overall_t0
    fit_counter = self.d.pop("_macro_fit_counter", {})
    fit_time = self.d.pop("_macro_fit_time", {})

    perf_rows = []
    for m_type in models:
        key = m_type.lower()
        traj = forecast_dfs[m_type][target_col].astype(float)
        perf_rows.append(
            {
                "model": m_type,
                "macro_fits": fit_counter.get(key, 0),
                "fit_time_s": round(fit_time.get(key, 0.0), 3),
                f"{target_col}_mean": round(traj.mean(), 4),
                f"{target_col}_std": round(traj.std(ddof=0), 4),
                f"{target_col}_min": round(traj.min(), 4),
                f"{target_col}_max": round(traj.max(), 4),
                f"{target_col}_last": round(traj.iloc[-1], 4),
            }
        )
    perf_df = pd.DataFrame(perf_rows)
    log.info(
        "forecast_macro_fn finished | elapsed=%.2fs | horizon=%d | target=%s | backtest=%s | models=%s",
        elapsed,
        abs(horizon),
        target_col,
        is_backtest,
        list(models),
    )
    log.log_dataframe(perf_df, title="forecast_macro performance & forecast summary")

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

    Runs Panel OLS with entity Fixed Effects and two-way clustered
    standard errors (entity + time). Time clustering is essential because
    macro regressors are constant across tickers within each period, so
    residuals are strongly cross-sectionally correlated and entity-only
    clustering would understate standard errors.

    Also reports the macro correlation matrix and Variance Inflation
    Factors to flag multicollinearity, which often explains a low within
    R-squared in this kind of panel.

    Args:
        self: Portfolio instance.
        verbose: If True, prints detailed results.

    Returns:
        Portfolio: self with results in self.d['macro_significance'].
    """
    from linearmodels.panel import PanelOLS
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # End-of-month DD per ticker — same convention as predict_dd_fn
    # (last observation of the month) to keep the significance analysis
    # consistent with the forecasting pipeline.
    df = self.d["portfolio"][["date", "ticker", "DD"] + cfg.MACRO_COLS].dropna().copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()
    dd_monthly = df.groupby(["ticker", "month"])["DD"].last()
    macro_monthly = df.groupby("month")[cfg.MACRO_COLS].last()
    monthly = dd_monthly.to_frame().join(macro_monthly, on="month").dropna()

    # Entity FE absorbs the constant; do NOT add one explicitly to avoid
    # a perfect-collinearity warning and keep coefficient indexing clean.
    panel_model = PanelOLS(
        monthly["DD"],
        monthly[cfg.MACRO_COLS],
        entity_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    ols_summary = {
        "r_squared": panel_model.rsquared,
        "r_squared_within": panel_model.rsquared_within,
        "f_statistic": panel_model.f_statistic.stat,
        "f_pvalue": panel_model.f_statistic.pval,
        "n_obs": int(panel_model.nobs),
        "n_entities": int(panel_model.entity_info.total),
        "coefficients": {},
    }
    for var in cfg.MACRO_COLS:
        ols_summary["coefficients"][var] = {
            "coef": panel_model.params[var],
            "std_err": panel_model.std_errors[var],
            "t_stat": panel_model.tstats[var],
            "p_value": panel_model.pvalues[var],
            "significant": panel_model.pvalues[var] < 0.05,
        }

    # Multicollinearity diagnostics on the time-varying macro panel.
    macro_design = macro_monthly.dropna()
    corr_matrix = macro_design.corr()
    vif_design = sm.add_constant(macro_design)
    vif = pd.Series(
        {
            col: variance_inflation_factor(vif_design.values, i)
            for i, col in enumerate(vif_design.columns)
            if col != "const"
        },
        name="VIF",
    )

    self.d["macro_significance"] = {
        "ols": ols_summary,
        "panel_model": panel_model,
        "macro_corr": corr_matrix,
        "macro_vif": vif,
    }

    if verbose:
        log.info("=" * 80)
        log.info("PANEL OLS (Entity FE, 2-way Clustered SE entity+time): DD ~ macro variables")
        log.info("=" * 80)
        log.info(
            "N obs=%d | entities=%d | time periods=%d | R2=%.4f (within=%.4f) | F=%.2f (p=%.2e)",
            ols_summary["n_obs"],
            ols_summary["n_entities"],
            int(panel_model.time_info.total),
            ols_summary["r_squared"],
            ols_summary["r_squared_within"],
            ols_summary["f_statistic"],
            ols_summary["f_pvalue"],
        )

        coef_data = []
        for var in cfg.MACRO_COLS:
            c = ols_summary["coefficients"][var]
            sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else ""
            coef_data.append(
                [var, f"{c['coef']:.4f}", f"{c['std_err']:.4f}", f"{c['t_stat']:.2f}", f"{c['p_value']:.4f}", sig]
            )
        coef_df = pd.DataFrame(coef_data, columns=["Variable", "Coef", "Std Err", "t-stat", "p-value", "Sig"])
        log.info("\n" + coef_df.to_string(index=False))

        log.info("Macro correlation matrix:\n" + corr_matrix.round(3).to_string())
        log.info("Macro VIF (>10 indicates strong multicollinearity):\n" + vif.round(2).to_string())

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
