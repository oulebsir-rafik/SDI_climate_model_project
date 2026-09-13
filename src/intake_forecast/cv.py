"""Walk-forward cross-validation and recursive multi-horizon scoring for the intake forecast.

See specs/intake-forecast-cv-and-metrics.md for the design this implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import BC_PARAMETERS, FeatureConfig, build_feature_table, build_features, select_feature_columns, select_xy
from .model import Model, fit_model

# Given a parameter name, return a fresh (untrained) model for it. Per-parameter rather than
# a single no-arg factory so that, during hyperparameter tuning, one parameter's model can use
# trial-sampled hyperparameters while the other 3 (needed only to build cross-parameter lag
# features during the recursive rollout) use a fixed default — see intake_forecast.tuning.
ModelFactory = Callable[[str], Model]


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(
    dates: pd.DatetimeIndex,
    n_folds: int = 4,
    init_train_days: int = 120,
    test_len_days: int = 30,
    gap_days: int = 21,
    holdout_days: int = 60,
) -> tuple[list[Fold], Fold]:
    """Expanding-window walk-forward CV folds plus a separate final holdout, generic over the
    actual date range (not hardcoded dates). With this project's real calendar and these
    defaults, reproduces exactly the schedule in
    diary/2026-08-22-cv-strategy-explainer.html.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates)))
    full_start, full_end = dates[0], dates[-1]

    holdout_start = full_end - pd.Timedelta(days=holdout_days - 1)
    cv_end = holdout_start - pd.Timedelta(days=1)
    holdout = Fold(train_start=full_start, train_end=cv_end, test_start=holdout_start, test_end=full_end)

    folds: list[Fold] = []
    train_end = full_start + pd.Timedelta(days=init_train_days - 1)
    for _ in range(n_folds):
        test_start = train_end + pd.Timedelta(days=gap_days + 1)
        test_end = test_start + pd.Timedelta(days=test_len_days - 1)
        if test_end > cv_end:
            break
        folds.append(Fold(train_start=full_start, train_end=train_end, test_start=test_start, test_end=test_end))
        train_end = test_end

    return folds, holdout


def fold_train_insample(history: pd.DataFrame, fold: Fold) -> dict[str, pd.Series]:
    train_slice = history.loc[fold.train_start : fold.train_end]
    return {parameter: train_slice[parameter] for parameter in BC_PARAMETERS}


def _valid_origins(test_start: pd.Timestamp, test_end: pd.Timestamp, max_horizon: int) -> list[pd.Timestamp]:
    last_valid = test_end - pd.Timedelta(days=max_horizon)
    if last_valid < test_start:
        return []
    return list(pd.date_range(test_start, last_valid, freq="D"))


def evaluate_fold(
    fold: Fold,
    history: pd.DataFrame,
    model_factory: ModelFactory,
    feature_config: FeatureConfig,
    max_horizon: int = 7,
    predictor_config: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Train one t+1 model per parameter on `fold`'s training window, then recursively roll
    each forward across every valid origin in the test block, recording every
    (origin, horizon, parameter, predicted, actual) tuple. `actual` is left as NaN where the
    real observation is missing — callers (see intake_forecast.metrics.aggregate_metrics) drop
    those before computing metrics, never impute them.

    The 4 parameters' rollouts advance in lockstep: at each horizon step, every parameter's
    prediction is appended to a shared rollout history before building the next step's
    features, so cross-parameter lag features at h>=2 are built from each parameter's own
    recursively-predicted trajectory rather than unavailable future ground truth.

    `predictor_config` optionally maps a parameter name to `select_xy`/`select_feature_columns`
    kwargs (`predictor_parameters`, `include_calendar`), restricting that parameter's model to
    a subset of the full feature set — see `features.RECOMMENDED_PREDICTOR_CONFIG`. A parameter
    absent from the mapping (or `predictor_config=None` entirely) uses the full feature set.
    """
    train_history = history.loc[fold.train_start : fold.train_end]
    table = build_feature_table(train_history, feature_config)
    predictor_config = predictor_config or {}

    models: dict[str, Model] = {}
    for parameter in BC_PARAMETERS:
        X, y = select_xy(table, parameter, **predictor_config.get(parameter, {}))
        models[parameter] = fit_model(model_factory(parameter), X, y)

    origins = _valid_origins(fold.test_start, fold.test_end, max_horizon)

    records: list[dict] = []
    for origin in origins:
        rollout_history = history.loc[:origin, list(BC_PARAMETERS)].copy()
        for horizon in range(1, max_horizon + 1):
            as_of = rollout_history.index[-1]
            target_day = as_of + pd.Timedelta(days=1)

            feats = build_features(rollout_history, as_of, feature_config)
            predicted = {}
            for p in BC_PARAMETERS:
                cols = select_feature_columns(list(feats.index), **predictor_config.get(p, {}))
                predicted[p] = float(models[p].predict(pd.DataFrame([feats[cols]]))[0])

            if target_day in history.index:
                actual_row = history.loc[target_day]
            else:
                actual_row = pd.Series(np.nan, index=BC_PARAMETERS)

            for p in BC_PARAMETERS:
                records.append(
                    {
                        "origin": origin,
                        "horizon": horizon,
                        "parameter": p,
                        "predicted": predicted[p],
                        "actual": actual_row.get(p, np.nan),
                    }
                )

            rollout_history.loc[target_day] = predicted

    return pd.DataFrame.from_records(records, columns=["origin", "horizon", "parameter", "predicted", "actual"])


def plot_obs_vs_pred_timeseries(predictions_df: pd.DataFrame, parameter: str, horizon: int, output_path: Path) -> None:
    subset = predictions_df[
        (predictions_df["parameter"] == parameter) & (predictions_df["horizon"] == horizon)
    ].dropna(subset=["actual"]).sort_values("origin")
    target_dates = subset["origin"] + pd.to_timedelta(horizon, unit="D")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(target_dates, subset["actual"], label="observed", marker="o")
    ax.plot(target_dates, subset["predicted"], label="predicted", marker="x")
    ax.set_title(f"{parameter} — observed vs predicted (h={horizon})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_obs_vs_pred_scatter(predictions_df: pd.DataFrame, parameter: str, horizon: int, output_path: Path) -> None:
    subset = predictions_df[
        (predictions_df["parameter"] == parameter) & (predictions_df["horizon"] == horizon)
    ].dropna(subset=["actual"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(subset["actual"], subset["predicted"], alpha=0.7)
    if len(subset):
        lo = min(subset["actual"].min(), subset["predicted"].min())
        hi = max(subset["actual"].max(), subset["predicted"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_title(f"{parameter} — observed vs predicted (h={horizon})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
