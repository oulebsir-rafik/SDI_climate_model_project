"""Shared metric implementations for the intake forecast.

See specs/intake-forecast-cv-and-metrics.md for the design this implements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def naive_insample_mae(y_train_insample: pd.Series) -> float:
    """The MASE scaling denominator: mean absolute naive lag-1 difference, computed only from
    a fold's own training window for one parameter (Hyndman & Koehler's original definition) —
    never a global or cross-fold value.
    """
    diffs = y_train_insample.diff().abs().dropna()
    return float(diffs.mean())


def mase(y_true, y_pred, y_train_insample: pd.Series) -> float:
    numerator = mean_absolute_error(y_true, y_pred)
    denominator = naive_insample_mae(y_train_insample)
    return float(numerator / denominator)


_METRIC_FUNCS = {"r2": r2, "mae": mae, "mse": mse, "smape": smape}


def _score_group(group: pd.DataFrame, insample: pd.Series) -> dict:
    y_true, y_pred = group["actual"], group["predicted"]
    scores = {name: func(y_true, y_pred) for name, func in _METRIC_FUNCS.items()}
    scores["mase"] = mase(y_true, y_pred, insample)
    scores["n"] = len(group)
    return scores


def aggregate_metrics(predictions_df: pd.DataFrame, train_insample: dict[str, pd.Series]) -> pd.DataFrame:
    """Given raw (origin, horizon, parameter, predicted, actual) predictions, drop samples
    whose actual value is NaN (excluded entirely, never imputed), then compute
    R²/MAE/MSE/sMAPE/MASE per (parameter, horizon), plus an "overall" row per parameter pooling
    every horizon together.
    """
    scored = predictions_df.dropna(subset=["actual"])

    rows: list[dict] = []
    for parameter, param_df in scored.groupby("parameter"):
        insample = train_insample[parameter]
        for horizon, group in param_df.groupby("horizon"):
            rows.append({"parameter": parameter, "horizon": horizon, **_score_group(group, insample)})
        rows.append({"parameter": parameter, "horizon": "overall", **_score_group(param_df, insample)})

    return pd.DataFrame(rows)
