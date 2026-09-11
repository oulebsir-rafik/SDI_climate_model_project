"""Model factory for the intake forecast's per-parameter XGBoost regressors.

See specs/intake-forecast-model-and-tuning.md for the design this implements.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# A reasonable, roughly-central point of the search space in
# specs/intake-forecast-model-and-tuning.md, used as a neutral stand-in wherever a parameter's
# model is needed but isn't the one currently being tuned (see intake_forecast.tuning).
DEFAULT_XGB_PARAMS: dict[str, float | int] = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# XGBoost's own default (n_jobs=-1, all cores) is measurably *slower* than a small fixed thread
# count on this project's training-set sizes (~100-300 rows/fold): spawning/synchronizing 20
# threads for that little data costs more than it saves. Empirically the fastest on this
# machine (~5x over n_jobs=-1); not a hyperparameter, so it's fixed here rather than tuned —
# see diary/2026-09-09-per-parameter-feature-selection.md.
_DEFAULT_N_JOBS = 6


def make_xgb_model(params: dict | None = None) -> XGBRegressor:
    params = dict(params or DEFAULT_XGB_PARAMS)
    params.setdefault("n_jobs", _DEFAULT_N_JOBS)
    return XGBRegressor(**params)


def fit_model(model: Model, X: pd.DataFrame, y: pd.Series) -> Model:
    """Fit `model`, dropping rows with a NaN target first. NaN feature values are passed
    through untouched — XGBRegressor (and any model honoring this contract) handles missing
    feature values natively, so no imputation happens here.
    """
    valid = y.notna()
    model.fit(X.loc[valid], y.loc[valid])
    return model
