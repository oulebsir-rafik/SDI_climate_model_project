"""Model factory for the intake forecast's per-parameter XGBoost regressors.

See specs/intake-forecast-model-and-tuning.md for the design this implements.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# A reasonable, roughly-central point of the search space in
# specs/intake-forecast-model-and-tuning.md, used as a neutral stand-in wherever a parameter's
# model is needed but isn't the one currently being tuned (see intake_forecast.tuning). Capped
# at 150 to match SEARCH_SPACE["n_estimators"]'s upper bound in tuning.py — both paths now rely
# on early stopping (below) as the real regularizer, so their ceilings stay aligned.
DEFAULT_XGB_PARAMS: dict[str, float | int] = {
    "max_depth": 4,
    "n_estimators": 150,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Early-stopping settings for fit_model's internal validation split. EVAL_SET_FRACTION is the
# fraction of each call's usable (non-NaN-target) rows held back, chronologically, as eval_set;
# MIN_EVAL_ROWS is a floor below which fit_model raises rather than fit on a validation slice
# too small to give a reliable stopping signal.
EVAL_SET_FRACTION = 0.15
MIN_EVAL_ROWS = 13
EARLY_STOPPING_ROUNDS = 20
EARLY_STOPPING_METRIC = "rmse"


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "Model": ...

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
    params.setdefault("eval_metric", EARLY_STOPPING_METRIC)
    params.setdefault("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
    return XGBRegressor(**params)


def fit_model(model: Model, X: pd.DataFrame, y: pd.Series) -> Model:
    """Fit `model` with early stopping against an internal validation split, dropping rows
    with a NaN target first. NaN feature values are passed through untouched — XGBRegressor
    (and any model honoring this contract) handles missing feature values natively, so no
    imputation happens here.

    The validation split is a chronological tail of this call's own training data only — never
    the fold's test block or the final holdout, which would leak the reported metric into model
    fitting. The last `EVAL_SET_FRACTION` of valid (non-NaN-target) rows, sorted by date, become
    `eval_set`; the rest are the actual fit rows. `n_eval` is rounded up (`ceil`) so
    "`EVAL_SET_FRACTION`" means "at least that fraction" rather than failing windows that would
    otherwise land just under `MIN_EVAL_ROWS` by a fraction of a row. Raises if even that isn't
    enough to clear `MIN_EVAL_ROWS` — a window too small for a reliable early-stopping signal is
    a data problem to surface, not paper over with a degenerate split.
    """
    valid = y.notna()
    X_valid = X.loc[valid].sort_index()
    y_valid = y.loc[valid].sort_index()

    n = len(y_valid)
    n_eval = math.ceil(n * EVAL_SET_FRACTION)
    if n_eval < MIN_EVAL_ROWS:
        raise ValueError(
            f"Only {n} usable training rows available; a {EVAL_SET_FRACTION:.0%} eval split "
            f"gives {n_eval} rows, below the required minimum of {MIN_EVAL_ROWS} for a "
            "reliable early-stopping signal."
        )

    X_train, X_eval = X_valid.iloc[:-n_eval], X_valid.iloc[-n_eval:]
    y_train, y_eval = y_valid.iloc[:-n_eval], y_valid.iloc[-n_eval:]

    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return model
