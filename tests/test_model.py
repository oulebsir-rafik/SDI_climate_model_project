import math

import numpy as np
import pandas as pd
import pytest

from src.intake_forecast.model import EVAL_SET_FRACTION, MIN_EVAL_ROWS, fit_model


class RecordingStubModel:
    def __init__(self):
        self.seen_X = None
        self.seen_y = None
        self.seen_kwargs = None

    def fit(self, X, y, **kwargs):
        self.seen_X = X
        self.seen_y = y
        self.seen_kwargs = kwargs
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_fit_model_drops_nan_target_rows_before_splitting_but_keeps_nan_features():
    dates = _dates(105)
    X = pd.DataFrame({"feat": np.arange(105, dtype=float)}, index=dates)
    y = pd.Series(np.arange(105, dtype=float), index=dates)
    X.loc[dates[50], "feat"] = np.nan  # NaN feature, valid target: row must survive
    y.loc[dates[60]] = np.nan  # NaN target: row must be dropped entirely, before the split

    model = RecordingStubModel()
    fit_model(model, X, y)

    usable_dates = dates.delete(60)
    n_eval = int(np.ceil(len(usable_dates) * EVAL_SET_FRACTION))
    expected_train_dates = usable_dates[:-n_eval]
    expected_eval_dates = usable_dates[-n_eval:]

    assert list(model.seen_y.index) == list(expected_train_dates)
    assert dates[60] not in model.seen_y.index
    assert np.isnan(model.seen_X.loc[dates[50], "feat"])

    (eval_X, eval_y), = model.seen_kwargs["eval_set"]
    assert list(eval_y.index) == list(expected_eval_dates)
    assert model.seen_kwargs["verbose"] is False


def test_fit_model_eval_split_is_chronological_tail_of_training_data_only():
    # Shuffle the input row order to prove the split is by date, not input order.
    dates = _dates(100)
    rng = np.random.default_rng(0)
    shuffled = dates[rng.permutation(len(dates))]
    X = pd.DataFrame({"feat": np.arange(100, dtype=float)}, index=shuffled)
    y = pd.Series(np.arange(100, dtype=float), index=shuffled)

    model = RecordingStubModel()
    fit_model(model, X, y)

    (eval_X, eval_y), = model.seen_kwargs["eval_set"]
    n_eval = int(np.ceil(100 * EVAL_SET_FRACTION))
    assert list(eval_y.index) == list(dates[-n_eval:])
    assert list(model.seen_y.index) == list(dates[:-n_eval])


def _smallest_n_where_eval_split_exactly_hits_floor() -> int:
    """The smallest usable-row count `n` where `ceil(n * EVAL_SET_FRACTION) == MIN_EVAL_ROWS`
    exactly — i.e. right on the boundary between "clears the floor by rounding up" and "would
    raise if rounded down instead". Derived from the live constants so this stays correct if
    either constant changes, rather than hardcoding a row count that quietly stops being the
    real boundary.
    """
    n = 1
    while math.ceil(n * EVAL_SET_FRACTION) < MIN_EVAL_ROWS:
        n += 1
    assert math.ceil(n * EVAL_SET_FRACTION) == MIN_EVAL_ROWS
    return n


def test_fit_model_rounds_eval_split_up_to_clear_the_floor_exactly():
    n = _smallest_n_where_eval_split_exactly_hits_floor()
    dates = _dates(n)
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=dates)
    y = pd.Series(np.arange(n, dtype=float), index=dates)

    model = RecordingStubModel()
    fit_model(model, X, y)  # must not raise

    (eval_X, eval_y), = model.seen_kwargs["eval_set"]
    assert len(eval_y) == MIN_EVAL_ROWS
    assert len(model.seen_y) == n - MIN_EVAL_ROWS


def test_fit_model_raises_when_usable_rows_too_few_for_min_eval_floor():
    dates = _dates(50)  # 50 * 15% = 7.5 -> 8, well below the 15-row floor
    X = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=dates)
    y = pd.Series(np.arange(50, dtype=float), index=dates)

    model = RecordingStubModel()
    with pytest.raises(ValueError, match=str(MIN_EVAL_ROWS)):
        fit_model(model, X, y)
