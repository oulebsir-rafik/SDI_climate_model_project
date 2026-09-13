import numpy as np
import pandas as pd
import pytest

from src.intake_forecast.cv import evaluate_fold, make_folds
from src.intake_forecast.features import BC_PARAMETERS, FeatureConfig
from src.intake_forecast.metrics import aggregate_metrics


def test_make_folds_matches_the_documented_schedule_for_the_real_calendar():
    dates = pd.date_range("2024-01-01", "2025-01-31", freq="D")

    folds, holdout = make_folds(dates)

    expected = [
        ("2024-01-01", "2024-04-29", "2024-05-21", "2024-06-19"),
        ("2024-01-01", "2024-06-19", "2024-07-11", "2024-08-09"),
        ("2024-01-01", "2024-08-09", "2024-08-31", "2024-09-29"),
        ("2024-01-01", "2024-09-29", "2024-10-21", "2024-11-19"),
    ]
    assert len(folds) == 4
    for fold, (train_start, train_end, test_start, test_end) in zip(folds, expected):
        assert fold.train_start == pd.Timestamp(train_start)
        assert fold.train_end == pd.Timestamp(train_end)
        assert fold.test_start == pd.Timestamp(test_start)
        assert fold.test_end == pd.Timestamp(test_end)

    assert holdout.test_start == pd.Timestamp("2024-12-03")
    assert holdout.test_end == pd.Timestamp("2025-01-31")
    assert holdout.train_end == pd.Timestamp("2024-12-02")


def test_make_folds_generalizes_to_a_different_date_range():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    folds, holdout = make_folds(dates, n_folds=4, init_train_days=60, test_len_days=15, gap_days=10, holdout_days=30)

    assert holdout.test_start == pd.Timestamp("2020-12-02")
    assert holdout.test_end == pd.Timestamp("2020-12-31")
    for fold in folds:
        # gap between train_end and test_start must be exactly gap_days
        assert (fold.test_start - fold.train_end).days == 11
        assert (fold.test_end - fold.test_start).days == 14
        assert fold.test_end <= holdout.train_end


class LagPlusOneModel:
    """Deterministic stub: predicts (its own parameter's lag-1 value) + 1. Used to verify the
    recursive rollout feeds its own predictions forward rather than real future values.
    """

    def __init__(self, parameter: str):
        self.parameter = parameter

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        # Mirrors XGBoost's real behavior of always returning a finite prediction even with a
        # NaN feature input (it learns a default split direction for missing values), so this
        # stub doesn't cascade NaNs in a way a real trained model wouldn't.
        lag1 = np.nan_to_num(X[f"{self.parameter}_lag1"].to_numpy(), nan=0.0)
        return lag1 + 1.0


def _steep_history(n_days: int = 60) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n_days, freq="D")
    offsets = np.arange(n_days, dtype=float)
    return pd.DataFrame({p: offsets * 10.0 for p in BC_PARAMETERS}, index=index)


def test_evaluate_fold_recursive_rollout_uses_predictions_not_future_ground_truth():
    # A 100-day train window (rather than a smaller one) so fit_model's early-stopping eval
    # split (15%, min 15 rows) has enough usable rows to not raise — see model.py.
    history = _steep_history(n_days=150)
    config = FeatureConfig(lag_days=[1], rolling_windows=[], include_day_of_year=False, include_day_of_week=False)

    from src.intake_forecast.cv import Fold

    fold = Fold(
        train_start=history.index[0],
        train_end=history.index[99],
        test_start=history.index[100],
        test_end=history.index[119],
    )

    predictions = evaluate_fold(fold, history, LagPlusOneModel, config, max_horizon=2)

    origin = fold.test_start
    origin_offset = (origin - history.index[0]).days

    h1 = predictions[(predictions.origin == origin) & (predictions.horizon == 1) & (predictions.parameter == "pH_BC")].iloc[0]
    h2 = predictions[(predictions.origin == origin) & (predictions.horizon == 2) & (predictions.parameter == "pH_BC")].iloc[0]

    expected_h1 = origin_offset * 10.0 + 1.0
    expected_h2_from_prediction_chain = expected_h1 + 1.0
    expected_h2_if_erroneously_using_real_future = (origin_offset + 1) * 10.0 + 1.0

    assert h1["predicted"] == pytest.approx(expected_h1)
    assert h2["predicted"] == pytest.approx(expected_h2_from_prediction_chain)
    assert h2["predicted"] != pytest.approx(expected_h2_if_erroneously_using_real_future)


def test_evaluate_fold_leaves_missing_actuals_as_nan_and_aggregate_metrics_drops_them():
    from src.intake_forecast.cv import Fold

    # A 100-day train window (rather than a smaller one) so fit_model's early-stopping eval
    # split (15%, min 15 rows) has enough usable rows to not raise — see model.py.
    history = _steep_history(n_days=150)
    missing_day = history.index[101]
    history.loc[missing_day, "pH_BC"] = np.nan

    config = FeatureConfig(lag_days=[1], rolling_windows=[], include_day_of_year=False, include_day_of_week=False)
    fold = Fold(
        train_start=history.index[0],
        train_end=history.index[99],
        test_start=history.index[100],
        test_end=history.index[119],
    )

    predictions = evaluate_fold(fold, history, LagPlusOneModel, config, max_horizon=2)

    ph_predictions = predictions[predictions.parameter == "pH_BC"]
    missing_rows = ph_predictions[
        (ph_predictions.origin + pd.to_timedelta(ph_predictions.horizon, unit="D")) == missing_day
    ]
    assert len(missing_rows) >= 1
    assert missing_rows["actual"].isna().all()

    train_slice = history.loc[fold.train_start : fold.train_end, "pH_BC"]
    agg = aggregate_metrics(ph_predictions, {"pH_BC": train_slice})

    horizon_of_missing_row = missing_rows.iloc[0]["horizon"]
    matching_agg_row = agg[(agg["parameter"] == "pH_BC") & (agg["horizon"] == horizon_of_missing_row)]
    total_rows_for_that_horizon = len(ph_predictions[ph_predictions.horizon == horizon_of_missing_row])
    assert matching_agg_row.iloc[0]["n"] == total_rows_for_that_horizon - 1


class ColumnSpyModel:
    """Records the exact feature columns it was fit/predicted on, per parameter instance."""

    def __init__(self, parameter: str):
        self.parameter = parameter
        self.fit_columns = None
        self.predict_columns = None

    def fit(self, X, y, **kwargs):
        self.fit_columns = list(X.columns)
        return self

    def predict(self, X):
        self.predict_columns = list(X.columns)
        return np.zeros(len(X))


def test_evaluate_fold_restricts_columns_per_parameter_via_predictor_config():
    from src.intake_forecast.cv import Fold

    # A 100-day train window (rather than a smaller one) so fit_model's early-stopping eval
    # split (15%, min 15 rows) has enough usable rows to not raise — see model.py.
    history = _steep_history(n_days=150)
    config = FeatureConfig(lag_days=[1], rolling_windows=[], include_day_of_year=False, include_day_of_week=False)
    fold = Fold(
        train_start=history.index[0],
        train_end=history.index[99],
        test_start=history.index[100],
        test_end=history.index[104],
    )

    predictor_config = {"pH_BC": {"predictor_parameters": ["pH_BC"], "include_calendar": False}}
    spies: dict[str, ColumnSpyModel] = {}

    def spy_factory(parameter: str) -> ColumnSpyModel:
        spies[parameter] = ColumnSpyModel(parameter)
        return spies[parameter]

    evaluate_fold(fold, history, spy_factory, config, max_horizon=1, predictor_config=predictor_config)

    # pH_BC is restricted to its own lag columns only
    assert spies["pH_BC"].fit_columns == ["pH_BC_lag1"]
    assert spies["pH_BC"].predict_columns == ["pH_BC_lag1"]
    # every other parameter still gets the full (unrestricted) feature set
    assert set(spies["Température_BC"].fit_columns) == {f"{p}_lag1" for p in BC_PARAMETERS}
