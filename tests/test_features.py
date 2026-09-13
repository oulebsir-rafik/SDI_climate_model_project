from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intake_forecast.features import (
    BC_PARAMETERS,
    RECOMMENDED_PREDICTOR_CONFIG,
    FeatureConfig,
    build_feature_table,
    build_features,
    load_bc_daily,
    select_feature_columns,
    select_xy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_history(n_days: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "A": rng.normal(size=n_days),
            "B": rng.normal(size=n_days),
        },
        index=index,
    )


def test_build_features_lag_values_are_causal():
    history = _synthetic_history()
    config = FeatureConfig(lag_days=[1, 2], rolling_windows=[], include_day_of_year=False, include_day_of_week=False)
    as_of = history.index[10]

    feats = build_features(history, as_of, config)

    # lag_k = y_{t-k} for target t = as_of + 1 day, so lag1 is the value AT as_of itself.
    assert feats["A_lag1"] == pytest.approx(history["A"].loc[as_of])
    assert feats["A_lag2"] == pytest.approx(history["A"].loc[as_of - pd.Timedelta(days=1)])
    assert feats["B_lag1"] == pytest.approx(history["B"].loc[as_of])


def test_build_features_never_uses_future_data():
    history = _synthetic_history()
    config = FeatureConfig(lag_days=[1, 3], rolling_windows=[3], rolling_stats=["mean"], include_day_of_year=False, include_day_of_week=False)
    as_of = history.index[10]

    truncated = history.loc[:as_of]
    full_feats = build_features(history, as_of, config)
    truncated_feats = build_features(truncated, as_of, config)

    pd.testing.assert_series_equal(full_feats, truncated_feats)


def test_build_features_rolling_is_trailing_and_nan_for_partial_window():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    history = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    config = FeatureConfig(lag_days=[], rolling_windows=[3], rolling_stats=["mean"], include_day_of_year=False, include_day_of_week=False)

    early_feats = build_features(history, index[1], config)
    assert np.isnan(early_feats["A_roll3_mean"])

    late_feats = build_features(history, index[3], config)
    assert late_feats["A_roll3_mean"] == pytest.approx((2.0 + 3.0 + 4.0) / 3)


def test_build_features_rolling_nan_if_any_value_in_window_missing():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    history = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=index)
    config = FeatureConfig(lag_days=[], rolling_windows=[3], rolling_stats=["mean"], include_day_of_year=False, include_day_of_week=False)

    feats = build_features(history, index[3], config)
    assert np.isnan(feats["A_roll3_mean"])


def test_build_features_calendar_cyclical_encoding():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    history = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    config = FeatureConfig(lag_days=[], rolling_windows=[], include_day_of_year=True, include_day_of_week=True)

    as_of = index[0]  # 2024-01-01, day-of-year 1, Monday (dayofweek 0)
    feats = build_features(history, as_of, config)

    expected_doy_sin = np.sin(2 * np.pi * 1 / 365.25)
    expected_doy_cos = np.cos(2 * np.pi * 1 / 365.25)
    assert feats["doy_sin"] == pytest.approx(expected_doy_sin)
    assert feats["doy_cos"] == pytest.approx(expected_doy_cos)
    assert feats["dow_sin"] == pytest.approx(np.sin(0.0))
    assert feats["dow_cos"] == pytest.approx(np.cos(0.0))


def test_build_feature_table_target_is_next_day_value():
    # build_feature_table's target columns are keyed off the real BC parameter names
    # (TARGET_COLUMNS), so the history frame must use them.
    history = _synthetic_history(n_days=10).rename(columns={"A": "pH_BC", "B": "Température_BC"})
    history["Conductivité_BC"] = 0.0
    history["TDS_BC"] = 0.0
    config = FeatureConfig(lag_days=[1], rolling_windows=[], include_day_of_year=False, include_day_of_week=False)

    table = build_feature_table(history, config)

    for i in range(len(history) - 1):
        this_day, next_day = history.index[i], history.index[i + 1]
        assert table.loc[this_day, "y_pH_BC"] == pytest.approx(history.loc[next_day, "pH_BC"])

    last_day = history.index[-1]
    assert np.isnan(table.loc[last_day, "y_pH_BC"])


def test_select_xy_splits_features_and_target():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    table = pd.DataFrame(
        {
            "pH_BC_lag1": [1.0, 2.0, 3.0],
            "doy_sin": [0.1, 0.2, 0.3],
            "y_pH_BC": [7.0, 8.0, 9.0],
            "y_TDS_BC": [30.0, 31.0, 32.0],
        },
        index=index,
    )

    X, y = select_xy(table, "pH_BC")

    assert list(X.columns) == ["pH_BC_lag1", "doy_sin"]
    assert list(y) == [7.0, 8.0, 9.0]


def test_select_feature_columns_default_includes_everything():
    columns = ["pH_BC_lag1", "Température_BC_roll7_mean", "doy_sin", "dow_cos", "y_pH_BC"]
    result = select_feature_columns(columns)
    assert result == ["pH_BC_lag1", "Température_BC_roll7_mean", "doy_sin", "dow_cos"]


def test_select_feature_columns_restricts_to_named_parameters_and_calendar_flag():
    columns = ["pH_BC_lag1", "pH_BC_roll7_mean", "TDS_BC_lag1", "doy_sin", "dow_cos", "y_pH_BC"]

    own_only_no_calendar = select_feature_columns(columns, predictor_parameters=["pH_BC"], include_calendar=False)
    assert own_only_no_calendar == ["pH_BC_lag1", "pH_BC_roll7_mean"]

    own_only_with_calendar = select_feature_columns(columns, predictor_parameters=["pH_BC"], include_calendar=True)
    assert own_only_with_calendar == ["pH_BC_lag1", "pH_BC_roll7_mean", "doy_sin", "dow_cos"]


def test_select_xy_applies_predictor_config():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    table = pd.DataFrame(
        {
            "pH_BC_lag1": [1.0, 2.0, 3.0],
            "TDS_BC_lag1": [10.0, 20.0, 30.0],
            "doy_sin": [0.1, 0.2, 0.3],
            "y_pH_BC": [7.0, 8.0, 9.0],
        },
        index=index,
    )

    X, y = select_xy(table, "pH_BC", predictor_parameters=["pH_BC"], include_calendar=False)

    assert list(X.columns) == ["pH_BC_lag1"]
    assert list(y) == [7.0, 8.0, 9.0]


def test_recommended_predictor_config_keeps_conductivity_cross_parameter():
    # Empirically validated: pH/temperature/TDS are restricted to their own history;
    # conductivity keeps the full cross-parameter set (see the feature-selection investigation).
    assert RECOMMENDED_PREDICTOR_CONFIG["pH_BC"]["predictor_parameters"] == ["pH_BC"]
    assert RECOMMENDED_PREDICTOR_CONFIG["Conductivité_BC"]["predictor_parameters"] == list(BC_PARAMETERS)


def test_load_bc_daily_averages_and_reindexes_with_nan_gaps(tmp_path):
    csv_path = tmp_path / "clean_data.csv"
    csv_path.write_text(
        "DATE,HEURE,pH_BC,Température_BC,Conductivité_BC,TDS_BC\n"
        "1/01/2024,8h00,8.0,17.0,55.0,33.0\n"
        "1/01/2024,10h30,8.2,17.4,55.2,33.2\n"
        "3/01/2024,8h00,8.4,17.8,55.4,33.4\n"
    )

    daily = load_bc_daily(csv_path)

    assert list(daily.index) == list(pd.date_range("2024-01-01", "2024-01-03", freq="D"))
    assert daily.loc["2024-01-01", "pH_BC"] == pytest.approx((8.0 + 8.2) / 2)
    assert daily.loc["2024-01-02"].isna().all()
    assert daily.loc["2024-01-03", "pH_BC"] == pytest.approx(8.4)
    assert list(daily.columns) == list(BC_PARAMETERS)
