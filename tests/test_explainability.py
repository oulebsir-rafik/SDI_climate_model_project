import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.intake_forecast.explainability import (
    build_holdout_feature_table,
    compute_shap_values,
    explain_parameter,
    load_shap_values,
    plot_mean_abs_shap,
    plot_shap_beeswarm,
    plot_shap_waterfall_for_date,
    save_shap_values,
)
from src.intake_forecast.features import BC_PARAMETERS, FeatureConfig, build_features


def _fitted_model_and_X(n_rows: int = 40, seed: int = 0) -> tuple[XGBRegressor, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    X = pd.DataFrame({"a": rng.normal(size=n_rows), "b": rng.normal(size=n_rows)}, index=index)
    X.index.name = "date"
    y = 3.0 * X["a"] - X["b"] + rng.normal(scale=0.01, size=n_rows)
    model = XGBRegressor(n_estimators=20, max_depth=2, n_jobs=1).fit(X, y)
    return model, X


def test_compute_shap_values_reconstructs_predictions():
    model, X = _fitted_model_and_X()

    shap_df, expected_value = compute_shap_values(model, X)

    assert list(shap_df.columns) == list(X.columns)
    assert list(shap_df.index) == list(X.index)
    reconstructed = shap_df.sum(axis=1).to_numpy() + expected_value
    np.testing.assert_allclose(reconstructed, model.predict(X), atol=1e-4)


def test_save_and_load_shap_values_round_trip(tmp_path):
    model, X = _fitted_model_and_X()
    shap_df, expected_value = compute_shap_values(model, X)

    save_shap_values(shap_df, expected_value, tmp_path, "train")
    loaded_df, loaded_expected = load_shap_values(tmp_path, "train")

    # CSV round-tripping loses XGBoost/SHAP's native float32 dtype (read back as float64) — the
    # values themselves must still match, just not the dtype.
    pd.testing.assert_frame_equal(loaded_df, shap_df, check_names=False, check_freq=False, check_dtype=False)
    assert loaded_expected == pytest.approx(expected_value)


def test_build_holdout_feature_table_matches_build_features_for_each_origin():
    history = pd.DataFrame(
        {p: np.linspace(0, 10, 30) for p in BC_PARAMETERS},
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
    )
    config = FeatureConfig(lag_days=[1, 2], rolling_windows=[3], rolling_stats=["mean"])
    origins = pd.DatetimeIndex([history.index[10], history.index[15], history.index[20]])

    table = build_holdout_feature_table(history, origins, config)

    assert list(table.index) == list(origins)
    for origin in origins:
        direct = build_features(history, origin, config)
        for col in direct.index:
            expected, actual = direct[col], table.loc[origin, col]
            if pd.isna(expected):
                assert pd.isna(actual)
            else:
                assert actual == pytest.approx(expected)
    # target column is the real next-day value, not a recursively-predicted one
    assert table.loc[origins[0], "y_pH_BC"] == pytest.approx(history.loc[origins[0] + pd.Timedelta(days=1), "pH_BC"])


def test_plot_mean_abs_shap_writes_a_file(tmp_path):
    model, X = _fitted_model_and_X()
    shap_df, _ = compute_shap_values(model, X)

    output_path = tmp_path / "mean_abs_shap.png"
    plot_mean_abs_shap(shap_df, "pH_BC", output_path)

    assert output_path.exists()


def test_plot_shap_beeswarm_writes_a_file(tmp_path):
    model, X = _fitted_model_and_X()
    shap_df, _ = compute_shap_values(model, X)

    output_path = tmp_path / "beeswarm.png"
    plot_shap_beeswarm(shap_df, X, "pH_BC", output_path)

    assert output_path.exists()


def test_plot_shap_waterfall_for_date_writes_a_file(tmp_path):
    model, X = _fitted_model_and_X()
    shap_df, expected_value = compute_shap_values(model, X)
    row_date = X.index[5]

    output_path = tmp_path / "waterfall.png"
    plot_shap_waterfall_for_date(shap_df, X, expected_value, "pH_BC", row_date, output_path)

    assert output_path.exists()


def test_plot_shap_waterfall_for_date_raises_for_unknown_date(tmp_path):
    model, X = _fitted_model_and_X()
    shap_df, expected_value = compute_shap_values(model, X)

    with pytest.raises(ValueError, match="no SHAP row"):
        plot_shap_waterfall_for_date(shap_df, X, expected_value, "pH_BC", pd.Timestamp("2099-01-01"), tmp_path / "x.png")


def test_explain_parameter_persists_both_datasets(tmp_path):
    model, X_train = _fitted_model_and_X(n_rows=40)
    _, X_holdout = _fitted_model_and_X(n_rows=10, seed=1)

    explain_parameter(model, X_train, X_holdout, tmp_path)

    train_df, _ = load_shap_values(tmp_path, "train")
    holdout_df, _ = load_shap_values(tmp_path, "holdout")
    assert len(train_df) == len(X_train)
    assert len(holdout_df) == len(X_holdout)
