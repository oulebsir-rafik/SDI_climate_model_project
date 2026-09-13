import numpy as np
import pandas as pd

from src.intake_forecast.cv import make_folds
from src.intake_forecast.features import BC_PARAMETERS, FeatureConfig
from src.intake_forecast.tuning import finalize_parameter, run_pipeline, run_study


def _synthetic_bc_history(n_days: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="D")
    base = np.linspace(0, 10, n_days)
    return pd.DataFrame({p: base + rng.normal(scale=0.1, size=n_days) for p in BC_PARAMETERS}, index=index)


def test_run_study_and_finalize_parameter_smoke(tmp_path):
    history = _synthetic_bc_history()
    folds, holdout = make_folds(history.index, n_folds=1, init_train_days=40, test_len_days=10, gap_days=5, holdout_days=15)
    assert len(folds) == 1

    config = FeatureConfig(lag_days=[1, 2], rolling_windows=[3], rolling_stats=["mean"])
    run_id = "test_run"

    study = run_study("pH_BC", history, folds, run_id, tmp_path, feature_config=config, n_trials=2, max_horizon=3)
    assert len(study.trials) == 2
    assert np.isfinite(study.best_value)

    agg = finalize_parameter("pH_BC", history, study, holdout, run_id, tmp_path, feature_config=config, max_horizon=3)
    assert not agg.empty

    param_dir = tmp_path / run_id / "pH_BC"
    assert (param_dir / "models" / "optuna_study.db").exists()
    assert (param_dir / "models" / "model.joblib").exists()
    assert (param_dir / "tables" / "holdout_metrics.csv").exists()
    assert (param_dir / "tables" / "holdout_predictions.csv").exists()
    assert (param_dir / "plots" / "holdout_timeseries_h1.png").exists()
    assert (param_dir / "plots" / "holdout_scatter_h1.png").exists()


def test_run_pipeline_smoke_all_four_parameters(tmp_path):
    history = _synthetic_bc_history(n_days=80)
    config = FeatureConfig(lag_days=[1, 2], rolling_windows=[3], rolling_stats=["mean"])

    run_id = run_pipeline(
        history,
        output_root=tmp_path,
        feature_config=config,
        n_folds=1,
        init_train_days=35,
        test_len_days=8,
        gap_days=4,
        holdout_days=12,
        n_trials=1,
        max_horizon=2,
        run_id="pipeline_test",
        feature_table_path=tmp_path / "intake_bc_features.parquet",
    )

    assert run_id == "pipeline_test"
    assert (tmp_path / "intake_bc_features.parquet").exists()
    run_dir = tmp_path / run_id
    assert (run_dir / "run_config.json").exists()
    for parameter in BC_PARAMETERS:
        assert (run_dir / parameter / "models" / "model.joblib").exists()
        assert (run_dir / parameter / "tables" / "holdout_metrics.csv").exists()
