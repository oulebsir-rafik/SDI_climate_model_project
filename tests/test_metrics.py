from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intake_forecast.features import load_bc_daily
from src.intake_forecast.metrics import aggregate_metrics, mase, naive_insample_mae, smape

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_CSV = REPO_ROOT / "data" / "clean" / "clean_data.csv"


def test_smape_hand_computed():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # per-point: 2*2/22*100=18.18..., 2*2/38*100=10.526...
    expected = np.mean([2 * 2 / 22 * 100, 2 * 2 / 38 * 100])
    assert smape(y_true, y_pred) == pytest.approx(expected)


def test_naive_insample_mae_hand_computed():
    series = pd.Series([1.0, 3.0, 2.0, 6.0])  # diffs: 2, 1, 4 -> mean 7/3
    assert naive_insample_mae(series) == pytest.approx(7 / 3)


def test_mase_scales_by_insample_naive_mae():
    y_true = np.array([5.0, 5.0])
    y_pred = np.array([6.0, 4.0])  # MAE = 1.0
    insample = pd.Series([1.0, 3.0, 2.0, 6.0])  # naive MAE = 7/3
    assert mase(y_true, y_pred, insample) == pytest.approx(1.0 / (7 / 3))


def test_mase_matches_real_pH_BC_fold1_worked_example():
    """Grounded regression check against the worked example in
    diary/2026-08-22-cv-strategy-explainer.html and specs/intake-forecast-cv-and-metrics.md:
    Fold 1's pH_BC training window (2024-01-01..2024-04-29) has an in-sample naive-lag-1 MAE
    of ~0.0248.
    """
    daily = load_bc_daily(CLEAN_DATA_CSV)
    train_window = daily.loc["2024-01-01":"2024-04-29", "pH_BC"]

    assert naive_insample_mae(train_window) == pytest.approx(0.0248, abs=1e-4)


def test_aggregate_metrics_drops_nan_actual_and_reports_per_horizon_plus_overall():
    predictions_df = pd.DataFrame(
        [
            {"origin": pd.Timestamp("2024-01-01"), "horizon": 1, "parameter": "pH_BC", "predicted": 8.0, "actual": 8.1},
            {"origin": pd.Timestamp("2024-01-02"), "horizon": 1, "parameter": "pH_BC", "predicted": 8.2, "actual": np.nan},
            {"origin": pd.Timestamp("2024-01-01"), "horizon": 2, "parameter": "pH_BC", "predicted": 8.3, "actual": 8.2},
        ]
    )
    insample = {"pH_BC": pd.Series([8.0, 8.1, 8.05, 8.2])}

    agg = aggregate_metrics(predictions_df, insample)

    h1_row = agg[(agg["parameter"] == "pH_BC") & (agg["horizon"] == 1)].iloc[0]
    assert h1_row["n"] == 1  # the NaN-actual sample is excluded, not imputed

    overall_row = agg[(agg["parameter"] == "pH_BC") & (agg["horizon"] == "overall")].iloc[0]
    assert overall_row["n"] == 2  # both remaining (non-NaN) samples across horizons
