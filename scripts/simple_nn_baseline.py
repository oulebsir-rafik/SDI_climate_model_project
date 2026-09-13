"""Simple neural-net baseline for intake t+1 forecasting.

A quick, standalone comparison script — NOT part of the tree-based pipeline in
src/intake_forecast/. It reuses the same feature engineering (so it's an apples-to-apples
comparison), but uses a plain sklearn MLPRegressor and a single chronological train/test split
instead of walk-forward CV, to sanity-check whether a different model family shows the same
t+1 issues seen with the XGBoost pipeline.

Simplifications vs. the main pipeline (deliberate, for a quick comparison — not a rigorous
benchmark):
  - Single 80/20 chronological split, no purge gap between train and test.
  - t+1 only, no recursive multi-horizon rollout.
  - One MLP per parameter (matching the "one model per parameter" convention), t+1 target.
  - A plain MLP can't handle NaN or unscaled features the way XGBoost can, so (unlike the main
    pipeline, which deliberately never imputes) this script median-imputes and standardizes
    features here — fit on the training split only, to avoid leaking test information.

Usage:
    .venv/bin/python scripts/simple_nn_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intake_forecast.features import BC_PARAMETERS, FeatureConfig, build_feature_table, load_bc_daily, select_xy
from src.intake_forecast.metrics import mae, mase, mse, naive_insample_mae, r2, smape

SOURCE_CSV = Path("data/clean/clean_data.csv")
TRAIN_FRACTION = 0.8


def chronological_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(table) * train_fraction)
    return table.iloc[:split_idx], table.iloc[split_idx:]


def main() -> None:
    history = load_bc_daily(SOURCE_CSV)
    table = build_feature_table(history, FeatureConfig())

    train_table, test_table = chronological_split(table)
    print(f"Train: {train_table.index.min().date()} -> {train_table.index.max().date()} ({len(train_table)} days)")
    print(f"Test:  {test_table.index.min().date()} -> {test_table.index.max().date()} ({len(test_table)} days)")

    for parameter in BC_PARAMETERS:
        X_train, y_train = select_xy(train_table, parameter)
        X_test, y_test = select_xy(test_table, parameter)

        # can't train/score on a missing target
        train_valid, test_valid = y_train.notna(), y_test.notna()
        X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid]
        X_test, y_test = X_test.loc[test_valid], y_test.loc[test_valid]

        # unlike XGBoost, a plain MLP needs NaN-free, scaled input (and a scaled target —
        # MLPRegressor's default optimizer converges poorly on raw-scale targets) — fit only
        # on train to avoid leaking test information
        imputer = SimpleImputer(strategy="median").fit(X_train)
        x_scaler = StandardScaler().fit(imputer.transform(X_train))
        X_train_ready = x_scaler.transform(imputer.transform(X_train))
        X_test_ready = x_scaler.transform(imputer.transform(X_test))

        y_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
        y_train_ready = y_scaler.transform(y_train.to_numpy().reshape(-1, 1)).ravel()

        model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=1e-2,
            max_iter=2000,
            early_stopping=True,
            random_state=0,
        )
        model.fit(X_train_ready, y_train_ready)
        y_pred = y_scaler.inverse_transform(model.predict(X_test_ready).reshape(-1, 1)).ravel()

        print(f"\n=== {parameter} (t+1, {len(y_test)} test days) ===")
        print(f"  R2:    {r2(y_test, y_pred):.3f}")
        print(f"  MAE:   {mae(y_test, y_pred):.4f}")
        print(f"  MSE:   {mse(y_test, y_pred):.4f}")
        print(f"  sMAPE: {smape(y_test, y_pred):.2f}%")
        print(f"  MASE:  {mase(y_test, y_pred, y_train):.3f}  (naive in-sample MAE: {naive_insample_mae(y_train):.4f})")


if __name__ == "__main__":
    main()
