"""SHAP-based model explainability for the intake forecast's per-parameter XGBoost models.

See specs/intake-forecast-explainability.md for the design this implements.

This module only computes and persists SHAP values (`explain_parameter`, called from
`intake_forecast.tuning.finalize_parameter`) and builds the holdout horizon-1 feature table
they're computed against (`build_holdout_feature_table`). Turning saved SHAP values back into
plots is a separate, on-demand step (`plot_mean_abs_shap`, `plot_shap_beeswarm`,
`plot_shap_waterfall_for_date`), driven by `scripts/plot_explanations.py` — so a tuning run never
pays for plot rendering it may not need, and plots can be regenerated later without recomputing
SHAP.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .features import FeatureConfig, build_feature_table
from .model import Model

DATASETS: tuple[str, ...] = ("train", "holdout")


def compute_shap_values(model: Model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute per-row, per-feature SHAP values for a fitted tree model.

    Uses `feature_perturbation="tree_path_dependent"`, which reads the missing-value split
    directions the tree already learned instead of requiring a background dataset — the same
    native-NaN handling `model.fit_model` relies on, so features with NaN (recent lags/rolling
    stats early in a window, or genuinely missing sensor readings) need no imputation here either.
    """
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, index=X.index, columns=X.columns)
    shap_df.index.name = X.index.name or "date"
    return shap_df, float(explainer.expected_value)


def build_holdout_feature_table(
    history: pd.DataFrame,
    origins: pd.DatetimeIndex,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Build the horizon-1 holdout feature table: one row per `origins` date, with features
    resolved from the full `history` (not just the holdout window) so lags/rolling stats match
    exactly what `cv.evaluate_fold` fed the model at h=1 for that same origin. Restricting to
    horizon 1 only avoids the alternative — capturing the recursive h=2..7 rollout, where later
    steps' features are partly built from the model's own earlier predictions rather than
    observed data — which is out of scope for now (see specs/intake-forecast-explainability.md).
    """
    return build_feature_table(history, feature_config, dates=origins)


def _dataset_dir(param_dir: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return param_dir / "explainability" / dataset


def save_shap_values(shap_df: pd.DataFrame, expected_value: float, param_dir: Path, dataset: str) -> None:
    dataset_dir = _dataset_dir(param_dir, dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(dataset_dir / "shap_values.csv")
    (dataset_dir / "expected_value.json").write_text(json.dumps({"expected_value": expected_value}))


def load_shap_values(param_dir: Path, dataset: str) -> tuple[pd.DataFrame, float]:
    dataset_dir = _dataset_dir(param_dir, dataset)
    shap_df = pd.read_csv(dataset_dir / "shap_values.csv", index_col=0, parse_dates=True)
    expected_value = json.loads((dataset_dir / "expected_value.json").read_text())["expected_value"]
    return shap_df, float(expected_value)


def explain_parameter(model: Model, X_train: pd.DataFrame, X_holdout: pd.DataFrame, param_dir: Path) -> None:
    """Compute and persist SHAP values for one parameter's final model, against both its
    training feature table and its horizon-1 holdout feature table. Called from
    `tuning.finalize_parameter` when `--explain` is on; produces no plots (see module docstring).
    """
    train_shap, train_expected = compute_shap_values(model, X_train)
    save_shap_values(train_shap, train_expected, param_dir, "train")

    holdout_shap, holdout_expected = compute_shap_values(model, X_holdout)
    save_shap_values(holdout_shap, holdout_expected, param_dir, "holdout")


def _to_explanation(shap_df: pd.DataFrame, feature_values: pd.DataFrame, expected_value: float) -> shap.Explanation:
    aligned = feature_values.reindex(index=shap_df.index)[shap_df.columns]
    return shap.Explanation(
        values=shap_df.to_numpy(),
        base_values=np.full(len(shap_df), expected_value),
        data=aligned.to_numpy(),
        feature_names=list(shap_df.columns),
    )


def plot_mean_abs_shap(shap_df: pd.DataFrame, parameter: str, output_path: Path) -> None:
    """Global importance: mean absolute SHAP value per feature, ranked descending."""
    ranked = shap_df.abs().mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(ranked))))
    ax.barh(ranked.index, ranked.to_numpy())
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"{parameter} — global feature importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shap_beeswarm(shap_df: pd.DataFrame, feature_values: pd.DataFrame, parameter: str, output_path: Path) -> None:
    """Global importance + direction + spread: SHAP beeswarm summary plot."""
    # beeswarm never reads base_values, so a placeholder is fine here (unlike the waterfall plot).
    explanation = _to_explanation(shap_df, feature_values, expected_value=0.0)

    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    fig = plt.gcf()
    fig.suptitle(f"{parameter} — SHAP summary")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shap_waterfall_for_date(
    shap_df: pd.DataFrame,
    feature_values: pd.DataFrame,
    expected_value: float,
    parameter: str,
    row_date: pd.Timestamp,
    output_path: Path,
) -> None:
    """Local importance for one row: how each feature pushed this specific day's prediction away
    from the model's expected (average) output.
    """
    row_date = pd.Timestamp(row_date)
    if row_date not in shap_df.index:
        available = f"{shap_df.index.min().date()}..{shap_df.index.max().date()}"
        raise ValueError(f"{row_date.date()} has no SHAP row in this dataset (available range: {available}).")

    explanation = _to_explanation(shap_df, feature_values, expected_value)
    row_explanation = explanation[shap_df.index.get_loc(row_date)]

    plt.figure()
    shap.plots.waterfall(row_explanation, show=False)
    fig = plt.gcf()
    fig.suptitle(f"{parameter} — {row_date.date()}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
