"""Per-parameter Optuna hyperparameter tuning and final refit/holdout evaluation.

See specs/intake-forecast-model-and-tuning.md for the design this implements.

Note on cross-parameter recursive coupling: each parameter's Optuna study is independent (one
study per parameter, per the spec), but a recursive rollout beyond h=1 needs *every* parameter's
model to build cross-parameter lag features (see intake_forecast.cv.evaluate_fold). While tuning
parameter P, the other 3 parameters' models use DEFAULT_XGB_PARAMS (a fixed, untuned baseline)
rather than trial-sampled hyperparameters — only P's model varies across trials. This keeps the
4 studies independent while still exercising the real multi-horizon, cross-parameter rollout
mechanism during tuning, rather than a simplified single-parameter approximation of it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import optuna
import pandas as pd

from .cv import Fold, evaluate_fold, fold_train_insample, make_folds, plot_obs_vs_pred_scatter, plot_obs_vs_pred_timeseries
from .features import RECOMMENDED_PREDICTOR_CONFIG, BC_PARAMETERS, FeatureConfig, build_feature_table, select_xy
from .metrics import aggregate_metrics
from .model import DEFAULT_XGB_PARAMS, fit_model, make_xgb_model

SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "max_depth": (2, 6),
    "n_estimators": (50, 500),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 10),
    "reg_alpha": (1e-3, 10),
    "reg_lambda": (1e-3, 10),
}


def _sample_params(trial: optuna.Trial) -> dict:
    lo, hi = SEARCH_SPACE["max_depth"]
    max_depth = trial.suggest_int("max_depth", int(lo), int(hi))
    lo, hi = SEARCH_SPACE["n_estimators"]
    n_estimators = trial.suggest_int("n_estimators", int(lo), int(hi))
    lo, hi = SEARCH_SPACE["learning_rate"]
    learning_rate = trial.suggest_float("learning_rate", lo, hi, log=True)
    lo, hi = SEARCH_SPACE["subsample"]
    subsample = trial.suggest_float("subsample", lo, hi)
    lo, hi = SEARCH_SPACE["colsample_bytree"]
    colsample_bytree = trial.suggest_float("colsample_bytree", lo, hi)
    lo, hi = SEARCH_SPACE["min_child_weight"]
    min_child_weight = trial.suggest_int("min_child_weight", int(lo), int(hi))
    lo, hi = SEARCH_SPACE["reg_alpha"]
    reg_alpha = trial.suggest_float("reg_alpha", lo, hi, log=True)
    lo, hi = SEARCH_SPACE["reg_lambda"]
    reg_lambda = trial.suggest_float("reg_lambda", lo, hi, log=True)
    return {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }


def _trial_model_factory(parameter: str, params: dict):
    def factory(target_parameter: str):
        chosen = params if target_parameter == parameter else DEFAULT_XGB_PARAMS
        return make_xgb_model(chosen)

    return factory


def _mean_mse_across_horizons(predictions_df: pd.DataFrame, parameter: str, history: pd.DataFrame, fold: Fold) -> float:
    param_predictions = predictions_df[predictions_df["parameter"] == parameter]
    insample = {parameter: fold_train_insample(history, fold)[parameter]}
    agg = aggregate_metrics(param_predictions, insample)
    horizon_rows = agg[agg["horizon"] != "overall"]
    if horizon_rows.empty:
        return float("nan")
    return float(horizon_rows["mse"].mean())


def run_study(
    parameter: str,
    history: pd.DataFrame,
    folds: list[Fold],
    run_id: str,
    output_root: Path,
    feature_config: FeatureConfig | None = None,
    n_trials: int = 100,
    max_horizon: int = 7,
    predictor_config: dict[str, dict] | None = None,
) -> optuna.Study:
    """One Optuna study for `parameter`, minimizing mean MSE across horizons h=1..7,
    averaged over `folds`. Persisted to a resumable SQLite file inside this run's directory.

    `predictor_config` restricts each parameter's feature set — see
    `features.RECOMMENDED_PREDICTOR_CONFIG` and `cv.evaluate_fold`.
    """
    feature_config = feature_config or FeatureConfig()
    models_dir = output_root / run_id / parameter / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{models_dir / 'optuna_study.db'}"

    study = optuna.create_study(
        study_name=f"{parameter}_{run_id}",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        model_factory = _trial_model_factory(parameter, params)
        fold_scores = []
        for fold in folds:
            predictions = evaluate_fold(
                fold, history, model_factory, feature_config, max_horizon=max_horizon, predictor_config=predictor_config
            )
            fold_scores.append(_mean_mse_across_horizons(predictions, parameter, history, fold))
        return float(pd.Series(fold_scores).mean())

    study.optimize(objective, n_trials=n_trials)
    return study


def finalize_parameter(
    parameter: str,
    history: pd.DataFrame,
    study: optuna.Study,
    holdout: Fold,
    run_id: str,
    output_root: Path,
    feature_config: FeatureConfig | None = None,
    max_horizon: int = 7,
    predictor_config: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Score `study`'s best hyperparameters once against the untouched final holdout, persist
    the holdout metrics/predictions/plots, then refit a final model on the entire CV region
    (holdout.train_start..train_end) and persist it.
    """
    feature_config = feature_config or FeatureConfig()
    best_params = study.best_params
    model_factory = _trial_model_factory(parameter, best_params)

    predictions = evaluate_fold(
        holdout, history, model_factory, feature_config, max_horizon=max_horizon, predictor_config=predictor_config
    )
    param_predictions = predictions[predictions["parameter"] == parameter]
    insample = {parameter: fold_train_insample(history, holdout)[parameter]}
    agg = aggregate_metrics(param_predictions, insample)

    param_dir = output_root / run_id / parameter
    tables_dir, plots_dir, models_dir = param_dir / "tables", param_dir / "plots", param_dir / "models"
    for d in (tables_dir, plots_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    agg.to_csv(tables_dir / "holdout_metrics.csv", index=False)
    param_predictions.to_csv(tables_dir / "holdout_predictions.csv", index=False)

    for horizon in range(1, max_horizon + 1):
        plot_obs_vs_pred_timeseries(param_predictions, parameter, horizon, plots_dir / f"holdout_timeseries_h{horizon}.png")
        plot_obs_vs_pred_scatter(param_predictions, parameter, horizon, plots_dir / f"holdout_scatter_h{horizon}.png")

    train_history = history.loc[holdout.train_start : holdout.train_end]
    table = build_feature_table(train_history, feature_config)
    X, y = select_xy(table, parameter, **(predictor_config or {}).get(parameter, {}))
    table.to_csv(tables_dir / "train_feature_table.csv")
    final_model = fit_model(make_xgb_model(best_params), X, y)
    joblib.dump(final_model, models_dir / "model.joblib")

    return agg


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def run_pipeline(
    history: pd.DataFrame,
    output_root: Path = Path("results/intake_forecast"),
    feature_config: FeatureConfig | None = None,
    n_folds: int = 4,
    init_train_days: int = 120,
    test_len_days: int = 30,
    gap_days: int = 21,
    holdout_days: int = 60,
    n_trials: int = 100,
    max_horizon: int = 7,
    run_id: str | None = None,
    feature_table_path: Path | None = Path("data/ML/intake_bc_features.parquet"),
    predictor_config: dict[str, dict] | None = RECOMMENDED_PREDICTOR_CONFIG,
) -> str:
    """Run the full tuning + final-evaluation pipeline for all 4 BC parameters and return the
    run_id under which every artifact was written.

    Also (re)writes the shared preprocessing output from
    specs/intake-forecast-preprocessing-features.md — the full engineered feature table plus
    next-day targets for all 4 parameters — to `feature_table_path`. Pass `None` to skip this
    (e.g. in tests, to avoid writing into the real project's data/ML/ directory).

    `predictor_config` defaults to `features.RECOMMENDED_PREDICTOR_CONFIG` — the empirically
    validated per-parameter feature restriction (pH/temperature/TDS use their own history only;
    conductivity keeps the full cross-parameter set). Pass `None` to use the full feature set
    for every parameter instead (the original, pre-investigation behavior).
    """
    feature_config = feature_config or FeatureConfig()
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if feature_table_path is not None:
        feature_table_path.parent.mkdir(parents=True, exist_ok=True)
        build_feature_table(history, feature_config).to_parquet(feature_table_path)

    folds, holdout = make_folds(
        history.index,
        n_folds=n_folds,
        init_train_days=init_train_days,
        test_len_days=test_len_days,
        gap_days=gap_days,
        holdout_days=holdout_days,
    )

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "feature_config": asdict(feature_config),
        "data_range": {"start": str(history.index.min().date()), "end": str(history.index.max().date())},
        "fold_settings": {
            "n_folds": n_folds,
            "init_train_days": init_train_days,
            "test_len_days": test_len_days,
            "gap_days": gap_days,
            "holdout_days": holdout_days,
        },
        "search_space": SEARCH_SPACE,
        "n_trials": n_trials,
        "predictor_config": predictor_config,
        "git_commit": _current_git_commit(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    for parameter in BC_PARAMETERS:
        study = run_study(
            parameter, history, folds, run_id, output_root, feature_config,
            n_trials=n_trials, max_horizon=max_horizon, predictor_config=predictor_config,
        )
        finalize_parameter(
            parameter, history, study, holdout, run_id, output_root, feature_config,
            max_horizon=max_horizon, predictor_config=predictor_config,
        )

    return run_id
