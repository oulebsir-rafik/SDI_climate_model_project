"""Launch a full intake-forecast tuning + evaluation run.

Loads the BC-station daily series, runs one Optuna study per parameter (pH, temperature,
conductivity, TDS), and writes each parameter's tuned model, holdout metrics/predictions, and
diagnostic plots under `--output-root/{run_id}/{parameter}/`, plus a `run_config.json`
describing the run. See specs/intake-forecast-cv-and-metrics.md and
specs/intake-forecast-model-and-tuning.md for the design this runs.

Usage:
    .venv/bin/python scripts/run_intake_forecast.py
    .venv/bin/python scripts/run_intake_forecast.py --n-trials 20 --run-id quick_check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intake_forecast.features import load_bc_daily
from src.intake_forecast.tuning import run_pipeline

DEFAULT_SOURCE_CSV = Path("data/clean/clean_data.csv")
DEFAULT_OUTPUT_ROOT = Path("results/intake_forecast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV, help="Raw physical-chemical CSV to load.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where run artifacts are written.")
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        default=None,
        help=(
            "Restrict loaded history to on/before this date (e.g. '2024-12-21'), dropping any "
            "later days entirely. Use this to exclude a known-bad tail segment (e.g. the "
            "unexplained Conductivite_BC/TDS_BC step-jump starting 2024-12-22, not corroborated "
            "by pH/temperature/turbidity -- see diary notes) from both training and the holdout "
            "window, which is measured back from the end of the loaded history. Default: use "
            "all available history."
        ),
    )
    parser.add_argument("--run-id", type=str, default=None, help="Defaults to a UTC timestamp.")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per parameter (spec default: 100).")
    parser.add_argument("--max-horizon", type=int, default=7, help="Recursive forecast horizon in days.")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of walk-forward CV folds.")
    parser.add_argument("--init-train-days", type=int, default=120)
    parser.add_argument("--test-len-days", type=int, default=30)
    parser.add_argument("--gap-days", type=int, default=21, help="Purge gap between train and test in each fold.")
    parser.add_argument("--holdout-days", type=int, default=60, help="Final untouched holdout period, in days.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    history = load_bc_daily(args.source_csv, end=args.end_date)
    print(f"Loaded {len(history)} calendar days ({history.index.min().date()} -> {history.index.max().date()})")
    print("Missingness per parameter:")
    print(history.isna().mean().round(3).to_string())

    start = time.time()
    run_id = run_pipeline(
        history,
        output_root=args.output_root,
        n_folds=args.n_folds,
        init_train_days=args.init_train_days,
        test_len_days=args.test_len_days,
        gap_days=args.gap_days,
        holdout_days=args.holdout_days,
        n_trials=args.n_trials,
        max_horizon=args.max_horizon,
        run_id=args.run_id,
    )
    elapsed_minutes = (time.time() - start) / 60

    print(f"\nRun '{run_id}' complete in {elapsed_minutes:.1f} min.")
    print(f"Artifacts under: {args.output_root / run_id}")
