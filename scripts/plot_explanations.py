"""Regenerate SHAP explainability plots from values already persisted by a
scripts/run_intake_forecast.py --explain run.

Never recomputes SHAP values — only reads what run_intake_forecast.py already saved under
`{output_root}/{run_id}/{parameter}/explainability/{train,holdout}/` and
`{output_root}/{run_id}/{parameter}/tables/{train,holdout}/feature_table.csv`, and writes a plot
under `.../{parameter}/explainability/{dataset}/plots/`.

Usage:
    .venv/bin/python scripts/plot_explanations.py --run-id quick_check --parameter pH_BC --plot mean
    .venv/bin/python scripts/plot_explanations.py --run-id quick_check --parameter pH_BC --plot beeswarm --dataset holdout
    .venv/bin/python scripts/plot_explanations.py --run-id quick_check --parameter pH_BC --plot waterfall --dataset holdout --row-date 2024-11-03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intake_forecast.explainability import (
    load_shap_values,
    plot_mean_abs_shap,
    plot_shap_beeswarm,
    plot_shap_waterfall_for_date,
)
from src.intake_forecast.features import BC_PARAMETERS

DEFAULT_OUTPUT_ROOT = Path("results/intake_forecast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where run artifacts were written.")
    parser.add_argument("--run-id", type=str, required=True, help="Run to read SHAP values from.")
    parser.add_argument("--parameter", type=str, required=True, choices=BC_PARAMETERS, help="Which BC parameter's model.")
    parser.add_argument("--dataset", type=str, default="train", choices=("train", "holdout"), help="Which feature table's SHAP values to plot.")
    parser.add_argument("--plot", type=str, required=True, choices=("mean", "beeswarm", "waterfall"), help="Which plot to (re)generate.")
    parser.add_argument(
        "--row-date",
        type=pd.Timestamp,
        default=None,
        help="Required for --plot waterfall: the date (e.g. 2024-11-03) whose row to explain.",
    )
    args = parser.parse_args()
    if args.plot == "waterfall" and args.row_date is None:
        parser.error("--plot waterfall requires --row-date")
    return args


def _feature_table_path(param_dir: Path, dataset: str) -> Path:
    return param_dir / "tables" / dataset / "feature_table.csv"


if __name__ == "__main__":
    args = parse_args()
    param_dir = args.output_root / args.run_id / args.parameter

    shap_df, expected_value = load_shap_values(param_dir, args.dataset)
    plots_dir = param_dir / "explainability" / args.dataset / "plots"

    if args.plot == "mean":
        output_path = plots_dir / "mean_abs_shap.png"
        plot_mean_abs_shap(shap_df, args.parameter, output_path)
    else:
        feature_values = pd.read_csv(_feature_table_path(param_dir, args.dataset), index_col=0, parse_dates=True)
        if args.plot == "beeswarm":
            output_path = plots_dir / "beeswarm.png"
            plot_shap_beeswarm(shap_df, feature_values, args.parameter, output_path)
        else:
            output_path = plots_dir / f"waterfall_{args.row_date.date()}.png"
            plot_shap_waterfall_for_date(shap_df, feature_values, expected_value, args.parameter, args.row_date, output_path)

    print(f"Wrote {output_path}")
