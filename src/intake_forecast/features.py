"""Preprocessing and feature generation for the intake (BC station) forecast.

See specs/intake-forecast-preprocessing-features.md for the design this implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BC_PARAMETERS: tuple[str, ...] = (
    "pH_BC",
    "Température_BC",
    "Conductivité_BC",
    "TDS_BC",
)

# ASCII-normalized target column names, per spec, so downstream column names stay portable.
TARGET_COLUMNS: dict[str, str] = {
    "pH_BC": "y_pH_BC",
    "Température_BC": "y_Temperature_BC",
    "Conductivité_BC": "y_Conductivite_BC",
    "TDS_BC": "y_TDS_BC",
}

_HEURE_BUCKETS: dict[str, str] = {
    "8h00": "morning",
    "8h30": "morning",
    "10h00": "midday",
    "10h30": "midday",
    "14h00": "afternoon",
    "15h00": "afternoon",
    "15h30": "afternoon",
}


@dataclass
class FeatureConfig:
    lag_days: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7, 14])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std"])
    include_day_of_year: bool = True
    include_day_of_week: bool = True


def _categorize_heure(heure: object) -> object:
    if pd.isna(heure):
        return np.nan
    return _HEURE_BUCKETS.get(str(heure).strip(), "unknown")


def load_bc_daily(
    source_csv: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load the raw physical-chemical CSV, average sub-daily readings to one row per day
    (reusing the existing morning/midday/afternoon convention), select the BC columns, and
    reindex onto a continuous daily calendar. Missing days/values are left as NaN.
    """
    raw = pd.read_csv(source_csv)
    # The source DATE column mixes dd/mm/yyyy and dd/mm/yy formatting (e.g. some days in
    # Oct-Dec 2024 use a 2-digit year). format="mixed" parses each value independently
    # instead of inferring one format for the whole column, which would otherwise silently
    # coerce every mismatched row to NaT and drop it in the groupby below.
    raw["DATE"] = pd.to_datetime(raw["DATE"], format="mixed", dayfirst=True, errors="coerce")
    for col in raw.columns:
        if col not in ("DATE", "HEURE"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["HEURE"] = raw["HEURE"].apply(_categorize_heure)
    daily = raw.drop(columns=["HEURE"]).groupby("DATE").mean(numeric_only=True)

    bc = daily[list(BC_PARAMETERS)]

    range_start = start if start is not None else bc.index.min()
    range_end = end if end is not None else bc.index.max()
    full_index = pd.date_range(range_start, range_end, freq="D")
    return bc.reindex(full_index)


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def build_features(history: pd.DataFrame, as_of: pd.Timestamp, config: FeatureConfig) -> pd.Series:
    """Pure function: build one row of causal features from `history` as of `as_of`.

    Never reads data later than `as_of` — this is what makes it safe to call repeatedly during
    a recursive multi-step rollout (see intake_forecast.cv.evaluate_fold), as well as once per
    day to build the batch training table below.
    """
    as_of = pd.Timestamp(as_of)
    past = history.loc[:as_of]

    features: dict[str, float] = {}

    for param in history.columns:
        series = history[param]
        for lag in config.lag_days:
            # lag_k = the value k days before the *target* (as_of + 1 day) — i.e. y_{t-k} for
            # target t = as_of+1, matching the standard AR(k) lag definition. lag1 is therefore
            # the value at as_of itself (the most recent known day), not as_of - 1.
            lag_idx = as_of - pd.Timedelta(days=lag - 1)
            features[f"{param}_lag{lag}"] = series.get(lag_idx, np.nan)

        past_series = past[param]
        for window in config.rolling_windows:
            windowed = past_series.rolling(window=window, min_periods=window)
            for stat in config.rolling_stats:
                result = getattr(windowed, stat)()
                features[f"{param}_roll{window}_{stat}"] = result.iloc[-1] if len(result) else np.nan

    if config.include_day_of_year:
        sin_v, cos_v = _cyclical_encode(as_of.dayofyear, 365.25)
        features["doy_sin"] = sin_v
        features["doy_cos"] = cos_v

    if config.include_day_of_week:
        sin_v, cos_v = _cyclical_encode(as_of.dayofweek, 7)
        features["dow_sin"] = sin_v
        features["dow_cos"] = cos_v

    return pd.Series(features, name=as_of)


def build_feature_table(history: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build the full training feature table: one engineered feature row per calendar day in
    `history`, plus next-day (t+1) target columns for each BC parameter.
    """
    rows = [build_features(history, as_of, config) for as_of in history.index]
    table = pd.DataFrame(rows)
    table.index.name = "date"

    for param, target_col in TARGET_COLUMNS.items():
        table[target_col] = history[param].shift(-1)

    return table


_CALENDAR_COLUMNS = {"doy_sin", "doy_cos", "dow_sin", "dow_cos"}

# Empirically validated per-parameter predictor sets (see diary/2026-09-09-feature-selection-
# per-parameter.md): pH, temperature, and TDS are dominated by their own recent history, and
# including the other 3 parameters' lags/rolling stats hurts more than it helps (fits noise on
# ~250-300 training rows rather than real signal). Conductivity is tightly, physically coupled
# to TDS/salinity and genuinely benefits from the full cross-parameter feature set. This is the
# default used by intake_forecast.tuning.run_pipeline; pass a different mapping (or None per
# parameter) to select_xy/select_feature_columns to override it.
RECOMMENDED_PREDICTOR_CONFIG: dict[str, dict] = {
    "pH_BC": {"predictor_parameters": ["pH_BC"], "include_calendar": False},
    "Température_BC": {"predictor_parameters": ["Température_BC"], "include_calendar": False},
    "Conductivité_BC": {"predictor_parameters": list(BC_PARAMETERS), "include_calendar": True},
    "TDS_BC": {"predictor_parameters": ["TDS_BC"], "include_calendar": False},
}


def select_feature_columns(
    columns: list[str],
    predictor_parameters: list[str] | None = None,
    include_calendar: bool = True,
) -> list[str]:
    """Filter a set of feature column names down to those belonging to `predictor_parameters`
    (own-variable lags/rolling stats for each named BC parameter) plus calendar columns if
    `include_calendar`. `predictor_parameters=None` (the default) means "all 4 BC parameters",
    matching the original identical-X-for-every-model behavior.
    """
    predictor_parameters = list(predictor_parameters) if predictor_parameters is not None else list(BC_PARAMETERS)

    def is_included(col: str) -> bool:
        if col.startswith("y_"):
            return False
        if col in _CALENDAR_COLUMNS:
            return include_calendar
        return any(col.startswith(f"{p}_lag") or col.startswith(f"{p}_roll") for p in predictor_parameters)

    return [c for c in columns if is_included(c)]


def select_xy(
    table: pd.DataFrame,
    parameter: str,
    predictor_parameters: list[str] | None = None,
    include_calendar: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into (X, y) for one parameter's model.

    By default X is every non-target column (identical across all 4 parameters, since
    cross-parameter lags mean every parameter's model needs every other parameter's history
    too); y is that parameter's own next-day target column. Pass `predictor_parameters` /
    `include_calendar` (see `RECOMMENDED_PREDICTOR_CONFIG`) to restrict X to a subset — e.g. a
    parameter's own history only.
    """
    target_col = TARGET_COLUMNS[parameter]
    feature_cols = select_feature_columns(list(table.columns), predictor_parameters, include_calendar)
    return table[feature_cols], table[target_col]
