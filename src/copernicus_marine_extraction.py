"""Extract Copernicus Marine data at the desalination plant's seawater intake point.

The reference point is the plant's seawater intake (lat=36.75, lon=3.06), not the
Corso watershed centroid used by era5_data_extraction.py / chirps_data_extraction.py:
marine conditions are tied to where the plant draws seawater offshore, not to the
land catchment. Each product's neighborhood search radius is sized from its own
native grid resolution (resolved via the Copernicus Marine metadata API) rather than
a single fixed radius, since products span roughly 9-28km native spacing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import median
from typing import Iterable

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr

KM_PER_DEG_LAT = 111.32

REFERENCE_LAT = 36.75
REFERENCE_LON = 3.06

DEFAULT_MIN_DEPTH_M = 0.49402499198913574  # first native depth level (surface layer)
DEFAULT_MAX_DEPTH_M = 5.0
DEFAULT_RADIUS_MULTIPLIER = 2.5
DEFAULT_START_DATETIME = "2024-02-01T00:00:00"
DEFAULT_END_DATETIME = "2025-02-28T23:59:59"
DEFAULT_DATE_RANGE_LABEL = "Feb2024_Feb2025"

USERNAME = "rafik.oulebsir@usthb.edu.dz"
PASSWORD = "bYs98T!LKb5#C8G"


@dataclass(frozen=True)
class ProductConfig:
    name: str
    dataset_id: str
    variables: tuple[str, ...]
    has_depth: bool


def default_product_catalog() -> tuple[ProductConfig, ...]:
    return (
        ProductConfig("Physics_Temperature", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ("thetao",), True),
        ProductConfig("Physics_Salinity", "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ("so",), True),
        ProductConfig("Physics_Currents", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ("uo", "vo"), True),
        ProductConfig("Sea_Surface_Height", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ("zos",), False),
        ProductConfig("Waves", "cmems_mod_glo_wav_my_0.2deg_PT3H-i", ("VHM0",), False),
        ProductConfig(
            "Wind", "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H", ("eastward_wind", "northward_wind"), False
        ),
        ProductConfig(
            "Ocean_Color_SPM_Turbidity_Regional",
            "cmems_obs_oc_med_bgc_tur-spm-chl_nrt_l3-hr-mosaic_P1D-m",
            ("TUR", "SPM"),
            False,
        ),
        ProductConfig("Biogeochemistry_Carbon_pH", "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m", ("ph", "dissic"), True),
        ProductConfig("Biogeochemistry_Oxygen", "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m", ("o2",), True),
        ProductConfig(
            "Biogeochemistry_Plankton_Biomass", "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m", ("chl", "phyc"), True
        ),
    )


@dataclass
class MarineExtractionConfig:
    reference_lat: float = REFERENCE_LAT
    reference_lon: float = REFERENCE_LON
    start_datetime: str = DEFAULT_START_DATETIME
    end_datetime: str = DEFAULT_END_DATETIME
    min_depth_m: float = DEFAULT_MIN_DEPTH_M
    max_depth_m: float = DEFAULT_MAX_DEPTH_M
    radius_multiplier: float = DEFAULT_RADIUS_MULTIPLIER
    date_range_label: str = DEFAULT_DATE_RANGE_LABEL
    products: tuple[ProductConfig, ...] = field(default_factory=default_product_catalog)
    username: str = USERNAME
    password: str = PASSWORD
    rawdata_dir: Path = Path("data/rawdata")
    clean_dir: Path = Path("data/clean")


# ==========================================================================
# Pure geometry / aggregation logic (network-independent, unit tested)
# ==========================================================================


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float):
    """Great-circle distance in km. Broadcasts like numpy for array inputs."""
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def compute_neighborhood_radius_km(native_grid_spacing_km: float, multiplier: float = DEFAULT_RADIUS_MULTIPLIER) -> float:
    if native_grid_spacing_km <= 0:
        raise ValueError("native_grid_spacing_km must be positive.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")
    return native_grid_spacing_km * multiplier


def bbox_from_radius_km(ref_lat: float, ref_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (min_lon, max_lon, min_lat, max_lat) bounding the given radius around the reference point."""
    delta_lat = radius_km / KM_PER_DEG_LAT
    delta_lon = radius_km / (KM_PER_DEG_LAT * np.cos(np.radians(ref_lat)))
    return (ref_lon - delta_lon, ref_lon + delta_lon, ref_lat - delta_lat, ref_lat + delta_lat)


@dataclass(frozen=True)
class MatchedPoint:
    lat: float
    lon: float
    day: date
    value: float


@dataclass(frozen=True)
class DailyAggregate:
    day: date
    median_value: float
    n_points: int
    min_distance_km: float
    median_distance_km: float
    max_distance_km: float


def aggregate_daily_median(
    points: Iterable[MatchedPoint],
    ref_lat: float,
    ref_lon: float,
    radius_km: float,
) -> list[DailyAggregate]:
    """Median-aggregate matched grid points per day, keeping only points within radius_km.

    This is the core correctness logic of the extraction: it decides which candidate
    points contribute to a day's value and records how far each contributing point
    actually was from the reference point.
    """
    by_day: dict[date, list[tuple[float, float]]] = {}

    for point in points:
        distance_km = float(haversine_distance_km(ref_lat, ref_lon, point.lat, point.lon))
        if distance_km > radius_km:
            continue
        by_day.setdefault(point.day, []).append((point.value, distance_km))

    aggregates = []
    for day, entries in sorted(by_day.items()):
        values = [v for v, _ in entries]
        distances = sorted(d for _, d in entries)
        aggregates.append(
            DailyAggregate(
                day=day,
                median_value=median(values),
                n_points=len(entries),
                min_distance_km=distances[0],
                median_distance_km=median(distances),
                max_distance_km=distances[-1],
            )
        )
    return aggregates


# ==========================================================================
# Network-calling wrapper functions (thin I/O around copernicusmarine SDK)
# ==========================================================================


def resolve_native_grid_spacing_km(
    dataset_id: str,
    ref_lat: float,
    ref_lon: float,
    probe_window_deg: float = 0.5,
    max_widenings: int = 4,
) -> float:
    """Resolve a product's native grid spacing near the reference point via the metadata API.

    Opens only a small window around the reference point (rather than the full global
    grid) so this stays fast even for very high-resolution regional products; widens
    the window if fewer than two grid points fall inside it.
    """
    window = probe_window_deg
    for _ in range(max_widenings):
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=ref_lon - window,
            maximum_longitude=ref_lon + window,
            minimum_latitude=ref_lat - window,
            maximum_latitude=ref_lat + window,
            coordinates_selection_method="inside",
        )
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if len(lat) >= 2 and len(lon) >= 2:
            lat_spacing_km = abs(float(lat[1] - lat[0])) * KM_PER_DEG_LAT
            lon_spacing_km = abs(float(lon[1] - lon[0])) * KM_PER_DEG_LAT * np.cos(np.radians(ref_lat))
            return max(lat_spacing_km, float(lon_spacing_km))
        window *= 2

    raise RuntimeError(
        f"Could not resolve native grid spacing for dataset '{dataset_id}': "
        "fewer than 2 grid points found near the reference point even after widening the probe window."
    )


def download_product_points(product: ProductConfig, cfg: MarineExtractionConfig, radius_km: float) -> Path:
    """Download the neighborhood of matched grid points for one product via subset()."""
    min_lon, max_lon, min_lat, max_lat = bbox_from_radius_km(cfg.reference_lat, cfg.reference_lon, radius_km)

    output_path = cfg.rawdata_dir / f"copernicus_marine_{product.name}_{cfg.date_range_label}.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request_kwargs = dict(
        dataset_id=product.dataset_id,
        variables=list(product.variables),
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=cfg.start_datetime,
        end_datetime=cfg.end_datetime,
        coordinates_selection_method="inside",
        output_filename=str(output_path),
        overwrite=True,
        disable_progress_bar=True,
    )
    if product.has_depth:
        request_kwargs["minimum_depth"] = cfg.min_depth_m
        request_kwargs["maximum_depth"] = cfg.max_depth_m

    copernicusmarine.subset(**request_kwargs)
    return output_path


def load_raw_points_dataframe(nc_path: Path, ref_lat: float, ref_lon: float) -> pd.DataFrame:
    """Load a downloaded product's matched points into a flat, distance-annotated table."""
    with xr.open_dataset(nc_path) as ds:
        df = ds.to_dataframe().reset_index()

    df["day"] = pd.to_datetime(df["time"]).dt.date
    df["distance_km"] = haversine_distance_km(ref_lat, ref_lon, df["latitude"].to_numpy(), df["longitude"].to_numpy())
    return df


def points_for_variable(raw_df: pd.DataFrame, variable: str) -> list[MatchedPoint]:
    valid = raw_df.dropna(subset=[variable])
    return [
        MatchedPoint(lat=row.latitude, lon=row.longitude, day=row.day, value=float(getattr(row, variable)))
        for row in valid.itertuples(index=False)
    ]


@dataclass
class ProductExtractionResult:
    product: ProductConfig
    native_grid_spacing_km: float
    radius_km: float
    raw_csv_path: Path
    daily_csv_path: Path
    daily_frame: pd.DataFrame


def extract_product(product: ProductConfig, cfg: MarineExtractionConfig) -> ProductExtractionResult:
    """Resolve resolution, download, and daily-aggregate one product's matched points."""
    native_spacing_km = resolve_native_grid_spacing_km(product.dataset_id, cfg.reference_lat, cfg.reference_lon)
    radius_km = compute_neighborhood_radius_km(native_spacing_km, cfg.radius_multiplier)

    nc_path = download_product_points(product, cfg, radius_km)
    raw_df = load_raw_points_dataframe(nc_path, cfg.reference_lat, cfg.reference_lon)

    raw_csv_path = cfg.rawdata_dir / f"copernicus_marine_{product.name}_{cfg.date_range_label}.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    per_variable_frames = []
    for variable in product.variables:
        points = points_for_variable(raw_df, variable)
        aggregates = aggregate_daily_median(points, cfg.reference_lat, cfg.reference_lon, radius_km)
        per_variable_frames.append(
            pd.DataFrame(
                {
                    "date": [a.day for a in aggregates],
                    f"{variable}_median": [a.median_value for a in aggregates],
                    f"{variable}_n_points": [a.n_points for a in aggregates],
                    f"{variable}_dist_min_km": [a.min_distance_km for a in aggregates],
                    f"{variable}_dist_median_km": [a.median_distance_km for a in aggregates],
                    f"{variable}_dist_max_km": [a.max_distance_km for a in aggregates],
                }
            )
        )

    daily_frame = per_variable_frames[0]
    for var_df in per_variable_frames[1:]:
        daily_frame = daily_frame.merge(var_df, on="date", how="outer")
    daily_frame = daily_frame.sort_values("date").reset_index(drop=True)

    daily_csv_path = cfg.clean_dir / f"copernicus_marine_{product.name}_daily_{cfg.date_range_label}.csv"
    daily_csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily_frame.to_csv(daily_csv_path, index=False)

    return ProductExtractionResult(
        product=product,
        native_grid_spacing_km=native_spacing_km,
        radius_km=radius_km,
        raw_csv_path=raw_csv_path,
        daily_csv_path=daily_csv_path,
        daily_frame=daily_frame,
    )


def extract_marine_data(cfg: MarineExtractionConfig | None = None) -> dict:
    """Orchestration entrypoint: extract every configured product and merge into one daily dataset.

    Every configured product is extracted regardless of resulting coverage - sparse
    products (e.g. regional ocean color, waves) are kept in the output so downstream
    per-project data cleaning can decide whether to use, impute, or drop them.
    """
    cfg = cfg or MarineExtractionConfig()
    cfg.rawdata_dir.mkdir(parents=True, exist_ok=True)
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)

    copernicusmarine.login(username=cfg.username, password=cfg.password, force_overwrite=True)

    results: list[ProductExtractionResult] = []
    for product in cfg.products:
        print(f"--- Extracting {product.name} ---")
        try:
            results.append(extract_product(product, cfg))
        except Exception as exc:
            print(f"Error processing {product.name}: {exc}")

    start = pd.to_datetime(cfg.start_datetime).date()
    end = pd.to_datetime(cfg.end_datetime).date()
    merged = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D").date})

    for result in results:
        merged = merged.merge(result.daily_frame, on="date", how="left")

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)

    merged_csv_path = cfg.clean_dir / f"copernicus_marine_daily_{cfg.date_range_label}.csv"
    merged.to_csv(merged_csv_path, index=False)
    print(f"Saved merged marine dataset: {merged_csv_path}")

    return {
        "merged_frame": merged,
        "merged_csv_path": merged_csv_path,
        "products": {
            result.product.name: {
                "dataset_id": result.product.dataset_id,
                "native_grid_spacing_km": result.native_grid_spacing_km,
                "radius_km": result.radius_km,
                "raw_csv_path": result.raw_csv_path,
                "daily_csv_path": result.daily_csv_path,
            }
            for result in results
        },
    }
