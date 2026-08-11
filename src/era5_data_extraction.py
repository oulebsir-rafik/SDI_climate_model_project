"""Download ERA5 climate data for the Corso watershed and export daily aggregated data to a CSV file.

The extraction point is the centroid of the delineated watershed
(output/corso_watershed/watershed.geojson), not an arbitrary coordinate: at ERA5's
native 0.25 degree grid resolution the whole watershed falls inside a single grid
cell, so the nearest cell to the centroid is representative of the catchment.
"""
import json
import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

# ==========================================================
# WATERSHED-DERIVED EXTRACTION POINT
# ==========================================================

WATERSHED_GEOJSON = Path("output/corso_watershed/watershed.geojson")

with WATERSHED_GEOJSON.open() as f:
    _watershed = json.load(f)

_centroid = shape(_watershed["features"][0]["geometry"]).centroid
lat_point = _centroid.y
lon_point = _centroid.x

out_dir = Path("era5_downloads/corso_watershed")
out_dir.mkdir(parents=True, exist_ok=True)

variables = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "surface_solar_radiation_downwards"
]

years = ["2024", "2025"]

months_by_year = {
    "2024": [f"{m:02d}" for m in range(1, 13)],
    "2025": ["01", "02"]
}

days = [f"{d:02d}" for d in range(1, 32)]
hours = [f"{h:02d}:00" for h in range(24)]

buffer = 0.125
area = [
    lat_point + buffer,
    lon_point - buffer,
    lat_point - buffer,
    lon_point + buffer
]

# ==========================================================
# DOWNLOAD DATA
# ==========================================================

client = cdsapi.Client()

downloaded_files = []

for year in years:
    for month in months_by_year[year]:

        month_dir = out_dir / f"era5_{year}_{month}"
        zip_file = out_dir / f"era5_{year}_{month}.zip"

        existing_nc_files = list(month_dir.glob("*.nc")) if month_dir.exists() else []

        if existing_nc_files:
            print(f"NetCDF already exists: {month_dir}")
            downloaded_files.extend(existing_nc_files)
            continue

        if not zip_file.exists():
            print(f"Downloading {year}-{month}...")

            request = {
                "product_type": ["reanalysis"],
                "variable": variables,
                "year": [year],
                "month": [month],
                "day": days,
                "time": hours,
                "data_format": "netcdf",
                "download_format": "zip",
                "area": area
            }

            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(zip_file)
            )

        print(f"Extracting {zip_file}...")

        month_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(month_dir)

        # CDS splits the request into an "instant" file (t2m, d2m, u10, v10, sp)
        # and an "accum" file (tp, ssrd) - both are needed, so keep every .nc
        # extracted rather than picking just one.
        month_nc_files = list(month_dir.glob("*.nc"))
        print("Extracted files:", [f.name for f in month_nc_files])

        downloaded_files.extend(month_nc_files)


# ==========================================================
# OPEN DATA
# ==========================================================

ds = xr.open_mfdataset(downloaded_files, combine="by_coords", engine="netcdf4")

if "valid_time" in ds.coords:
    time_name = "valid_time"
elif "time" in ds.coords:
    time_name = "time"
else:
    raise ValueError("No time coordinate found.")

point_ds = ds.sel(
    latitude=lat_point,
    longitude=lon_point,
    method="nearest"
)

# ==========================================================
# UNIT CONVERSIONS
# ==========================================================

point_ds["t2m_c"] = point_ds["t2m"] - 273.15
point_ds["d2m_c"] = point_ds["d2m"] - 273.15
point_ds["tp_mm"] = point_ds["tp"] * 1000.0
point_ds["sp_hpa"] = point_ds["sp"] / 100.0
point_ds["ssrd_mj_m2"] = point_ds["ssrd"] / 1_000_000.0
point_ds["wind_speed_10m"] = np.sqrt(point_ds["u10"]**2 + point_ds["v10"]**2)

# ==========================================================
# DAILY AGGREGATION
# ==========================================================

daily = xr.Dataset()

daily["t2m_mean_c"] = point_ds["t2m_c"].resample({time_name: "1D"}).mean()
daily["t2m_min_c"] = point_ds["t2m_c"].resample({time_name: "1D"}).min()
daily["t2m_max_c"] = point_ds["t2m_c"].resample({time_name: "1D"}).max()

daily["dewpoint_mean_c"] = point_ds["d2m_c"].resample({time_name: "1D"}).mean()

daily["precipitation_mm"] = point_ds["tp_mm"].resample({time_name: "1D"}).sum()

daily["wind_speed_mean_ms"] = point_ds["wind_speed_10m"].resample({time_name: "1D"}).mean()
daily["wind_speed_max_ms"] = point_ds["wind_speed_10m"].resample({time_name: "1D"}).max()

daily["surface_pressure_mean_hpa"] = point_ds["sp_hpa"].resample({time_name: "1D"}).mean()

daily["solar_radiation_mj_m2"] = point_ds["ssrd_mj_m2"].resample({time_name: "1D"}).sum()

# ==========================================================
# EXPORT CSV
# ==========================================================

df = daily.to_dataframe().reset_index()

df["date"] = pd.to_datetime(df[time_name]).dt.date
df = df.drop(columns=[time_name])

df["requested_latitude"] = lat_point
df["requested_longitude"] = lon_point
df["era5_latitude"] = float(point_ds.latitude.values)
df["era5_longitude"] = float(point_ds.longitude.values)

cols = [
    "date",
    "requested_latitude",
    "requested_longitude",
    "era5_latitude",
    "era5_longitude",
    "t2m_mean_c",
    "t2m_min_c",
    "t2m_max_c",
    "dewpoint_mean_c",
    "precipitation_mm",
    "wind_speed_mean_ms",
    "wind_speed_max_ms",
    "surface_pressure_mean_hpa",
    "solar_radiation_mj_m2"
]

df = df[cols]

output_csv = Path("data/clean/era5_daily_point_Jan2024_Feb2025.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)

print(df.head())
print(f"Saved file: {output_csv}")
