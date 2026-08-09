"""Includes code to process ERA5 climate data for a specific point (latitude, longitude) and export daily aggregated data to a CSV file."""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

lat_point = 36.75
lon_point = 3.06

# List of directories containing the .nc files (Update with your actual paths)
folders = [
    "./era5_downloads/era5_2024_01",
    "./era5_downloads/era5_2024_02",
    "./era5_downloads/era5_2024_03",
    "./era5_downloads/era5_2024_04",
    "./era5_downloads/era5_2024_05",
    "./era5_downloads/era5_2024_06",
    "./era5_downloads/era5_2024_07",
    "./era5_downloads/era5_2024_08",
    "./era5_downloads/era5_2024_09",
    "./era5_downloads/era5_2024_10",
    "./era5_downloads/era5_2024_11",
    "./era5_downloads/era5_2024_12",
    "./era5_downloads/era5_2025_01",
    "./era5_downloads/era5_2025_02"
]

downloaded_files = []
for folder in folders:
    # Use rglob if the .nc files are inside subdirectories
    downloaded_files.extend(list(Path(folder).rglob("*.nc")))

# Optional: Ensure files were found before proceeding
if not downloaded_files:
    raise FileNotFoundError("No .nc files found in the specified folders.")

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

output_csv = "era5_daily_point_Jan2024_Feb2025.csv"
df.to_csv(output_csv, index=False)

print(df.head())
print(f"Saved file: {output_csv}")
