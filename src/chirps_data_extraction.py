"""Download CHIRPS daily precipitation for the Corso watershed and compute zonal
statistics (mean/max/min) plus a per-pixel diagnostic table.

The zonal geometry is the delineated watershed (output/corso_watershed/watershed.geojson),
not a bounding box, and pixels are selected with all_touched=True since the basin only
spans a handful of CHIRPS 0.05 degree pixels - being strict about pixel centers would risk
dropping a pixel that is mostly inside the watershed.
"""
import gzip
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import xy as transform_xy
from shapely.geometry import mapping, shape

# ==========================================================
# WATERSHED GEOMETRY
# ==========================================================

WATERSHED_GEOJSON = Path("output/corso_watershed/watershed.geojson")

with WATERSHED_GEOJSON.open() as f:
    _watershed = json.load(f)

watershed_polygon = shape(_watershed["features"][0]["geometry"])
watershed_geometry = [mapping(watershed_polygon)]

# ==========================================================
# DOWNLOAD CONFIG
# ==========================================================

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 2, 28)

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05"

DOWNLOAD_DIR = Path("chirps_downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def chirps_url(day: date) -> str:
    return f"{BASE_URL}/{day.year}/chirps-v2.0.{day.year}.{day.month:02d}.{day.day:02d}.tif.gz"


def local_gz_path(day: date) -> Path:
    year_dir = DOWNLOAD_DIR / str(day.year)
    year_dir.mkdir(exist_ok=True)
    return year_dir / f"chirps-v2.0.{day.year}.{day.month:02d}.{day.day:02d}.tif.gz"


def download_day(day: date) -> Path:
    gz_path = local_gz_path(day)
    if gz_path.exists():
        return gz_path

    url = chirps_url(day)
    try:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"CHIRPS download failed for {day} with HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"CHIRPS download failed for {day}: {exc.reason}") from exc

    gz_path.write_bytes(payload)
    return gz_path


# ==========================================================
# ZONAL EXTRACTION
# ==========================================================

def extract_day(day: date, gz_path: Path) -> pd.DataFrame:
    with gzip.open(gz_path, "rb") as f:
        tif_bytes = f.read()

    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            nodata = src.nodata if src.nodata is not None else -9999.0
            clipped, clipped_transform = mask(
                src,
                watershed_geometry,
                crop=True,
                all_touched=True,
                nodata=nodata,
            )

    band = clipped[0]
    rows, cols = np.where(~np.isclose(band, nodata))

    if rows.size == 0:
        raise RuntimeError(f"No CHIRPS pixels intersect the watershed on {day}.")

    values = band[rows, cols].astype(float)
    xs, ys = transform_xy(clipped_transform, rows, cols)

    day_max = values.max()

    return pd.DataFrame({
        "date": day,
        "pixel_lat": ys,
        "pixel_lon": xs,
        "precipitation_mm": values,
        "is_max": np.isclose(values, day_max),
    })


# ==========================================================
# MAIN LOOP
# ==========================================================

if __name__ == "__main__":
    pixel_rows = []
    daily_rows = []

    for day in daterange(START_DATE, END_DATE):
        print(f"Processing {day}...")
        gz_file = download_day(day)
        day_df = extract_day(day, gz_file)
        pixel_rows.append(day_df)

        daily_rows.append({
            "date": day,
            "precip_mean_mm": day_df["precipitation_mm"].mean(),
            "precip_max_mm": day_df["precipitation_mm"].max(),
            "precip_min_mm": day_df["precipitation_mm"].min(),
            "pixel_count": len(day_df),
        })

    pixel_df = pd.concat(pixel_rows, ignore_index=True)
    daily_df = pd.DataFrame(daily_rows)

    # ==========================================================
    # EXPORT CSVs
    # ==========================================================

    pixel_csv = Path("data/rawdata/chirps_pixels_watershed_Jan2024_Feb2025.csv")
    pixel_csv.parent.mkdir(parents=True, exist_ok=True)
    pixel_df.to_csv(pixel_csv, index=False)

    daily_csv = Path("data/clean/chirps_daily_watershed_Jan2024_Feb2025.csv")
    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(daily_csv, index=False)

    print(daily_df.head())
    print(f"Saved per-pixel data: {pixel_csv}")
    print(f"Saved daily summary: {daily_csv}")
