from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import rasterio
from pyproj import CRS, Transformer

if not hasattr(np, "in1d"):
    def _np_in1d(ar1, ar2, assume_unique: bool = False, invert: bool = False):
        return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)

    np.in1d = _np_in1d

from pysheds.grid import Grid
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

WGS84 = CRS.from_epsg(4326)
SRTM30 = "srtm30"
COPERNICUS30 = "copernicus30"


@dataclass
class WatershedConfig:
    outlet_lat: float
    outlet_lon: float
    bbox_margin_deg: float = 0.35
    accumulation_threshold: float = 800.0
    snap_distance_m: float = 1500.0
    snap_warning_distance_m: float = 1200.0
    snapping_enabled: bool = True
    png_dpi: int = 200
    dem_source: str = SRTM30
    opentopography_api_key: str | None = None
    output_dir: Path = Path("output/corso_watershed")


def validate_outlet_coordinates(lat: float, lon: float) -> None:
    if lat is None or lon is None:
        raise ValueError("Outlet coordinates are required.")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Outlet latitude {lat} is outside the valid range [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Outlet longitude {lon} is outside the valid range [-180, 180].")


def validate_config(cfg: WatershedConfig) -> None:
    validate_outlet_coordinates(cfg.outlet_lat, cfg.outlet_lon)

    if cfg.bbox_margin_deg <= 0:
        raise ValueError("DEM download margin must be greater than 0 degrees.")
    if cfg.accumulation_threshold <= 0:
        raise ValueError("Flow accumulation threshold must be greater than 0.")
    if cfg.snap_distance_m <= 0:
        raise ValueError("Maximum snapping distance must be greater than 0 meters.")
    if cfg.snap_warning_distance_m <= 0:
        raise ValueError("Snap warning distance must be greater than 0 meters.")
    if cfg.png_dpi <= 0:
        raise ValueError("PNG DPI must be greater than 0.")
    if cfg.dem_source not in {SRTM30, COPERNICUS30}:
        raise ValueError(f"Unsupported DEM source '{cfg.dem_source}'.")


def choose_utm_crs(lon: float, lat: float) -> CRS:
    if not (-80.0 <= lat <= 84.0):
        raise RuntimeError(
            f"Latitude {lat} is outside UTM coverage [-80, 84]. "
            "A local projected CRS cannot be selected automatically."
        )
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_bbox(lon: float, lat: float, margin_deg: float) -> tuple[float, float, float, float]:
    return (
        lon - margin_deg,
        lat - margin_deg,
        lon + margin_deg,
        lat + margin_deg,
    )


def _pick_dem_asset(item) -> str:
    preferred_keys = ("data", "dem")
    for key in preferred_keys:
        if key in item.assets:
            return planetary_computer.sign(item.assets[key].href)

    for asset in item.assets.values():
        href = asset.href.lower()
        if href.endswith(".tif") or href.endswith(".tiff"):
            return planetary_computer.sign(asset.href)

    raise RuntimeError(f"Could not find a DEM GeoTIFF asset for item {item.id}")


def _download_srtm30_from_opentopography(
    bbox_4326: tuple[float, float, float, float],
    output_path: Path,
    api_key: str | None,
) -> Path:
    if not api_key:
        raise RuntimeError(
            "SRTM download requested but OpenTopography API key is missing. "
            "Set OPENTOPOGRAPHY_API_KEY or pass opentopography_api_key."
        )

    west, south, east, north = bbox_4326
    query = urlencode(
        {
            "demtype": "SRTMGL1",
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
    )
    url = f"https://portal.opentopography.org/API/globaldem?{query}"
    request = Request(url, headers={"User-Agent": "desalination-modeling-project/0.1"})

    try:
        with urlopen(request, timeout=180) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        raise RuntimeError(f"SRTM download failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"SRTM download failed: {exc.reason}") from exc

    if "xml" in content_type.lower() or "json" in content_type.lower() or "text" in content_type.lower():
        snippet = payload.decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"SRTM provider returned an error payload: {snippet}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    try:
        with rasterio.open(output_path):
            pass
    except Exception as exc:
        raise RuntimeError("SRTM payload was downloaded but is not a valid GeoTIFF raster.") from exc

    return output_path


def download_copernicus_dem(bbox_4326: tuple[float, float, float, float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox_4326)
    items = list(search.items())

    if not items:
        raise RuntimeError(
            "No Copernicus DEM tiles were found for the requested area. "
            "Try increasing bbox_margin_deg."
        )

    sources = []
    try:
        for item in items:
            sources.append(rasterio.open(_pick_dem_asset(item)))

        mosaic, transform = merge(sources, bounds=bbox_4326)
        profile = sources[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "compress": "deflate",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in sources:
            src.close()

    return output_path


def download_dem(
    bbox_4326: tuple[float, float, float, float],
    output_path: Path,
    source: str,
    opentopography_api_key: str | None,
) -> dict[str, Any]:
    if source == COPERNICUS30:
        download_copernicus_dem(bbox_4326, output_path)
        return {
            "requested_source": COPERNICUS30,
            "used_source": COPERNICUS30,
            "fallback_triggered": False,
            "fallback_reason": None,
        }

    if source != SRTM30:
        raise RuntimeError(f"Unsupported DEM source '{source}'.")

    try:
        _download_srtm30_from_opentopography(bbox_4326, output_path, opentopography_api_key)
        return {
            "requested_source": SRTM30,
            "used_source": SRTM30,
            "fallback_triggered": False,
            "fallback_reason": None,
        }
    except Exception as srtm_error:
        try:
            download_copernicus_dem(bbox_4326, output_path)
        except Exception as cop_error:
            raise RuntimeError(
                "DEM download failed. SRTM attempt failed and Copernicus fallback also failed. "
                f"SRTM error: {srtm_error}. Copernicus error: {cop_error}"
            ) from cop_error

        return {
            "requested_source": SRTM30,
            "used_source": COPERNICUS30,
            "fallback_triggered": True,
            "fallback_reason": str(srtm_error),
        }


def reproject_raster_to_crs(src_path: Path, dst_path: Path, dst_crs: CRS) -> Path:
    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise RuntimeError("Input DEM has no CRS metadata.")

            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
            )
            profile = src.profile.copy()
            nodata = src.nodata if src.nodata is not None else -9999
            profile.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": nodata,
                    "compress": "deflate",
                }
            )

            with rasterio.open(dst_path, "w", **profile) as dst:
                for band_index in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_index),
                        destination=rasterio.band(dst, band_index),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        src_nodata=src.nodata,
                        dst_nodata=nodata,
                        resampling=Resampling.bilinear,
                    )
    except Exception as exc:
        raise RuntimeError(f"DEM reprojection failed: {exc}") from exc

    return dst_path


def validate_projected_dem_crs(projected_dem_path: Path, expected_crs: CRS) -> str:
    with rasterio.open(projected_dem_path) as src:
        if src.crs is None:
            raise RuntimeError("CRS check failed: projected DEM has no CRS metadata.")
        if src.crs != expected_crs:
            raise RuntimeError(
                "CRS check failed: projected DEM CRS does not match target CRS. "
                f"Expected {expected_crs.to_string()}, got {src.crs.to_string()}."
            )
        return src.crs.to_string()


def point_transformer(src_crs: CRS | str, dst_crs: CRS | str) -> Transformer:
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def snap_to_stream(
    accumulation: np.ndarray,
    affine,
    outlet_x: float,
    outlet_y: float,
    accumulation_threshold: float,
    max_snap_distance_m: float,
) -> tuple[float, float, dict[str, Any]]:
    acc = np.asarray(accumulation, dtype=float)
    if acc.size == 0:
        raise RuntimeError("Flow accumulation raster is empty.")

    rows, cols = np.indices(acc.shape)
    xs = affine.c + (cols + 0.5) * affine.a + (rows + 0.5) * affine.b
    ys = affine.f + (cols + 0.5) * affine.d + (rows + 0.5) * affine.e

    valid = np.isfinite(acc)
    if not np.any(valid):
        raise RuntimeError("Flow accumulation raster has no finite values.")

    distances = np.sqrt((xs - outlet_x) ** 2 + (ys - outlet_y) ** 2)
    stream_candidates = valid & (acc >= accumulation_threshold)
    nearby_streams = stream_candidates & (distances <= max_snap_distance_m)

    fallback_triggered = False
    snap_method = "threshold_stream"
    nearby_stream_count = int(np.count_nonzero(nearby_streams))

    if nearby_stream_count > 0:
        candidate_distances = np.where(nearby_streams, distances, np.inf)
        row, col = np.unravel_index(np.argmin(candidate_distances), candidate_distances.shape)
    else:
        fallback_triggered = True
        snap_method = "max_accumulation_fallback"
        nearby_cells = valid & (distances <= max_snap_distance_m)
        if not np.any(nearby_cells):
            raise RuntimeError(
                "No valid snapping candidate found within snap_distance_m. "
                "Increase snap_distance_m or bbox_margin_deg."
            )
        candidate_acc = np.where(nearby_cells, acc, -np.inf)
        row, col = np.unravel_index(np.argmax(candidate_acc), candidate_acc.shape)

    snapped_x = float(xs[row, col])
    snapped_y = float(ys[row, col])
    snap_distance = float(math.hypot(snapped_x - outlet_x, snapped_y - outlet_y))
    snapped_accumulation = float(acc[row, col])

    return snapped_x, snapped_y, {
        "snap_distance_m": snap_distance,
        "snapped_accumulation": snapped_accumulation,
        "candidate_stream_cells": int(np.count_nonzero(stream_candidates)),
        "candidate_stream_cells_within_radius": nearby_stream_count,
        "fallback_triggered": fallback_triggered,
        "snap_method": snap_method,
    }


def polygonize_mask(mask: np.ndarray, affine):
    polygons = [
        shape(geom)
        for geom, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=affine)
        if value == 1
    ]
    if not polygons:
        raise RuntimeError("Polygonization failed: watershed mask is empty.")

    polygon = unary_union(polygons).buffer(0)
    if polygon.is_empty:
        raise RuntimeError("Polygonization failed: watershed polygon is empty after cleanup.")

    if polygon.geom_type == "GeometryCollection":
        polygon_parts = [part for part in polygon.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        if not polygon_parts:
            raise RuntimeError("Polygonization failed: no polygon geometry was produced.")
        polygon = unary_union(polygon_parts).buffer(0)

    if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
        raise RuntimeError(f"Polygonization failed: unexpected geometry type '{polygon.geom_type}'.")

    if not polygon.is_valid:
        raise RuntimeError("Polygonization failed: watershed polygon is invalid.")

    return polygon


def touches_array_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def save_geojson(features: Iterable[dict], output_path: Path) -> None:
    geojson = {"type": "FeatureCollection", "features": list(features)}
    try:
        output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Output writing failed for '{output_path}': {exc}") from exc


def plot_results(
    dem: np.ndarray,
    dem_nodata: float | None,
    affine,
    watershed_polygon,
    outlet_point,
    snapped_point,
    snapping_enabled: bool,
    png_path: Path,
    png_dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    image = np.asarray(dem, dtype=float)
    if dem_nodata is not None:
        image = np.where(np.isclose(image, dem_nodata), np.nan, image)
    image = np.ma.masked_invalid(image)

    left, bottom, right, top = array_bounds(image.shape[0], image.shape[1], affine)
    ax.imshow(image, extent=[left, right, bottom, top], origin="upper", cmap="terrain")

    if watershed_polygon.geom_type == "Polygon":
        xs, ys = watershed_polygon.exterior.xy
        ax.plot(xs, ys, color="crimson", linewidth=2, label="Watershed boundary")
    else:
        first = True
        for polygon in watershed_polygon.geoms:
            xs, ys = polygon.exterior.xy
            label = "Watershed boundary" if first else None
            ax.plot(xs, ys, color="crimson", linewidth=2, label=label)
            first = False

    ax.scatter([outlet_point.x], [outlet_point.y], color="yellow", edgecolor="black", s=70, label="Original outlet")
    snapped_label = "Snapped outlet" if snapping_enabled else "Final outlet (snapping disabled)"
    ax.scatter([snapped_point.x], [snapped_point.y], color="red", edgecolor="black", s=70, label=snapped_label)

    ax.set_title("Watershed delineation")
    ax.set_xlabel("Projected X (m)")
    ax.set_ylabel("Projected Y (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=png_dpi)
    plt.close(fig)


def _delineate_watershed(
    projected_dem_path: Path,
    outlet_lon: float,
    outlet_lat: float,
    cfg: WatershedConfig,
    dem_download_info: dict[str, Any],
) -> dict[str, Any]:
    projected_crs = choose_utm_crs(outlet_lon, outlet_lat)
    to_projected = point_transformer(WGS84, projected_crs)
    to_wgs84 = point_transformer(projected_crs, WGS84)

    outlet_x, outlet_y = to_projected.transform(outlet_lon, outlet_lat)
    original_outlet = Point(outlet_x, outlet_y)

    with rasterio.open(projected_dem_path) as src:
        dem_for_plot = src.read(1)
        dem_nodata = src.nodata

    grid = Grid.from_raster(str(projected_dem_path))
    dem = grid.read_raster(str(projected_dem_path))

    try:
        filled_pits = grid.fill_pits(dem)
        flooded = grid.fill_depressions(filled_pits)
        inflated = grid.resolve_flats(flooded)
    except Exception as exc:
        raise RuntimeError(f"Hydrologic preprocessing failed: {exc}") from exc

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    try:
        flow_direction = grid.flowdir(inflated, dirmap=dirmap)
        accumulation = grid.accumulation(flow_direction, dirmap=dirmap)
    except Exception as exc:
        raise RuntimeError(f"Flow routing computation failed: {exc}") from exc

    if cfg.snapping_enabled:
        snapped_x, snapped_y, snap_info = snap_to_stream(
            accumulation=np.asarray(accumulation),
            affine=grid.affine,
            outlet_x=outlet_x,
            outlet_y=outlet_y,
            accumulation_threshold=cfg.accumulation_threshold,
            max_snap_distance_m=cfg.snap_distance_m,
        )
    else:
        snapped_x, snapped_y = outlet_x, outlet_y
        snap_info = {
            "snap_distance_m": 0.0,
            "snapped_accumulation": None,
            "candidate_stream_cells": None,
            "candidate_stream_cells_within_radius": None,
            "fallback_triggered": False,
            "snap_method": "disabled",
        }

    snapped_outlet = Point(snapped_x, snapped_y)

    catchment = grid.catchment(
        x=snapped_x,
        y=snapped_y,
        fdir=flow_direction,
        dirmap=dirmap,
        xytype="coordinate",
    )
    catchment_mask = np.asarray(catchment).astype(bool)
    if not np.any(catchment_mask):
        raise RuntimeError("Watershed mask is empty.")

    watershed_polygon = polygonize_mask(catchment_mask, grid.affine)

    watershed_wgs84 = shapely_transform(to_wgs84.transform, watershed_polygon)
    snapped_outlet_wgs84 = shapely_transform(to_wgs84.transform, snapped_outlet)
    original_outlet_wgs84 = shapely_transform(to_wgs84.transform, original_outlet)

    if watershed_wgs84.is_empty:
        raise RuntimeError("Geometry check failed: watershed polygon is empty.")
    if not watershed_wgs84.is_valid:
        raise RuntimeError("Geometry check failed: watershed polygon is invalid.")

    area_km2 = watershed_polygon.area / 1_000_000.0
    edge_warning = touches_array_edge(catchment_mask)
    large_snap_warning = bool(
        cfg.snapping_enabled and snap_info["snap_distance_m"] > cfg.snap_warning_distance_m
    )

    watershed_geojson_path = cfg.output_dir / "watershed.geojson"
    outlet_geojson_path = cfg.output_dir / "outlet_snapped.geojson"
    png_path = cfg.output_dir / "watershed.png"
    report_path = cfg.output_dir / "report.json"

    save_geojson(
        [
            {
                "type": "Feature",
                "properties": {
                    "area_km2": area_km2,
                    "projected_crs": projected_crs.to_string(),
                    "edge_warning": edge_warning,
                },
                "geometry": mapping(watershed_wgs84),
            }
        ],
        watershed_geojson_path,
    )

    save_geojson(
        [
            {
                "type": "Feature",
                "properties": {
                    "kind": "original_outlet",
                    "crs": "EPSG:4326",
                },
                "geometry": mapping(original_outlet_wgs84),
            },
            {
                "type": "Feature",
                "properties": {
                    "kind": "snapped_outlet",
                    "crs": "EPSG:4326",
                    **snap_info,
                },
                "geometry": mapping(snapped_outlet_wgs84),
            },
        ],
        outlet_geojson_path,
    )

    plot_results(
        dem=dem_for_plot,
        dem_nodata=dem_nodata,
        affine=grid.affine,
        watershed_polygon=watershed_polygon,
        outlet_point=original_outlet,
        snapped_point=snapped_outlet,
        snapping_enabled=cfg.snapping_enabled,
        png_path=png_path,
        png_dpi=cfg.png_dpi,
    )

    warnings: list[str] = []
    if edge_warning:
        warnings.append(
            "Watershed touches the edge of the DEM extent; basin may be truncated. "
            "Increase bbox_margin_deg and rerun."
        )
    if large_snap_warning:
        warnings.append(
            "Snapping distance is larger than the warning threshold. "
            "Visually inspect outlet location and flow accumulation settings."
        )
    if bool(snap_info.get("fallback_triggered")):
        warnings.append(
            "Snapping fallback was triggered: no threshold-based stream candidate was found "
            "within the snapping radius."
        )
    if bool(dem_download_info.get("fallback_triggered")):
        warnings.append(
            "DEM source fallback was triggered from SRTM to Copernicus. "
            "Provide an OpenTopography API key to use SRTM directly."
        )

    report = {
        "outlet_input": {
            "latitude": outlet_lat,
            "longitude": outlet_lon,
            "crs": "EPSG:4326",
            "coordinate_order": "latitude, longitude",
        },
        "outlet_final": {
            "latitude": snapped_outlet_wgs84.y,
            "longitude": snapped_outlet_wgs84.x,
            "crs": "EPSG:4326",
        },
        "working_crs": projected_crs.to_string(),
        "area_km2": area_km2,
        "snapping": {
            "enabled": cfg.snapping_enabled,
            "max_distance_m": cfg.snap_distance_m,
            "warning_distance_m": cfg.snap_warning_distance_m,
            "accumulation_threshold": cfg.accumulation_threshold,
            **snap_info,
            "large_snap_warning": large_snap_warning,
        },
        "dem": {
            **dem_download_info,
            "download_margin_deg": cfg.bbox_margin_deg,
        },
        "validation": {
            "edge_truncation_warning": edge_warning,
            "geometry_valid": bool(watershed_wgs84.is_valid),
        },
        "parameters": {
            "bbox_margin_deg": cfg.bbox_margin_deg,
            "accumulation_threshold": cfg.accumulation_threshold,
            "snap_distance_m": cfg.snap_distance_m,
            "snap_warning_distance_m": cfg.snap_warning_distance_m,
            "snapping_enabled": cfg.snapping_enabled,
            "png_dpi": cfg.png_dpi,
            "dem_source": cfg.dem_source,
        },
        "warnings": warnings,
        "files": {
            "dem_4326": str(cfg.output_dir / "dem_4326.tif"),
            "dem_projected": str(cfg.output_dir / "dem_projected.tif"),
            "watershed_geojson": str(watershed_geojson_path),
            "outlet_geojson": str(outlet_geojson_path),
            "watershed_png": str(png_path),
            "report_json": str(report_path),
        },
    }

    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Output writing failed for '{report_path}': {exc}") from exc

    return report


def delineate_corso_watershed(
    outlet_lat: float,
    outlet_lon: float,
    *,
    bbox_margin_deg: float = 0.35,
    accumulation_threshold: float = 800.0,
    snap_distance_m: float = 1500.0,
    snap_warning_distance_m: float = 1200.0,
    snapping_enabled: bool = True,
    png_dpi: int = 200,
    dem_source: str = SRTM30,
    opentopography_api_key: str | None = None,
    output_dir: str | Path = Path("output/corso_watershed"),
) -> dict[str, Any]:
    """Download DEM data and delineate a watershed from an outlet point.

    Returns a report dictionary and writes raster/vector/PNG outputs into output_dir.
    """
    cfg = WatershedConfig(
        outlet_lat=outlet_lat,
        outlet_lon=outlet_lon,
        bbox_margin_deg=bbox_margin_deg,
        accumulation_threshold=accumulation_threshold,
        snap_distance_m=snap_distance_m,
        snap_warning_distance_m=snap_warning_distance_m,
        snapping_enabled=snapping_enabled,
        png_dpi=png_dpi,
        dem_source=dem_source,
        opentopography_api_key=opentopography_api_key or os.getenv("OPENTOPOGRAPHY_API_KEY"),
        output_dir=Path(output_dir),
    )
    validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    bbox_4326 = make_bbox(cfg.outlet_lon, cfg.outlet_lat, cfg.bbox_margin_deg)
    projected_crs = choose_utm_crs(cfg.outlet_lon, cfg.outlet_lat)

    raw_dem_path = cfg.output_dir / "dem_4326.tif"
    projected_dem_path = cfg.output_dir / "dem_projected.tif"

    dem_download_info = download_dem(
        bbox_4326=bbox_4326,
        output_path=raw_dem_path,
        source=cfg.dem_source,
        opentopography_api_key=cfg.opentopography_api_key,
    )

    reproject_raster_to_crs(raw_dem_path, projected_dem_path, projected_crs)
    validate_projected_dem_crs(projected_dem_path, projected_crs)

    return _delineate_watershed(projected_dem_path, cfg.outlet_lon, cfg.outlet_lat, cfg, dem_download_info)


