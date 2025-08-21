"""
Open-Meteo client (free, no API key) – normalized hourly forecast.

Normalized columns:
- timestamp (UTC)
- forecast_speed (m/s)
- temperature (°C)
- humidity (%)
- pressure (hPa)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


@dataclass
class OpenMeteoConfig:
    lat: float
    lon: float
    hours: int = 48


def fetch_open_meteo_hourly(lat: float, lon: float) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,temperature_2m,relativehumidity_2m,pressure_msl",
        "windspeed_unit": "ms",  # m/s
        "timeformat": "unixtime",
        "timezone": "UTC",
    }
    resp = requests.get(base, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def normalize_open_meteo_hourly(
    json_data: Dict[str, Any], max_hours: int = 48
) -> pd.DataFrame:
    hourly = json_data.get("hourly", {})
    times = hourly.get("time", [])
    ws = hourly.get("windspeed_10m", [])
    t2m = hourly.get("temperature_2m", [])
    rh = hourly.get("relativehumidity_2m", [])
    pmsl = hourly.get("pressure_msl", [])

    n = min(max_hours, len(times))
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times[:n], unit="s", utc=True),
            "forecast_speed": ws[:n],
            "temperature": t2m[:n],
            "humidity": rh[:n],
            "pressure": pmsl[:n],
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def save_dataframe_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(
        description="Fetch Open-Meteo hourly forecast and save CSV"
    )
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--out", type=str, default="data/raw/forecast_openmeteo.csv")
    args = p.parse_args(argv)
    return vars(args)


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv or sys.argv[1:]
    args = _parse_args(argv)

    js = fetch_open_meteo_hourly(args["lat"], args["lon"])
    df = normalize_open_meteo_hourly(js, max_hours=args["hours"])
    path = save_dataframe_csv(df, args["out"])
    print(f"Open-Meteo hourly forecast saved: {path}")


if __name__ == "__main__":
    main()
