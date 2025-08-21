"""
Temporal alignment between forecast and local measurements.

Key functions:
- align_dataframes: tolerance-based asof join + optional resampling
- CLI: read forecast CSV and measurements CSV, output aligned CSV
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class AlignmentConfig:
    forecast_path: str
    measurements_path: str
    output_path: str = "data/processed/aligned.csv"
    forecast_time_col: str = "timestamp"
    measure_time_col: str = "timestamp"
    tolerance_minutes: int = 30
    resample_rule: Optional[str] = None  # e.g., '1H'


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True)
    return s


def align_dataframes(
    forecast_df: pd.DataFrame,
    measure_df: pd.DataFrame,
    cfg: AlignmentConfig,
) -> pd.DataFrame:
    df_f = forecast_df.copy()
    df_m = measure_df.copy()

    df_f[cfg.forecast_time_col] = _to_datetime_utc(df_f[cfg.forecast_time_col])
    df_m[cfg.measure_time_col] = _to_datetime_utc(df_m[cfg.measure_time_col])

    if cfg.resample_rule:
        df_f = (
            df_f.set_index(cfg.forecast_time_col)
            .sort_index()
            .resample(cfg.resample_rule)
            .nearest()
        ).reset_index()
        df_m = (
            df_m.set_index(cfg.measure_time_col)
            .sort_index()
            .resample(cfg.resample_rule)
            .mean()
        ).reset_index()

    # Merge asof (nearest within tolerance)
    df_f = df_f.sort_values(cfg.forecast_time_col)
    df_m = df_m.sort_values(cfg.measure_time_col)

    aligned = pd.merge_asof(
        df_f,
        df_m,
        left_on=cfg.forecast_time_col,
        right_on=cfg.measure_time_col,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=cfg.tolerance_minutes),
        suffixes=("_forecast", "_measured"),
    )

    if "wind_speed_measured" in aligned.columns:
        aligned = aligned.rename(columns={"wind_speed_measured": "wind_speed"})

    aligned = (
        aligned.dropna(subset=["wind_speed"])
        if "wind_speed" in aligned.columns
        else aligned
    )

    return aligned


def _parse_args(argv):
    import argparse

    p = argparse.ArgumentParser(description="Align forecast and local measurements")
    p.add_argument(
        "--forecast", required=True, help="Forecast CSV (must contain 'timestamp')"
    )
    p.add_argument(
        "--measures", required=True, help="Measurements CSV (must contain 'timestamp')"
    )
    p.add_argument("--out", default="data/processed/aligned.csv")
    p.add_argument("--tolerance-min", type=int, default=30)
    p.add_argument("--resample", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    argv = argv or sys.argv[1:]
    args = _parse_args(argv)

    cfg = AlignmentConfig(
        forecast_path=args.forecast,
        measurements_path=args.measures,
        output_path=args.out,
        tolerance_minutes=args.tolerance_min,
        resample_rule=args.resample,
    )

    forecast_df = pd.read_csv(cfg.forecast_path)
    measures_df = pd.read_csv(cfg.measurements_path)
    aligned = align_dataframes(forecast_df, measures_df, cfg)

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(cfg.output_path, index=False)
    print(f"Aligned dataset saved: {cfg.output_path}")


if __name__ == "__main__":
    main()
