#!/usr/bin/env python3
"""
Aggregate high-frequency wind data into fixed-interval (hourly by default) statistics
for the ML pipeline.
"""

import argparse

import pandas as pd


def aggregate_hourly(input_csv: str, output_csv: str, freq: str = "1h") -> None:
    # Load raw CSV
    df = pd.read_csv(input_csv)

    # Parse timestamp (UTC) and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # Coerce numeric columns we expect
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    # Normalize frequency to lowercase to avoid pandas FutureWarning
    resample_freq = str(freq).lower()

    # Resample & aggregate
    resampler = df.resample(resample_freq)

    agg_df = pd.DataFrame()
    if "wind_speed" in df.columns:
        agg_df["wind_speed"] = resampler["wind_speed"].mean()
        agg_df["wind_speed_std"] = resampler["wind_speed"].std()
        agg_df["wind_speed_min"] = resampler["wind_speed"].min()
        agg_df["wind_speed_max"] = resampler["wind_speed"].max()

    # Save
    agg_df = agg_df.reset_index()
    agg_df.to_csv(output_csv, index=False)
    print(f"Aggregated data saved: {output_csv} ({len(agg_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate wind data to fixed intervals"
    )
    parser.add_argument(
        "--input", required=True, help="Path to raw high-frequency measurements CSV"
    )
    parser.add_argument("--output", required=True, help="Path to save aggregated CSV")
    parser.add_argument(
        "--freq", default="1h", help="Aggregation frequency (default: 1h)"
    )
    args = parser.parse_args()

    aggregate_hourly(args.input, args.output, args.freq)
