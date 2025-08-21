#!/usr/bin/env python3
"""
DVC script to generate simulated raw data.
"""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from wind_ml_project.data_generator import WindDataGenerator  # noqa: E402


def expand_to_fine_grain(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    """
    Simulate high-frequency measurements from hourly data.
    Each hourly step is expanded into several interpolated points with noise.
    """
    fine_data = []
    for _, row in df.iterrows():
        base_time = row["timestamp"]
        for offset in range(0, 60, freq_minutes):
            ts = base_time + timedelta(minutes=offset)
            # Add noise
            speed = max(row["wind_speed"] + (0.3 * (0.5 - offset / 60)), 0)
            fine_data.append(
                {
                    "timestamp": ts,
                    "wind_speed": round(speed, 2),
                }
            )
    return pd.DataFrame(fine_data)


def main():
    # Load params
    with open("configs/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    gen_params = params.get(
        "data_generation",
        {
            "start_date": "2025-01-01",
            "num_days": 60,
            "noise_level": 0.1,
            "seed": 42,
        },
    )

    # Generate synthetic hourly data
    generator = WindDataGenerator(seed=gen_params["seed"])
    df_hourly = generator.generate_dataset(
        start_date=gen_params.get("start_date", "2025-01-01"),
        days=gen_params.get("num_days", 60),
    )

    # Convert timestamp to UTC datetime
    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)

    # Simulate 5-minute raw data
    df_raw = expand_to_fine_grain(df_hourly, freq_minutes=5)

    # Save
    output_path = Path("data/raw/wind_data_raw.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(output_path, index=False)

    print(f"Simulated raw data: {len(df_raw)} points -> {output_path}")


if __name__ == "__main__":
    main()
