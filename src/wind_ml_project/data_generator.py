"""
Synthetic data generator for testing the ML pipeline.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration of logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WindDataGenerator:
    """Realistic synthetic wind data generator."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

    def generate_daily_pattern(self, hours: int = 24) -> np.ndarray:
        """Generate a typical daily wind pattern."""
        # Typical pattern: stronger in the afternoon, calmer at night
        time_hours = np.linspace(0, 24, hours)
        base_pattern = 2 + 3 * np.sin(2 * np.pi * (time_hours - 6) / 24)
        return np.maximum(base_pattern, 0.5)  # Minimum 0.5 m/s

    def add_noise(self, values: np.ndarray, noise_factor: float = 0.3) -> np.ndarray:
        """Add realistic noise to data."""
        noise = np.random.normal(0, noise_factor, len(values))
        return np.maximum(values + noise, 0.1)  # Minimum 0.1 m/s

    

    def add_forecast_bias(
        self, actual_values: np.ndarray, bias_factor: float = 0.1
    ) -> np.ndarray:
        """Add a realistic bias to forecast values."""
        # Forecasts are typically slightly different from actuals
        bias = np.random.normal(1, bias_factor, len(actual_values))
        forecast_noise = np.random.normal(0, 0.2, len(actual_values))
        return np.maximum(actual_values * bias + forecast_noise, 0.1)

    def generate_dataset(
        self, start_date: str = "2025-01-01", days: int = 30, hours_per_day: int = 24
    ) -> pd.DataFrame:
        """
        Generate a full wind dataset.

        Args:
            start_date: Start date (YYYY-MM-DD)
            days: Number of days to generate
            hours_per_day: Number of points per day

        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating {days} days of data from {start_date}")

        # Creation of timestamps
        start = datetime.strptime(start_date, "%Y-%m-%d")
        timestamps = []

        for day in range(days):
            current_day = start + timedelta(days=day)
            for hour in range(0, 24, 24 // hours_per_day):
                timestamps.append(current_day + timedelta(hours=hour))

        n_points = len(timestamps)

        # Generate daily patterns
        daily_pattern = self.generate_daily_pattern(hours_per_day)

        # Repeat the pattern for all days with variations
        wind_speeds = []
        for day in range(days):
            # Daily variation factor (changing weather)
            day_factor = np.random.uniform(0.7, 1.3)
            daily_speeds = daily_pattern * day_factor
            wind_speeds.extend(daily_speeds)

        # Add realistic noise
        wind_speeds = self.add_noise(np.array(wind_speeds))

        # Generate forecasts with bias
        forecast_speeds = self.add_forecast_bias(wind_speeds)

        # Create DataFrame
        data = {
            "timestamp": timestamps,
            "wind_speed": wind_speeds,
            "forecast_speed": forecast_speeds,
            "temperature": np.random.normal(15, 8, n_points),  # Temperature in °C
            "humidity": np.random.uniform(30, 90, n_points),  # Humidity in %
            "pressure": np.random.normal(1013, 15, n_points),  # Pressure in hPa
        }

        df = pd.DataFrame(data)

        # Round values for realism
        df["wind_speed"] = df["wind_speed"].round(1)
        df["forecast_speed"] = df["forecast_speed"].round(1)
        df["temperature"] = df["temperature"].round(1)
        df["humidity"] = df["humidity"].round(0)
        df["pressure"] = df["pressure"].round(1)

        logger.info(f"Generated dataset: {len(df)} data points")
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved: {filepath}")


def main():
    """Generate and save a test dataset."""
    generator = WindDataGenerator()

    # Generate 30 days of data
    df = generator.generate_dataset(start_date="2025-01-01", days=30, hours_per_day=24)

    # Save
    output_path = "data/raw/wind_data.csv"
    generator.save_dataset(df, output_path)

    # Display stats
    print("\n=== Generated dataset ===")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data points: {len(df)}")
    print("\nWind speed stats:")
    print(df[["wind_speed", "forecast_speed"]].describe())


if __name__ == "__main__":
    main()
