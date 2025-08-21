#!/usr/bin/env python3
"""
Data preparation and feature engineering module.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for wind data."""

    def __init__(
        self,
        target_mode: str = "wind_speed",
        temporal_split: bool = True,
        artifacts_dir: str = "artifacts",
        temporal_features: bool = True,
        lag_features: bool = True,
        rolling_features: bool = True,
        rolling_window: int = 3,
    ):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = "wind_speed"
        self.target_mode = target_mode
        self.temporal_split = temporal_split
        self.artifacts_dir = Path(artifacts_dir)
        self.temporal_features_flag = temporal_features
        self.lag_features_flag = lag_features
        self.rolling_features_flag = rolling_features
        self.rolling_window = rolling_window

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        logger.info(f"Loading data: {filepath}")
        df = pd.read_csv(filepath)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features."""
        logger.info("Creating features...")

        df_features = df.copy()

        # Temporal features
        if self.temporal_features_flag and "timestamp" in df_features.columns:
            df_features["hour"] = df_features["timestamp"].dt.hour
            df_features["day_of_week"] = df_features["timestamp"].dt.dayofweek
            df_features["month"] = df_features["timestamp"].dt.month

            df_features["hour_sin"] = np.sin(2 * np.pi * df_features["hour"] / 24)
            df_features["hour_cos"] = np.cos(2 * np.pi * df_features["hour"] / 24)
            df_features["day_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
            df_features["day_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

        # Forecast - actual difference
        if (
            "forecast_speed" in df_features.columns
            and "wind_speed" in df_features.columns
        ):
            df_features["speed_diff"] = (
                df_features["forecast_speed"] - df_features["wind_speed"]
            )

        # Lags
        if self.lag_features_flag and "wind_speed" in df_features.columns:
            df_features["wind_speed_lag1"] = df_features["wind_speed"].shift(1)
            df_features["wind_speed_lag2"] = df_features["wind_speed"].shift(2)

        # Rolling statistics
        if self.rolling_features_flag and "wind_speed" in df_features.columns:
            df_features[f"wind_speed_mean_{self.rolling_window}"] = (
                df_features["wind_speed"]
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
            )
            df_features[f"wind_speed_std_{self.rolling_window}"] = (
                df_features["wind_speed"]
                .rolling(window=self.rolling_window, min_periods=1)
                .std()
                .fillna(0)
            )

        logger.info(f"Features created: {len(df_features.columns)} columns")
        return df_features

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features for training."""

        base_features = ["forecast_speed", "temperature", "humidity", "pressure"]

        temporal_features = (
            [
                "hour",
                "day_of_week",
                "month",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
            ]
            if self.temporal_features_flag
            else []
        )

        lag_features = (
            [
                "wind_speed_lag1",
                "wind_speed_lag2",
            ]
            if self.lag_features_flag
            else []
        )

        rolling_features = (
            [
                f"wind_speed_mean_{self.rolling_window}",
                f"wind_speed_std_{self.rolling_window}",
            ]
            if self.rolling_features_flag
            else []
        )

        agg_features = [
            "wind_speed_min",
            "wind_speed_max",
            "wind_speed_std",
        ]

        all_features = (
            base_features
            + temporal_features
            + lag_features
            + rolling_features
            + agg_features
        )
        available_features = [f for f in all_features if f in df.columns]

        self.feature_columns = available_features
        logger.info(f"Selected features: {len(available_features)}")
        logger.info(f"List: {available_features}")

        return df[available_features + [self.target_column]]

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training."""

        df_with_features = self.create_features(df)
        df_final = self.select_features(df_with_features)
        df_final = df_final.dropna()
        logger.info(f"Data after NaN drop: {len(df_final)} rows")

        X = df_final[self.feature_columns]
        if self.target_mode == "delta" and "forecast_speed" in df_final.columns:
            y = df_final["wind_speed"] - df_final["forecast_speed"]
        else:
            y = df_final[self.target_column]

        if self.temporal_split and "timestamp" in df.columns:
            df_sorted = df_with_features.sort_values("timestamp")
            df_sorted = self.select_features(df_sorted)
            df_sorted = df_sorted.dropna(
                subset=(self.feature_columns or []) + [self.target_column]
            )
            X = df_sorted[self.feature_columns]
            if self.target_mode == "delta" and "forecast_speed" in df_sorted.columns:
                y = df_sorted["wind_speed"] - df_sorted["forecast_speed"]
            else:
                y = df_sorted[self.target_column]
            split_index = int((1 - test_size) * len(X))
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        logger.info(f"Training rows: {len(X_train_scaled)}")
        logger.info(f"Test rows: {len(X_test_scaled)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = "data/processed",
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.artifacts_dir / "scaler.pkl")
        with open(self.artifacts_dir / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)
        with open(self.artifacts_dir / "target_mode.txt", "w") as f:
            f.write(self.target_mode)

        logger.info(f"Data saved to {output_dir}")
        logger.info(f"Artifacts saved to {self.artifacts_dir}")


def main():
    # Read config
    with open("configs/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    fe_params = params["data_preprocessing"]["feature_engineering"]

    preprocessor = DataPreprocessor(
        target_mode=params["data_preprocessing"]["target_mode"],
        temporal_split=params["data_preprocessing"]["temporal_split"],
        artifacts_dir=params["data_preprocessing"]["artifacts_dir"],
        temporal_features=fe_params.get("temporal_features", True),
        lag_features=fe_params.get("lag_features", True),
        rolling_features=fe_params.get("rolling_features", True),
        rolling_window=fe_params.get("rolling_window", 3),
    )

    df = preprocessor.load_data(params["data_preprocessing"]["input_path"])
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df,
        test_size=params["data_preprocessing"]["test_size"],
        random_state=params["data_preprocessing"]["random_state"],
    )
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)

    print("\n=== Prepared data ===")
    print(f"Used features: {preprocessor.feature_columns}")
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")


if __name__ == "__main__":
    main()
