#!/usr/bin/env python3
"""
Example usage of a trained model to predict local wind speed.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifacts():
    scaler_path = Path("artifacts/scaler.pkl")
    features_path = Path("artifacts/feature_columns.json")
    target_mode_path = Path("artifacts/target_mode.txt")
    if not scaler_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Preprocessing artifacts missing. Run data preparation first."
        )
    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        feature_columns = json.load(f)
    target_mode = (
        target_mode_path.read_text().strip()
        if target_mode_path.exists()
        else "wind_speed"
    )
    return scaler, feature_columns, target_mode


def load_best_model():
    """Load a trained model (defaults to linear regression)."""
    for name in ["linear_regression", "ridge_regression", "random_forest"]:
        model_path = Path(f"models/{name}.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Loaded model: {model_path}")
            return model
    raise FileNotFoundError(
        "No trained model found in models/. Train the models first."
    )


def predict_local_wind_speed(
    model,
    forecast_speed,
    temperature,
    humidity,
    pressure,
    hour=12,
    day_of_week=2,
    month=6,
):
    # Temporal features
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Approximate lag features (use forecast values as proxy)
    wind_speed_lag1 = forecast_speed * 0.95  # Simulated 1h lag
    wind_speed_lag2 = forecast_speed * 0.9  # Simulated 2h lag
    wind_speed_mean_3 = forecast_speed * 0.98  # 3h mean (approx.)
    wind_speed_std_3 = forecast_speed * 0.1  # 3h std (approx.)

    # Create DataFrame with a superset of possible columns
    raw_features = {
        "forecast_speed": [forecast_speed],
        "temperature": [temperature],
        "humidity": [humidity],
        "pressure": [pressure],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "hour_sin": [np.sin(2 * np.pi * hour / 24)],
        "hour_cos": [np.cos(2 * np.pi * hour / 24)],
        "day_sin": [day_sin],
        "day_cos": [day_cos],
        "wind_speed_lag1": [wind_speed_lag1],
        "wind_speed_lag2": [wind_speed_lag2],
        "wind_speed_mean_3": [wind_speed_mean_3],
        "wind_speed_std_3": [wind_speed_std_3],
    }
    features_df = pd.DataFrame(raw_features)

    # Load artifacts, order columns, and apply identical scaling
    scaler, feature_columns, target_mode = load_artifacts()
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[feature_columns]
    features_scaled = pd.DataFrame(
        scaler.transform(features_df), columns=feature_columns
    )

    pred = model.predict(features_scaled)[0]
    if target_mode == "delta":
        return forecast_speed + pred
    return pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict local wind speed from forecast and weather conditions."
    )
    parser.add_argument(
        "--forecast_speed", type=float, help="Forecasted wind speed (m/s)"
    )
    parser.add_argument("--temperature", type=float, help="Temperature (°C)")
    parser.add_argument("--humidity", type=float, help="Humidity (%)")
    parser.add_argument("--pressure", type=float, help="Pressure (hPa)")
    parser.add_argument("--hour", type=int, default=12, help="Hour of day (0-23)")
    parser.add_argument("--day_of_week", type=int, default=2, help="0=Monday, 6=Sunday")
    parser.add_argument("--month", type=int, default=6, help="Month (1-12)")
    parser.add_argument(
        "--scenario_name", type=str, default="CLI Scenario", help="Name of the scenario"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed feature info"
    )
    return parser.parse_args()


def print_prediction(name, forecast_speed, temperature, humidity, prediction):
    print(f"\n{name}:")
    print(f"   Forecast speed: {forecast_speed} m/s")
    print(f"   Temp: {temperature}°C")
    print(f"   Humidity: {humidity}%")
    print(f"   Adjusted local speed: {prediction:.2f} m/s")
    diff = prediction - forecast_speed
    if diff > 0:
        print(f"   Correction: +{diff:.2f} m/s (stronger than forecast)")
    else:
        print(f"   Correction: {diff:.2f} m/s (weaker than forecast)")


def main():
    scaler, feature_columns, target_mode = load_artifacts()
    model = load_best_model()

    print("\n" + "=" * 60)
    print("LOCAL WIND SPEED PREDICTIONS")
    print("=" * 60)

    if len(sys.argv) > 1:
        args = parse_args()
        pred = predict_local_wind_speed(
            model,
            args.forecast_speed,
            args.temperature,
            args.humidity,
            args.pressure,
            args.hour,
            args.day_of_week,
            args.month,
        )
        print_prediction(
            args.scenario_name,
            args.forecast_speed,
            args.temperature,
            args.humidity,
            pred,
        )
    else:
        scenarios = [
            {
                "name": "Strong forecast in summer",
                "forecast_speed": 12.0,
                "temperature": 25.0,
                "humidity": 60.0,
                "pressure": 1015.0,
                "hour": 14,
                "day_of_week": 2,
                "month": 7,
            },
            {
                "name": "Moderate forecast in winter",
                "forecast_speed": 8.0,
                "temperature": 5.0,
                "humidity": 80.0,
                "pressure": 1020.0,
                "hour": 10,
                "day_of_week": 0,
                "month": 1,
            },
            {
                "name": "Weak forecast in spring",
                "forecast_speed": 4.0,
                "temperature": 15.0,
                "humidity": 70.0,
                "pressure": 1010.0,
                "hour": 16,
                "day_of_week": 4,
                "month": 4,
            },
        ]
        for scenario in scenarios:
            name = scenario.pop("name")
            prediction = predict_local_wind_speed(model, **scenario)
            print_prediction(
                name,
                scenario["forecast_speed"],
                scenario["temperature"],
                scenario["humidity"],
                prediction,
            )

    print("\n" + "=" * 60)
    print("Applications:")
    print("   • Forecast correction")
    print("   • Local adjustment of global models")
    print("   • Wind energy optimization")
    print("   • Site-specific accurate forecasts")
    print("\nUsage:")
    print("   python predict_example.py")
    print(
        "   python predict_example.py --forecast_speed 10 --temperature 20 --humidity 60 --pressure 1015 --hour 14 --day_of_week 2 --month 7"
    )


if __name__ == "__main__":
    main()
