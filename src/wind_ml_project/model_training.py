"""
Training and evaluation module for local wind speed prediction models with MLflow tracking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def availability_metrics(y_true, y_pred, u=3.0):
    """Compute availability metrics for a given threshold u (m/s)."""
    y_true_cls = (y_true >= u).astype(int)
    y_pred_cls = (y_pred >= u).astype(int)
    return {
        "avail_accuracy": accuracy_score(y_true_cls, y_pred_cls),
        "avail_precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "avail_recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "avail_f1": f1_score(y_true_cls, y_pred_cls, zero_division=0),
    }


class ModelTrainer:
    """ML model trainer for local wind speed prediction with MLflow tracking."""

    def __init__(
        self,
        experiment_name: str = "wind_speed_prediction",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            client = MlflowClient()
            existing = mlflow.get_experiment_by_name(experiment_name)

            if existing and getattr(existing, "lifecycle_stage", None) == "deleted":
                client.restore_experiment(existing.experiment_id)

            if existing is None:
                try:
                    client.create_experiment(experiment_name)
                except Exception:
                    pass

            mlflow.set_experiment(experiment_name)
        except MlflowException as e:
            logger.warning(
                "MLflow: problem selecting/creating experiment '%s': %s",
                experiment_name,
                e,
            )
            mlflow.set_experiment(experiment_name)

    def get_models(self) -> Dict[str, Any]:
        """Return a dictionary of baseline models to evaluate."""
        models = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }
        return models

    def evaluate_model(
        self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if isinstance(y_true, pd.Series):
            from typing import cast
            y_true = cast(np.ndarray, y_true.values)

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        return metrics

    def train_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Train a model and log results with MLflow."""

        logger.info(f"Training model: {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_metrics = self.evaluate_model(y_train, y_pred_train)
            test_metrics = self.evaluate_model(y_test, y_pred_test)

            # Availability metrics (test set)
            avail = availability_metrics(y_test, y_pred_test, u=3.0)
            test_metrics.update(avail)

            # Log params
            if hasattr(model, "get_params"):
                params = model.get_params()
                for param, value in params.items():
                    mlflow.log_param(param, value)

            # Log all metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)

            # Log model artifact
            import mlflow.sklearn as mlflow_sklearn
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train.iloc[:5]
            mlflow_sklearn.log_model(
                sk_model=model,
                name=f"model_{model_name}",
                signature=signature,
                input_example=input_example,
            )

            # Save local model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            joblib.dump(model, model_dir / f"{model_name}.pkl")

            # Store results in memory
            self.models[model_name] = model
            self.results[model_name] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "predictions": {"train": y_pred_train, "test": y_pred_test},
            }

            logger.info(f"Model {model_name} - Test RMSE: {test_metrics['rmse']:.3f}")

        return test_metrics

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict]:
        """Train all baseline models."""
        logger.info("Beginning training of all models...")
        models = self.get_models()
        all_results = {}
        for model_name, model in models.items():
            try:
                results = self.train_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error occurred while training {model_name}: {e}")
                continue
        return all_results

    def get_best_model(self, metric: str = "rmse") -> Tuple[str, Any]:
        """Return the best model by a given metric."""
        if not self.results:
            raise ValueError("No model has been trained")
        best_score = float("inf")
        best_model_name = None
        for model_name, results in self.results.items():
            score = results["test_metrics"][metric]
            if score < best_score:
                best_score = score
                best_model_name = model_name
        if best_model_name is None:
            raise ValueError("No valid model found")
        return best_model_name, self.models[best_model_name]

    def print_comparison(self):
        """Print a comparison of trained models."""
        if not self.results:
            print("No results to display")
            return
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 70)
        for model_name, results in self.results.items():
            metrics = results["test_metrics"]
            print(
                f"{model_name:<20} "
                f"{metrics['rmse']:<10.3f} "
                f"{metrics['mae']:<10.3f} "
                f"{metrics['r2']:<10.3f}"
            )
        best_name, _ = self.get_best_model()
        print("-" * 70)
        print(f"Best model: {best_name}")

    def save_results(self, filepath: str = "results/model_comparison.yaml"):
        """Save results to a YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_data = {}
        for model_name, results in self.results.items():
            tm = results["train_metrics"]
            te = results["test_metrics"]
            save_data[model_name] = {
                "train_metrics": {
                    "rmse": float(tm["rmse"]),
                    "mae": float(tm["mae"]),
                    "r2": float(tm["r2"]),
                },
                "test_metrics": {
                    "rmse": float(te["rmse"]),
                    "mae": float(te["mae"]),
                    "r2": float(te["r2"]),
                    "avail_accuracy": float(te["avail_accuracy"]),
                    "avail_precision": float(te["avail_precision"]),
                    "avail_recall": float(te["avail_recall"]),
                    "avail_f1": float(te["avail_f1"]),
                },
            }
        with open(filepath, "w") as f:
            yaml.dump(save_data, f, default_flow_style=False)
        logger.info(f"Results saved: {filepath}")


def main():
    """Manual test of the trainer using prepared data files."""
    logging.basicConfig(level=logging.INFO)
    try:
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        from typing import cast
        y_train = cast(pd.Series, y_train)
        y_test = cast(pd.Series, y_test)
        print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    except FileNotFoundError:
        print("Error: Prepared data not found. Run data preparation first.")
        return

    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.print_comparison()
    trainer.save_results()
    print("\nTraining pipeline completed!")
    print("Open MLflow UI with: mlflow ui")


if __name__ == "__main__":
    main()
