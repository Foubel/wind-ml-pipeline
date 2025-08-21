"""
Main pipeline to train local wind speed prediction models.
"""

import logging
from pathlib import Path

import yaml

from wind_ml_project.data_generator import WindDataGenerator
from wind_ml_project.data_preprocessing import DataPreprocessor
from wind_ml_project.model_training import ModelTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WindMLPipeline:
    """End-to-end pipeline for local wind speed prediction models."""

    def __init__(self, config_path: str = "configs/params.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from a YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded: {}".format(self.config_path))
            return config
        except FileNotFoundError:
            logger.warning(
                "Config not found: {}, using default values".format(self.config_path)
            )
            return {
                "data": {"raw_path": "data/raw/fake_day.csv"},
                "model": {"test_size": 0.2, "random_state": 42, "name": "pipeline_run"},
            }

    def step_1_generate_data(self, force_regenerate: bool = False):
        """Step 1: Generate synthetic data"""
        logger.info("=== STEP 1: Data generation ===")

        raw_path = self.config["data"]["raw_path"]

        if Path(raw_path).exists() and not force_regenerate:
            logger.info(f"Existing data found: {raw_path}")
            return

        logger.info("Generating new synthetic data...")
        generator = WindDataGenerator(seed=42)

        # Generate 60 days of data (2 months)
        df = generator.generate_dataset(
            start_date="2025-01-01", days=60, hours_per_day=24
        )

        # Save
        generator.save_dataset(df, raw_path)
        logger.info(f"Data generated: {len(df)} points")

    def step_2_preprocess_data(self):
        """Step 2: Data preparation"""
        logger.info("=== STEP 2: Data preparation ===")

        # Read options from config
        target_mode = self.config.get("data_preprocessing", {}).get(
            "target_mode", "wind_speed"
        )
        temporal_split = self.config.get("data_preprocessing", {}).get(
            "temporal_split", True
        )
        artifacts_dir = self.config.get("data_preprocessing", {}).get(
            "artifacts_dir", "artifacts"
        )
        preprocessor = DataPreprocessor(
            target_mode=target_mode,
            temporal_split=temporal_split,
            artifacts_dir=artifacts_dir,
        )

        # Load data (prefer aligned data if available)
        raw_path = self.config["data"]["raw_path"]
        aligned_candidate = self.config.get("alignment", {}).get("output_path")
        if aligned_candidate and Path(aligned_candidate).exists():
            logger.info(
                f"Aligned data detected, using: {aligned_candidate}"
            )
            raw_path = aligned_candidate
        df = preprocessor.load_data(raw_path)

        # Preparation
        test_size = self.config.get("data_preprocessing", {}).get("test_size", 0.2)
        random_state = self.config.get("data_preprocessing", {}).get("random_state", 42)

        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            df, test_size=test_size, random_state=random_state
        )

        # Save
        preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
        logger.info(
            f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test"
        )

        return X_train, X_test, y_train, y_test

    def step_3_train_models(self, X_train, X_test, y_train, y_test):
        """Step 3: Train models"""
        logger.info("=== STEP 3: Model training ===")

        # MLflow experiment name
        experiment_name = self.config.get("mlflow", {}).get(
            "experiment_name", "wind_prediction"
        )
        tracking_uri = self.config.get("mlflow", {}).get("tracking_uri")
        trainer = ModelTrainer(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        # Train all models
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)

        # Show results
        trainer.print_comparison()

        # Save results
        trainer.save_results()

        # Best model
        best_name, best_model = trainer.get_best_model()
        logger.info(f"Best model: {best_name}")

        return trainer, results

    def run_full_pipeline(self, force_regenerate_data: bool = False):
        """Run the full pipeline"""
        logger.info("STARTING ML PIPELINE")
        logger.info("=" * 50)

        try:
            # Step 1: Data generation
            self.step_1_generate_data(force_regenerate=force_regenerate_data)

            # Step 2: Data preparation
            X_train, X_test, y_train, y_test = self.step_2_preprocess_data()

            # Step 3: Model training
            trainer, results = self.step_3_train_models(
                X_train, X_test, y_train, y_test
            )

            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)

            # Summary
            print("\nPIPELINE SUMMARY:")
            print(f"Data: {len(X_train) + len(X_test)} total points")
            print(f"Features: {len(X_train.columns)} variables")
            print(f"Trained models: {len(results)}")

            best_name, _ = trainer.get_best_model()
            best_rmse = trainer.results[best_name]["test_metrics"]["rmse"]
            print(f"Best model: {best_name} (RMSE: {best_rmse:.3f})")

            print("\nCheck MLflow UI: mlflow ui")
            print("Models saved in: models/")

            return trainer

        except Exception as e:
            logger.error(f"PIPELINE ERROR: {e}")
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ML pipeline for local wind speed prediction"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/params.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force data regeneration",
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Step to run (1=data, 2=preparation, 3=models, all=full)",
    )

    args = parser.parse_args()

    # Initialize the pipeline
    pipeline = WindMLPipeline(config_path=args.config)

    # Execute according to the requested step
    if args.step == "1":
        pipeline.step_1_generate_data(force_regenerate=args.regenerate_data)
    elif args.step == "2":
        pipeline.step_2_preprocess_data()
    elif args.step == "3":
        # Load preprocessed data
        import pandas as pd

        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]

        pipeline.step_3_train_models(X_train, X_test, y_train, y_test)
    else:
        pipeline.run_full_pipeline(force_regenerate_data=args.regenerate_data)


if __name__ == "__main__":
    main()
