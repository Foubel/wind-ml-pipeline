#!/usr/bin/env python3
"""
Simple test script to validate the wind ML pipeline.
"""


def test_data_generation(tmp_path, monkeypatch):
    """Test data generation (use tmp_path)."""
    print("Test 1: Data generation...")

    from wind_ml_project.data_generator import WindDataGenerator

    # Isolate all writes to the temporary directory
    monkeypatch.chdir(tmp_path)

    generator = WindDataGenerator(seed=42)

    # Generate
    print("  - Generating wind data...")
    df = generator.generate_dataset(days=7)
    print(f"    {len(df)} points generated")
    print(f"    Columns: {list(df.columns)}")

    # Basic checks
    assert len(df) == 168  # 7 days * 24 hours
    assert "wind_speed" in df.columns
    assert "forecast_speed" in df.columns
    assert "temperature" in df.columns
    assert df["wind_speed"].min() >= 0
    print("    Validations passed")

    # Save in temporary directory
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "test_data.csv"
    generator.save_dataset(df, str(raw_file))

    assert raw_file.exists()
    return True


def test_preprocessing(tmp_path, monkeypatch):
    """Test data preparation (use tmp_path)."""
    print("\nTest 2: Data preparation...")

    from wind_ml_project.data_generator import WindDataGenerator
    from wind_ml_project.data_preprocessing import DataPreprocessor

    monkeypatch.chdir(tmp_path)

    # Generate raw data dedicated to this test
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "test_data.csv"
    df_raw = WindDataGenerator(seed=42).generate_dataset(days=7)
    WindDataGenerator(seed=42).save_dataset(df_raw, str(raw_file))

    preprocessor = DataPreprocessor(artifacts_dir=str(tmp_path / "artifacts"))

    # Prepare
    print("  - Preparing data...")
    df = preprocessor.load_data(str(raw_file))
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, test_size=0.3)
    print(
        f"    Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}"
    )

    # Checks
    assert X_train.shape[1] > 0
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    print("    Validations passed")

    # Save prepared data in tmp_path
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_processed_data(
        X_train, X_test, y_train, y_test, output_dir=str(processed_dir)
    )

    assert (processed_dir / "X_train.csv").exists()
    return True


def test_model_training(tmp_path, monkeypatch):
    """Test simple model training (use tmp_path)."""
    print("\nTest 3: Train a simple model...")

    import pandas as pd
    from wind_ml_project.data_generator import WindDataGenerator
    from wind_ml_project.data_preprocessing import DataPreprocessor
    from wind_ml_project.model_training import ModelTrainer

    monkeypatch.chdir(tmp_path)

    # Prepare temporary training data
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "train_data.csv"
    df_raw = WindDataGenerator(seed=42).generate_dataset(days=7)
    WindDataGenerator(seed=42).save_dataset(df_raw, str(raw_file))

    preprocessor = DataPreprocessor(artifacts_dir=str(tmp_path / "artifacts"))
    df = preprocessor.load_data(str(raw_file))
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, test_size=0.3)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_processed_data(
        X_train, X_test, y_train, y_test, output_dir=str(processed_dir)
    )

    # Ensure y are Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    # Ensure y_train and y_test are Series in the expected format
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Use a local MLflow tracking at tmp_path
    trainer = ModelTrainer(
        experiment_name="test_wind_prediction",
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )

    print("  - Training a linear regression...")
    results = trainer.train_model(
        "test_linear", model, X_train, y_train, X_test, y_test
    )

    # Checks
    assert "rmse" in results
    assert "r2" in results
    assert results["rmse"] >= 0
    assert results["r2"] <= 1
    print(f"    RMSE: {results['rmse']:.3f}")
    print(f"    R²: {results['r2']:.3f}")
    print("    Validations passed")

    return True


def test_full_pipeline(tmp_path, monkeypatch):
    """Test the full pipeline (use tmp_path and custom config)."""
    print("\nTest 4: Full pipeline...")

    import yaml
    from wind_ml_project import data_preprocessing as dp_module
    from wind_ml_project import model_training as mt_module
    from wind_ml_project.pipeline import WindMLPipeline

    # Isolate the CWD and all relative paths
    monkeypatch.chdir(tmp_path)

    # Temporary configuration adapted for the pipeline
    cfg = {
        "data": {"raw_path": str(tmp_path / "data" / "raw" / "wind_data.csv")},
        "data_preprocessing": {
            "test_size": 0.2,
            "random_state": 42,
            "target_mode": "wind_speed",
            "temporal_split": True,
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "mlflow": {
            "tracking_uri": f"file:{tmp_path / 'mlruns'}",
            "experiment_name": "wind_prediction_test",
        },
        "alignment": {
            "output_path": str(tmp_path / "data" / "processed" / "aligned.csv")
        },
    }
    cfg_path = tmp_path / "params.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Monkeypatch to save prepared data in tmp_path
    original_save_processed = dp_module.DataPreprocessor.save_processed_data

    def _save_processed(
        self, X_train, X_test, y_train, y_test, output_dir="data/processed"
    ):
        return original_save_processed(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
            output_dir=str(tmp_path / "data" / "processed"),
        )

    monkeypatch.setattr(
        dp_module.DataPreprocessor, "save_processed_data", _save_processed
    )

    # Monkeypatch to save model results in tmp_path
    original_save_results = mt_module.ModelTrainer.save_results

    def _save_results(self, filepath: str = "results/model_comparison.yaml"):
        return original_save_results(
            self, filepath=str(tmp_path / "results" / "model_comparison.yaml")
        )

    monkeypatch.setattr(mt_module.ModelTrainer, "save_results", _save_results)

    # Instantiate pipeline
    pipeline = WindMLPipeline(config_path=str(cfg_path))

    print("  - Step 1: Generation...")
    pipeline.step_1_generate_data(force_regenerate=True)

    print("  - Step 2: Preparation...")
    X_train, X_test, y_train, y_test = pipeline.step_2_preprocess_data()

    print("  - Step 3: Training...")
    trainer, results = pipeline.step_3_train_models(X_train, X_test, y_train, y_test)

    # Checks
    assert len(results) > 0
    assert len(trainer.results) > 0
    print(f"    {len(results)} models trained")
    print("    Full pipeline validated")

    return True


def test_data_quality(tmp_path, monkeypatch):
    """Test data quality (use tmp_path)."""
    print("\nTest 5: Data quality...")

    import pandas as pd
    from wind_ml_project.data_generator import WindDataGenerator

    monkeypatch.chdir(tmp_path)

    # Generate dedicated data file
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "wind_data.csv"
    df_src = WindDataGenerator(seed=42).generate_dataset(days=7)
    WindDataGenerator(seed=42).save_dataset(df_src, str(raw_file))

    # Load generated data
    df = pd.read_csv(str(raw_file))

    print("  - Checking structure...")
    expected_columns = [
        "timestamp",
        "wind_speed",
        "forecast_speed",
        "temperature",
        "humidity",
        "pressure",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    print("  - Checking values...")
    # Consistency checks
    assert df["wind_speed"].min() >= 0, "Negative wind speed"
    assert df["forecast_speed"].min() >= 0, "Negative forecast speed"
    assert df["humidity"].min() >= 0 and df["humidity"].max() <= 100, "Invalid humidity"
    assert (
        df["pressure"].min() > 900 and df["pressure"].max() < 1100
    ), "Invalid pressure"

    # Missing value check
    assert not df.isnull().any().any(), "Missing values detected"

    print("    Structure validated")
    print("    Values coherent")
    print("    No missing values")

    return True


def test_model_performance(tmp_path, monkeypatch):
    """Test model performance (self-contained, use tmp_path)."""
    print("\nTest 6: Model performance...")

    import yaml
    from wind_ml_project.data_generator import WindDataGenerator
    from wind_ml_project.data_preprocessing import DataPreprocessor
    from wind_ml_project.model_training import ModelTrainer

    monkeypatch.chdir(tmp_path)

    # Prepare a small dataset and train models
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "perf_data.csv"
    df_raw = WindDataGenerator(seed=42).generate_dataset(days=10)
    WindDataGenerator(seed=42).save_dataset(df_raw, str(raw_file))

    preprocessor = DataPreprocessor(artifacts_dir=str(tmp_path / "artifacts"))
    df = preprocessor.load_data(str(raw_file))
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, test_size=0.2)

    trainer = ModelTrainer(
        experiment_name="perf_test",
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    trainer.train_all_models(X_train, y_train, X_test, y_test)

    results_path = tmp_path / "results" / "model_comparison.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_results(filepath=str(results_path))

    # Load results from the temp path
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    print("  - Checking metrics...")
    for model_name, metrics in results.items():
        test_metrics = metrics["test_metrics"]

        # Basic checks
        assert "rmse" in test_metrics
        assert "r2" in test_metrics
        assert test_metrics["rmse"] >= 0
        assert test_metrics["r2"] <= 1

        print(
            f"    {model_name}: RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.3f}"
        )

    # Ensure at least one model has acceptable performance (relaxed threshold for synthetic data)
    best_r2 = max(results[model]["test_metrics"]["r2"] for model in results)
    assert best_r2 > 0.3, f"Best R² too low: {best_r2}"

    print("    Acceptable performance validated")

    return True


def main():
    """Run all tests."""
    print("COMPLETE TESTS FOR WIND ML PIPELINE")
    print("=" * 50)

    tests = [
        ("Data generation", test_data_generation),
        ("Data preparation", test_preprocessing),
        ("Model training", test_model_training),
        ("Full pipeline", test_full_pipeline),
        ("Data quality", test_data_quality),
        ("Model performance", test_model_performance),
    ]

    passed = 0
    total = len(tests)

    try:
        for test_name, test_func in tests:
            try:
                test_func()
                passed += 1
                print(f"    {test_name}: PASSED")
            except Exception as e:
                print(f"    {test_name}: FAILED - {e}")

        print("\n" + "=" * 50)
        print(f"RESULTS: {passed}/{total} tests passed")

        if passed == total:
            print("ALL TESTS PASSED!")
            print("Your ML pipeline is fully validated!")
            print("=" * 50)

            print("\nYOUR PORTFOLIO IS READY:")
            print("1. End-to-end ML pipeline working")
            print("2. Realistic synthetic data")
            print("3. ML models with strong performance")
            print("4. MLflow tracking integrated")
            print("5. Comprehensive automated tests")

            print("\nUSEFUL COMMANDS:")
            print("• Full pipeline: python src/wind_ml_project/pipeline.py")
            print("• MLflow UI: mlflow ui")
            print("• Tests: python test_pipeline.py")

        else:
            print(f"{total - passed} tests failed")
            print("Check errors above")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    return passed == total


if __name__ == "__main__":
    main()
