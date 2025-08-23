# Wind Speed Prediction using Machine Learning

## MLOps Pipeline for Local Wind Forecast Correction

### Overview

End-to-end ML pipeline that **corrects coarse weather forecasts** with **local wind measurements**. Built to demonstrate reproducible ML with DVC and MLflow.

---

### Prerequisites

* Python 3.10–3.12 (repo uses 3.12)
* [uv](https://github.com/astral-sh/uv)
* [DVC](https://dvc.org/doc/install)
* [MLflow](https://mlflow.org/docs/latest/index.html)

---

### Project Structure

```
wind-ml-pipeline/
├─ src/wind_ml_project/
│  ├─ data_sources/
│  │  └─ weather_providers/
│  │     └─ open_meteo.py
│  ├─ alignment.py
│  ├─ data_generator.py
│  ├─ data_preprocessing.py
│  ├─ model_training.py
│  └─ pipeline.py
├─ scripts/
│  ├─ generate_data.py
│  └─ aggregate_hourly.py
├─ configs/params.yaml
├─ tests/
├─ dvc.yaml
├─ dvc.lock
├─ pyproject.toml
└─ README.md
```

---

### Quickstart

1. **Install dependencies**

```bash
uv sync
```

2. **Run the pipeline (DVC)**

```bash
dvc repro
```

This triggers: `generate_data → fetch_forecast → aggregate_data → align_data → prepare_data → train_models`.

3. **MLflow UI**

```bash
uv run mlflow ui
# open http://localhost:5000
```

4. **Tests**

```bash
uv run pytest
```

---

### Useful CLI snippets

**Fetch Open-Meteo forecast as normalized CSV:**

```bash
PYTHONPATH=src uv run python -m wind_ml_project.data_sources.weather_providers.open_meteo \
  --lat 49.223 --lon 18.739 --hours 48 --out data/raw/forecast_openmeteo.csv
```

**Align forecast and measurements:**

```bash
PYTHONPATH=src uv run python -m wind_ml_project.alignment \
  --forecast data/raw/forecast_openmeteo.csv \
  --measures data/raw/wind_data.csv \
  --out data/processed/aligned.csv
```

**Predict example:**

```bash
uv run python predict_example.py --forecast_speed 10 \
  --temperature 20 --humidity 60 --pressure 1015 \
  --hour 14 --day_of_week 2 --month 7
```

---

### Features

* Multi-model training: Linear, Ridge, Random Forest
* Feature engineering: temporal features, lags, rolling window
* Reproducibility: DVC stages and locks
* Experiment tracking: MLflow params, metrics, artifacts

---

### Roadmap

* Phase 1 done: synthetic data, pipeline, tracking, tests
* Phase 2: real data integration and retraining schedule
* Phase 3: optional service (FastAPI), UI (Streamlit), Docker, monitoring

---

### Notes

* Artifacts and model outputs are generated at run time.
* They are not committed by default to keep the repo clean.
* Configure a DVC remote if you want to share them.

---

### License

MIT. See [LICENSE](LICENSE).
