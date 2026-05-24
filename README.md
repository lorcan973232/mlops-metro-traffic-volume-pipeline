# MLOps Metro Traffic Volume Predictor

## Project Summary

This is a Metro traffic MLOps project. It predicts whether hourly traffic volume is
normal or high using time, weather, holiday, and recent traffic-volume inputs. The
project includes data processing, model training, testing, a Flask API, a browser
page, Docker, Kind Kubernetes, GitHub Actions, monitoring checks, and saved reports.

The model gives useful results for this project, but it is not a perfect traffic
forecasting system. The main point is to show the full MLOps workflow around the
model.

## What This Project Shows

- Data ingestion and preprocessing.
- Model training, model selection, and evaluation with scikit-learn.
- A Flask API for predictions.
- A browser page for the live demo.
- A Docker container for the Flask app and saved model.
- A Kind Kubernetes deployment for a local cluster demo.
- GitHub Actions for CI, training, Docker, deployment, monitoring, and security checks.
- Continuous Training with a quality gate.
- Monitoring and drift checks.
- Tests and saved reports.

## Quick Start

1. Set up the environment.

   PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1
   .\.venv\Scripts\Activate.ps1
   ```

   Bash or Git Bash:

   ```bash
   bash scripts/setup_local.sh
   source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
   ```

2. Run the pipeline.

   PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
   ```

   Bash:

   ```bash
   bash scripts/run_pipeline.sh
   ```

3. Start the Flask app.

   ```powershell
   python -m app.main
   ```

4. Open the browser page.

   ```text
   http://127.0.0.1:5000/
   ```

5. Test the API from another terminal.

   PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:5000
   ```

   Bash:

   ```bash
   bash scripts/smoke_test_api.sh http://127.0.0.1:5000
   ```

## Project Structure

```text
.
├── app/                    # Flask API, browser page, schemas, and dashboard routes
├── data/raw/               # UCI Metro traffic CSV gzip used by the pipeline
├── data/processed/         # Processed CSV created by preprocessing
├── deployment/kind/        # Local Kubernetes manifests for Kind
├── models/                 # Saved scikit-learn model bundle
├── reports/                # Metrics, monitoring, security, and analysis reports
├── scripts/                # Setup, smoke-test, monitoring, Docker, and Kind helpers
├── src/                    # Data, preprocessing, training, evaluation, and registry code
├── tests/                  # Unit, integration, API, workflow, and report tests
├── .github/workflows/      # GitHub Actions workflows
├── Dockerfile              # Flask API image
├── Makefile                # Short local command targets
└── README.md
```

## Dataset and Prediction Task

| Item | Value |
|---|---|
| Dataset | UCI Metro Interstate Traffic Volume |
| Source | <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume> |
| Raw file | `data/raw/Metro_Interstate_Traffic_Volume.csv.gz` |
| Raw SHA-256 | `0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07` |
| Source target | `traffic_volume` |
| Model target | `high_traffic` |
| Task type | Classification |
| Positive class | `high traffic`, where `traffic_volume >= 3800` |
| Negative class | `normal traffic` |

The model uses these features:

```text
temp, rain_1h, snow_1h, clouds_all, hour, month, day_of_week,
is_weekend, is_holiday, weather_main, lag_1h_volume, lag_24h_volume,
lag_168h_volume, rolling_3h_volume, rolling_24h_volume
```

The current `traffic_volume` value is not used as a model feature. Lag and rolling
features use earlier traffic observations only. This makes the dataset suitable for
the project because it has real public data, a clear target, enough rows for tests,
and a useful mix of weather, time, holiday, and recent traffic features.

## Model Training and Results

Training builds a scikit-learn pipeline with `StandardScaler`, `OneHotEncoder`, and
a `HistGradientBoostingClassifier`. Model selection can use `RandomizedSearchCV`.
The final test split is kept separate from model selection.

| Metric | Value |
|---|---:|
| Accuracy | 0.979 |
| Balanced accuracy | 0.978 |
| Macro F1 | 0.979 |
| Weighted F1 | 0.979 |
| ROC AUC | 0.998 |
| Baseline accuracy | 0.541 |
| Quality gate result | Passed |

Full metric values are saved in:

- `reports/metrics/latest_metrics.json`
- `reports/metrics/baseline_metrics.json`
- `reports/metrics/quality_gate_report.json`
- `reports/metrics/model_metadata.json`

## Flask API and Web Page

Start the app with:

```powershell
python -m app.main
```

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Opens the browser prediction page |
| `/health` | GET | Checks that the API and model are available |
| `/predict` | POST | Returns a traffic-level prediction, label, probabilities, confidence, and model version |
| `/dashboard/` | GET | Shows optional charts from saved reports |

The browser page calls the real `/predict` route. It is not a separate mock demo.

## Running Tests

```powershell
python -m compileall app src tests scripts
pytest -q
ruff check src tests scripts
```

The tests check data loading, preprocessing, model reports, API responses,
monitoring outputs, workflow files, and the browser form schema.

## Docker

Docker is used so the Flask app and saved model can run in the same way on another
machine.

```powershell
docker build -t mlops-flask-api:latest .
docker run --rm -d --name mlops-flask-demo -p 5001:5000 mlops-flask-api:latest
powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:5001
docker stop mlops-flask-demo
```

The image uses `MODEL_PATH=models/traffic_volume_classifier.joblib`, serves the
Flask app with Gunicorn, and exposes `/health`.

## Kind Kubernetes

Kind is used to run Kubernetes locally. This lets the project show a Kubernetes
deployment without needing a cloud VM.

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1
powershell -ExecutionPolicy Bypass -File scripts/deploy_kind.ps1
kubectl port-forward service/mlops-flask-api 8080:80
powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:8080
python scripts/monitor.py --api-url http://127.0.0.1:8080
```

Bash:

```bash
bash scripts/create_kind_cluster.sh
bash scripts/deploy_kind.sh
kubectl port-forward service/mlops-flask-api 8080:80
bash scripts/smoke_test_api.sh http://127.0.0.1:8080
python scripts/monitor.py --api-url http://127.0.0.1:8080
```

If Docker, Kind, or kubectl are not ready locally, run:

```powershell
python scripts/check_deployment_readiness.py
```

If local tools are missing, the readiness script can report
`github_actions_current_sha` when the current commit has matching Docker and Kind
workflow checks.

## CI/CD with GitHub Actions

The workflows run the same kind of checks as the local commands, but on GitHub
runners.

| Workflow file | Display name | What it checks |
|---|---|---|
| `.github/workflows/ci.yml` | CI | Setup, lint, tests, Flask import, ML smoke path, and monitoring scripts |
| `.github/workflows/data-preprocessing.yml` | Data Preprocessing | Data ingestion and deterministic preprocessing |
| `.github/workflows/train-and-evaluate.yml` | Train and Evaluate | Model selection, training, evaluation, registry metadata, explainability, proxy checks, and cost-benefit reports |
| `.github/workflows/continuous-training.yml` | Continuous Training | Retrains the model and accepts it only if the quality gate passes |
| `.github/workflows/docker-build.yml` | Docker Build | Image build, container run, API smoke test, and latency benchmark |
| `.github/workflows/deploy.yml` | Deploy Kind | Kind cluster deployment, rollout, port-forward, and API smoke test |
| `.github/workflows/monitoring.yml` | Monitoring | Data-quality checks, drift checks, and retraining flags |
| `.github/workflows/model-analysis.yml` | Model Analysis | SHAP or permutation explainability, proxy subgroup checks, cost-benefit example, and monitoring reports |
| `.github/workflows/security-scan.yml` | Security Scan | Secret scan summary, dependency scan, Docker checks, image scan output, and SBOM |
| `.github/workflows/bash-script-verification.yml` | Bash Script Verification | Bash setup and API smoke scripts on Ubuntu |
| `.github/workflows/final-readiness.yml` | Final Readiness | Compile, tests, lint, workflow checks, security summary, and current report generation |

## Continuous Training

Continuous Training reruns the data, model-selection, training, evaluation,
explainability, proxy subgroup, and cost-benefit steps. The new model is only
accepted if it passes the quality gate.

Useful files:

- `.github/workflows/continuous-training.yml`
- `reports/metrics/latest_metrics.json`
- `reports/metrics/quality_gate_report.json`
- `reports/metrics/model_metadata.json`
- `reports/model_registry/version_history.json`

## Monitoring and Drift Checks

Monitoring is lightweight. It checks the data schema, missing values, feature
summaries, and drift signals. It can also call the deployed API when a URL is
passed in.

Offline monitoring:

```powershell
python scripts/monitor.py
python scripts/check_drift.py
```

API-aware monitoring:

```powershell
python scripts/monitor.py --api-url http://127.0.0.1:8080
```

Monitoring is simulated unless an API URL is passed in. The drift check can look
at traffic-related inputs such as weather, time fields, holiday indicators, and
recent traffic-volume distributions. It also creates a deterministic shifted batch
so the retraining signal can be checked.

Reports are saved in:

- `reports/monitoring/monitoring_report.json`
- `reports/monitoring/data_quality_report.json`
- `reports/monitoring/drift_report.json`
- `reports/monitoring/api_monitoring_report.json`

## Extra Evidence

### Feature Importance and SHAP

Explainability reports are saved in `reports/explainability/`. SHAP is used when
it is available. If SHAP cannot run in the environment, the script uses permutation
importance and labels that fallback clearly.

### Fairness Proxy Check

Fairness reports are saved in `reports/fairness/`. The dataset does not include
personal protected attributes, so the subgroup check is based on non-sensitive
proxy groups from the traffic data, such as time bands and weather. It is not a
demographic fairness audit.

### Cost-Benefit Example

Cost-benefit reports are saved in `reports/business/`. The cost values are made-up
examples, not real business values.

### Security Checks

Security reports are saved in `reports/security/`. The checks include a local
secret scan summary, dependency scan output, Docker runtime user check, image scan
output, and SBOM.

## Troubleshooting

| Problem | Quick fix |
|---|---|
| Model file not found | Run `python -m src.train`, or start `python -m app.main` so the app can train the model if the file is missing |
| Flask app will not start | Run `python -m compileall app src tests scripts` and check that `requirements.txt` has been installed |
| Docker is not running | Start Docker Desktop, then run `docker info` |
| Kind cannot deploy | Run `python scripts/check_deployment_readiness.py` to check Docker, Kind, and kubectl |
| API smoke test fails | Check `/health` first, then confirm the smoke test URL matches the port you exposed |

## Traceability

| Project need | Where it is shown | How to check it |
|---|---|---|
| Model training and testing | `src/train.py`, `src/evaluate.py`, `reports/metrics/latest_metrics.json` | Run `python -m src.train` and `python -m src.evaluate --fail-on-rejection` |
| Flask prediction API | `app/`, `scripts/smoke_test_api.ps1`, `scripts/smoke_test_api.sh` | Start Flask and run a smoke test |
| Docker container | `Dockerfile`, `.github/workflows/docker-build.yml` | Build and run the image |
| Kind deployment | `deployment/kind/`, `.github/workflows/deploy.yml` | Run the Kind scripts or inspect the deploy workflow |
| GitHub Actions | `.github/workflows/` | Check the Actions tab for the commit |
| Continuous Training | `.github/workflows/continuous-training.yml` | Run the workflow manually or inspect its latest run |
| Monitoring | `scripts/monitor.py`, `scripts/check_drift.py`, `reports/monitoring/` | Run the monitoring commands |
| Tests | `tests/` | Run `pytest -q` |

## Demo Steps

These steps give a quick way to show the project working.

1. Show the repository and current commit SHA.
2. Show `.github/workflows/` and the latest relevant workflow runs.
3. Open `reports/metrics/latest_metrics.json`, `reports/metrics/quality_gate_report.json`, and `reports/metrics/model_metadata.json`.
4. Run `python -m compileall app src tests scripts`.
5. Run `pytest -q`.
6. Start Flask with `python -m app.main`.
7. Open `http://127.0.0.1:5000/`.
8. Use the example values on the page and make a prediction.
9. Run the API smoke test against local Flask.
10. Show Docker or Kind if the local machine has the required tools.
11. Show `reports/monitoring/drift_report.json`, `reports/fairness/fairness_report.json`, and `reports/security/security_scan_summary.md`.

## Limitations

This is a student project, not a live traffic-control system.

Monitoring is simulated unless an API URL is passed in.

If subgroup checks are included, they use proxy groups from traffic or weather
features, not protected personal attributes.

The cost-benefit values are made-up examples, not real business values.

Kind is used for a local Kubernetes demo. It is not a public cloud service.
