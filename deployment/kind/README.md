# Kind Deployment

This artefact uses Kind Kubernetes only for deployment evidence. It deploys the Flask prediction API image `mlops-flask-api:latest`, which serves the model at `models/wine_quality_classifier.joblib`.

Run from the repository root:

```bash
docker build -t mlops-flask-api:latest .
bash scripts/create_kind_cluster.sh
kind load docker-image mlops-flask-api:latest --name mlops-kind
kubectl apply -f deployment/kind/
kubectl rollout status deployment/mlops-flask-api --timeout=180s
kubectl port-forward service/mlops-flask-api 8080:80
```

In a second terminal:

```bash
bash scripts/smoke_test_api.sh http://127.0.0.1:8080
```

Windows PowerShell equivalent:

```powershell
docker build -t mlops-flask-api:latest .
powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1 -ClusterName mlops-kind -NodeImage kindest/node:v1.30.2
kind load docker-image mlops-flask-api:latest --name mlops-kind
kubectl apply -f deployment/kind/
kubectl rollout status deployment/mlops-flask-api --timeout=180s
kubectl port-forward service/mlops-flask-api 8080:80
```

In a second PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:8080
```

Expected evidence:

- `/health` returns `status=healthy`, `model_loaded=true`, `model_version`, and `feature_count=11`.
- `/predict` returns a `low`, `medium`, or `high` prediction, class probabilities, and `model_version`.

The smoke test uses the UCI Wine Quality feature schema: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, and `alcohol`.
