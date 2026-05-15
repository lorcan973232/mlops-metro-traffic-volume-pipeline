# Kind Deployment

This artefact uses Kind Kubernetes only for deployment evidence. It deploys the Flask
prediction image `mlops-flask-api:latest`, which serves the UCI red wine quality
classifier at `models/wine_quality_classifier.joblib`.

Run from the repository root:

```bash
scripts/create_kind_cluster.sh
scripts/deploy_kind.sh
kubectl port-forward service/mlops-flask-api 8080:80
scripts/smoke_test_api.sh http://127.0.0.1:8080
```

PowerShell equivalents:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1
powershell -ExecutionPolicy Bypass -File scripts/deploy_kind.ps1
kubectl port-forward service/mlops-flask-api 8080:80
powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:8080
```

Open the live-demo UI at:

```text
http://127.0.0.1:8080/
```

The smoke test uses the real Wine Quality feature schema: `fixed_acidity`,
`volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`,
`free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `ph`, `sulphates`,
and `alcohol`.
