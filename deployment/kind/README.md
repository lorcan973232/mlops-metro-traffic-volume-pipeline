# Kind Deployment

This artefact uses Kind Kubernetes only for deployment evidence. It deploys the Flask
prediction image `mlops-flask-api:latest`, which serves the UCI Energy Efficiency
heating-load model at `models/energy_efficiency_heating_load_regressor.joblib`.

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

The smoke test uses the real Energy Efficiency feature schema:
`relative_compactness`, `surface_area`, `wall_area`, `roof_area`,
`overall_height`, `orientation`, `glazing_area`, and
`glazing_area_distribution`.
